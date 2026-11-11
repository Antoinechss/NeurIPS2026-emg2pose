"""VEMG2Pose+ : VEMG2Pose trained for cross-user generalization (submitted model).

Training record:
    n_train_subjects = 158
    training time    = 3.2 h (191.6 min, 25 000 steps) on 1x RTX 6000 Ada 48 GB (RunPod),
                       + ~25 min data loading
    calibration      = ~128 steps/min on that GPU, same speed at 8 and 158 users
    command          = benchopt run <benchmark> -d "EMG2Pose[n_train_subjects=158,num_workers=12]" \
                           -s src/vemg2pose_plus.py

Changes vs. the VEMG2Pose baseline (same braindecode backbone, same left-context padding):
- Augmentations aimed at new users: band rotation (+-1 electrode, the band is a ring of 16),
  per-channel and global gain jitter, small additive noise.
- AdamW + linear warmup + cosine decay over a fixed step budget, grad clipping.
- Mixed precision on the convolutional encoder only; the pose rollout stays fp32 (it
  integrates velocities over 250 steps, fp16 would drift).
- EMA of the weights; the EMA weights are the ones saved and evaluated.
- Optional ensemble of `n_models` seeds, and test-time averaging over band rotations,
  batched into a single forward pass so it costs far less than k x the inference time.

Everything is deterministic given `seed`: `fit` reproduces the shipped weights.
"""

import copy
import math
import time

import torch
from braindecode.models import VEMG2Pose
from torch import nn
from torch.nn import functional as F

from benchmark_utils.base_solver import CompetSolver

WEIGHTS = "weights.pt"
LEFT_CONTEXT = 1790  # samples VEMG2Pose consumes before its first prediction


class AmpEncoder(nn.Module):
    """Run the wrapped encoder under fp16 autocast on CUDA, return fp32."""

    def __init__(self, encoder):
        super().__init__()
        self.inner = encoder

    def forward(self, x):
        if x.is_cuda:
            with torch.autocast("cuda", dtype=torch.float16):
                return self.inner(x).float()
        return self.inner(x)


class PoseNet(nn.Module):
    def __init__(self, n_chans, n_joints, n_times, sfreq, drop_prob=0.0):
        super().__init__()
        self.net = VEMG2Pose(n_chans=n_chans, n_outputs=n_joints, sfreq=sfreq,
                             n_times=n_times + LEFT_CONTEXT, drop_prob=drop_prob)
        self.net.encoder = AmpEncoder(self.net.encoder)

    def forward(self, X):
        mode = "reflect" if X.shape[-1] > LEFT_CONTEXT else "replicate"
        X = F.pad(X.float(), (LEFT_CONTEXT, 0), mode=mode)
        return self.net(X).transpose(1, 2)  # (B, J, T)


class Ensemble(nn.Module):
    """Average of PoseNets, with test-time averaging over band rotations."""

    def __init__(self, members, tta_shifts=(0,)):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.tta_shifts = tuple(tta_shifts)

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        X = X.float()
        B = X.shape[0]
        # All rotations in one batch: the rollout is launch-bound, so a bigger
        # batch is nearly free compared with k sequential passes.
        Xs = torch.cat([torch.roll(X, s, dims=1) for s in self.tta_shifts])
        out = 0.0
        for m in self.members:
            out = out + m(Xs).view(len(self.tta_shifts), B, -1,
                                   Xs.shape[-1]).mean(0)
        return out / len(self.members)


# ---------------------------------------------------------------- augmentation

def augment(X, g, max_shift=1, p_shift=0.5, chan_gain=0.15, glob_gain=0.25,
            noise=0.05):
    """Batch augmentation on X's device. X: (B, C, T); `g` a seeded Generator
    on the same device."""
    B, C, _ = X.shape
    dev = X.device
    if max_shift > 0:
        shifts = torch.randint(-max_shift, max_shift + 1, (B,), generator=g,
                               device=dev)
        keep = torch.rand(B, generator=g, device=dev) < p_shift
        shifts = torch.where(keep, shifts, torch.zeros_like(shifts))
        idx = (torch.arange(C, device=dev)[None, :] - shifts[:, None]) % C
        X = torch.gather(X, 1, idx[:, :, None].expand_as(X))
    gain = torch.exp(chan_gain * torch.randn(B, C, 1, generator=g, device=dev)
                     + glob_gain * torch.randn(B, 1, 1, generator=g, device=dev))
    X = X * gain
    if noise > 0:
        std = X.std(dim=-1, keepdim=True)
        X = X + noise * std * torch.randn(X.shape, generator=g, device=dev)
    return X


# -------------------------------------------------------------------- training

def train_member(model, loader, *, seed, lr, weight_decay, max_steps,
                 warmup_steps, ema_decay, max_hours, aug):
    torch.manual_seed(seed)
    dev = next(model.parameters()).device
    g = torch.Generator(device=dev).manual_seed(seed + 1)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    params = list(model.parameters())
    ema_params = list(ema.parameters())

    cuda = next(model.parameters()).is_cuda
    if cuda:  # compile the convolutional encoder only (see the baseline)
        model.net.encoder.inner = torch.compile(model.net.encoder.inner)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)

    def lr_at(step):
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    start = time.time()
    deadline = start + 3600 * max_hours if max_hours else None
    step, epoch = 0, 0
    model.train()
    while step < max_steps:
        total, n = 0.0, 0
        for X, y, _info in loader:
            for group in opt.param_groups:
                group["lr"] = lr_at(step)
            X = augment(X, g, **aug)
            labelled = ~torch.isnan(y)
            loss = (model(X) - torch.nan_to_num(y)).abs()[labelled].mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            with torch.no_grad():
                d = min(ema_decay, (1 + step) / (10 + step))
                torch._foreach_lerp_(ema_params, [p.detach() for p in params],
                                     1 - d)
            total, n, step = total + loss.detach(), n + 1, step + 1
            if step % 500 == 0:
                print(f"[seed {seed}] step {step}/{max_steps} "
                      f"lr {lr_at(step):.2e} L1 {float(total) / n:.4f} rad "
                      f"{(time.time() - start) / 60:.1f} min", flush=True)
            if step >= max_steps or (deadline and time.time() > deadline):
                break
        epoch += 1
        print(f"[seed {seed}] epoch {epoch} done, step {step}, "
              f"L1 {float(total) / max(n, 1):.4f} rad, "
              f"{(time.time() - start) / 60:.1f} min", flush=True)
        if deadline and time.time() > deadline:
            print(f"[seed {seed}] stopped by the {max_hours} h budget", flush=True)
            break
    # Buffers (none expected in VEMG2Pose) follow the live model.
    for b_ema, b in zip(ema.buffers(), model.buffers()):
        b_ema.copy_(b)
    return ema


class Solver(CompetSolver):

    name = "vemg2pose_plus"

    requirements = ["pip::braindecode"]

    parameters = {
        "n_models": [1],
        "max_steps": [25000],  # ~3.3 h at ~128 steps/min on an RTX 6000 Ada
        "batch_size": [256],
        "lr": [1e-3],
        "weight_decay": [1e-2],
        "warmup_steps": [1000],
        "ema_decay": [0.999],
        "drop_prob": [0.1],
        "max_shift": [1],
        "tta": [True],
        "max_hours": [11.0],  # hard stop for the organisers' 12 h V100 retrain
        "seed": [0],
    }

    def _build(self, meta):
        return PoseNet(meta["n_chans"], meta["n_joints"], meta["n_times"],
                       meta["sfreq"], drop_prob=self.drop_prob)

    def load_model(self, meta):
        weights = meta["submission_dir"] / WEIGHTS
        n_models = self.n_models
        state = None
        if weights.exists():
            state = torch.load(weights, map_location="cpu")
            n_models = state["n_models"]
        shifts = (-1, 0, 1) if self.tta else (0,)
        model = Ensemble([self._build(meta) for _ in range(n_models)], shifts)
        if state is not None:
            model.load_state_dict(state["state_dict"])
        return model.to(meta["device"])

    def fit(self, model, train_set):
        loader = self.make_loader(train_set, batch_size=self.batch_size)
        hours_each = self.max_hours / len(model.members)
        aug = dict(max_shift=self.max_shift)
        trained = []
        for i, member in enumerate(model.members):
            trained.append(train_member(
                member, loader, seed=self.seed + 1000 * i, lr=self.lr,
                weight_decay=self.weight_decay, max_steps=self.max_steps,
                warmup_steps=self.warmup_steps, ema_decay=self.ema_decay,
                max_hours=hours_each, aug=aug))
        model.members = nn.ModuleList(trained)

    def save_model(self, model, path):
        torch.save({"n_models": len(model.members),
                    "state_dict": model.state_dict()}, path / WEIGHTS)
