"""PoseBiNet: an offline, bidirectional EMG-to-pose decoder (best local score, not submitted).

Training record:
    n_train_subjects = 158
    training time    = 2.5 h (148.7 min, 25 000 steps, both members side by side) on
                       1x RTX 6000 Ada 48 GB (RunPod), + ~25 min data loading
    calibration      = ~168 batches/min on that GPU, data-bound (the GPU waited for data
                       26% of the time)
    result           = 15.147 deg public (subject std 1.289); member 0 alone 15.454
    command          = benchopt run <benchmark> --config configs/final.yml

Why: the challenge scores whole 5-s windows offline, so the pose at time t may use the EMG
recorded after t too. The published emg2pose models are causal, built for streaming: they
only look at the past. This model looks both ways.
- Encoder: the VEMG2Pose TDS convolutional encoder (2 kHz -> 25 Hz features), with its
  1790-sample receptive field centred on t (895 reflected samples padded on each side)
  instead of ending at t.
- Decoder: a 2-layer bidirectional LSTM over the features resampled to 50 Hz, so every frame
  sees the whole window, then a linear read-out of the 20 absolute joint angles. No velocity
  integration, hence no drift. The read-out starts from the median training pose.
- Training: band rotation / gain / noise augmentation, AdamW (no decay on biases and
  norms), warmup + cosine, grad clipping, EMA weights, fp16 encoder, test-time averaging
  over band rotations.
- Ensemble of `n_models` members trained side by side: every loaded batch feeds each
  member (own initial weights, augmentation, optimizer, EMA), and their predictions are
  averaged. Loading the data is the bottleneck, so the second member costs little extra
  time. With the L1 metric, the average of two models is never worse than the mean
  error of the two.

Everything is seeded by `seed`: `fit` reproduces the shipped weights up to GPU noise.
"""

import copy
import math
import time

import torch
from braindecode.models import VEMG2Pose
from torch import nn
from torch.nn import functional as F

from benchmark_utils.base_solver import CompetSolver

WEIGHTS = "posebinet.pt"
CONTEXT = 1790  # receptive field of the VEMG2Pose encoder, in samples


class PoseBiNet(nn.Module):
    def __init__(self, n_chans, n_joints, sfreq, enc_channels=256, feature_dim=64,
                 hidden=256, layers=2, rate=50.0, drop_prob=0.1):
        super().__init__()
        # Borrow the encoder of VEMG2Pose; its causal LSTM decoder is dropped.
        self.encoder = VEMG2Pose(
            n_chans=n_chans, n_outputs=n_joints, sfreq=sfreq, n_times=2 * CONTEXT,
            encoder_channels=enc_channels, feature_dim=feature_dim,
            drop_prob=drop_prob).encoder
        self.sfreq, self.rate = float(sfreq), float(rate)
        self.lstm = nn.LSTM(feature_dim, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True,
                            dropout=drop_prob if layers > 1 else 0.0)
        self.head = nn.Linear(2 * hidden, n_joints)

    def forward(self, X):
        T = X.shape[-1]
        left = CONTEXT // 2
        mode = "reflect" if T > CONTEXT - left else "replicate"
        X = F.pad(X.float(), (left, CONTEXT - left), mode=mode)
        # Convolutions in fp16 (the bulk of the compute), the LSTM in fp32.
        with torch.autocast("cuda", dtype=torch.float16, enabled=X.is_cuda):
            feats = self.encoder(X)                       # (B, F, T / 80 + 1)
        with torch.autocast("cuda", enabled=False):
            n_frames = max(1, round(T / self.sfreq * self.rate))
            feats = F.interpolate(feats.float(), size=n_frames, mode="linear",
                                  align_corners=False)
            seq, _ = self.lstm(feats.transpose(1, 2))    # (B, n_frames, 2 * hidden)
            pose = self.head(seq).transpose(1, 2)         # (B, J, n_frames)
            return F.interpolate(pose, size=T, mode="linear", align_corners=False)


class Ensemble(nn.Module):
    """Average of PoseBiNets, with test-time averaging over band rotations."""

    def __init__(self, members, tta_shifts=(0,)):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.tta_shifts = tuple(tta_shifts)

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        X = X.float()
        B = X.shape[0]
        # All rotations in one batch: one big forward pass instead of k small ones.
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

def pose_loss(pred, y):
    """L1 on the joint angles over the labelled samples. Masked sums rather than
    boolean indexing: no GPU -> CPU sync per step."""
    mask = (~torch.isnan(y)).float()
    y = torch.nan_to_num(y)
    return ((pred - y).abs() * mask).sum() / mask.sum().clamp_min(1)


def train_members(members, loader, *, seed, lr, weight_decay, max_steps, warmup_steps,
                  ema_decay, max_hours, aug):
    """Train the members side by side: each batch is loaded once and used by every
    member, each with its own initial weights, augmentation, optimizer and EMA."""
    torch.manual_seed(seed)
    dev = next(members[0].parameters()).device
    cuda = dev.type == "cuda"

    # Start every read-out at the median pose of one batch (L1-optimal constant).
    _, y0, _ = next(iter(loader))
    with torch.no_grad():
        y0 = y0.transpose(0, 1).reshape(y0.shape[1], -1)
        median = torch.nanmedian(y0, dim=1).values
        for model in members:
            model.head.bias.copy_(median)
    del y0

    runs = []
    for i, model in enumerate(members):
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
        params = list(model.parameters())
        # Weight decay on weight matrices and kernels only, not on biases and norms.
        opt = torch.optim.AdamW(
            [{"params": [p for p in params if p.ndim >= 2],
              "weight_decay": weight_decay},
             {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}],
            lr=lr)
        if cuda:  # compile the convolutional encoder only
            model.encoder = torch.compile(model.encoder)
        model.train()
        runs.append(dict(
            model=model, ema=ema, params=params, ema_params=list(ema.parameters()),
            opt=opt, scaler=torch.amp.GradScaler("cuda", enabled=cuda),
            g=torch.Generator(device=dev).manual_seed(seed + 1000 * i + 1)))

    def lr_at(step):
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    def losses(totals, n):  # mean training L1 of each member, in rad and degrees
        l1 = [float(t) / max(n, 1) for t in totals]
        return (" / ".join(f"{v:.4f}" for v in l1) + " ("
                + " / ".join(f"{math.degrees(v):.1f}" for v in l1) + " deg)")

    start = time.time()
    deadline = start + 3600 * max_hours if max_hours else None
    step, epoch, waited = 0, 0, 0.0
    while step < max_steps:
        totals, n = [0.0] * len(runs), 0
        batches = iter(loader)
        while step < max_steps:
            t0 = time.time()
            try:
                X, y, _info = next(batches)
            except StopIteration:
                break
            waited += time.time() - t0
            for k, r in enumerate(runs):
                for group in r["opt"].param_groups:
                    group["lr"] = lr_at(step)
                loss = pose_loss(r["model"](augment(X, r["g"], **aug)), y)
                r["opt"].zero_grad(set_to_none=True)
                r["scaler"].scale(loss).backward()
                r["scaler"].unscale_(r["opt"])
                nn.utils.clip_grad_norm_(r["params"], 1.0)
                r["scaler"].step(r["opt"])
                r["scaler"].update()
                with torch.no_grad():
                    d = min(ema_decay, (1 + step) / (10 + step))
                    torch._foreach_lerp_(r["ema_params"],
                                         [p.detach() for p in r["params"]], 1 - d)
                totals[k] = totals[k] + loss.detach()
            n, step = n + 1, step + 1
            if step % 500 == 0:
                elapsed = time.time() - start
                print(f"step {step}/{max_steps} lr {lr_at(step):.2e} "
                      f"loss {losses(totals, n)} {elapsed / 60:.1f} min, "
                      f"waiting for data {100 * waited / elapsed:.0f}%", flush=True)
            if deadline and time.time() > deadline:
                break
        epoch += 1
        print(f"epoch {epoch} done, step {step}, loss {losses(totals, n)}, "
              f"{(time.time() - start) / 60:.1f} min", flush=True)
        if deadline and time.time() > deadline:
            print(f"stopped by the {max_hours} h budget", flush=True)
            break
    emas = []
    for r in runs:
        for b_ema, b in zip(r["ema"].buffers(), r["model"].buffers()):
            b_ema.copy_(b)
        r["ema"].lstm.flatten_parameters()
        emas.append(r["ema"])
    return emas


class Solver(CompetSolver):

    name = "posebinet"

    requirements = ["pip::braindecode"]

    parameters = {
        "n_models": [2],  # members trained side by side, averaged at test time
        "max_steps": [25000],  # 2.5 h with 2 members on an RTX 6000 Ada (data-bound)
        "batch_size": [256],
        "lr": [1e-3],
        "weight_decay": [1e-2],
        "warmup_steps": [1000],
        "ema_decay": [0.999],
        "drop_prob": [0.1],
        "enc_channels": [256],
        "hidden": [256],
        "max_shift": [1],
        "tta": [True],
        "max_hours": [11.0],  # hard stop for the organisers' 12 h V100 retrain
        "seed": [0],
        "member": [-1],  # -1: average all members; i: re-evaluate member i alone
    }

    def _build(self, meta):
        return PoseBiNet(meta["n_chans"], meta["n_joints"], meta["sfreq"],
                         enc_channels=self.enc_channels, hidden=self.hidden,
                         drop_prob=self.drop_prob)

    def load_model(self, meta):
        weights = meta["submission_dir"] / WEIGHTS
        n_models = self.n_models
        state = None
        if weights.exists():
            state = torch.load(weights, map_location="cpu")
            n_models = state["n_models"]
        torch.manual_seed(self.seed)  # same initial weights on every run
        shifts = (-1, 0, 1) if self.tta else (0,)
        model = Ensemble([self._build(meta) for _ in range(n_models)], shifts)
        if state is not None:
            model.load_state_dict(state["state_dict"])
            if self.member >= 0:  # score one member alone (re-evaluation only)
                model.members = model.members[self.member:self.member + 1]
        model = model.to(meta["device"])
        for m in model.members:
            m.lstm.flatten_parameters()
        return model

    def fit(self, model, train_set):
        loader = self.make_loader(train_set, batch_size=self.batch_size)
        model.members = nn.ModuleList(train_members(
            list(model.members), loader, seed=self.seed, lr=self.lr,
            weight_decay=self.weight_decay, max_steps=self.max_steps,
            warmup_steps=self.warmup_steps, ema_decay=self.ema_decay,
            max_hours=self.max_hours, aug=dict(max_shift=self.max_shift)))

    def save_model(self, model, path):
        torch.save({"n_models": len(model.members),
                    "state_dict": model.state_dict()}, path / WEIGHTS)
