# EMG2Pose: hand pose from wrist EMG

## Overview

The task is to predict the 20 joint angles of the hand over a 5 s window, from 16 channels of
surface EMG recorded at the wrist at 2 kHz. The data is the emg2pose corpus (Salter et al., 2024).
This is my entry to the Codabench challenge "Datacamp 2026 - EMG2Pose" (M2 course). The score is
the mean angular error in degrees on users never seen during training.

## Approach

The difficulty is new users: every arm is different and the band is never worn the same way.
I built on VEMG2Pose, the emg2pose reference model (in braindecode). It is a convolutional
encoder followed by an LSTM that rolls out the pose step by step, looking only at the past.

- **Train on all 158 training users.** With 8 users the model was worse than predicting the
  mean pose. Users mattered more than any other setting.
- **Augmentations for new users.** Rotating the band by one electrode, random gain per
  electrode and overall, and noise. They imitate band placement and skin differences.
- **Stable training.** AdamW with warmup and cosine decay, gradient clipping, and an average
  of the weights over training (EMA). At test time I average predictions over the three band
  rotations. This is the submitted model (`src/vemg2pose_plus.py`).
- **Look both ways in time.** The score is computed offline on whole windows, so the pose at
  time t can use the EMG after t. `src/posebinet.py` keeps the VEMG2Pose encoder, centres its
  receptive field, and replaces the causal decoder with a bidirectional LSTM. It predicts
  absolute angles, so there is no drift from integrating velocities.
- **Average two models.** Training was limited by data loading: the GPU waited for data 57%
  of the time. Two models trained on the same batches at almost no extra cost. With an L1
  metric, their average is never worse than their mean error.

Each model is a single file because the platform only accepts one `submission.py`.

## Results

Mean angular error in degrees (std across users). Held-out users and held-out movement stages,
regression mode (no ground-truth starting pose).

| Model | Source | Error (deg) |
|---|---|---|
| Mean pose of 8 training users | ours | 19.27 (2.02) |
| SensingDynamics | Salter et al. 2024, Table 4 | 18.7 (1.6) |
| NeuroPose | Salter et al. 2024, Table 4 | 17.5 (1.5) |
| vemg2pose | Salter et al. 2024, Table 4 | 15.8 (1.4) |
| vemg2pose, best variant (Pos MT) | Hadidi et al. 2026, Table 2 | 14.58 (1.29) |
| VEMG2Pose+ (submitted) | ours | 15.98 (1.41) |
| PoseBiNet, one model | ours | 15.45 (1.27) |
| PoseBiNet, two models (not submitted) | ours | **15.15 (1.29)** |

Our numbers are on the challenge public split (20 users). The paper numbers are on the
emg2pose "user, stage" test set: same condition and metric, but not the same people, so the
comparison is only rough. The submitted model is VEMG2Pose+: Codabench shows 15.98, the same as
the local score. PoseBiNet was scored locally only; its weights could not be copied off the
training machine in time, so it was never submitted.

Using the EMG after t and averaging two models gained 0.83 deg over the causal model. The best
published result is still lower; it comes from multi-task training, which I did not try. The
training error ends at 8.4 deg against 15.15 deg on new users, so new users remain the main
problem. Full log: `results/experiments.md`.

## What didn't work

- Training on 8 users: 19.96 deg, worse than the mean pose (19.27).
- Ridge regression on raw EMG samples: 19.27 deg, no better than the mean pose.
- A wider encoder (384 channels instead of 256): 16.49 vs 16.53 deg after 3000 steps, within
  noise.

## Possible improvements

1. Average 4 models instead of 2. Going from 1 to 2 gave 0.31 deg and extra models are cheap
   to train here. The inference time on the platform's T4 must be checked first.
2. Multi-task training with a tracking objective, which gave Hadidi et al. their best result
   (15.63 to 14.58 deg).
3. Faster data loading, for example keeping the labels at 50 Hz instead of 2 kHz. The GPU
   waited for data 26 to 57% of the time, so the same budget would allow longer training.
4. Randomly dropping electrodes during training, to imitate bad skin contact. Untested.

## Compute

One RTX 6000 Ada 48 GB rented on RunPod (listed at $0.99/h), data on an 800 GB network volume
(listed at $1.85/day).

| Run | Wall-clock |
|---|---|
| VEMG2Pose+, 25k steps (submitted) | 191.6 min training + ~25 min data loading |
| PoseBiNet screen, 2 runs of 3000 steps | 19.5 + 19.0 min training + ~25 min data loading |
| PoseBiNet, 2 models, 25k steps | 148.7 min training + ~25 min data loading |

Total cost: about $12 (RunPod billing, GPU and storage). Total GPU hours were not recorded.

## Reproduce

```bash
git clone https://github.com/Antoinechss/NeurIPS2026-emg2pose.git
cd NeurIPS2026-emg2pose
bash scripts/setup.sh /path/to/data   # organisers' benchmark, .venv, env.sh
source env.sh                         # $B is the benchmark folder

benchopt run $B --config configs/smoke.yml          # fake data, ~1.5 min on a laptop CPU

benchopt prepare $B -d "EMG2Pose[n_train_subjects=158]"   # about 340 GB
benchopt run $B --config configs/final.yml          # train, score public, export the zip
benchopt run $B --config configs/reeval.yml         # score the export without training
python scripts/show_results.py 2
```

`final.yml` exports `$B/outputs/posebinet.zip`, ready to upload. `configs/` also holds the 3000-step
screen and the submitted model. Training is seeded, but GPU runs can differ slightly.

## References

- Salter et al. 2024, emg2pose: A Large and Diverse Benchmark for Surface Electromyographic
  Hand Pose Estimation. arXiv:2412.02725.
- Hadidi et al. 2026, Re-evaluating Position and Velocity Decoding for Hand Pose Estimation
  with Surface Electromyography. arXiv:2603.08212.
