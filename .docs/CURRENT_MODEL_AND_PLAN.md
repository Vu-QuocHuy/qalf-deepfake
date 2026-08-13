# Current model and protocol

Status: texture-only SBI model, 2026-08-12.

## Architecture

The retained classifier has one visual branch:

1. select a 32-frame temporal window from each video;
2. uniformly sample 8 aligned full-face RGB frames during training;
3. encode every frame independently with ImageNet-pretrained EfficientNet-B0;
4. project each frame to a 192-dimensional embedding;
5. score each frame embedding with a one-layer temporal attention scorer and
   weighted-pool the sequence into one clip embedding;
6. predict `P(fake)` for the clip;
7. at evaluation, use 12 frames in each of three clips, horizontal-flip TTA,
   and mean aggregation to obtain one score per video.

Landmarks remain preprocessing metadata only. They align the full-face crop and
construct the SBI blending mask. They are not converted into classifier
features and are not passed to the model.

## Training

- Dataset: FaceForensics++ c23 official train/validation split.
- Sampling: 50% real, 25% original fake, 25% self-blended real frames.
- Loss: one binary cross-entropy on the clip prediction.
- Optimizer: AdamW with separate head/backbone learning rates.
- Selection: maximum FF++ validation AUC, patience 5.

## Evaluation

- Cross-dataset development benchmark: Celeb-DF-v2 official 518-video list.
- Threshold: Youden-J calibrated only on FF++ validation.
- Labels: `0=real`, `1=fake`; score is `P(fake)`.
- Celeb-DF has already been inspected repeatedly and is not an untouched final
  test set. A final paper claim requires another untouched dataset.

## Commands

```bash
./run_train.sh
./run_test.sh
```

Geometry, fixed/learned SRM, fusion gates, reliability loss, modality dropout,
MixStyle, EfficientNet-B1, and their diagnostic profiles are retired from the
active codebase. The active candidate is texture-only full-face RGB with SBI,
EfficientNet-B0, lightweight attention pooling, EMA decay 0.999, 50 maximum
epochs, and early stopping patience 5. Mean pooling remains supported as a
backward-compatible control. Landmark caches remain only for preprocessing
alignment and SBI mask generation; they are not used as learned features.
The attention scorer is zero-initialized so its starting behavior is uniform
mean pooling; entropy regularization is intentionally not enabled in this
diagnostic revision.
