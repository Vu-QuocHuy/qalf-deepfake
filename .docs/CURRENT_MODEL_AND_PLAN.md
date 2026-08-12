# Current model and protocol

Status: texture-only SBI model, 2026-08-12.

## Architecture

The retained classifier has one visual branch:

1. select a 32-frame temporal window from each video;
2. uniformly sample 8 aligned full-face RGB frames during training;
3. encode every frame independently with ImageNet-pretrained EfficientNet-B0;
4. project each frame to a 192-dimensional embedding;
5. mean-pool frame embeddings into one clip embedding;
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
./run_train_cross_dataset.sh
./run_test_cross_dataset.sh
```

Geometry, fixed/learned SRM, fusion gates, reliability loss, modality dropout,
MixStyle, EfficientNet-B1, and their diagnostic profiles are retired from the
active codebase. The baseline remains raw-weight training; `run_train_ema.sh`
and `run_test_ema.sh` provide one controlled EMA comparison using decay 0.999.
Historical experiment outputs can be described as negative ablations in the
paper, but are no longer loadable by the cleaned source tree.
