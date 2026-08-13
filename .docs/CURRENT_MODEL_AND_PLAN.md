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
7. at evaluation, use 8 frames in each of three clips, horizontal-flip TTA,
   and mean aggregation to obtain one score per video. A 12-frame evaluation
   is a separate frame-count ablation, not the locked baseline.

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

For a practical robustness table, run `./run_robustness.sh`. This is an
evaluation-only protocol: it keeps the texture-only EfficientNet-B0 model
unchanged and applies controlled JPEG, blur, downscale, and noise corruptions
to RGB frames with the locked 8-frame inference protocol. Set
`QALF_ROBUSTNESS_TEXTURE_FRAMES` only for a separate frame-count ablation.
The operating threshold is fitted on clean FF++ validation, not on Celeb-DF.

For a resumable full comparison, run `./run_ablation_suite.sh`. It keeps the
locked baseline unchanged and writes training controls, inference-protocol
ablations, and corruption robustness to separate experiment directories.

Geometry, fixed/learned SRM, fusion gates, reliability loss, modality dropout,
MixStyle, EfficientNet-B1, and their diagnostic profiles are retired from the
active codebase. The active baseline is texture-only full-face RGB with SBI,
EfficientNet-B0, EMA decay 0.999, 50 maximum epochs, and early stopping
patience 5. Landmark caches remain only for preprocessing alignment and SBI
mask generation; they are not used as learned features.
