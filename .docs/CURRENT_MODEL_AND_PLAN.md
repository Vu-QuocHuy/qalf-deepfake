# Current model and protocol

Status: QALF v2 texture-only with frequency/multiscale/temporal-attention, 2026-08-12.

## Architecture

The QALF v2 classifier extends the texture-only v1 baseline with three
lightweight modules while keeping the total parameter count under 6M:

### Pipeline (v2 full)

```text
video -> extracted frame sequence -> landmark face alignment (preprocessing)
      -> RGB face frames (224×224)
      -> [NEW] SRM high-pass filter layer (fixed, 3 kernels) → 12ch → 1×1 conv → 3ch
      -> EfficientNet-B0 per frame
         ├─ features[:3] → low (24ch)  ─┐
         ├─ features[3:6] → mid (112ch) ├─ [NEW] Multi-Scale Aggregation → 192D
         └─ features[6:] → high (1280ch)┘
      -> (B, T, 192D) frame embeddings
      -> [NEW] Temporal Attention Pooling (attention + variance)
      -> clip P(fake) -> mean of three clips = video P(fake)
```

### V1 baseline (retained as ablation default)

1. select a 32-frame temporal window from each video;
2. uniformly sample 8 aligned full-face RGB frames during training;
3. encode every frame independently with ImageNet-pretrained EfficientNet-B0;
4. project each frame to a 192-dimensional embedding;
5. mean-pool frame embeddings into one clip embedding;
6. predict `P(fake)` for the clip;
7. at evaluation, use 12 frames in each of three clips, horizontal-flip TTA,
   and mean aggregation to obtain one score per video.

### New modules (v2)

- **FrequencyPreprocess** (`qalf/models/frequency.py`): 3 fixed SRM kernels
  applied per-channel, concatenated with RGB, and projected back to 3ch via a
  learnable 1×1 Conv + BatchNorm + SiLU adapter. Non-learnable filters reveal
  high-frequency compression artefacts invisible to ImageNet features.

- **MultiScaleAggregation** (`qalf/models/multiscale.py`): Taps into three
  stages of EfficientNet-B0 (low/mid/high), projects each to 64/64/128 dims
  via AdaptiveAvgPool + Linear, concatenates (256D), and merges to 192D. Captures
  both fine-grained boundary artefacts and high-level semantic patterns.

- **TemporalAttentionPooling** (`qalf/models/temporal.py`): Learns per-frame
  importance scores via a 2-layer MLP → softmax, computes attention-weighted
  mean embedding and attention-weighted temporal variance, concatenates (384D),
  and projects to 192D. Detects frame-to-frame inconsistencies that mean-pooling
  discards.

Landmarks remain preprocessing metadata only. They align the full-face crop and
construct the SBI blending mask. They are not converted into classifier
features and are not passed to the model.

## Training

- Dataset: FaceForensics++ c23 official train/validation split.
- Sampling: 50% real, 25% original fake, 25% self-blended real frames.
- Loss: one binary cross-entropy on the clip prediction with label smoothing.
- Optimizer: AdamW with separate head/backbone learning rates.
- Selection: maximum FF++ validation AUC, patience 7.
- Image size: 224×224 (v2) / 160×160 (v1).
- Texture frames: 10 (v2 training) / 8 (v1 training).

## Evaluation

- Cross-dataset development benchmark: Celeb-DF-v2 official 518-video list.
- Threshold: Youden-J calibrated only on FF++ validation.
- Labels: `0=real`, `1=fake`; score is `P(fake)`.
- Celeb-DF has already been inspected repeatedly and is not an untouched final
  test set. A final paper claim requires another untouched dataset.

## Commands

```bash
# V2 (new)
./run_train_v2.sh
./run_test_v2.sh

# V1 (baseline)
./run_train.sh
./run_test.sh
```

## Parameter budget

| Component             | V1 params | V2 params |
|-----------------------|-----------|-----------|
| EfficientNet-B0       | ~5.29M    | ~5.29M    |
| Projection head       | ~250K     | ~250K     |
| SRM Frequency Preproc | 0         | ~54       |
| Multi-Scale Agg       | 0         | ~222K     |
| Temporal Attn Pool    | 0         | ~86K      |
| **Total**             | **~5.54M**| **~5.85M**|

Geometry, fixed/learned SRM branches, fusion gates, reliability loss, modality
dropout, MixStyle, EfficientNet-B1, and their diagnostic profiles are retired
from the active codebase. The v2 SRM is a fixed preprocessing layer, not a
learned branch.
