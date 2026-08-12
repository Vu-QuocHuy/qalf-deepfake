# QALF current model and research plan

Status: SRM seed-42 ablation implementation ready, 2026-08-12.

## Locked protocol

| Item | Value |
| --- | --- |
| Train / validation | FaceForensics++ c23 official split |
| Zero-shot test | Celeb-DF-v2 official 518-video test list |
| Labels / score | `0=real`, `1=fake`, score=`P(fake)` |
| Input | aligned full face, 160 x 160 RGB |
| Frames | 8 texture frames in training, 12 in evaluation |
| Video inference | 3 clips, mean aggregation, horizontal-flip TTA |
| Threshold | Youden-J selected only on FF++ validation |
| Checkpoint | maximum FF++ validation AUC, patience 5 |

Celeb-DF is a fixed cross-dataset development benchmark because its results have
already been inspected repeatedly. It must not be described as an untouched
test set. No thresholds are selected on it, and the pre-registered gate below
must not be changed after seeing the SRM result. A final generalization claim
requires a separate, untouched dataset.

## Retained controls

- `texture_only_sbi`: EfficientNet-B0 texture-only accuracy control.
- `full_face_sbi`: original pose-decoupled landmark geometry + texture baseline A.
- SBI sampling: 50% real, 25% original fake, 25% self-blended real frames.

Across seeds 17/42/73, texture-only mean Celeb-DF AUC was 0.8403 and geometry
baseline mean AUC was 0.8352. In baseline A, geometry had a small positive
seed-42 zero-geometry counterfactual gain (+0.0039), but reliability supervision
collapsed its gate weight to approximately zero. Modality dropout, reliability
loss, attentive geometry, and the associated diagnostic profiles are closed and
removed from active code.

## SRM candidate

The candidate replaces landmarks with a residual branch derived from the same
full-face RGB frames:

1. three fixed zero-sum 5 x 5 high-pass filters are applied independently to
   each RGB channel;
2. a small depthwise-separable CNN encodes the nine residual maps;
3. frame embeddings are mean-pooled into the same 192-dimensional space as the
   EfficientNet-B0 texture embedding;
4. the existing two-way quality-aware gate fuses SRM and texture features;
5. the loss is fused BCE + 0.25 SRM BCE + 0.25 texture BCE.

There is no modality dropout, reliability loss, EMA, MixStyle, or learned SRM
filter. The SRM branch is intentionally simple so the first experiment measures
whether residual evidence is complementary at all.

## Decision gate

Run seed 42:

```bash
./run_srm_ablation.sh all
```

SRM advances to seeds 17 and 73 only if all conditions hold:

- Celeb-DF AUC is at least 0.005 above the texture-only control;
- normal-minus-zero-SRM counterfactual AUC is at least 0.003;
- mean SRM gate weight is at least 0.05;
- fused AUC is greater than the SRM model's own texture-branch AUC.

If SRM passes, run the remaining seeds and report mean ± sample SD, paired
bootstrap confidence intervals, parameters, latency, peak memory, and threshold
provenance. If it fails, do not tune on Celeb-DF; inspect residual activation and
define a separate development protocol before changing the branch.

## Removed directions

EfficientNet-B1, EMA, MixStyle, canonical-skin input, dual view, learned texture
dynamics pooling, attentive/graph/two-stream/self-supervised geometry,
class-balanced geometry loss, modality dropout, and reliability supervision are
not active. Historical results remain in git and experiment artifacts.
