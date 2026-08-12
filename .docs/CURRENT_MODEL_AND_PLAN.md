# QALF current model and research plan

Status: fixed-SRM and staged softmax fusion are closed. The active model is the
learnable constrained residual stream with residual-interaction fusion,
2026-08-12.

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

## Seed-42 result

| Model/score | Celeb-DF AUC |
| --- | ---: |
| Texture-only SBI control | 0.8120 |
| Geometry baseline A | 0.8325 |
| SRM fused output | 0.8481 |
| Texture score inside SRM checkpoint | 0.8524 |
| SRM score inside SRM checkpoint | 0.6078 |

The SRM gate weight was 0.0057 and fused AUC was 0.0043 below the internal
texture score. Fixed-SRM fusion therefore failed the registered gate even
though SRM-assisted training produced the strongest observed texture score.
Seeds 17 and 73 must not be run for the original fusion claim.

The zero-auxiliary result is not equivalent to texture-only inference because
zeroing changes the fusion MLP input distribution. Direct texture score and
branch-specific FF++ thresholds are the valid diagnostic comparison.

## Historical decision gate

Seed 42 was run with:

```bash
./run_srm_ablation.sh all
```

SRM advances to seeds 17 and 73 only if all conditions hold:

- Celeb-DF AUC is at least 0.005 above the texture-only control;
- normal-minus-zero-SRM counterfactual AUC is at least 0.003;
- mean SRM gate weight is at least 0.05;
- fused AUC is greater than the SRM model's own texture-branch AUC.

The candidate failed because its mean SRM weight was below 0.05 and fused AUC
was below its own texture AUC. This gate remains unchanged in the record.

## Closed SRM diagnostics

The completed post-hoc diagnostic is reproducible with:

```bash
./run_srm_diagnostics.sh
```

It showed low texture/SRM correlation and 65 SRM-only-correct videos, but an
FF++-selected linear blend assigned zero weight to SRM. Five branch-only warmup
epochs also failed: Celeb-DF fused/texture/SRM AUC was
0.8086/0.8096/0.5769 and mean SRM routing weight was 0.0005. The fixed branch,
softmax gate, and staged variant are therefore closed.

## Active upgraded model

The retained upgrade addresses the identified structural failures directly:

1. SRM kernels initialize a trainable filter bank; every update is projected to
   zero spatial mean, preserving zero DC response.
2. A roughly half-million-parameter residual CNN replaces the 28K-parameter
   fixed-SRM encoder.
3. An explicit component seed makes texture initialization independent of the
   auxiliary architecture while preserving legacy-profile reproducibility.
4. Fusion uses texture logit as a skip connection and learns a bounded,
   quality-conditioned correction from texture/residual products and absolute
   differences. It no longer makes the branches compete in a softmax mixture.
5. Eight branch-only epochs train both evidence streams before joint fusion;
   the fusion module is frozen and warmup epochs cannot become best checkpoints.

Train and evaluate the final profile once:

```bash
./run_train_cross_dataset.sh learned_srm
./run_test_cross_dataset.sh learned_srm
```

## Removed directions

EfficientNet-B1, EMA, MixStyle, canonical-skin input, dual view, learned texture
dynamics pooling, attentive/graph/two-stream/self-supervised geometry,
class-balanced geometry loss, modality dropout, and reliability supervision are
not active. Historical results remain in git and experiment artifacts.
