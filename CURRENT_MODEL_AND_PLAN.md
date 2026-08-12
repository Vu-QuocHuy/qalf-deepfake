# QALF current model and research plan

Status: P1 implementation ready, 2026-08-12

This is the single source of truth for the retained model, the evidence collected
so far, and the next decision. `SBI_IMPLEMENTATION_PLAN.md` and
`IMPLEMENTATION_REPORT_VI.md` are historical records, not active roadmaps.

## 1. Current decision

There is not yet enough evidence to promote the Geometry candidate to the final
model.

- **Trusted reference:** full-face EfficientNet-B0 + SBI + baseline geometry
  (`full_face_sbi`).
- **Retained research candidate:** the same model with attentive geometry pooling,
  modality dropout, and reliability-gate supervision (`geometry_candidate`).
- **Reason for holding the decision:** the candidate has a slightly better
  three-seed mean and lower variance, but its learned geometry weight is almost
  zero. The gain may come from regularizing the texture path rather than using
  geometry at inference.

The next experiment must therefore be a texture-only SBI control. No new feature
or hyperparameter should be added before that control is complete.

## 2. Locked data and evaluation protocol

| Item | Value |
| --- | --- |
| Train / validation | FaceForensics++ c23, official split |
| Zero-shot test | Celeb-DF-v2 official 518-video test list |
| Labels / score | `0=real`, `1=fake`, score=`P(fake)` |
| Geometry clip | 32 frames |
| Texture frames | 8 train, 12 evaluation |
| Face input | aligned full face, 160 x 160 RGB |
| Video inference | 3 clips, mean aggregation, horizontal-flip TTA |
| Threshold | Youden-J selected only on FF++ validation |
| Checkpoint selection | FF++ validation AUC, patience 5 |
| Seeds | 17, 42, 73 |

Celeb-DF has already been inspected repeatedly during development. From this
point onward it is frozen: do not use it to change SBI ratios, architecture,
loss weights, or training duration. Any further tuning requires a separate
cross-domain development protocol.

## 3. Retained architecture

### Shared components

- **Geometry input:** 59 selected MediaPipe landmarks with pose-decoupled
  `aligned_motion_3d` features. The per-frame input dimension is 532.
- **Geometry encoder:** a three-layer causal depthwise TCN, hidden width 128,
  producing a 192-dimensional embedding and an auxiliary fake logit.
- **Texture encoder:** ImageNet-pretrained EfficientNet-B0 over full-face RGB
  frames. Frame embeddings are mean-pooled into a 192-dimensional video
  embedding with an auxiliary fake logit.
- **Quality-aware fusion:** a learned gate consumes both embeddings, five quality
  values per branch, and detached branch uncertainty. The weighted embedding is
  classified by the fused head.
- **Loss:** fused BCE + 0.25 geometry BCE + 0.25 texture BCE. Geometry BCE is
  masked for SBI samples because their geometry remains authentic.
- **Optimization:** AdamW, head LR `3e-4`, backbone LR `3e-5`, weight decay
  `3e-4`, cosine decay, AMP, raw weights, deterministic execution.

The baseline has 4,495,361 parameters. The candidate has 4,553,218 parameters.

### SBI training distribution

The explicit balanced sampler targets 50% pristine real, 25% original fake,
and 25% SBI generated only from training-real clips. SBI is training-only and
uses temporally coherent blend parameters across a clip.

### Candidate-only changes

- attentive statistics pooling over geometry TCN outputs (weighted mean,
  standard deviation, and maximum);
- modality dropout probability 0.15 per branch during training;
- reliability-gate loss weight 0.10 on modality-dropped samples.

No EMA, EfficientNet-B1, MixStyle, dual-view input, canonical-skin masking, or
texture dynamics pooling remains in the supported model.

## 4. Three-seed evidence

Celeb-DF video-level AUC:

| Seed | SBI baseline | Geometry candidate | Candidate delta |
| ---: | ---: | ---: | ---: |
| 17 | 0.8486 | 0.8464 | -0.0022 |
| 42 | 0.8325 | 0.8323 | -0.0002 |
| 73 | 0.8245 | 0.8399 | +0.0154 |
| Mean ± sample SD | 0.8352 ± 0.0123 | **0.8395 ± 0.0071** | +0.0043 |

Aggregate diagnostics:

| Metric | SBI baseline | Geometry candidate |
| --- | ---: | ---: |
| Average precision | 0.9027 ± 0.0095 | **0.9082 ± 0.0075** |
| EER (lower is better) | 0.2476 ± 0.0079 | **0.2414 ± 0.0059** |
| Balanced accuracy | 0.7569 ± 0.0036 | **0.7688 ± 0.0089** |
| ACER (lower is better) | 0.2431 ± 0.0036 | **0.2312 ± 0.0089** |
| FF++→Celeb domain gap | 0.1357 ± 0.0104 | **0.1317 ± 0.0040** |
| Geometry AUC | 0.5912 ± 0.0508 | 0.5917 ± 0.0127 |
| Texture AUC | 0.8343 ± 0.0148 | **0.8384 ± 0.0065** |
| Mean geometry gate weight | 0.1219 ± 0.0745 | **0.0015 ± 0.0014** |

Interpretation: the candidate is empirically promising, especially in AP and
variance, but these results do not establish a useful geometry contribution.
Its near-zero geometry weight is the central unresolved issue.

## 5. Next plan and decision gates

### P0 — review and freeze this cleanup (complete)

- Review this document, the active runners, and the supported architecture.
- Do not train a new model until the cleanup commit is accepted.

### P1 — texture-only SBI control (complete)

The exact locked protocol was run for seeds 17, 42, and 73 with
`fusion_mode=texture`. Mean Celeb-DF AUC was 0.8352 for SBI baseline, 0.8395 for
the retained geometry candidate, and 0.8403 for texture-only SBI. Candidate
minus texture-only was -0.0007 AUC on average and the candidate won only one of
three seeds. Its mean geometry weight was approximately 0.0015 and its mean
zero-geometry counterfactual gain was approximately 0.0001.

The evaluator now also reports a zero-geometry counterfactual without changing
the primary prediction or threshold. It records the counterfactual AUC, normal
minus counterfactual AUC, score shifts, gate-weight percentiles, and separate
real/fake mean gate weights. This is a no-retraining counterfactual diagnostic;
because zero geometry may be out of distribution, it is not by itself a causal
identification result. It is not a replacement for the texture-only control.

The archived P1 runner remains available with:

```bash
./run_geometry_ablation.sh all
```

For seeds 17 and 73, set `QALF_SEED` before the command. Existing baseline and
candidate checkpoints are reused rather than overwritten; only the missing
texture-only checkpoint is trained.

Decision reached:

- Texture-only is the clean-data accuracy reference under the pre-registered
  P1 rule. The candidate does not justify a geometry contribution claim.

### P1b — geometry gate failure diagnostic (implementation ready)

Before closing geometry, isolate the one missing factor on seed 42 while keeping
`tcn_mean`, full-face SBI, optimizer, data, and evaluation fixed:

| Config | Modality dropout | Reliability loss |
| --- | ---: | ---: |
| A — SBI baseline | 0.00 | 0.00 |
| C — dropout only | 0.15 | 0.00 |
| D — historical I2 | 0.15 | 0.10 |

Run:

```bash
./run_geometry_failure_diagnostic.sh all
```

Only C is trained. A is reused and D is retested from its completed historical
checkpoint. If C has geometry weight below 0.01 and absolute counterfactual gain
below 0.001, modality dropout alone reproduces collapse. If C retains geometry
weight at least 0.05 and counterfactual gain at least 0.003, reliability
supervision is the proximate suspect. Do not run more seeds until this diagnostic
selects the next intervention.

### P2 — geometry robustness test (only after P1b)

Evaluate already-trained checkpoints under deterministic texture degradation:
JPEG compression, blur, downsampling, noise, and partial face occlusion. Do not
retrain and do not tune corruption severity on Celeb-DF.

Geometry earns a robustness claim only if the fused candidate consistently
beats texture-only and the gate shifts toward geometry as texture quality falls.
Clean AUC alone cannot support that claim.

### P3 — final reporting

- freeze one final architecture before the final test pass;
- report mean ± sample SD for all three seeds and per-seed values;
- add paired bootstrap confidence intervals for AUC/AP differences;
- report parameters, latency, peak memory, branch AUCs, gate weights, and the
  exact FF++ threshold provenance;
- separate the clean-accuracy claim from any robustness claim.

## 6. Stopped directions

The following were tested and are no longer active: EfficientNet-B1, EMA,
MixStyle, canonical-skin input, dual view, learned texture dynamics pooling,
graph geometry, rigid/non-rigid two-stream geometry, geometry self-supervision,
and class-balanced geometry loss. They should not be reintroduced without a new
pre-registered hypothesis and development protocol.
