# QALF cross-dataset experiments

The current architecture and roadmap live in
[`CURRENT_MODEL_AND_PLAN.md`](CURRENT_MODEL_AND_PLAN.md). This file documents
only how to reproduce the retained profiles and their focused comparisons.

## Retained profiles

| Profile | Purpose |
| --- | --- |
| `full_face` | Historical pre-SBI full-face reference (seed-42 AUC 0.8209) |
| `full_face_sbi` | Trusted 50/25/25 SBI reference |
| `geometry_candidate` | SBI + attentive geometry + reliability routing |
| `texture_only_sbi` | Required P1 control: identical SBI protocol without geometry/fusion |
| `geometry_dropout_only` | Seed-42 diagnostic: SBI baseline plus modality dropout, no reliability loss |
| `geometry_reliability_combined` | Historical I2 control: modality dropout plus reliability loss |

All profiles use EfficientNet-B0, 160-pixel full-face input, eight texture frames
during training, and raw weights. Evaluation uses 12 texture frames, three clips,
mean aggregation, horizontal-flip TTA, and an FF++-validation threshold.

## Three-seed result

| Seed | SBI baseline AUC | Geometry candidate AUC | Texture-only AUC |
| ---: | ---: | ---: | ---: |
| 17 | 0.8486 | 0.8464 | 0.8501 |
| 42 | 0.8325 | 0.8323 | 0.8120 |
| 73 | 0.8245 | 0.8399 | 0.8587 |
| Mean ± sample SD | 0.8352 ± 0.0123 | 0.8395 ± 0.0071 | **0.8403 ± 0.0249** |

The candidate improves mean EER, balanced accuracy, and ACER, but texture-only
has slightly higher mean AUC/AP and the candidate's mean geometry gate weight is
only 0.0015. Under the pre-registered P1 rule, texture-only is the clean-data
accuracy reference. The seed-42 failure diagnostic below determines why the
candidate gate collapsed; it does not reopen P1 model selection.

## Run one profile

From Git Bash:

```bash
./run_train_cross_dataset.sh full_face_sbi
./run_test_cross_dataset.sh full_face_sbi

./run_train_cross_dataset.sh geometry_candidate
./run_test_cross_dataset.sh geometry_candidate

./run_train_cross_dataset.sh texture_only_sbi
./run_test_cross_dataset.sh texture_only_sbi
```

The focused geometry failure diagnostic is intentionally locked to seed 42. It
reuses the completed SBI baseline, automatically trains either missing C/D
control, refreshes their diagnostics when needed, and writes one A-C-D
comparison:

```bash
./run_geometry_failure_diagnostic.sh all
```

Output: `E:/DeepFakeData/experiments/geometry_failure_diagnostic_seed42.md`.

For a non-default seed in PowerShell:

```powershell
$env:QALF_SEED = "17"
& "C:\Program Files\Git\bin\bash.exe" ./run_geometry_ablation.sh all
Remove-Item Env:QALF_SEED
```

Seed 42 uses historical paths. Other seeds automatically receive `_seedN`, so
checkpoints and evaluation artifacts cannot overwrite one another.

## Compare outputs

```bash
./run_geometry_ablation.sh train
./run_geometry_ablation.sh test
./run_geometry_ablation.sh all
```

The runner writes `p1_texture_control_comparison[_seedN].csv` and `.md` under
`E:/DeepFakeData/experiments`. It includes FF++ validation AUC, Celeb-DF AUC,
domain gap, branch AUCs, fusion gain, gate weights, zero-geometry counterfactual AUC,
AP, EER, balanced accuracy, and ACER. Existing checkpoints are reused during
`train`/`all`; the runner does not overwrite completed baseline or candidate runs.

Training artifacts include `best.pt`, `last.pt`, `config.json`,
`run_metadata.json`, `training_summary.json`, `history.json`, `train.log`, and
`plots/training_history.png`. Evaluation writes `metrics.txt`, `metrics.json`,
`evaluation_protocol.txt`, prediction CSV files, `eval.log`, and diagnostic
plots.

## Stopped experiments

Canonical-skin input, dual view, EfficientNet-B1, EMA, MixStyle, learned texture
dynamics, graph geometry, rigid/non-rigid two-stream geometry, self-supervised
geometry, and class-balanced geometry loss were negative ablations. Their active
implementations were removed to keep the research code aligned with the evidence.
