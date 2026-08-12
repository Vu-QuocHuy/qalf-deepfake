# QALF cross-dataset experiments

The current architecture and roadmap live in
[`CURRENT_MODEL_AND_PLAN.md`](CURRENT_MODEL_AND_PLAN.md). This file documents
only how to reproduce the two retained profiles and their three-seed comparison.

## Retained profiles

| Profile | Purpose |
| --- | --- |
| `full_face` | Historical pre-SBI full-face reference (seed-42 AUC 0.8209) |
| `full_face_sbi` | Trusted 50/25/25 SBI reference |
| `geometry_candidate` | SBI + attentive geometry + reliability routing |
| `texture_only_sbi` | Required P1 control: identical SBI protocol without geometry/fusion |

All profiles use EfficientNet-B0, 160-pixel full-face input, eight texture frames
during training, and raw weights. Evaluation uses 12 texture frames, three clips,
mean aggregation, horizontal-flip TTA, and an FF++-validation threshold.

## Three-seed result

| Seed | SBI baseline AUC | Geometry candidate AUC | Delta |
| ---: | ---: | ---: | ---: |
| 17 | 0.8486 | 0.8464 | -0.0022 |
| 42 | 0.8325 | 0.8323 | -0.0002 |
| 73 | 0.8245 | 0.8399 | +0.0154 |
| Mean ± sample SD | 0.8352 ± 0.0123 | **0.8395 ± 0.0071** | +0.0043 |

The candidate also improves mean AP, EER, balanced accuracy, and ACER, but its
mean geometry gate weight is only 0.0015. It remains a candidate rather than the
final model until the texture-only SBI control is complete.

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
