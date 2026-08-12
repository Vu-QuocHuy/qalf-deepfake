# QALF cross-dataset experiments

The active profiles all use EfficientNet-B0, 160-pixel full-face input, the
locked 50/25/25 SBI distribution, raw weights, eight training texture frames,
and 12 evaluation frames.

| Profile | Purpose |
| --- | --- |
| `texture_only_sbi` | Texture-only SBI control |
| `full_face_sbi` | Retained landmark geometry + texture baseline A |
| `srm_sbi` | Fixed SRM residual + texture candidate |

Run the complete seed-42 comparison from Git Bash:

```bash
./run_srm_ablation.sh all
```

The runner reuses completed controls, trains SRM only when its checkpoint is
missing, evaluates any missing comparison row with FF++-validation threshold
calibration, and writes:

- `E:/DeepFakeData/experiments/srm_ablation_seed42.csv`
- `E:/DeepFakeData/experiments/srm_ablation_seed42.md`

Run an individual profile with:

```bash
./run_train_cross_dataset.sh srm_sbi
./run_test_cross_dataset.sh srm_sbi
```

After a seed-42 pass, use PowerShell for the remaining locked seeds:

```powershell
$env:QALF_SEED = "17"
& "C:\Program Files\Git\bin\bash.exe" ./run_srm_ablation.sh all
Remove-Item Env:QALF_SEED
```

Training artifacts are `best.pt`, `config.json`, `run_metadata.json`,
`training_summary.json`, `history.json`, `train.log`, and the training plot.
Evaluation artifacts are `metrics.txt`, `metrics.json`, protocol metadata,
prediction CSV files, `eval.log`, and diagnostic plots.
