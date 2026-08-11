# Cross-dataset texture experiments

## Evidence so far

All rows below use EfficientNet-B0 and the same deterministic FF++ training
split. Celeb-DF is evaluated zero-shot with 12 texture frames, three clips,
mean video aggregation, horizontal-flip TTA, and a threshold selected on FF++
validation.

| Profile | Celeb-DF AUC | Texture AUC | Geometry AUC |
| --- | ---: | ---: | ---: |
| Canonical-skin control | 0.7279 | 0.7205 | 0.5810 |
| Full face | **0.8209** | **0.8162** | 0.5616 |
| Full face + dual view | 0.8037 | 0.8064 | 0.5906 |
| Canonical skin + dynamics | 0.7562 | 0.7578 | 0.5506 |
| Canonical skin + MixStyle | 0.7369 | 0.7298 | 0.5808 |
| Full face + 224px + 16f + dynamics + MixStyle + EMA | 0.8128 | 0.8106 | **0.6233** |

Full face is therefore the current baseline. The earlier combined profile is
not retained as a default: it costs substantially more compute and scores 0.81
AUC points below the simpler full-face model. Attention pooling and dedicated
B1/EMA wrapper scripts were also removed after their completed ablations
underperformed. EfficientNet-B1 remains available through
`--texture-backbone efficientnet_b1` for controlled experiments.

## Active profiles

Train and test one profile with:

```bash
./run_train_cross_dataset.sh PROFILE
./run_test_cross_dataset.sh PROFILE
```

For `full_face_sbi`, inspect a small training-only preview before the full run:

```powershell
& ".\.venv\Scripts\python.exe" scripts/preview_sbi.py `
  --manifest "E:/DeepFakeData/data/landmarks/ffpp-landmark/manifests/ffpp_train_landmarks.jsonl" `
  --frame-root "E:/DeepFakeData/data/extracted/ffpp" `
  --landmark-root "E:/DeepFakeData/data/landmarks/ffpp-landmark/landmarks" `
  --output "E:/DeepFakeData/experiments/qalf_sbi_preview.png"
```

| Profile | Purpose |
| --- | --- |
| `full_face` | Reproduce the established 0.8209-AUC baseline |
| `full_face_sbi` | Locked 50/25/25 hybrid SBI experiment; current primary profile |
| `full_face_ema` | Isolate EMA on the full-face baseline |
| `full_face_dynamics` | Isolate temporal dynamics on full-face inputs |
| `full_face_mixstyle` | Isolate MixStyle on full-face inputs |
| `dual_view` | Completed negative ablation of full-face and canonical-skin fusion |

Dual-view is not the next primary experiment: it reached 0.8037 Celeb-DF AUC,
below the 0.8209 full-face baseline, and its texture AUC also decreased. It is
retained only for reproducibility. The next controlled experiment is the
training-only SBI hybrid defined in `SBI_IMPLEMENTATION_PLAN.md`.

Run only one change at a time before combining mechanisms. Do not select epochs,
fusion rules, or hyperparameters from Celeb-DF test AUC for a final paper; use a
separate cross-domain development set and reserve the locked test set for the
final report.

## Reporting artifacts

Training preserves `best.pt` and `last.pt` at their existing paths and adds
`run_metadata.json`, `training_summary.json`, `history.json`, `train.log`, and
`plots/training_history.png`. Evaluation writes a stable human-readable table in
`metrics.txt`, a machine-readable `metrics.json`, `eval.log`, video- and
clip-level prediction CSV files, the exact protocol, raw/normalized confusion
matrices, ROC and precision-recall curves, and the real/fake score distribution.

## Motivation

Full-face input restores eyes, mouth, and face-boundary evidence removed by the
legacy four-rectangle skin mask. The retained dual-view experiment tests whether
the restricted skin signal is complementary rather than replacing the stronger
full-face signal. The rationale is consistent with cross-manipulation artifact
work such as Face X-Ray and Self-Blended Images:

- https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Face_X-Ray_for_More_General_Face_Forgery_Detection_CVPR_2020_paper.html
- https://openaccess.thecvf.com/content/CVPR2022/html/Shiohara_Detecting_Deepfakes_With_Self-Blended_Images_CVPR_2022_paper.html
- https://openreview.net/forum?id=6xHJ37MVxxp
