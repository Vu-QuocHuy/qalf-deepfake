# SBI Ablation: Statistical Results

## Experimental design

The ablation evaluates five configurations over five training seeds
(`0, 17, 42, 73, 123`) on Celeb-DF-v2 and FF++ c23. The paired analyses use
the same seed across configurations and therefore compare training-seed
deltas rather than treating runs as independent samples.

| Profile | SBI | Temporal consistency | EMA |
| --- | --- | --- | --- |
| `baseline` | Yes | Clip-consistent | Yes |
| `sbi_frame` | Yes | Frame-wise | Yes |
| `no_sbi` | No | -- | Yes |
| `no_ema` | Yes | Clip-consistent | No |
| `texture_only` | No | -- | No |

The complete experiment contains `2 datasets x 5 profiles x 5 seeds = 50
runs`. Statistical significance is assessed using two-sided paired t-tests;
95% bootstrap confidence intervals resample the five paired training seeds.
The reported p-values are unadjusted for multiple comparisons.

## Aggregate results

Values are mean +/- sample standard deviation across training seeds. Values are
percentages.

### Celeb-DF-v2

| Profile | AUC (%) | Accuracy (%) | Macro-F1 (%) | EER (%) |
| --- | ---: | ---: | ---: | ---: |
| Clip-consistent SBI + EMA | 85.88 +/- 0.55 | 78.76 +/- 0.49 | 77.14 +/- 0.73 | 21.15 +/- 1.12 |
| Frame-wise SBI + EMA | 81.69 +/- 0.86 | 73.71 +/- 1.85 | 72.65 +/- 1.61 | 25.46 +/- 1.53 |
| No SBI + EMA | 83.27 +/- 0.74 | 75.25 +/- 1.95 | 74.21 +/- 1.78 | 23.11 +/- 0.66 |
| Clip-consistent SBI, no EMA | 84.63 +/- 1.72 | 76.33 +/- 2.65 | 74.79 +/- 2.34 | 23.93 +/- 2.24 |
| No SBI, no EMA | 81.80 +/- 1.22 | 72.70 +/- 2.03 | 71.74 +/- 1.68 | 25.08 +/- 2.09 |

### FF++ c23

| Profile | AUC (%) | Accuracy (%) | Macro-F1 (%) | EER (%) |
| --- | ---: | ---: | ---: | ---: |
| Clip-consistent SBI + EMA | 96.70 +/- 0.16 | 92.09 +/- 0.50 | 88.64 +/- 0.70 | 7.71 +/- 0.73 |
| Frame-wise SBI + EMA | 96.82 +/- 0.42 | 91.83 +/- 1.02 | 88.30 +/- 1.45 | 8.21 +/- 1.00 |
| No SBI + EMA | 97.33 +/- 0.09 | 92.94 +/- 0.52 | 89.87 +/- 0.66 | 6.55 +/- 0.52 |
| Clip-consistent SBI, no EMA | 96.27 +/- 0.39 | 90.89 +/- 1.09 | 87.15 +/- 1.37 | 8.62 +/- 1.34 |
| No SBI, no EMA | 97.10 +/- 0.42 | 92.80 +/- 0.39 | 89.66 +/- 0.61 | 7.02 +/- 0.61 |

## Paired statistical tests for AUC

The comparison is always candidate minus the clip-consistent SBI + EMA
baseline. Negative values favor the baseline.

| Dataset | Comparison | Delta AUC (pp) | 95% CI (pp) | p-value | Significant at 0.05? |
| --- | --- | ---: | ---: | ---: | --- |
| Celeb-DF-v2 | No EMA - baseline | -1.25 | [-2.80, 0.32] | 0.2418 | No |
| Celeb-DF-v2 | No SBI - baseline | -2.61 | [-2.94, -2.30] | 0.0001 | Yes |
| Celeb-DF-v2 | Frame-wise SBI - baseline | -4.18 | [-4.92, -3.05] | 0.0018 | Yes |
| Celeb-DF-v2 | No SBI, no EMA - baseline | -4.08 | [-5.18, -3.06] | 0.0027 | Yes |
| FF++ c23 | No EMA - baseline | -0.43 | [-0.67, -0.19] | 0.0386 | Yes |
| FF++ c23 | No SBI - baseline | +0.63 | [0.45, 0.82] | 0.0038 | Yes |
| FF++ c23 | Frame-wise SBI - baseline | +0.12 | [-0.11, 0.37] | 0.4335 | No |
| FF++ c23 | No SBI, no EMA - baseline | +0.40 | [-0.09, 0.73] | 0.1891 | No |

## Main findings

1. **Temporal consistency is important for Celeb-DF-v2 generalization.**
   Replacing clip-consistent SBI with frame-wise SBI reduces AUC by 4.18 pp
   (`p = 0.0018`, 95% CI `[-4.92, -3.05]` pp). The same comparison on FF++
   changes AUC by only +0.12 pp and is not significant (`p = 0.4335`).

2. **SBI improves Celeb-DF-v2 performance.** Removing SBI from the EMA model
   reduces AUC by 2.61 pp (`p = 0.0001`). This effect is not reproduced on
   FF++, where the no-SBI + EMA model has the highest AUC (+0.63 pp relative
   to the baseline).

3. **The EMA effect is dataset-dependent.** Removing EMA reduces AUC by 0.43
   pp on FF++ (`p = 0.0386`), but the 1.25 pp reduction on Celeb-DF-v2 is not
   significant (`p = 0.2418`). The lower standard deviation with EMA is a
   descriptive stability result and was not tested as a separate variance
   hypothesis.

4. **The complete configuration is strongest on Celeb-DF-v2, not on FF++.**
   Clip-consistent SBI + EMA has the best Celeb-DF-v2 AUC and the lowest
   across-seed standard deviation. On FF++, no-SBI + EMA performs best by AUC.

## Paper-ready conclusion

The five-seed ablation indicates that clip-consistent SBI substantially
improves cross-dataset performance on Celeb-DF-v2, while this advantage does
not generalize to FF++ c23. Frame-wise SBI causes a significant AUC decrease
on Celeb-DF-v2 but has no significant effect on FF++. EMA provides a modest,
dataset-dependent benefit: its AUC improvement is significant on FF++ but not
on Celeb-DF-v2. Therefore, the results support temporal consistency as an
important factor for Celeb-DF-v2 generalization, but do not support claiming
that either temporal consistency, SBI, or EMA is universally beneficial across
all evaluation domains.

## Reproducibility

The report was generated from the 50 existing evaluation runs; no additional
training or evaluation was performed. To regenerate it in PowerShell:

```powershell
git pull origin main
\.venv\Scripts\python.exe scripts\summarize_sbi_ablation.py `
  --ablation-root "E:\DeepFakeData\experiments\ablation"
```

The generated machine-readable statistical output is
`sbi_summary/paired_statistics.csv`.
