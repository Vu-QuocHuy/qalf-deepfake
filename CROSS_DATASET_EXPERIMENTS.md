# Cross-dataset texture experiments

The original EfficientNet-B0 checkpoint remains the reference result.  The new
features are opt-in and old checkpoints keep strict state-dict compatibility.

## Scientific motivation

- `full_face` exposes the eyes, mouth, and face boundary removed by the legacy
  four-rectangle skin mask.  Blending-boundary supervision is known to transfer
  across face-manipulation methods (Face X-Ray, CVPR 2020; Self-Blended Images,
  CVPR 2022).
- `dynamics` retains mean pooling as a residual and adds learnable statistics of
  embedding variance, first differences, and second differences.  This directly
  represents temporal texture instability instead of assuming that more frames
  make permutation-invariant mean/attention pooling temporal.
- Video MixStyle follows MixStyle domain generalization (ICLR 2021/IJCV 2024),
  but shares the partner and interpolation coefficient across all frames in one
  video.  It perturbs FF++ appearance style without manufacturing frame-wise
  flicker.
- `temporal_dg` uses available memory for B0 at its native 224px resolution and
  16 texture frames, then combines full-face visibility, dynamics, MixStyle, and
  EMA.  B1 is intentionally excluded because the completed B1 ablation was worse.

Primary references:

- https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Face_X-Ray_for_More_General_Face_Forgery_Detection_CVPR_2020_paper.html
- https://openaccess.thecvf.com/content/CVPR2022/html/Shiohara_Detecting_Deepfakes_With_Self-Blended_Images_CVPR_2022_paper.html
- https://openreview.net/forum?id=6xHJ37MVxxp
- https://arxiv.org/abs/2107.02053
- https://openaccess.thecvf.com/content/CVPR2021/html/Haliassos_Lips_Dont_Lie_A_Generalisable_and_Robust_Approach_To_Face_CVPR_2021_paper.html
- https://arxiv.org/abs/2604.16808

## Profiles

Train a profile with:

```bash
./run_train_cross_dataset.sh PROFILE
```

Test it using the fixed Celeb-DF protocol (three clips, mean aggregation, flip
TTA, and FF++ validation threshold calibration):

```bash
./run_test_cross_dataset.sh PROFILE
```

Available profiles:

| Profile | Isolated change |
| --- | --- |
| `control` | deterministic reproduction of B0 160px/8f/mean |
| `full_face` | `canonical_skin` to `full_face` |
| `mixstyle` | Video MixStyle after shallow EfficientNet feature stages 1 and 2 |
| `dynamics` | temporal dynamics pooling instead of mean |
| `temporal_dg` | B0 224px/16f, full face, dynamics, MixStyle, EMA |

Run `control`, `full_face`, `mixstyle`, and `dynamics` with the same seed before
attributing an improvement to a component.  `temporal_dg` is the performance
candidate, not a one-variable ablation.  Do not select epochs or hyperparameters
on Celeb-DF test AUC in a final paper; reserve a separate cross-domain development
set and keep the official test set for the final locked evaluation.
