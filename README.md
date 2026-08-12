# QALF Deepfake Detection

Lightweight video deepfake detection trained on FaceForensics++ and evaluated
cross-dataset on Celeb-DF-v2. Labels are always `0=real`, `1=fake`, and every
reported score is `P(fake)`.

The current architecture and pre-registered SRM decision gate are documented in
[`.docs/CURRENT_MODEL_AND_PLAN.md`](.docs/CURRENT_MODEL_AND_PLAN.md).

## Active models

All active profiles share aligned 160 x 160 full-face crops, EfficientNet-B0,
SBI training, eight training frames, and 12 evaluation frames.

- `texture_only_sbi`: RGB texture control.
- `full_face_sbi`: retained landmark-geometry + RGB baseline A.
- `srm_sbi`: fixed high-pass SRM residual + RGB candidate.
- `learned_srm_v2`: lightweight diverse learned-SRM residual + RGB model.

The upgraded SRM branch applies 30 diverse learnable zero-DC 5 x 5 filters to
grayscale frames, rectifies the residuals, compresses them to three channels,
and uses a small CNN with spatial attention and temporal statistics. It keeps
EfficientNet-B0 as the RGB backbone and does not copy the heavier B4/Transformer
design from the reference repository.

## Environment

Use Python 3.11 and install the CUDA-compatible PyTorch build appropriate for
the target GPU before installing the remaining dependencies:

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install the correct torch/torchvision wheel for this machine first.
python -m pip install -r requirements.txt
python -m pip install -e ".[dev,export]"
```

Do not install a second OpenCV distribution alongside
`opencv-contrib-python`.

## Data workflow

The scripts are the source of truth:

```text
raw videos
  -> scripts/extract_frames.py
  -> scripts/extract_landmarks.py
  -> scripts/audit_manifest.py       # optional explicit audit
  -> scripts/train.py
  -> scripts/evaluate.py
```

Training and evaluation do not rerun a dataset audit. They read the supplied
manifest, extracted frames, and landmark cache directly.

Typical extraction commands are:

```powershell
python scripts\extract_frames.py ffpp `
  --dataset-root $FFPP_RAW `
  --split-root $FFPP_SPLITS `
  --output-root $FFPP_FRAMES `
  --device cuda

python scripts\extract_frames.py celebdf `
  --dataset-root $CELEB_RAW `
  --output-root $CELEB_FRAMES `
  --device cuda

python scripts\extract_landmarks.py `
  --manifest "$FFPP_FRAMES\manifests\ffpp_train.jsonl" `
  --frame-root $FFPP_FRAMES `
  --landmark-root "$FFPP_LM\landmarks" `
  --output-manifest "$FFPP_LM\manifests\ffpp_train_landmarks.jsonl"
```

Repeat landmark extraction for FF++ validation/test and the official Celeb-DF
test manifest. `scripts/audit_manifest.py` remains available when an explicit
integrity report is needed; it is not invoked by train/test runners.

## Locked SRM ablation

Completed texture and geometry controls are reused. This command trains only a
missing seed-42 SRM checkpoint, evaluates all missing rows, and writes a CSV and
Markdown decision report:

```powershell
& "C:\Program Files\Git\bin\bash.exe" ./run_srm_ablation.sh all
```

Individual profiles can be run with:

```powershell
& "C:\Program Files\Git\bin\bash.exe" ./run_train_cross_dataset.sh srm_sbi
& "C:\Program Files\Git\bin\bash.exe" ./run_test_cross_dataset.sh srm_sbi
```

Run additional seeds only after the seed-42 gate passes:

```powershell
$env:QALF_SEED = "17"
& "C:\Program Files\Git\bin\bash.exe" ./run_srm_ablation.sh all
Remove-Item Env:QALF_SEED

$env:QALF_SEED = "73"
& "C:\Program Files\Git\bin\bash.exe" ./run_srm_ablation.sh all
Remove-Item Env:QALF_SEED
```

The fixed-SRM candidate failed that gate. Do not run seeds 17/73 for the
original fusion claim. Diagnose the completed seed-42 checkpoint without
retraining:

```powershell
& "C:\Program Files\Git\bin\bash.exe" ./run_srm_diagnostics.sh
Get-Content "E:/DeepFakeData/experiments/srm_complementarity_seed42.md"
```

The fixed and staged SRM variants are closed. The upgraded model uses a
constrained learnable high-pass bank, stronger residual encoder, reproducible
component initialization, and residual-interaction fusion. Run it once with:

```powershell
& "C:\Program Files\Git\bin\bash.exe" ./run_train_cross_dataset.sh learned_srm_v2
& "C:\Program Files\Git\bin\bash.exe" ./run_test_cross_dataset.sh learned_srm_v2
```

Supported runner modes are `train`, `test`, and `all`. Evaluation always uses
three clips, mean aggregation, 12 frames, horizontal-flip TTA, and a threshold
calibrated on FF++ validation.

## Outputs

Training writes `best.pt`, `config.json`, `run_metadata.json`,
`training_summary.json`, `history.json`, `train.log`, and a training-history
plot. Evaluation writes `metrics.txt`, `metrics.json`, protocol metadata,
prediction CSV files, `eval.log`, and diagnostic plots.

`scripts/benchmark.py`, `scripts/export_onnx.py`, and
`scripts/export_torchscript_int8.py` support deployment measurements and export.
Cached model/preprocessing latency must be reported separately from raw video
decode, face detection, and landmark extraction.

## Scientific protocol

- Select the checkpoint and decision threshold only on FF++ validation.
- Keep splits, preprocessing, SBI mixture, seed, and inference settings fixed
  across the three active profiles.
- Treat Celeb-DF as a fixed cross-dataset development benchmark because its
  results have already been inspected; do not call it an untouched test set.
- Do not alter the pre-registered SRM gate after seeing its result.
- Use a separate untouched dataset for a final unbiased generalization claim.
