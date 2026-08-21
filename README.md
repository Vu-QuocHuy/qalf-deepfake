# TextureSBI deepfake detector

Lightweight video-level deepfake detection with aligned full-face RGB texture,
an EfficientNet-B0 backbone, and Self-Blended Images (SBI) training.

## Current pipeline

```text
video -> extracted frame sequence -> landmark face alignment (preprocessing)
      -> RGB face frames -> EfficientNet-B0 per frame -> mean temporal pooling
      -> clip P(fake) -> mean of three clips = video P(fake)
```

Landmarks are used only to align faces and create the SBI mask. The classifier
does not contain a geometry or SRM branch.

## Environment

Use Python 3.11 and install a CUDA-compatible PyTorch build for the target GPU
before installing the remaining dependencies:

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data workflow

```text
raw videos -> scripts/extract_frames.py -> scripts/extract_landmarks.py
           -> train/evaluate
```

Training and evaluation read the supplied manifests, extracted frames, and
landmark caches directly. They do not run a separate dataset audit during
training or evaluation.

## Train and evaluate

From Git Bash on Windows:

```powershell
& "C:\Program Files\Git\bin\bash.exe" ./run_train.sh
& "C:\Program Files\Git\bin\bash.exe" ./run_test.sh
```

The canonical runner trains for up to 50 epochs with early stopping patience
5, uses EMA decay `0.999`, updates EMA after every optimizer step, selects EMA
weights for FF++ validation, and saves those weights in `best.pt`.

The locked baseline protocol uses FF++ c23, 32-frame clip windows, and 8
texture frames during both training and evaluation. Evaluation uses three
clips per video, mean clip aggregation, horizontal-flip TTA, and the closest-to-
EER threshold calibrated on FF++ validation. Set `QALF_TEST_TEXTURE_FRAMES=12` only for a
separate frame-count ablation.

### Multi-seed baseline

Run the same baseline independently with seeds `17`, `42`, and `73`, then
write per-seed and mean ± standard-deviation reports:

```powershell
& "C:\Program Files\Git\bin\bash.exe" ./run_baseline_seeds.sh
```

Useful overrides are space-separated `QALF_SEEDS`, `QALF_EPOCHS`,
`QALF_TEST_TEXTURE_FRAMES`, `QALF_FORCE_TRAIN=1`, and `QALF_FORCE_TEST=1`.
Each seed receives its own checkpoint and Celeb-DF evaluation directory; the
summary is saved as `qalf_ffpp4_effb0_160_8f_texture_sbi_ema_multiseed.csv/.md`.

### 32-frame / seed-4 experiment

This branch includes a deployment-oriented non-inferiority experiment that
keeps the baseline's eight texture frames and three-clip evaluation while
halving the stored preprocessing sequence. It derives frames `0, 2, ..., 62`
from each canonical 64-frame sequence, producing an approximately 5 FPS,
32-frame view without copying JPEG files. Landmark arrays are subset into
separate cache roots, so the canonical data remains untouched.

Prepare the derived manifests/caches, train seed `4`, and evaluate Celeb-DF:

```powershell
& "C:\Program Files\Git\bin\bash.exe" ./run_32f_seed4.sh
```

The matching protocol is `num_frames=16`, `texture_frames=8`, and three
evaluation clips with starts `0`, `8`, and `16`. Run individual stages with
`prepare_32f_data.sh`, `run_train_32f_seed4.sh`, and
`run_test_32f_seed4.sh`. Set `QALF_SKIP_32F_PREPARE=1` to reuse an already
prepared derived cache. Direct extraction of future videos should use
`scripts/extract_frames.py --frames-per-video 32 --target-fps 5`; subsampling
an already-extracted sequence does not retroactively reduce extraction time.

### Robustness evaluation

Run the evaluation-only corruption protocol with the trained baseline:

```powershell
& "C:\Program Files\Git\bin\bash.exe" ./run_robustness.sh
```

It measures clean performance and degradation under JPEG compression, blur,
downscaling, and additive noise using the locked 8-frame baseline protocol.
Set `QALF_ROBUSTNESS_TEXTURE_FRAMES` only for a separate frame-count
ablation. The script uses the clean FF++ validation threshold and never fits
anything on Celeb-DF. Corruptions are applied after denormalizing RGB tensors
and normalized again before inference.

The default operating-point rule is the closest finite ROC threshold to EER.
Threshold calibration can be repeated without retraining:

```powershell
$env:QALF_THRESHOLD_SELECTION="eer"
& "C:\Program Files\Git\bin\bash.exe" ./run_test.sh
Remove-Item Env:QALF_THRESHOLD_SELECTION
```

Existing checkpoints can be re-evaluated with this rule; the runner recalibrates
on the FF++ validation manifest and writes a separate `_eer_` output directory.
Accuracy, Precision, Recall and F1 from older Youden-J reports must not be reused.

Threshold calibration remains restricted to FF++ validation; the selected
threshold is frozen before Celeb-DF inference. Set the same variable before
`run_ablation_suite.sh` or `run_robustness.sh` to use EER operating points in
those reports.

### One-shot ablation suite

To run the registered comparisons in one resumable job:

```powershell
& "C:\Program Files\Git\bin\bash.exe" ./run_ablation_suite.sh
```

The suite runs five seeds for the full SBI/EMA grid (`baseline`, `no_sbi`,
`no_ema`, and `texture_only`), one seed for the pretrained, augmentation, and
SBI-mixture controls, then evaluates frame
count, clip count, aggregation, TTA, and corruption robustness. Existing
checkpoints and metrics are skipped, so it can safely be restarted. Results
default to `E:/DeepFakeData/experiments/ablation` with short names such as
`baseline_seed42`, `no_sbi_seed42`, `no_ema_seed42`, and
`texture_only_seed42`.
The core comparisons now default to five seeds: `0 17 42 73 123`. After each
video-level evaluation the suite writes `bootstrap_ci.json/.md`, using 2,000
video resamples by default. Set `QALF_BOOTSTRAP_REPS` to change the number of
resamples; the FF++ validation threshold is held fixed during resampling.
Use `QALF_ABLATION_MODE=train`, `eval`, or `robustness` to run one phase only.
Set `QALF_ABLATION_PROFILES="texture_only"` to train/evaluate only the plain
texture control (`--no-sbi --ema-decay 0 --validation-weights raw`) without
starting the other profiles. For example:

```powershell
$env:QALF_ABLATION_PROFILES="texture_only"
$env:QALF_ABLATION_MODE="train"
& "C:\Program Files\Git\bin\bash.exe" ./run_ablation_suite.sh
$env:QALF_ABLATION_MODE="eval"
& "C:\Program Files\Git\bin\bash.exe" ./run_ablation_suite.sh
Remove-Item Env:QALF_ABLATION_PROFILES
Remove-Item Env:QALF_ABLATION_MODE
```

### FF++ in-domain evaluation

To evaluate the already-trained ablation checkpoints on the official FF++ test
split, without retraining:

```powershell
& "C:\Program Files\Git\bin\bash.exe" ./run_ffpp_indomain_ablation.sh
```

The runner evaluates `baseline`, `no_sbi`, `no_ema`, `texture_only`,
`no_pretrain`, `no_aug`, and `sbi_half` using the existing checkpoints under
`E:/DeepFakeData/experiments/ablation`. It uses FF++ validation only for
threshold calibration, evaluates three clips with mean aggregation, and
explicitly includes only `Deepfakes`, `Face2Face`, `FaceSwap`, and
`NeuralTextures`; `FaceShifter` is excluded. Results are written under
`E:/DeepFakeData/experiments/ablation/ffpp_test` with a consolidated
`summary_eer.md/.csv`; the report keeps per-seed rows and also writes
`summary_eer_by_method.csv` with method-level mean ± standard deviation.
The FF++ and Celeb-DF summaries use the same primary metric schema: Accuracy,
Precision, Recall and F1 for the positive class (`label=1`, fake), plus ROC-AUC,
AP, EER, balanced accuracy and ACER. Real is the negative class (`label=0`);
separate real-class Precision/Recall/F1 columns are retained only in raw
`metrics.json` diagnostics, not in the paper summary table.
Set `QALF_FFPP_TEST_MANIFEST` if the official test
manifest is stored at a different path. If only the FF++ validation manifest
exists, do not use it as a final in-domain test: it is already used for model
selection and threshold calibration.
Set `QALF_FFPP_PROFILES="texture_only"` to evaluate only the new texture-only
checkpoints on FF++.

```powershell
$env:QALF_FFPP_PROFILES="texture_only"
& "C:\Program Files\Git\bin\bash.exe" ./run_ffpp_indomain_ablation.sh
Remove-Item Env:QALF_FFPP_PROFILES
```

If FF++ metrics have already been evaluated and only the report must be
recomputed, use summary-only mode. It discovers existing FF++ `metrics.json`
files and never calls the evaluator:

```powershell
$env:QALF_FFPP_SUMMARY_ONLY="1"
$env:QALF_THRESHOLD_SELECTION="eer"
& "C:\Program Files\Git\bin\bash.exe" ./run_ffpp_indomain_ablation.sh
Remove-Item Env:QALF_FFPP_SUMMARY_ONLY
Remove-Item Env:QALF_THRESHOLD_SELECTION
```

## Outputs

Training writes `best.pt`, configuration, history, logs, and a training plot.
Evaluation writes video/clip predictions, metrics in JSON/text, the evaluation
protocol, and ROC/PR/confusion/score-distribution plots. Console output is
limited to training progress and final metrics; hardware and dataset-audit
details are not printed or saved as run metadata.

Landmark caches remain a preprocessing dependency: they align the full-face
crop and provide the SBI face mask. They are not a model branch or a runtime
dataset audit.

Historical geometry/SRM experiments are retired from the source tree and are
not part of the active model.
