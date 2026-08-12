# QALF Deepfake Detection

Quality-Aware Landmark–Texture Fusion trained on FaceForensics++ and evaluated
zero-shot on Celeb-DF-v2. The label convention is always `0=real`, `1=fake`, and
the model score is `P(fake)`.

The retained architecture, three-seed evidence, and next decision are documented
in [`CURRENT_MODEL_AND_PLAN.md`](CURRENT_MODEL_AND_PLAN.md). Review that file
before starting another experiment.

## Canonical workflow

The supported workflow is local and script-based on Windows:

```text
raw videos
  -> scripts/extract_frames.py
  -> scripts/extract_landmarks.py
  -> scripts/audit_manifest.py (optional, explicit)
  -> scripts/train.py
  -> scripts/evaluate.py
```

The notebooks under `notebooks/` are archived Kaggle utilities. They are not the
source of truth for dependencies, defaults, QC rules, or local execution. All new
experiments should use the Python scripts above so extraction and training share
the same package implementation.

Protocol defaults:

- 64 ordered face crops per source video at a target 10 FPS.
- 256x256 JPEG crops with MTCNN main-face tracking.
- FF++ methods: Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures.
- MediaPipe Tasks Face Landmarker, IMAGE mode, 468 XYZ points.
- Detection-ratio threshold 75% for both frame and landmark QC.
- Training clip length 32 frames sampled from the extracted 64-frame sequence.
- `0=real`, `1=fake` in every generated manifest.

## 1. Windows environment

Use 64-bit Python 3.11 and a current NVIDIA driver. RTX 50-series requires a
Blackwell-capable PyTorch wheel; the setup script installs the CUDA 13.0 build
before any other package can alter it.

From PowerShell in the repository root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_windows.ps1
.\.venv-local\Scripts\Activate.ps1
```

The setup script performs this ordered installation:

1. Create `.venv-local` with Python 3.11.
2. Install CUDA `torch` and `torchvision` from the `cu130` index.
3. Install the shared dependencies from `requirements.txt`.
4. Install `facenet-pytorch==2.6.0` with `--no-deps` so its obsolete
   `torch<2.3` metadata cannot downgrade the RTX 50-compatible wheel.
5. Install this project and run the environment verifier.

Run the verifier again at any time:

```powershell
python scripts\verify_environment.py --require-cuda
```

Do not run `pip install facenet-pytorch` without `--no-deps`, and do not install
a second OpenCV distribution into the same environment. The canonical OpenCV
package is `opencv-contrib-python`, required by MediaPipe and also used by QALF.

## 2. Raw dataset layout

The FF++ loader accepts either the official nested structure or this flattened
C23 structure:

```text
F:\DeepFakedata\raw\ffpp\FaceForensics++_C23\
  original\
  Deepfakes\
  Face2Face\
  FaceShifter\
  FaceSwap\
  NeuralTextures\

F:\DeepFakedata\raw\split_ffpp\
  train.json
  val.json
  test.json
```

Celeb-DF-v2 must contain its official test list:

```text
F:\DeepFakedata\raw\celebdf\
  Celeb-real\
  Celeb-synthesis\
  YouTube-real\
  List_of_testing_videos.txt
```

## 3. Extract frames locally

Set paths once in PowerShell:

```powershell
$FFPP_RAW = "F:\DeepFakedata\raw\ffpp\FaceForensics++_C23"
$FFPP_SPLITS = "F:\DeepFakedata\raw\split_ffpp"
$CELEB_RAW = "F:\DeepFakedata\raw\celebdf"

$FFPP_FRAMES = "F:\DeepFakedata\outputs_duong_huy\data\extracted\ffpp"
$CELEB_FRAMES = "F:\DeepFakedata\outputs_duong_huy\data\extracted\celebdf"
```

Extract all official FF++ splits:

```powershell
python scripts\extract_frames.py ffpp `
  --dataset-root $FFPP_RAW `
  --split-root $FFPP_SPLITS `
  --output-root $FFPP_FRAMES `
  --device cuda
```

Extract the official Celeb-DF-v2 test list:

```powershell
python scripts\extract_frames.py celebdf `
  --dataset-root $CELEB_RAW `
  --output-root $CELEB_FRAMES `
  --device cuda
```

Both commands resume safely by default. Configuration fingerprints prevent a
run with changed FPS, crop, detector, or QC settings from silently reusing old
records. Expected output:

```text
data\extracted\ffpp\
  frames\ffpp\{train,val,test}\...
  manifests\ffpp_train.jsonl
  manifests\ffpp_val.jsonl
  manifests\ffpp_test.jsonl
  frame_extraction_qc.json

data\extracted\celebdf\
  frames\celebdf_v2\test\...
  manifests\celebdf_test.jsonl
  frame_extraction_qc.json
```

Configured face-QC exclusions are documented and do not make the command fail.
Decode, missing-file, or other unexpected errors still fail the command. Add
`--strict` when an experiment requires zero exclusions.

## 4. Extract landmarks locally

```powershell
$FFPP_LM = "F:\DeepFakedata\outputs_duong_huy\data\landmarks\ffpp-landmark"
$CELEB_LM = "F:\DeepFakedata\outputs_duong_huy\data\landmarks\celebdf-landmark"
```

The first command automatically downloads and SHA-256 verifies the official
`models\face_landmarker.task` file.

```powershell
python scripts\extract_landmarks.py `
  --manifest "$FFPP_FRAMES\manifests\ffpp_train.jsonl" `
  --frame-root $FFPP_FRAMES `
  --landmark-root "$FFPP_LM\landmarks" `
  --output-manifest "$FFPP_LM\manifests\ffpp_train_landmarks.jsonl"

python scripts\extract_landmarks.py `
  --manifest "$FFPP_FRAMES\manifests\ffpp_val.jsonl" `
  --frame-root $FFPP_FRAMES `
  --landmark-root "$FFPP_LM\landmarks" `
  --output-manifest "$FFPP_LM\manifests\ffpp_val_landmarks.jsonl"

python scripts\extract_landmarks.py `
  --manifest "$FFPP_FRAMES\manifests\ffpp_test.jsonl" `
  --frame-root $FFPP_FRAMES `
  --landmark-root "$FFPP_LM\landmarks" `
  --output-manifest "$FFPP_LM\manifests\ffpp_test_landmarks.jsonl"

python scripts\extract_landmarks.py `
  --manifest "$CELEB_FRAMES\manifests\celebdf_test.jsonl" `
  --frame-root $CELEB_FRAMES `
  --landmark-root "$CELEB_LM\landmarks" `
  --output-manifest "$CELEB_LM\manifests\celebdf_test_landmarks.jsonl"
```

The canonical default is IMAGE mode to match the completed dataset. VIDEO mode
is available only as an explicit ablation via `--video-mode`; do not mix IMAGE
and VIDEO caches in one experiment.

## 5. Audit before training

```powershell
python scripts\audit_manifest.py `
  --manifest `
    "$FFPP_LM\manifests\ffpp_train_landmarks.jsonl" `
    "$FFPP_LM\manifests\ffpp_val_landmarks.jsonl" `
  --frame-root $FFPP_FRAMES `
  --landmark-root "$FFPP_LM\landmarks" `
  --expected-frames 64

python scripts\audit_manifest.py `
  --manifest "$CELEB_LM\manifests\celebdf_test_landmarks.jsonl" `
  --frame-root $CELEB_FRAMES `
  --landmark-root "$CELEB_LM\landmarks" `
  --expected-frames 64
```

Proceed only when both reports contain `"failures": []`. Documented exclusions
remain visible in the extraction reports and must be reported in the paper.

## 6. Train FF++ with 32-frame clips

```powershell
python scripts\train.py `
  --config configs\ffpp_to_celebdf.json `
  --train-manifest "$FFPP_LM\manifests\ffpp_train_landmarks.jsonl" `
  --val-manifest "$FFPP_LM\manifests\ffpp_val_landmarks.jsonl" `
  --frame-root $FFPP_FRAMES `
  --landmark-root "$FFPP_LM\landmarks" `
  --output-dir outputs\qalf_ffpp_32f
```

The balanced sampler compensates for one real sequence versus five fake methods.
The best checkpoint and decision threshold are selected only on FF++ validation.
On Windows, set `training.num_workers=0` for the first smoke run; use 2 or 4 after
the pipeline is confirmed stable.

### Reproducible experiment profiles

The active profiles preserve the locked EfficientNet-B0 full-face protocol.
Flip TTA is evaluation-only, and the Celeb-DF decision threshold is calibrated
on FF++ validation:

```powershell
& "C:\Program Files\Git\bin\bash.exe" ./run_train_cross_dataset.sh full_face_sbi
& "C:\Program Files\Git\bin\bash.exe" ./run_test_cross_dataset.sh full_face_sbi
```

The only retained geometry candidate is run separately:

```powershell
& "C:\Program Files\Git\bin\bash.exe" ./run_train_cross_dataset.sh geometry_candidate
& "C:\Program Files\Git\bin\bash.exe" ./run_test_cross_dataset.sh geometry_candidate
```

The required texture-only SBI control uses the same data and optimization
protocol:

```powershell
& "C:\Program Files\Git\bin\bash.exe" ./run_train_cross_dataset.sh texture_only_sbi
& "C:\Program Files\Git\bin\bash.exe" ./run_test_cross_dataset.sh texture_only_sbi
```

To isolate modality dropout from reliability supervision without rerunning the
completed three-seed P1 suite:

```powershell
Remove-Item Env:QALF_SEED -ErrorAction SilentlyContinue
& "C:\Program Files\Git\bin\bash.exe" ./run_geometry_failure_diagnostic.sh all
```

This command is locked to seed 42 and trains only `geometry_dropout_only`. It
requires the completed historical `geometry_i2_reliability` checkpoint.

Use `QALF_SEED` for the locked seeds. Seed 42 keeps the historical path; other
seeds receive an automatic suffix and cannot overwrite it:

```powershell
$env:QALF_SEED = "17"
& "C:\Program Files\Git\bin\bash.exe" ./run_geometry_ablation.sh all
Remove-Item Env:QALF_SEED
```

The core now supports only the retained full-face B0 mean-pooling model. Failed
B1, EMA, MixStyle, dual-view, and texture-dynamics implementations have been
removed.

## 7. Evaluate without test-set tuning

FF++ test:

```powershell
python scripts\evaluate.py `
  --checkpoint outputs\qalf_ffpp_32f\best.pt `
  --manifest "$FFPP_LM\manifests\ffpp_test_landmarks.jsonl" `
  --frame-root $FFPP_FRAMES `
  --landmark-root "$FFPP_LM\landmarks" `
  --output-dir outputs\qalf_ffpp_32f\ffpp_test
```

Celeb-DF-v2 zero-shot test:

```powershell
python scripts\evaluate.py `
  --checkpoint outputs\qalf_ffpp_32f\best.pt `
  --manifest "$CELEB_LM\manifests\celebdf_test_landmarks.jsonl" `
  --frame-root $CELEB_FRAMES `
  --landmark-root "$CELEB_LM\landmarks" `
  --output-dir outputs\qalf_ffpp_32f\celebdf_test
```

The evaluator reuses the FF++ validation threshold stored in the checkpoint. It
never selects a threshold on Celeb-DF test.

## Research controls

- Keep extraction settings, split files, excluded-video rule, and random seed
  fixed across ablations.
- Treat `aligned_motion_3d` as the locked geometry representation.
- Compare only the SBI baseline, the retained geometry candidate, and the next
  required texture-only SBI control.
- Select every threshold and model choice on FF++ validation, then run each test
  set once for the final report.
- Do not use further Celeb-DF results to tune the model; see the decision gates in
  `CURRENT_MODEL_AND_PLAN.md`.
- Report cached model latency separately from raw video decode, MTCNN, landmark,
  and end-to-end streaming latency.
