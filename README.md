# QALF deepfake detector

Lightweight video-level deepfake detection with aligned full-face RGB frames,
EfficientNet-B0, and Self-Blended Images (SBI) training.

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

The locked protocol uses FF++ c23, 32-frame clip windows, 8 frames during
training, and 12 frames in each of three evaluation clips. Evaluation uses
mean clip aggregation, horizontal-flip TTA, and an FF++ validation Youden-J
threshold.

## Outputs

Training writes `best.pt`, configuration, metadata, history, logs, and a
training plot. Evaluation writes video/clip predictions, metrics in JSON/text,
the evaluation protocol, and ROC/PR/confusion/score-distribution plots.

Landmark caches remain a preprocessing dependency: they align the full-face
crop and provide the SBI face mask. They are not a model branch or a runtime
dataset audit.

See [.docs/CURRENT_MODEL_AND_PLAN.md](.docs/CURRENT_MODEL_AND_PLAN.md) for the
current scientific protocol. Historical geometry/SRM experiments are retired
from the source tree and are not part of the active model.
