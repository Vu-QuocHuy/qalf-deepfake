# QALF Self-Blended Images implementation plan

Status: completed historical implementation record, 2026-08-12

SBI was implemented and improved the seed-42 full-face result from 0.8209 to
0.8325 Celeb-DF AUC. The active model and next decision are now documented in
`CURRENT_MODEL_AND_PLAN.md`; this file is retained only for implementation
provenance.

## 1. Locked reference

The reference is the deterministic EfficientNet-B0 full-face experiment. Do not
change this checkpoint, overwrite its output directory, or combine another
technique with SBI in the first run.

| Item | Locked value |
| --- | --- |
| Celeb-DF AUC | 0.8209 |
| Celeb-DF average precision | 0.8964 |
| Celeb-DF texture AUC | 0.8162 |
| Celeb-DF ACER | 0.2522 |
| Backbone | EfficientNet-B0 |
| Train input | full face, 160 px, 8 texture frames |
| Texture pooling | mean |
| Checkpoint weights | raw, no EMA |
| Seed | 42, deterministic |
| Evaluation | 12 texture frames, 3 clips, mean, flip TTA |
| Threshold source | FF++ validation only |

Celeb-DF is a locked test set. It must not be used to choose the SBI ratio,
mask parameters, augmentation ranges, epoch, or checkpoint. If SBI needs
hyperparameter tuning after the first pre-registered run, use an FF++
leave-one-manipulation-out development protocol instead.

## 2. Scope of the first experiment

Create one new profile named `full_face_sbi`. Everything except the training
sample generator and the geometry-loss mask remains identical to `full_face`.

The target training mixture is:

| Sample stratum | Batch share | Fused target | Texture target | Geometry loss |
| --- | ---: | ---: | ---: | --- |
| Pristine FF++ real | 50% | 0 | 0 | enabled, target 0 |
| Original FF++ fake | 25% | 1 | 1 | enabled, target 1 |
| SBI from FF++ real | 25% | 1 | 1 | disabled |

SBI samples may only be generated from real videos in the FF++ training split.
Validation, threshold calibration, and all evaluation datasets remain entirely
unchanged.

The first experiment must not enable EMA, MixStyle, temporal dynamics pooling,
dual-view input, EfficientNet-B1, a larger image size, or more training frames.

## 3. SBI generator design

Add an isolated training-only module under `qalf/data/` rather than embedding
all SBI operations in `QALFVideoDataset.__getitem__`.

For every SBI clip:

1. Build the same aligned full-face RGB frames used by the current baseline.
2. Derive a face-region mask from the available landmarks in aligned image
   coordinates.
3. Create pseudo source and target views from the same pristine frame.
4. Apply source/target appearance differences such as color, contrast,
   resolution, sharpening or blur.
5. Apply a small spatial transform and smooth deformation to the face mask.
6. Randomize blend strength and blur the mask boundary before compositing.
7. Apply the existing whole-image texture augmentation after blending.

Spatial transform, mask family, blend strength, and appearance transform must
be coherent across all texture frames in one clip. Per-frame independent SBI
parameters are prohibited in the first experiment because they could introduce
an artificial flicker shortcut that is absent from ordinary video deepfakes.

The implementation should be based on the SBI method, but remain native to the
existing NumPy/OpenCV pipeline and avoid introducing Albumentations or copying
the research-only reference implementation.

References:

- https://openaccess.thecvf.com/content/CVPR2022/html/Shiohara_Detecting_Deepfakes_With_Self-Blended_Images_CVPR_2022_paper.html
- https://github.com/mapooon/SelfBlendedImages

## 4. Dataset and sampler changes

Represent the three strata explicitly; do not randomly change a record label
inside `__getitem__`. Dynamic labels would make the current balanced sampler's
weights incorrect and make a run harder to reproduce.

Required behavior:

- expose the effective label and sample type before sampling;
- sample the three strata at the locked 50/25/25 proportions;
- create SBI companions only for FF++ training-real records;
- retain the existing random temporal clip selection;
- return `sample_type` and `geometry_loss_mask` in each batch;
- preserve the old dataset behavior when SBI is disabled.

No dataset audit or full-manifest scan is to be added to train or evaluation
startup. A separate optional preview command may save a small real/SBI image
grid for manual inspection before the expensive run.

## 5. Loss changes

The current QALF loss applies one binary label to fused, geometry, and texture
outputs. Extend it so that:

- fused BCE is calculated for every sample;
- texture BCE is calculated for every sample;
- geometry BCE is calculated only where `geometry_loss_mask == 1`;
- the masked geometry loss is normalized by the number of valid samples;
- a batch with no valid geometry samples returns a differentiable zero geometry
  loss rather than NaN;
- old batches without the mask behave exactly as before.

Log the number or fraction of SBI samples per epoch so the intended mixture can
be verified from training artifacts.

## 6. Configuration and runners

Add explicit, checkpointed configuration fields for SBI. At minimum record:

- enabled/disabled;
- mixture proportions;
- blend-strength choices;
- affine/deformation ranges;
- source/target appearance transform ranges;
- temporal-coherence mode.

Add `full_face_sbi` to the existing cross-dataset train and test runners with a
new output directory. Do not alter the `full_face` profile or its paths.

Evaluation must print the SBI training metadata loaded from the checkpoint, but
must never generate SBI samples.

## 7. Verification before the full run

The implementation is not ready for GPU training until all of these checks pass:

- [ ] real and original-fake samples are unchanged when SBI is disabled;
- [ ] an SBI sample has the same tensor shape and normalization as full-face;
- [ ] an SBI sample differs from its pristine source inside the face region;
- [ ] blend masks are finite, bounded, non-empty, and not full-image masks;
- [ ] all frames in a clip share the intended coherent SBI parameters;
- [ ] only training-real records can produce SBI companions;
- [ ] SBI labels are fake for fused/texture and masked for geometry;
- [ ] masked geometry loss matches the legacy loss when every mask value is 1;
- [ ] validation and evaluation paths never invoke the SBI generator;
- [ ] deterministic runs reproduce generated SBI samples for the same seed;
- [ ] Python compilation, unit tests, and shell syntax checks pass;
- [ ] a short training smoke test produces finite loss and a loadable checkpoint.

## 8. Locked experiment and decision rule

Run seed 42 once with the exact baseline hyperparameters. Select the checkpoint
by FF++ validation AUC with patience 5, as in the baseline. Then run the locked
Celeb-DF protocol exactly once.

Primary decision metric: Celeb-DF video-level ROC AUC.

| Outcome | Decision |
| --- | --- |
| AUC >= 0.8250 and texture AUC > 0.8162 | candidate improvement; repeat seeds 17 and 123 |
| AUC 0.8162--0.8249 | inconclusive; do not combine techniques; estimate seed variance if budget permits |
| AUC < 0.8162 | reject the SBI hybrid profile |

Average precision and texture AUC are secondary ranking metrics. Threshold,
accuracy, F1, ACER, and EER must be reported, but they must not override the AUC
decision because the project objective is ranking performance. A claimed final
improvement requires the mean and standard deviation over seeds 42, 17, and 123.

## 9. Work order

- [x] Freeze and document the 0.8209 full-face baseline.
- [x] Record dual-view as a negative standalone ablation.
- [x] Implement and test the temporally coherent SBI generator.
- [x] Implement explicit three-stratum sampling.
- [x] Implement geometry-loss masking with backward compatibility.
- [x] Add configuration, metadata, reporting, and `full_face_sbi` runners.
- [ ] Run automated checks and a short smoke train.
- [ ] Review a small SBI preview grid manually.
- [ ] Train the locked seed-42 experiment.
- [ ] Evaluate once on the locked Celeb-DF protocol.
- [ ] Apply the decision rule before planning another experiment.

## 10. Deferred work

Only after SBI hybrid passes the decision rule may the project test, one at a
time, a different SBI mixture, pure real/SBI training, EMA, B1, or another
temporal model. None of those changes belong in the initial SBI implementation.
