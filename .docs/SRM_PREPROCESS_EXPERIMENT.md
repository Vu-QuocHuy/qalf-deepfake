# SRM preprocessing experiment

This branch tests a deliberately conservative frequency cue. Three fixed
zero-sum 5x5 high-pass kernels (one isotropic SRM-style residual and horizontal
and vertical directional residuals) are applied to the normalized RGB frames.
The bounded three-channel residual is added to RGB before the pretrained
EfficientNet-B0 input. A single learnable residual scale starts at `0.05`.

The experiment keeps the baseline protocol unchanged: full-face RGB, 160px,
32 sampled frames, 8 texture frames, SBI, mean aggregation over three clips,
and EMA (`0.999`). It adds no auxiliary encoder and no fusion gate. Existing
baseline checkpoints remain loadable because the preprocessing module is only
constructed when `model.srm_preprocess=true`.

## Run

```bash
./run_train_srm.sh
./run_test_srm.sh
```

The result is exploratory until it is compared with the untouched baseline on
the same seed and protocol. Report both AUC/AP/EER and the learned residual
scale; do not claim that SRM helps unless the cross-dataset improvement is
reproducible across seeds.
