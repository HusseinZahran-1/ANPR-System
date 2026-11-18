# Auxiliary OCR development experiment

The current CD/SN errors persist in the deployed OCR. A manually localized
letter-region diagnostic offers a direction to investigate: on five reused
current-format development views, the project recognizer reads 0/5 prefix
regions exactly, while an existing local pretrained English recognizer reads
4/5 after case folding. These are manually cropped letters, not automatic
localization or complete plate results. No correction is deployed from this
diagnostic.

Nine visually verified training prefix crops come from only two existing
physical training identities. Five probe regions remain outside training.
The new training experiment combines ordinary full-plate CTC with auxiliary
prefix-region CTC and teacher retention on numeric examples. Prefix-region
forwards keep batch-normalization running statistics fixed, so tiny letter
crops do not redefine the statistics used for full plates. Probe images are
not used to select checkpoints. Epoch-zero baseline weights are eligible;
training does not by itself establish improvement.

Only BatchNorm layers switch to evaluation mode for the auxiliary forward.
Recurrent layers remain in training mode so cuDNN can compute their backward
pass. A synthetic recurrent-network test verifies backward succeeds, affine
gradients remain available and BatchNorm running statistics stay unchanged.

`tools/ocr_auxiliary_losses.py` supplies a generic retention-loss helper. It
computes teacher KL over selected training rows' valid CTC timesteps, detaches
the teacher and excludes padding before probability computation. Unselected
rows and padded frames have zero gradient. An empty selection returns a
differentiable zero, including when excluded values are non-finite. The helper
has no plate-format prior, text replacement or runtime inference behavior.

Synthetic tests verify the gradient scope, non-finite padding exclusion,
empty selection, equal-logit loss and invalid arguments:

```sh
python -m pytest tests/test_ocr_auxiliary_losses.py -q
```

Per-image review receipts, training configuration, weights and model
experiments remain local. Review [experiment results](EXPERIMENT_STATUS.md)
for measured outcomes and [upgrade policy](MODEL_UPGRADES.md) for promotion
requirements. Broader numeric, authority, camera and localization regressions
must be checked before changing the deployed pipeline.

The completed full-model candidates improved the reused authority probe to
1/5 and 2/5 but regressed on an existing numeric scene. Classifier-only
follow-ups preserved numeric results but did not improve complete authority
readings. No candidate from these experiments is deployed. Isolated prefix
training fit must not be reported as full-plate accuracy.
