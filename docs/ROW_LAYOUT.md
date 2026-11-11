# Optional geometry-based two-row layout

`rectify_unfold_rows_v4` is an experimental OCR layout override. It remains
opt-in; the configuration default retains `rectify_unfold_wide_v3`. A later
local pilot combines v4 with the separately validated optional detector fallback.
No detector or OCR checkpoint is replaced by this change.

The existing aspect-only preprocessing can miss two-row plates whose crop is
wider than 1.8 times its height. The new mode considers crops with aspect ratios
from 1.8 to below 3.2, finds dark connected components, and requires two vertically
separated groups with compatible heights and overlapping horizontal spans.
At least two components above and three below are required. The split is placed
between their vertical extents, then the entire upper image is placed before the
entire lower image without deleting source regions or mirroring characters.
When evidence is insufficient, the mode uses the existing v3 transform exactly.
This uses image geometry only, without expected transcripts or OCR confidence.

`propose_two_row_cut` returns its component evidence and cut. `unfold_at_cut`
returns the transformed image and piecewise source-to-output mapping in
pixel-edge coordinates. Crop coordinates remain available independently from
detector boxes. The mapping is not a claim to localize individual OCR glyphs.

To try the override with compatible private checkpoints, set the private `.env`
value and restart the manually operated server:

```dotenv
OCR_LAYOUT_MODE=rectify_unfold_rows_v4
```

Record the override when comparing checkpoint results. The response metadata
exposes the configured OCR layout. The saved v4 primary CRNN weights retain their
original training metadata; this experiment changes preprocessing explicitly.

## Development evidence

The prototype and actual application were checked separately. Five proposals
occur in 1,032 training crops and none in 110 existing validation crops. All five
proposals were visually inspected. Four previously incorrect numeric readings
become correct, with no correct-to-incorrect changes. Including the existing
optional authority reader, training exact counts change from 1,005/1,032 to
1,009/1,032; validation remains 96/110. These are reused training-fit and
development counts, not independent accuracy. Three gains are from the newly
reviewed numeric batch and one from the preceding public numeric batch.

The five reused authority probe views remain 5/5 with the optional prefix reader.
All four scene regression cohorts remain 40/45, 12/14, 5/5 and 10/16. The frozen
camera cohort remains 112/113 readable exact and 4/5 no-visible-plate views empty.
Ninety-eight readable camera views have OCR training identities; the remaining
15/15 are separate from that training manifest but not independently sampled.
Across 920 derived perturbation views of existing validation/probe crops, every
final text is identical between v3 and the new mode. The earlier authority
perturbation sensitivity remains unresolved. Reused authority and foreign full
scene texts are also identical; 8/9 foreign views still match a Jordanian text
format. Country remains unverified and review is required.

## Full-scene detector limitation

Real-model HTTP uploads of all fourteen newly reviewed external source scenes
return HTTP 200 but **zero detections and zero OCR outputs in both layouts**.
No end-to-end improvement on these external scenes is established. The manual
demo uses the configured camera detector. On a diagnostic source, neither the
camera detector nor the older general detector yields a box at confidence 0.25;
lower-score boxes do not localize the visible plate. Blindly reducing confidence
is not supported by this observation. The crop improvement does not solve
this detector failure by itself. A later primary-first fallback recovers four
of ten reviewed external detector-validation scenes without the measured older
scene regressions; see [fallback evidence](FALLBACK_DETECTOR.md).

The test uses manual demo uploads without archiving them; the physical-camera
event log hash is unchanged. These sources were assigned to OCR training, so
this is a pipeline diagnostic rather than independent evaluation. Their
upstream boxes have been reviewed only as OCR crop envelopes, not certified as
new detector training ground truth. Detector annotation review and a controlled
training expansion are still needed. Images, labels, weights and private
per-image reports remain local; original and external held-out test pixels
remain unopened.
