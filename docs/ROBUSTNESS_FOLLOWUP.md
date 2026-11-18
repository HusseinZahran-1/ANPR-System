# Development robustness follow-up

Eight fixed perturbations were evaluated on 110 existing validation crops and five
authority probe crops. Every source was screened against protected names and byte
hashes before decoding. No held-out test pixels were opened. These are derived
views of reused data, not independent examples or full-scene detector evaluation.

| Variant | Validation primary / region exact | Authority primary / region exact |
|---|---:|---:|
| original | 96/110 / 96/110 | 0/5 / 5/5 |
| brightness_085 | 96/110 / 96/110 | 0/5 / 4/5 |
| brightness_115 | 95/110 / 95/110 | 0/5 / 4/5 |
| contrast_085 | 94/110 / 94/110 | 0/5 / 4/5 |
| contrast_115 | 96/110 / 96/110 | 0/5 / 5/5 |
| jpeg70 | 95/110 / 95/110 | 0/5 / 5/5 |
| downscale_075 | 97/110 / 97/110 | 0/5 / 3/5 |
| gaussian_040 | 96/110 / 96/110 | 0/5 / 5/5 |

No numeric readings change between primary and region OCR in any variant.
All five authority failures retain a geometry proposal. Four prefixes are read
as CD below the configured 0.8 threshold; one is read only as C. Visual inspection
confirms that the proposed crops contain both letters. The current failure is
prefix recognition sensitivity rather than a missing proposal. The threshold
was not reduced in response to these development failures. Runtime is unchanged.

## Further verified numeric expansion

Another 15 external source groups were visually inspected with all supplied
variants before showing model predictions. Fourteen distinct numeric identities
have readable digits and accepted OCR crop envelopes. One group is excluded
because all variants clip lower glyph strokes. These are assistant-reviewed
transcriptions; independent human signoff has not been obtained.
One selected variant per identity extends training from 1,018 to 1,032 crops;
validation stays at 110. All remaining variants stay quarantined. No authority
identity is added. Known-identity and parent similarity checks find no flagged
overlaps; the full cross-split audit finds zero exact or configured near candidates.
This does not establish absence of every near duplicate or overlap with the
sealed, untranscribed external test members. No external test image is decoded.

Baseline exact reading on the 14 newly selected training crops is 8/14. This
is a training-fit diagnostic, not an independent accuracy estimate. The controlled fine-tune uses the unchanged v4 checkpoint, seed 42, natural
sampling, batch size 16, backbone/head learning rates 5e-6/5e-5, one backbone-frozen
epoch, up to eight epochs, patience three, and validation CER checkpoint selection
with exact match as tie-breaker. Layout migration to the established runtime
preprocessor is explicitly allowed; other checkpoint metadata remains strict.
CUDA kernels emitted nondeterminism warnings despite deterministic settings;
bitwise reproducibility across reruns is not established.

The run stops after four epochs in 147.21 seconds. The selected epoch one reads
95/110 validation views exactly, with 29/709 character edits, versus 96/110 and
29/709 for the primary baseline. Epoch two reads 97/110 but has higher CER and
is not the checkpoint selected by this declared recipe; it is not separately
evaluated or promoted. On the 14 new training crops, the selected checkpoint
retains 8/14 exact, unchanged from the baseline. These remain fit diagnostics.

With the optional region reader enabled, full-scene regression cohorts change
40/45 to 38/45 and 10/16 to 8/16; the other cohorts remain 12/14 and 5/5.
The candidate fails the existing development gates and is not promoted. No
candidate camera or candidate authority robustness claim is made; those gates
were not run after the measured regressions ruled out promotion. The deployed
primary OCR and detector weights, region threshold and runtime configuration
remain unchanged by this follow-up.

Source attribution remains bau, number plate recognition Jordan version 4,
CC BY 4.0. Local provenance retains original archive/member hashes, source
attribution, reviewed crop boxes and transcripts. Images and labels are not
redistributed.
