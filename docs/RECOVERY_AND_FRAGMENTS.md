# Optional recovery and aligned plate fragments

When a full plate is split into overlapping partial detections, selecting only
one box can drop the registration prefix. Another failure is an empty primary
and fallback stage on a readable two-row plate. This opt-in pilot addresses
these two observed development failures while retaining the first nonempty
model stage and its original confidence scale.

## Configuration and stage order

```dotenv
FALLBACK_DETECTOR_PATH=models/local/fallback.pt
FALLBACK_DETECTION_CONF=0.05
RECOVERY_DETECTOR_PATH=models/local/recovery.pt
RECOVERY_DETECTION_CONF=0.05
RECOVERY_OCR_LAYOUT_MODE=rectify_unfold_rows_v5
FALLBACK_MERGE_FRAGMENTS=true
DETECTION_MAX_PLATES=1
```

Both features default off. Paths above are placeholders for trusted compatible
local weights, which are not distributed. Recovery requires a configured
fallback. A missing configured model returns 503; detector failures propagate
rather than being interpreted as an empty scene. Invalid merge booleans fail
configuration instead of silently enabling a feature.

The measured local composition uses the retained primary at 0.25, the balanced
external detector's best-by-mAP checkpoint at 0.05, and then its final-epoch
checkpoint at 0.05 only if both earlier stages return no boxes. Both added
checkpoints are YOLOv8n snapshots of the same completed training run. They are
not independent architectures, and this follow-up did not retrain a model.
Replacing the best snapshot outright gave no net exact-reading gain and was
rejected. OCR weights remain frozen.

Primary and first fallback retain v4 layout. Only recovery uses v5: its row
proposal allows components further left (0.15 rather than 0.28 of crop width)
and wider connected components (aspect up to 2.0 rather than 1.3). All other
row tests remain unchanged; without a proposal it falls back to v4. Layout
selection is explicit per crop and never mutates shared pipeline settings.
Width-512 and independent-row OCR ablations were rejected.

## Fragment geometry and evidence

The merge option applies only to the selected fallback or recovery model, and
only in the single-plate profile. Primary results and multi-plate profiles are
unchanged. There is no mixing of boxes or confidence scores across models.

Two components must satisfy every fixed condition: IoU at least 0.30,
horizontal overlap at least 0.60 of the smaller width, vertical overlap at
least 0.85 of the smaller height, height ratio at least 0.70, vertical-center
distance at most 0.25 of the smaller height, and union width at most 1.6 times
the larger width. Groups require compatibility between every original pair;
a transitive bridge cannot connect incompatible objects. Labels and OCR text
do not enter this decision. The union remains an axis-aligned source-image box.

Results retain `fragment_boxes`, `fragment_confidences`, and
`merged_fragment_count`. `det_conf` stays the highest original component score,
with `det_conf_scope=highest_scoring_fragment`; it is not a calibrated union
confidence. `fragment_input_candidate_count` reports original component count,
while `detector_candidate_count` reports post-geometry count before the output
cap. `fragment_merge_applied` exposes whether geometry changed. Raw model box
errors must still be reported separately from post-geometry results. Fallback
and recovery readings always require human review and never verify country.

## Measured development results

| Reused cohort | Deployed fallback | Recovery plus fragments |
| --- | ---: | ---: |
| Ten external detector-training scenes | 6/10 exact | 6/10 exact |
| Ten prior external validation scenes | 4/10 exact | 6/10 exact |
| Ten additional reviewed external scenes | 4/10 exact | 6/10 exact |
| Twenty derived plate-free crops | 20/20 empty | 20/20 empty |

Recovery adds two prior-validation readings. Fragment unions fix two partial
readings in the additional cohort; recovery alone does not improve that cohort.
On the additional ten, raw selected-model boxes still produce 7 TP, 3 unmatched
FP and 3 FN at IoU 0.5. Post-geometry boxes produce 7 TP, 1 unmatched FP and 3 FN.
Duplicate matches are zero in both; overlapping fragments can count as unmatched
boxes rather than duplicates when they fail the matching threshold. One primary
box still targets the wrong small region. First-nonempty composition cannot
repair that error. Four of ten remain incorrect in each external evaluation
cohort. Multi-vehicle fragment behavior remains unvalidated.

Older scene exact counts remain 40/45, 12/14, 5/5 and 10/16, with 80 TP,
0 FP and no duplicate matches across their uncapped detector outputs. The
120-view camera cohort remains 112/113 readable exact and 4/5 no-visible-plate
views empty. Ninety-eight readable views have OCR training identities; the
other fifteen are correct but are not independently sampled. Five reused
CD/SN views of three physical plates remain 5/5. Prefix sensitivity under
mild perturbations remains unresolved. Eight of nine foreign views still
match a Jordanian text format, so reliable country rejection is unresolved.

These are development counts, not production accuracy. Additional sources
were visually reviewed before prediction, then reused to design fragment
geometry; they are no longer an independent test. Their visual reference
boxes support approximate IoU-0.5 assessment, not precise independent box
annotation. Derived background crops share parent-vehicle train/validation
groups; they are not independent no-plate camera captures. Ten reviewed
training-background crops are available for future work but were not trained
on. Manual review is by the assistant, without independent human signoff.

## Integration and data boundaries

Real-model HTTP tests upload 55 sources through both camera ingestion and the
manual demo: five authority views, twenty prior external sources, ten additional
sources and twenty derived backgrounds. Demo and camera results match direct
inference, including stage and fragment counts. Every upload appears as the
latest live event, and all 55 retrieved images match source bytes. The temporary
archive contains 55 events; the physical-camera archive hash is unchanged.
The live feed deliberately returns the latest ten events. This verifies
software integration, not connectivity to a camera on a different network.

All 7,191 original raw file hashes remain unchanged. Original held-out test
pixels and sealed external test members remain unopened. Images, private
transcripts, weights, camera secrets and per-image logs are excluded from GitHub.
