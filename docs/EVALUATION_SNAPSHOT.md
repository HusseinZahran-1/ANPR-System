# Measured development snapshot - 18 November 2026

This report summarizes the already evaluated optional primary/fallback/recovery
composition with aligned fragment geometry. Cohorts were evaluated on
14 November; one deliberately selected corrected-fragmentation example was
rerun with the current plate-parts output on 18 November. This is a development
preview, not an independent held-out test or production accuracy claim.
No detector or OCR was retrained, no runtime settings were changed, and no
private image or transcript is published with this report.

## Twenty external positive development scenes

These combine the prior ten external evaluation sources with ten additional
reviewed sources. Both cohorts have been reused for development. Matching uses
IoU >= 0.50 and approximate assistant-reviewed reference boxes. Detection counts
are before the single-output cap; model-level max_det still applies.

| Metric | Result | Definition / scope |
| --- | ---: | --- |
| Complete plate exact match with localization | 60.0% (12/20) | Earlier two-stage composition: 40.0% (8/20) |
| End-to-end character error rate | 36.43% (51/140) | Levenshtein edits / reference characters; includes missed detections and wrong-region readings |
| Post-geometry detector precision | 92.86% | 13 TP / (13 TP + 1 FP) |
| Post-geometry detector recall | 65.0% | 13 TP / (13 TP + 7 FN) |
| Post-geometry detector F1 | 76.47% | 2 TP / (2 TP + FP + FN) |
| Raw selected-model component precision | 81.25% | 13 TP / (13 TP + 3 FP) |
| Raw selected-model component recall | 65.0% | 13 TP / (13 TP + 7 FN) |
| Raw selected-model component F1 | 72.22% | Preserves detector fragmentation errors in reporting |
| Duplicate matches | 0 | Both raw and post-geometry; unmatched fragments can be FP without qualifying as duplicates |
| Composed-system mAP@50 / mAP@50:95 | Not evaluated yet | Fixed-threshold counts are not an AP sweep |
| Independent code-boundary accuracy | Not evaluated yet | Code split is explicit format metadata, not independently verified visual boundary prediction |

Prior ten: 6/10 exact, 28 edits / 70 characters, post-geometry 6 TP / 0 FP /
4 FN. Additional ten: 6/10 exact, 23 edits / 70 characters, post-geometry
7 TP / 1 FP / 3 FN. Do not interpret 1 - CER as an independently measured
character-classification accuracy. This is not crop-only OCR CER.

## Other reused cohorts

| Cohort | Result | Limitation |
| --- | ---: | --- |
| Camera readable views | 112/113 exact (99.12%); CER 1/754 (0.13%) | 98 views have OCR training identities; remaining 15 are correct but not independently sampled |
| Older full-scene development cohorts | 67/80 exact (83.75%) | Separate cohorts: 40/45, 12/14, 5/5, 10/16; reused development |
| Camera no-visible-plate views | 4/5 empty | Small selected sample; not background false-positive rate |
| Authority views | 5/5 exact | Five reused views of three physical plates; perturbation fragility remains |
| Derived plate-free crops | 20/20 empty | Grouped with parent vehicles; not independent no-plate camera captures |
| Foreign format false matches | 8/9 | Country remains unverified; no reliable foreign rejection claim |

## Selected example rerun

Current full-image inference correctly reads the selected fragmentation example
and exposes complete text, code and number. Two original fallback fragments
become one geometry union, with IoU 0.856 against its approximate reviewed
reference. The highest original fragment score is 10.75%, and the OCR sequence
score is 93.40%. Neither score is measured accuracy or a calibrated probability
of a correct reading. Human review is required. Selection illustrates a known
successful correction and must not be described as a random or blind test.

Three warmups followed by eight measured full-pipeline runs on this laptop's
CUDA device give a median of 70.2 ms on this one image. Loading and network I/O
are excluded. This is not throughput, a multi-scene benchmark or a latency SLA.

Original raw data and held-out test pixels are unchanged/unopened. For stage
configuration and limitations, see [recovery and fragments](RECOVERY_AND_FRAGMENTS.md)
and [plate-parts semantics](PLATE_PARTS.md).
