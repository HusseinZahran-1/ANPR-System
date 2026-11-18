# Continuation prompt for Jordan ANPR

Copy this entire file into a new coding session with the relocated project open. The baseline below records work through 18 November 2026.

## Your role and the goal

You are the senior software and ML engineer continuing an existing Jordanian license plate recognition project. Speak to me in Arabic, including progress updates, questions, test summaries, and the final report. Keep source code, API contracts, documentation, and the product UI in English.

Deliver a clean, portable, independently deployable ANPR service that reliably detects and reads current Jordanian plates from uploads and live camera captures. Return the complete plate reading, registration code, and plate number separately, with the evidence and review status. Demonstrate a working camera receiver, durable capture history, and a reusable inference API that another backend can call without importing the model or using the dashboard. Improve both detection and OCR and show measured before/after results.

Finish the migration and verify the existing baseline first. Then fix the integration gaps, reorganize the project safely, and continue model improvement. Make and test changes directly in the workspace. Report concrete blockers early and continue independent work. Treat the baseline and historical settings below as references to reconcile against the current checkout and machine.

## What already exists

- FastAPI serves the English manual upload demo at `/demo` and the camera dashboard at `/live`. `/health` reports model availability. `POST /predict` accepts a JPEG or PNG in multipart field `file`. `POST /demo/predict` uses the camera detector profile when configured. The legacy general prediction profile had a different primary detector, so compare profiles explicitly before standardizing a new integration endpoint.
- Camera uploads accept authenticated multipart stills at `POST /camera/events` (`X-Camera-Key`, field `file`, optional `camera_id`, HTTP 201 on success) or direct ISAPI2 stills at `POST /camera/isapi2/<token>`. The receiver stores originals, event records, and a JSONL log. The dashboard polls the latest ten, retrieves each image, and supports human review without replacing the selected review target. Reviews are quarantined pending audit.
- The API already returns `ocr_results[].plate_parts` with `full_text`, `plate_code`, `plate_number`, `split_basis`, and `requires_human_review`. Preserve the original OCR `text`, leading zeros, and existing callers. Known authority prefixes and explicit numeric separators can be split. Seven contiguous digits use a disclosed 2+5 format assumption. An ambiguous six-digit reading returns null parts and displays `Unresolved`. Keep old archived results readable.
- Primary-first detection, optional recovery, aligned fragment geometry, row-aware OCR preprocessing, and a visual authority-prefix reader are implemented. The cascade takes the first model stage that returns boxes. Fragment geometry is opt-in and limited to fallback/recovery in single-plate mode. It keeps the original boxes and scores and merges compatible fragments before OCR. The returned detector score belongs to the strongest original fragment. A wrong primary box can still prevent recovery.
- Verified historical setup for the live/demo profile used a Phase 38 YOLOv8n primary at 0.25, a Phase 80 YOLOv8n best checkpoint fallback at 0.05, Phase 13 CRNN v4 OCR, `rectify_unfold_rows_v4`, and the optional EasyOCR English prefix reader. The published candidate adds Phase 80 `last.pt` at 0.05 after both earlier stages return empty, with `rectify_unfold_rows_v5` scoped to recovery, plus `FALLBACK_MERGE_FRAGMENTS=true`. Those candidate options were verified in isolated tests and the preview. Check the live configuration separately. Phase 80 best and last are snapshots of the same run. YOLO11n was compared but was not promoted. Promote architectures only after a fair comparison on the same cohorts.

Historical model paths are relative to the workspace root. Locate compatible weights and verify provenance and hashes before changing settings.

| Setting | Historical file |
| --- | --- |
| `CAMERA_DETECTOR_PATH` | `models/phase38_detector/jordan_yolov8n_v6_manual_recovery/weights/best.pt` |
| `MODEL_DETECTOR_PATH` | `models/phase6_detector/yolov8n_plate_baseline/weights/best.pt` for the legacy general profile |
| `FALLBACK_DETECTOR_PATH` | `models/phase80_detector/jordan_yolov8n_v10_public_balanced/weights/best.pt` |
| `RECOVERY_DETECTOR_PATH` | `models/phase80_detector/jordan_yolov8n_v10_public_balanced/weights/last.pt` |
| `MODEL_OCR_PATH` | `models/phase13_ocr/jordan_crnn_v4_layout/weights/best.pt` |
| `AUTHORITY_REGION_MODEL_PATH` | `models/pretrained/easyocr/english_g2.pth` |

## Evidence and resume point

| Measurement | Last verified result |
| --- | --- |
| Two reused external development cohorts | Complete plate match improved from 8/20 to 12/20, with 6/10 in each cohort |
| External end-to-end character error rate | 51 edits / 140 characters = 36.4%, including misses and wrong-region reads |
| External box metrics, IoU 0.50, before the output cap | Post-geometry TP 13, FP 1, FN 7, precision 92.9%, recall 65.0%, F1 76.5%. Raw components TP 13, FP 3, FN 7, F1 72.2% |
| Prior camera and scene cohorts | Camera 112/113 readable exact, 1/754 character edits, 4/5 no-visible-plate views empty. 98 readable views share OCR training identities. Older scenes 67/80 exact |
| Authority, backgrounds, foreign plates | 5/5 reused CD/SN views of three plates, 20/20 derived background crops empty, and 8/9 foreign views still matched Jordanian text patterns |
| Selected fragmentation example | Two boxes became one. IoU 0.856. The full reading and separate parts matched the reviewed example. Median warm CUDA inference was 70.2 ms on one image, excluding startup and network |
| Software checks | 291 project tests passed, 258 publication tests passed with 3 optional skips, and 55 isolated real-model upload/live/image checks passed. The physical archive was unchanged. Raw integrity verified for 7,191 files |

These are development observations on small or reused cohorts. Independent field accuracy, composed-system mAP, multi-vehicle fragment behavior, and registration-code boundary accuracy remain unmeasured. Confidence scores are not calibrated correctness probabilities. Use `PROJECT_PROGRESS.md`, `AI_DEVELOPMENT_PLAN.md`, `CAMERA_INTEGRATION.md`, and `COMPANY_DEMO.md` for the local history and architecture. The latest preview is under `logs/phase89_preview_20261118/`. The public documentation includes `docs/EVALUATION_SNAPSHOT.md`, `docs/RECOVERY_AND_FRAGMENTS.md`, `docs/MODEL_UPGRADES.md`, and `docs/PLATE_PARTS.md`. Some local evidence and weights are absent from a public clone.

Known published baseline is `61141b350b3b8d9b33649abe07eab3f163f058ee` on `feat/public-data-intake`, with source tests green at `ade630e65b1a749b9f17f2911515faa8ba2fa56c`. Inspect current heads and publication receipts before making changes. [GitHub PR #1](https://github.com/HusseinZahran-1/ANPR-System/pull/1)

## 1. Migrate paths and restore the baseline

1. Discover the actual workspace root, current checkout, entry points, virtual environment, data, model files, camera archive, review store, and pending changes. Preserve the user's existing edits. Use the project's dependency files and recreate a moved virtual environment if it points at the old interpreter. Read `.env.example` and check effective local settings without echoing secrets. Identify absent private assets precisely if the workspace is only a public clone.
2. Find stale paths in application configuration, `.env`, startup and evaluation scripts, dataset manifests, checkpoint references, archive and review loaders, tests, and documentation. Replace operational paths with paths under the detected new root, project-relative defaults, or explicit local overrides where appropriate. Resolve paths consistently when running from a different working directory. Preserve original provenance in historical reports. For active manifests and records, use an explicit mapping or a compatible reader and validate that each relocated file is the same asset.
3. Run the relevant existing tests, confirm model readiness, and run one real image through the normal inference entry point. Verify the expected profile and loaded checkpoint hashes. Record the new root, every updated path category, unresolved assets, and the command that starts the service from the new location.

## 2. Restore live camera integration and give me the new settings

- Preserve manual startup. The Windows scheduled task `Jordan ANPR Camera Receiver` was disabled by request. Use the new environment's actual interpreter and launcher. Check the listener, camera-facing interface, port, and network reachability. Dashboard loopback and a LAN camera destination serve different purposes. The camera needs the reachable LAN IP or configured hostname of the receiver. A moved filesystem path does not determine the host address. Use a verified trusted LAN binding or the deployment's approved HTTPS configuration as appropriate. Keep tokenized URLs out of access logs.
- Check `HOST`, `PORT`, the actual listener, and the settings shown by `GET /camera/isapi2/config` and **Show camera settings** in `/live`. The current settings endpoint derives an address from the default route and uses the configured port, so it can disagree with the camera-facing interface or explicit launcher arguments. Fix that mismatch where needed. A configurable camera-facing address is appropriate for multi-interface and proxy setups. Preserve usable local credentials and provision missing credentials privately.
- Verify that a test capture is received, the original bytes and record are saved at the new archive location, inference succeeds, and `/live` displays the correct image and result. Use an isolated archive for automated smoke tests. Verify that old records and reviews can still be opened after relocation. Test one physical camera capture when the camera is reachable. Otherwise report the concrete network blocker and the software checks that passed.
- Give me the verified **new settings in Arabic**, with the camera UI field labels in English. Include the manual startup command, Dashboard URL, verified transport, Host IP Address/Domain Name, Host Port, Host URL, Heartbeat Interval, Uploaded Picture Type, Authentication Mode, and Platform Response Verification. Current ISAPI2 defaults are HTTP on a trusted LAN, path `/camera/isapi2/<token>`, heartbeat `0`, `Detection Picture`, `None` for the camera's authentication dropdown because the URL carries the token, and response verification off for initial testing. Verify actual values for the new setup. Also provide the developer bridge URL `/camera/events`, required header name, body fields, and where to obtain the secret through the authenticated local interface. Keep secrets out of the prompt, GitHub, routine output, screenshots, and public documentation.

## 3. Make inference easy to integrate with another backend

- Reuse the existing model pipeline and upload route. Provide a stable, versioned inference contract with a consistent, explicitly selected profile. Preserve `/predict` and `/demo/predict` compatibility while introducing a new version if needed. Keep inference independent from camera event persistence and from the UI. Load models once per appropriate worker, document CPU/CUDA operation, and validate startup against explicit model paths.
- Document OpenAPI, a working multipart cURL request, and one backend client example. Cover image types and size, service authentication for remote clients, timeout and retry behavior, readiness, predictable errors, coordinate convention, and how to choose or identify the model profile. Return detections, OCR evidence, full text, separate code and number, split basis, review status, and model/version identifiers. Document empty and ambiguous responses. Keep camera credentials scoped to camera operations. Confirm integration by calling the inference service from a separate client without using the dashboard.

## 4. Organize the workspace and improve recognition

- Inventory the tree and references before moving files. Keep application code in `app/`, runnable utilities in `scripts/` or `tools/`, tests in `tests/`, maintained guides in `docs/`, and model/data/experiment outputs in clearly separated directories. Give the root a short README with the runnable commands and links. Group legacy experiments and duplicate notes into a labeled archive only after checking dependencies and references. Keep a migration map, update imports and links, and verify that tests, loaders, metrics, camera history, and training commands still work. Preserve recoverable history and require explicit approval before irreversible deletion.
- Keep `data/raw/` immutable. Keep existing held-out test pixels and sealed external test members unopened. Use separate reviewed data for independent evaluation. Exclude near duplicates and repeated vehicles or identities across splits. Use current Jordanian formats and confirm labels visually before admitting training data. Preserve licensing and provenance. Do not use legacy XML or model predictions as ground truth.
- Prioritize missed external detections, primary false boxes on country text/reflections, split or duplicate boxes before the single-plate cap, real multi-vehicle scenes, and hard negatives. Use the reviewed background crops as derived evidence and collect independent no-plate captures. Compare candidates on identical splits and operating settings, including a YOLO11 candidate only if the comparison adds useful evidence. Verify that gains survive on older cohorts and camera data.
- Continue work on current-format CD/SN confusions (`CD → C0`, `SN → 5N/5S`) using visual evidence and verified examples. Keep uncertain readings reviewable. Evaluate visual code boundaries for ambiguous numeric plates. Improve foreign rejection while measuring how often legitimate Jordanian plates are rejected. A matching text pattern alone cannot establish country.
- Report full-plate exact match, end-to-end CER, code/number accuracy where boundaries were labeled, detector precision/recall/F1 and mAP, false positives/misses/duplicates, foreign false acceptance and Jordanian false rejection, and warm latency with sample sizes and conditions. Keep raw and post-processing detection counts visible. Establish independent vehicle-grouped field baselines before setting numerical release targets with me. Require reliable upload and live workflows, reproducible evaluation, reviewed gains without material regressions, and a tested second-backend integration before claiming production readiness.

## 5. Publish and report

Publish reviewed changes incrementally on the existing GitHub PR branch and verify CI. Use a sanitized publication checkout or an equivalent reviewed public tree. The original workspace Git history is private. Never push that history or upload private images, weights, labels, camera settings, secrets, or logs. Keep PR #1 open for review. Update the progress log with verified results and the next unresolved experiment.

Report back in Arabic with what now works, the verified new camera settings, the inference base URL and integration example, the cleaned project map, validation results, the GitHub commit or PR link, and remaining blockers.
