# Public image acquisition

External images supplement local data; a download does not establish accuracy.
Never replace immutable raw images or decode the reserved test split during
acquisition. No external image is automatically admitted to training.

## Candidate sources checked on 2026-11-09

| Source | Published terms / access observed |
| --- | --- |
| [JO-LPR by user001](https://universe.roboflow.com/user001-8hq14/jo-lpr) | Page declares CC BY 4.0; full download may require login |
| [Jordan plates by bau](https://universe.roboflow.com/bau-l8e0s/number-plate-recognition-jordan) | Page declares CC BY 4.0; original-image download required login; versions include augmented/cropped images |
| [Mixed-country plates by Capstone](https://universe.roboflow.com/capstone-n8yld/ksa-uae-oman-jordan-qatar) | Page declares CC BY 4.0; no generated version observed; country review mandatory |
| [Wikimedia Commons: Jordan](https://commons.wikimedia.org/wiki/Category:License_plates_of_Jordan) | Public originals, individual author/license per file; includes both photographs and illustrations |

Retain source and license links, credit the author, and record transformations.
Source declarations do not verify annotation quality or third-party authorship.
Review share-alike obligations before distributing derived assets or models.
A paper's open-access license does not necessarily cover a separate dataset.
Do not bypass login requirements or commit API keys. Images and weights are
not redistributed by this repository.

## Quarantine importer

`tools/quarantine_public_images.py` accepts a private JSON manifest containing
a `sources` array. Each entry requires `id`, `title`, `author`, `source_url`,
`asset_url`, `license_spdx`, and `license_url`. Copy actual asset links from
reviewed source pages. Supported media hosts are `upload.wikimedia.org` and
`source.roboflow.com`, over HTTPS. Redirects are checked too.

Supply a CSV with `filename,sha256` columns containing hashes excluded from
decoding/import (for example, all original images). Exclusion happens before
image decoding. A new project with no existing protected data may explicitly
provide a header-only inventory; never omit known protected hashes. This check
does not establish absence of re-encoded or same-identity overlap.

```powershell
python -m tools.quarantine_public_images `
  --manifest data/reviewed/public_sources.json `
  --output data/processed/external/review_batch_01 `
  --protected-hashes-csv data/reviewed/excluded_image_hashes.csv
```

Output must be a new subdirectory of `data/processed/external`. It contains
unchanged image bytes and an `intake.json` receipt with attribution, byte/pixel
hashes, dimensions and decode status. Small previews are flagged. Transfers
and image size are bounded. Failures are recorded and return a nonzero exit
status. Existing output directories are never overwritten. No OCR labels are
guessed or copied from filenames.

Before training, visually verify complete text and country, reject unreadable
samples, and compare identities/near-duplicate source groups against every
split and warm-start inventory. Freeze a reproducible split; keep all views
and augmentations of one plate together. New evaluation images must remain
outside training and must not be selected using model predictions.

Keep manifests, images, receipts, annotations and experiment results under
ignored `data/` or `logs/`. Commit only generic code, synthetic tests and source
documentation. A download count is not a training-set size or accuracy metric.
