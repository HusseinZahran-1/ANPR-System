# Image and XML exports

The image demo and completed camera captures save the original JPEG/PNG
and an XML summary automatically. Set `ANPR_EXPORT_DIR` to change the
project-relative default `data/processed/anpr_exports`.

```text
data/processed/anpr_exports/
  2026-11-18/
    ANPR_1200345_Black_BMW_Car-SUV.jpg
    ANPR_1200345_Black_BMW_Car-SUV.xml
```

PNG uploads keep their original PNG bytes and extension. Each pair shares
the basename `ANPR_<full-plate-reading>_<color>_<make>_<type-category>`, preserving
leading zeros. Uncertain or missing attributes use `Unknown`; the capture ID
fills the plate position when there is no usable reading. All exports saved on
the same day go into one folder dated in the Asia/Amman time zone. When a name
is already in use on the same day, both new filenames add the capture ID.
The XML is published after the original image. Camera exports retain the camera
event ID and stay linked to the live timeline. An empty detection still produces a valid bundle.
A failed export returns an error. The camera inbox retains the original
received image for troubleshooting.

XML schema version 1 contains the source, timestamp, camera ID when available,
date folder and filenames, original filename, byte count and SHA-256,
full plate reading and parts, coordinates, OCR and detector evidence, validation
and review decisions, and the selected vehicle's broad type, detailed category,
color, make and associated plate indexes. Unknown
attributes and ambiguous plate parts are represented by empty elements.

The [versioned inference API](INFERENCE_API.md) keeps its default inference-only
behavior. Add multipart field `save_xml=true` to save a bundle on request.
The `export` response contains the capture ID, date folder and both filenames. The API
console under `/api` can issue a dedicated client key and test either mode.
