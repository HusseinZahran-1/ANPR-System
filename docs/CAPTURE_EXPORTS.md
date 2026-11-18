# Image and XML exports

The image demo and completed camera captures save the original JPEG/PNG
and an XML summary automatically. Set `ANPR_EXPORT_DIR` to change the
project-relative default `data/processed/anpr_exports`.

```text
data/processed/anpr_exports/
  <capture-id>/
    <capture-id>.jpg
    <capture-id>.xml
```

PNG uploads keep their original PNG bytes and extension. The two files share
a capture ID and basename and are published together from a staging directory.
Camera exports use the camera event ID and remain accessible alongside the
existing live timeline. An empty detection still produces a valid bundle.
A failed export returns an error. The camera inbox retains the original
received image for troubleshooting.

XML schema version 1 contains the source, timestamp, image name and SHA-256,
plate readings and individual parts, detection coordinates, review status,
and the selected vehicle's category, color and make when available. Unknown
attributes and ambiguous plate parts are represented by empty elements.

The [versioned inference API](INFERENCE_API.md) keeps its default inference-only
behavior. Add multipart field `save_xml=true` to save a bundle on request.
The `export` response contains the capture ID and both filenames. The API
console under `/api` can issue a dedicated client key and test either mode.
