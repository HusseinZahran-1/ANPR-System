"""Primary-first detection with an optional independently scored fallback."""

class PrimaryFirstDetector:
    """Use the fallback only when the primary returns no detections.

    Never merge box sets or compare scores from differently calibrated models.
    A primary inference error propagates; it is not interpreted as an empty scene.
    """
    def __init__(self, primary, fallback, *, fallback_source="fallback"):
        if fallback_source not in ("fallback", "recovery"):
            raise ValueError("unknown detector stage")
        self.fallback_source = fallback_source
        self.primary = primary
        self.fallback = fallback

    def detect(self, image_path):
        detections = self.primary.detect(image_path)
        if detections:
            return detections
        return [dict(detection, detector_source=self.fallback_source)
                for detection in self.fallback.detect(image_path)]
