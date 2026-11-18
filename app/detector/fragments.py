"""Conservative within-model fragment unions with original-box provenance.

Caller must provide one selected detector's outputs. This does not fuse scores
or select across detector stages. Complete-link groups prevent bridge chains
from joining otherwise incompatible objects.
"""
from __future__ import annotations
import math


def _components(detection):
    return detection.get("fragment_boxes", [detection["box"]])


def _compatible_boxes(a, b):
    if len(a) != 4 or len(b) != 4:
        return False
    if not all(math.isfinite(float(value)) for value in (*a, *b)):
        return False
    x0, y0, x1, y1 = a
    u0, v0, u1, v1 = b
    aw, ah, bw, bh = x1 - x0, y1 - y0, u1 - u0, v1 - v0
    if min(aw, ah, bw, bh) <= 0:
        return False
    overlap_x = max(0, min(x1, u1) - max(x0, u0))
    overlap_y = max(0, min(y1, v1) - max(y0, v0))
    intersection = overlap_x * overlap_y
    iou = intersection / (aw * ah + bw * bh - intersection)
    return (iou >= .30 and overlap_x / min(aw, bw) >= .60
            and overlap_y / min(ah, bh) >= .85
            and min(ah, bh) / max(ah, bh) >= .70
            and abs((y0 + y1 - v0 - v1) / 2) <= .25 * min(ah, bh)
            and max(x1, u1) - min(x0, u0) <= 1.6 * max(aw, bw))


def _compatible(a, b):
    if a.get("detector_source") != b.get("detector_source"):
        return False
    return all(_compatible_boxes(x, y) for x in _components(a) for y in _components(b))


def merge_plate_fragments(detections):
    """Return aligned unions, retaining every original component and score.

    The returned det_conf is the highest component's original score, explicitly
    scoped to that component; it is not a calibrated confidence for the union.
    No input dictionary or coordinate list is modified.
    """
    for detection in detections:
        if not math.isfinite(float(detection["det_conf"])):
            raise ValueError("non-finite detector score")
    groups = []
    for detection in sorted(detections, key=lambda item: item["det_conf"], reverse=True):
        compatible = next((group for group in groups
                           if all(_compatible(detection, other) for other in group)), None)
        if compatible is None:
            groups.append([detection])
        else:
            compatible.append(detection)
    output = []
    for group in groups:
        selected = dict(group[0], box=list(group[0]["box"]))
        if len(group) > 1:
            boxes = [list(box) for item in group for box in _components(item)]
            scores = [score for item in group
                      for score in item.get("fragment_confidences", [item["det_conf"]])]
            selected.update(
                box=[min(box[0] for box in boxes), min(box[1] for box in boxes),
                     max(box[2] for box in boxes), max(box[3] for box in boxes)],
                merged_fragment_count=len(boxes), fragment_boxes=boxes,
                fragment_confidences=scores, det_conf_scope="highest_scoring_fragment",
                detector_geometry="aligned_fragments_v1")
        output.append(selected)
    return output
