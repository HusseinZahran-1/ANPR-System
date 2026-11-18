"""Geometry safeguards use synthetic boxes, without plate identities or models."""
import copy
import pytest
from app.detector.fragments import merge_plate_fragments


def box(coords, score=.1, source="fallback"):
    return {"box": coords, "det_conf": score, "detector_source": source}


def test_union_preserves_component_evidence_without_mutating_inputs():
    inputs = [box([0, 0, 100, 20], .08), box([35, 1, 135, 21], .12)]
    before = copy.deepcopy(inputs)
    merged = merge_plate_fragments(inputs)
    assert inputs == before
    assert len(merged) == 1
    assert merged[0]["box"] == [0, 0, 135, 21]
    assert merged[0]["fragment_boxes"] == [inputs[1]["box"], inputs[0]["box"]]
    assert merged[0]["fragment_confidences"] == [.12, .08]
    assert merged[0]["det_conf"] == .12
    assert merged[0]["det_conf_scope"] == "highest_scoring_fragment"
    assert merged[0]["merged_fragment_count"] == 2
    assert merge_plate_fragments(merged) == merged


@pytest.mark.parametrize("other", [
    box([200, 0, 300, 20]), box([0, 15, 100, 35]),
    box([20, 0, 120, 50]), box([35, 0, 135, 20], source="recovery"),
    box([0, 0, 0, 20]), box([0, 0, float("nan"), 20]),
])
def test_separate_objects_sources_and_invalid_geometry_are_not_unioned(other):
    assert len(merge_plate_fragments([box([0, 0, 100, 20]), other])) == 2


def test_complete_link_blocks_transitive_bridge_on_reapplication():
    a, b, c = [box([x, 0, x + 100, 20], score) for x, score in [(0, .3), (35, .2), (70, .1)]]
    merged = merge_plate_fragments([a, b, c])
    assert len(merged) == 2
    assert merged[0]["merged_fragment_count"] == 2
    assert len(merge_plate_fragments([merged[0], c])) == 2


def test_nonfinite_score_fails_visibly():
    with pytest.raises(ValueError, match="non-finite"):
        merge_plate_fragments([box([0, 0, 100, 20], float("nan"))])
