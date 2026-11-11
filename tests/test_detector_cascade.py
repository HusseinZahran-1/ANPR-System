"""Primary-first invariants without model files or image decoders."""
import pytest
from app.detector.cascade import PrimaryFirstDetector


class FakeDetector:
    def __init__(self,results,events,name):
        self.results=results;self.events=events;self.name=name
    def detect(self,path):
        self.events.append((self.name,path))
        return self.results


def test_primary_boxes_are_returned_without_calling_fallback_or_merging_scores():
    events=[];boxes=[{"box":[0,0,10,10],"det_conf":.26},{"box":[20,0,30,10],"det_conf":.25}]
    detector=PrimaryFirstDetector(FakeDetector(boxes,events,"primary"),FakeDetector([{"det_conf":.99}],events,"fallback"))
    assert detector.detect("source")==boxes
    assert events==[("primary","source")]


def test_empty_primary_calls_fallback_and_retains_its_own_score_without_mutation():
    events=[];boxes=[{"box":[0,0,10,10],"det_conf":.08}]
    detector=PrimaryFirstDetector(FakeDetector([],events,"primary"),FakeDetector(boxes,events,"fallback"))
    result=detector.detect("source")
    assert result==[{**boxes[0],"detector_source":"fallback"}]
    assert boxes==[{"box":[0,0,10,10],"det_conf":.08}]
    assert events==[("primary","source"),("fallback","source")]


def test_two_empty_detectors_do_not_invent_boxes():
    events=[]
    assert PrimaryFirstDetector(FakeDetector([],events,"primary"),FakeDetector([],events,"fallback")).detect("source")==[]


def test_primary_error_is_not_interpreted_as_an_empty_scene():
    class Broken:
        def detect(self,path):raise RuntimeError("primary failure")
    events=[]
    detector=PrimaryFirstDetector(Broken(),FakeDetector([],events,"fallback"))
    with pytest.raises(RuntimeError,match="primary failure"):detector.detect("source")
    assert events==[]
