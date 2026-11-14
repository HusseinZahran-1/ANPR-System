"""Uncapped one-to-one matching and protected preflight, with synthetic bytes."""
import pytest
from tools.detector_metrics import box_iou,match_plate_boxes
from tools.evaluate_detector_development import preflight,read_labels


def test_duplicate_and_unrelated_boxes_are_separate_false_positives():
    truth=[[0,0,10,10]]
    result=match_plate_boxes([[0,0,10,10],[1,1,9,9],[30,30,40,40]],truth)
    assert result=={"tp":1,"fp":2,"fn":0,"duplicate_boxes":1,"other_false_boxes":1}


def test_two_plates_can_each_match_once_and_unmatched_truth_remains_visible():
    truth=[[0,0,10,10],[20,0,30,10],[40,0,50,10]]
    assert match_plate_boxes(truth[:2],truth)=={"tp":2,"fp":0,"fn":1,"duplicate_boxes":0,"other_false_boxes":0}
    assert match_plate_boxes([],truth)["fn"]==3
    assert match_plate_boxes(truth,[])["other_false_boxes"]==3


def test_prediction_overlapping_two_targets_matches_the_remaining_target():
    result=match_plate_boxes([[0,0,10,10],[0,0,15,10]],[[0,0,10,10],[5,0,15,10]])
    assert result["tp"]==2 and result["fp"]==0 and result["fn"]==0


def test_touching_edges_have_zero_overlap():
    assert box_iou([0,0,10,10],[10,0,20,10])==0
    assert box_iou([0,0,10,10],[0,0,10,10])==1


@pytest.mark.parametrize("box",[[0,0,0,10],[0,0,float("nan"),10],[0,0,10],[0,0,float("inf"),10]])
def test_invalid_boxes_do_not_produce_silent_metrics(box):
    with pytest.raises(ValueError):match_plate_boxes([box],[])


def fixture(tmp_path):
    root=tmp_path/"development";images=root/"val/images";labels=root/"val/labels"
    images.mkdir(parents=True);labels.mkdir(parents=True)
    protected=tmp_path/"sealed";protected.mkdir();(protected/"sealed.png").write_bytes(b"sealed bytes, never decoded")
    (images/"source.png").write_bytes(b"synthetic preflight fixture, never decoded")
    (labels/"source.txt").write_text("0 0.5 0.5 0.4 0.2\n")
    return root,images,labels,protected


def test_preflight_does_not_decode_protected_or_development_pixels(tmp_path):
    root,images,labels,protected=fixture(tmp_path)
    rows=preflight(root,"val",protected)
    assert len(rows)==1 and rows[0]["normalized_boxes"]==[[.3,.4,.7,.6]]


@pytest.mark.parametrize("same_name",[False,True])
def test_protected_names_or_bytes_are_rejected_before_decoder(tmp_path,same_name):
    root,images,labels,protected=fixture(tmp_path)
    if same_name:
        (images/"source.png").rename(images/"sealed.png")
        (labels/"source.txt").rename(labels/"sealed.txt")
    else:(images/"source.png").write_bytes((protected/"sealed.png").read_bytes())
    with pytest.raises(ValueError,match="protected image rejected"):
        preflight(root,"val",protected)


def test_test_split_and_missing_background_annotation_are_rejected(tmp_path):
    root,images,labels,protected=fixture(tmp_path)
    with pytest.raises(ValueError,match="development"):
        preflight(root,"test",protected)
    (labels/"source.txt").unlink()
    with pytest.raises(ValueError,match="backgrounds require"):
        preflight(root,"val",protected)


def test_background_is_explicit_and_empty_protection_is_an_error(tmp_path):
    root,images,labels,protected=fixture(tmp_path)
    (labels/"source.txt").write_text("")
    assert preflight(root,"val",protected)[0]["normalized_boxes"]==[]
    (protected/"sealed.png").unlink()
    with pytest.raises(ValueError,match="empty protected"):
        preflight(root,"val",protected)


@pytest.mark.parametrize("label",["0 .5 .5 nan .2","0 .95 .5 .4 .2","1 .5 .5 .4 .2","0 .5 .5 0 .2"])
def test_bad_labels_cannot_be_silently_counted_as_ground_truth(tmp_path,label):
    path=tmp_path/"label.txt";path.write_text(label)
    with pytest.raises(ValueError):read_labels(path)


def test_multiclass_checkpoint_is_rejected_before_image_inference(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from tools.evaluate_detector_development import main
    root,images,labels,protected=fixture(tmp_path)
    weights=tmp_path/"trusted-local-fixture.pt";weights.write_bytes(b"stub, never deserialized")
    class MultiClass:
        names={0:"first",1:"second"}
        def predict(self,*args,**kwargs):
            pytest.fail("decoder reached with a multiclass checkpoint")
    monkeypatch.setitem(sys.modules,"ultralytics",SimpleNamespace(YOLO=lambda path:MultiClass()))
    monkeypatch.setattr(sys,"argv",["evaluate","--data-dir",str(root),"--protected-images",str(protected),"--weights",str(weights),"--output",str(tmp_path/"report.json")])
    with pytest.raises(ValueError,match="single-class plate"):
        main()
