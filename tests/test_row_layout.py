"""Geometry contracts use synthetic shapes, never private plate images."""
import numpy as np
import pytest
from PIL import Image, ImageDraw
from app.ocr.layout import prepare_layout
from app.ocr.row_layout import propose_two_row_cut, unfold_at_cut


def synthetic_rows(*,diagonal=False):
    image=Image.new("RGB",(240,110),"white")
    draw=ImageDraw.Draw(image)
    for x in ([80,110] if diagonal else [120,152]):
        draw.rectangle((x,10,x+15,40),fill="black")
    for x in ([150,180,210] if diagonal else [80,110,140,170,200]):
        draw.rectangle((x,65,x+15,100),fill="black")
    return image


def test_rows_are_localized_from_geometry_without_mutating_source():
    image=synthetic_rows();before=np.asarray(image).copy()
    proposal=propose_two_row_cut(image)
    assert proposal is not None and 41<proposal["cut"]<65
    assert len(proposal["top"])==2 and len(proposal["bottom"])==5
    assert np.array_equal(np.asarray(image),before)
    output,mapping=unfold_at_cut(image,proposal["cut"])
    assert output.width>image.width and output.height<image.height
    assert np.array_equal(np.asarray(prepare_layout(image,"rectify_unfold_rows_v4")),np.asarray(output))
    assert mapping["coordinate_convention"]=="pixel_edges"


@pytest.mark.parametrize("cut",[31,52,73])
def test_mapping_preserves_both_complete_source_rows_and_reading_order(cut):
    # Both source rectangles map to consecutive output pieces without clipping.
    source=Image.new("RGB",(80,100),"white")
    draw=ImageDraw.Draw(source);draw.rectangle((0,0,79,cut-1),fill="red");draw.rectangle((0,cut,79,99),fill="blue")
    result,mapping=unfold_at_cut(source,cut)
    for piece in mapping["pieces"]:
        x0,y0,x1,y1=piece["source_box_xyxy"]
        corners=np.array([[x0,y0,1],[x1,y1,1]]).T
        transformed=np.asarray(piece["source_to_output"])@corners
        a,b,c,d=piece["output_box_xyxy"]
        assert np.allclose(transformed[:2].T,[[a,b],[c,d]])
    boundary=mapping["pieces"][0]["output_box_xyxy"][2]
    assert result.getpixel((boundary//2,result.height//2))==(255,0,0)
    assert result.getpixel(((boundary+result.width)//2,result.height//2))==(0,0,255)
    assert mapping["pieces"][1]["output_box_xyxy"][0]==boundary


def test_diagonal_single_sequence_is_not_mistaken_for_stacked_rows():
    assert propose_two_row_cut(synthetic_rows(diagonal=True)) is None


def test_single_row_and_left_hand_two_row_labels_do_not_trigger():
    image=Image.new("RGB",(240,110),"white");draw=ImageDraw.Draw(image)
    for x in (80,110,140,170,200):draw.rectangle((x,25,x+15,80),fill="black")
    for y in (10,65):
        for x in (10,30,50):draw.rectangle((x,y,x+9,y+28),fill="black")
    assert propose_two_row_cut(image) is None
    assert np.array_equal(np.asarray(prepare_layout(image,"rectify_unfold_rows_v4")),np.asarray(prepare_layout(image,"rectify_unfold_wide_v3")))


@pytest.mark.parametrize("size",[(15,10),(400,100),(100,100)])
def test_outside_new_proposal_range_keeps_legacy_behavior(size):
    image=Image.new("RGB",size,"white")
    assert propose_two_row_cut(image) is None
    assert np.array_equal(np.asarray(prepare_layout(image,"rectify_unfold_rows_v4")),np.asarray(prepare_layout(image,"rectify_unfold_wide_v3")))


@pytest.mark.parametrize("cut",[0,100,-1,True,2.5])
def test_invalid_cut_is_rejected(cut):
    with pytest.raises(ValueError,match="cut"):
        unfold_at_cut(Image.new("RGB",(80,100)),cut)


def test_v5_recovers_rows_with_left_shifted_and_touching_glyphs_without_changing_v4():
    image=Image.new("RGB",(240,110),"white");draw=ImageDraw.Draw(image)
    for x in (120,152):draw.rectangle((x,10,x+15,40),fill="black")
    for x0,x1 in ((50,66),(90,148),(180,195),(210,225)):
        draw.rectangle((x0,65,x1,100),fill="black")
    before=np.asarray(image).copy()
    assert propose_two_row_cut(image) is None
    proposal=propose_two_row_cut(image,minimum_x_ratio=.15,maximum_component_aspect=2.)
    assert proposal is not None and 41<proposal["cut"]<65
    result=prepare_layout(image,"rectify_unfold_rows_v5")
    expected,_=unfold_at_cut(image,proposal["cut"])
    assert np.array_equal(np.asarray(result),np.asarray(expected))
    assert np.array_equal(np.asarray(image),before)


def test_v5_insufficient_geometry_retains_v4_pixels():
    image=Image.new("RGB",(240,110),"white")
    assert np.array_equal(np.asarray(prepare_layout(image,"rectify_unfold_rows_v5")),np.asarray(prepare_layout(image,"rectify_unfold_rows_v4")))
