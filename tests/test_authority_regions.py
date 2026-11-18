"""Synthetic geometry, CTC alignment and optional reader contract guards."""
from types import SimpleNamespace
import cv2
import numpy as np
from PIL import Image
import pytest
import torch
from app.ocr.regions import propose_first_pair
from app.ocr.spatial import vgg_ctc_tokens
import app.ocr.authority_regions as authority


def test_first_pair_geometry_and_source_mapping_preserve_input():
    pixels=np.full((100,500,3),255,np.uint8)
    cv2.putText(pixels,'CD 12345',(85,70),cv2.FONT_HERSHEY_SIMPLEX,1.8,(0,0,0),3)
    image=Image.fromarray(pixels)
    original=np.array(image)
    proposal=propose_first_pair(image)
    assert proposal is not None
    assert np.array_equal(np.array(image),original)
    assert proposal.box_xyxy[0] < 95 and 140 < proposal.box_xyxy[2] < 200
    mapped=cv2.perspectiveTransform(np.array([proposal.source_polygon],np.float32),np.array(proposal.source_to_plate))[0]
    x1,y1,x2,y2=proposal.box_xyxy
    assert np.allclose(mapped,[[x1,y1],[x2-1,y1],[x2-1,y2-1],[x1,y2-1]],atol=1e-3)


@pytest.mark.parametrize('color,size',[('white',(300,100)),('black',(300,100)),('white',(10,10))])
def test_blank_or_tiny_crop_has_no_proposal(color,size):
    assert propose_first_pair(Image.new('RGB',size,color)) is None


def test_ctc_run_peaks_and_blank_separated_repetition_have_spatial_positions():
    logits=torch.full((6,4),-9.)
    # Alphabet A12: A, blank, 1 repeated over two frames, blank, 1 again.
    for t,symbol in enumerate([1,0,2,2,0,2]):
        logits[t,symbol]=8.
    logits[3,2]=10.
    tokens=vgg_ctc_tokens(logits,'A12',source_width=320,resized_width=32)
    assert ''.join(t['character'] for t in tokens)=='A11'
    assert [t['time_index'] for t in tokens]==[0,3,5]
    assert [t['source_x_center'] for t in tokens]==[40.,160.,240.]
    with pytest.raises(ValueError,match='padded'):
        vgg_ctc_tokens(torch.zeros(8,4),'A12',source_width=320,resized_width=32)


def install_proposal(monkeypatch):
    proposal=SimpleNamespace(crop=Image.new('RGB',(30,30)),plate=Image.new('RGB',(100,40)),
                             box_xyxy=[10,5,40,35],source_polygon=[[10,5],[39,5],[39,34],[10,34]],
                             source_to_plate=np.eye(3).tolist())
    monkeypatch.setattr(authority,'propose_first_pair',lambda image:proposal)


def test_visual_prefix_and_spatial_suffix_are_independent_evidence(monkeypatch):
    install_proposal(monkeypatch)
    tokens=[{'character':'0','source_x_center':30,'confidence':.99},
            {'character':'1','source_x_center':55,'confidence':.9},
            {'character':'2','source_x_center':75,'confidence':.8}]
    reader=authority.AuthorityRegionReader(lambda image:[{'text':'cd','confidence':.92}],lambda image:tokens)
    result=reader.read(Image.new('RGB',(100,40)))
    assert result['text']=='CD12'
    assert result['numeric_suffix']=='12'
    assert result['prefix_confidence']==.92
    assert result['suffix_confidence']==pytest.approx(.85)
    assert result['country_verified'] is False and result['requires_human_review'] is True


@pytest.mark.parametrize('readings',[[],[{'text':'CD','confidence':.3}],
    [{'text':'CD','confidence':float('nan')}],[{'text':'12','confidence':.99}],
    [{'text':'CD','confidence':.99},{'text':'SN','confidence':.99}]])
def test_numeric_ambiguous_or_low_confidence_prefix_does_not_touch_suffix(monkeypatch,readings):
    install_proposal(monkeypatch)
    def forbidden(_):
        pytest.fail('suffix reader called without a strong unique authority prefix')
    assert authority.AuthorityRegionReader(lambda image:readings,forbidden).read(Image.new('RGB',(100,40))) is None


@pytest.mark.parametrize('text',['A2','','1234567'])
def test_suffix_is_not_guessed_or_truncated(monkeypatch,text):
    install_proposal(monkeypatch)
    tokens=[{'character':c,'source_x_center':50+i,'confidence':.9} for i,c in enumerate(text)]
    reader=authority.AuthorityRegionReader(lambda image:[{'text':'SN','confidence':.95}],lambda image:tokens)
    assert reader.read(Image.new('RGB',(100,40))) is None


def test_optional_model_is_verified_before_any_optional_import(tmp_path):
    fake=tmp_path/'english_g2.pth'
    fake.write_bytes(b'untrusted model')
    with pytest.raises(authority.AuthorityReaderUnavailable,match='verification'):
        authority.load_prefix_reader(str(fake),'cpu')
