"""Public acquisition guards tested with synthetic in-memory images only."""
import hashlib
import io
import pytest
from PIL import Image
from urllib.request import Request
from tools.quarantine_public_images import inspect_bytes,validate_asset_url,validate_entry,PublicRedirects

def test_excluded_hash_never_reaches_image_decoder(monkeypatch):
    payload=b'protected synthetic bytes'
    monkeypatch.setattr(Image,'open',lambda *a,**k: pytest.fail('protected pixels decoded'))
    result=inspect_bytes(payload,{hashlib.sha256(payload).hexdigest()})
    assert result['status']=='excluded_existing_hash'

@pytest.mark.parametrize('url',[
    'http://upload.wikimedia.org/image.jpg','https://localhost/image.jpg',
    'https://upload.wikimedia.org.attacker.example/image.jpg',
    'https://user:password@source.roboflow.com/image.jpg',
    'https://source.roboflow.com:8443/image.jpg'])
def test_rejects_unapproved_media_hosts(url):
    with pytest.raises(ValueError):
        validate_asset_url(url)

def test_small_preview_is_flagged_and_integrity_is_measured():
    buffer=io.BytesIO()
    Image.new('RGB',(58,16),'white').save(buffer,format='PNG')
    result=inspect_bytes(buffer.getvalue(),set())
    assert result['small_image'] and result['status']=='needs_visual_review'
    assert result['width']==58 and len(result['pixel_sha256'])==64

def test_unknown_license_and_path_ids_are_not_admitted():
    with pytest.raises(ValueError):
        validate_entry({'id':'../../escape','license_spdx':'CC-BY-4.0'})
    with pytest.raises(ValueError):
        validate_entry({'id':'sample','license_spdx':'unknown'})

def test_redirect_cannot_escape_public_host_allowlist():
    with pytest.raises(ValueError):
        PublicRedirects().redirect_request(Request('https://upload.wikimedia.org/source.jpg'),
                                          None,302,'Found',{},'https://localhost/private')

def test_pixel_hash_includes_shape_not_only_flat_pixels():
    hashes=[]
    for shape in ((160,40),(80,80)):
        buffer=io.BytesIO()
        Image.new('RGB',shape,'white').save(buffer,format='PNG')
        hashes.append(inspect_bytes(buffer.getvalue(),set())['pixel_sha256'])
    assert hashes[0]!=hashes[1]
