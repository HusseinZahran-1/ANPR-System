"""HTTP contract and file lifecycle checks; models are stubbed."""
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api import routes
from app.api.config import Settings
from app.main import app


@pytest.fixture
def api(monkeypatch, tmp_path):
    detector, ocr = tmp_path / 'det.pt', tmp_path / 'ocr.pt'
    detector.touch()
    ocr.touch()
    settings = Settings(detector_model_path=str(detector), ocr_model_path=str(ocr))
    monkeypatch.setattr(routes, 'get_settings', lambda: settings)
    monkeypatch.setattr(routes, '_pipeline', None)
    monkeypatch.setattr(routes, '_camera_pipeline', None)
    with TestClient(app) as client:
        yield client, settings


def png_bytes():
    out = io.BytesIO()
    Image.new('RGB', (30, 20), 'white').save(out, format='PNG')
    return out.getvalue()


def test_health_does_not_load_models(api):
    client, settings = api
    assert client.get('/health').json()['status'] == 'ready'
    assert routes._pipeline is None
    Path(settings.ocr_model_path).unlink()
    assert client.get('/health').json()['status'] == 'model_missing'


def test_company_demo_serves_upload_interface(api):
    client, _ = api
    assert client.get('/', follow_redirects=False).headers['location'] == '/demo'
    response = client.get('/demo')
    assert response.status_code == 200
    assert 'text/html' in response.headers['content-type']
    assert 'Analyze image' in response.text
    assert 'lang="en"' in response.text
    assert 'Jordanian format match' in response.text
    assert "does not confirm the plate's country" in response.text
    assert "fetch('/demo/predict'" in response.text
    assert routes._pipeline is None


@pytest.mark.parametrize('content,mime,status', [
    (b'not an image', 'text/plain', 415),
    (b'not an image', 'image/jpeg', 422),
    (b'', 'image/png', 422),
])
def test_upload_rejected(api, content, mime, status):
    client, _ = api
    assert client.post('/predict', files={'file': ('x.jpg', content, mime)}).status_code == status
    assert routes._pipeline is None


def test_upload_limit(api):
    client, settings = api
    settings.max_upload_bytes = 8
    assert client.post('/predict', files={'file': ('x.png', png_bytes(), 'image/png')}).status_code == 413


def test_success_cleans_temp_file_and_forwards_resolution(api, monkeypatch):
    client, settings = api
    settings.detection_imgsz = 1280
    observed = {}

    class StubPipeline:
        def __init__(self, **kwargs):
            observed['config'] = kwargs

        def run(self, path):
            observed['path'] = path
            assert path.exists()
            return {'image_path': str(path), 'detections': [], 'ocr_results': []}

    monkeypatch.setattr(routes, 'Pipeline', StubPipeline)
    response = client.post('/predict', files={'file': ('scene.png', png_bytes(), 'image/png')})
    assert response.status_code == 200
    assert response.json()['image_path'] == 'scene.png'
    assert observed['config']['imgsz'] == 1280
    assert observed['config']['nms_iou'] == 0.5
    assert observed['config']['max_plates'] == 1
    assert observed['config']['layout_mode'] == 'rectify_unfold_wide_v3'
    assert not observed['path'].exists()


def test_missing_model_returns_503(api):
    client, settings = api
    Path(settings.ocr_model_path).unlink()
    response = client.post('/predict', files={'file': ('scene.png', png_bytes(), 'image/png')})
    assert response.status_code == 503


def test_upload_demo_uses_camera_detector_without_archiving(api,monkeypatch,tmp_path):
    client,settings=api
    camera=tmp_path/'camera.pt'
    camera.touch()
    settings.camera_detector_model_path=str(camera)
    seen={}
    class StubPipeline:
        def __init__(self,**kwargs):
            seen.update(kwargs)
        def run(self,path):
            return {'image_path':str(path),'detections':[],'ocr_results':[]}
    monkeypatch.setattr(routes,'Pipeline',StubPipeline)
    response=client.post('/demo/predict',files={'file':('x.png',png_bytes(),'image/png')})
    assert response.status_code==200
    assert seen['detector_path']==camera
    assert routes._pipeline is None
    assert routes._camera_pipeline is not None


def test_missing_configured_authority_model_is_visible_before_loading(api,tmp_path):
    client,settings=api
    settings.authority_model_path=str(tmp_path/'missing'/'english_g2.pth')
    health=client.get('/health').json()
    assert health['authority_region_enabled'] is True
    assert health['authority_region_available'] is False
    assert health['status']=='model_missing'
    assert client.post('/predict',files={'file':('x.png',png_bytes(),'image/png')}).status_code==503


def test_failure_cleans_temp_file(api, monkeypatch):
    client, _ = api
    observed = {}

    class BrokenPipeline:
        def run(self, path):
            observed['path'] = path
            raise RuntimeError('private implementation detail')

    monkeypatch.setattr(routes, '_pipeline', BrokenPipeline())
    response = client.post('/predict', files={'file': ('scene.png', png_bytes(), 'image/png')})
    assert response.status_code == 500
    assert response.json()['detail'] == 'inference failed'
    assert not observed['path'].exists()


@pytest.mark.parametrize('size', [0, -32, 961])
def test_invalid_resolution_rejected(size):
    with pytest.raises(ValueError, match='positive multiple of 32'):
        Settings(detection_imgsz=size)


def test_resolution_from_environment(monkeypatch):
    monkeypatch.setenv('DETECTION_IMGSZ', '1280')
    assert Settings.from_env().detection_imgsz == 1280


def test_invalid_ocr_layout_rejected():
    with pytest.raises(ValueError, match='OCR_LAYOUT_MODE'):
        Settings(ocr_layout_mode='unknown')


@pytest.mark.parametrize('iou', [0, 1, -0.1, 1.1])
def test_invalid_detection_nms_iou_rejected(iou):
    with pytest.raises(ValueError, match='DETECTION_NMS_IOU'):
        Settings(detection_nms_iou=iou)


def test_invalid_detection_max_plates_rejected():
    with pytest.raises(ValueError, match='DETECTION_MAX_PLATES'):
        Settings(detection_max_plates=0)


def test_missing_configured_fallback_is_visible_before_loading(api,tmp_path):
    client,settings=api
    settings.fallback_detector_model_path=str(tmp_path/"missing-fallback.pt")
    health=client.get("/health").json()
    assert health["fallback_detector_enabled"] is True
    assert health["fallback_detector_available"] is False
    assert health["status"]=="model_missing"
    assert client.post("/demo/predict",files={"file":("source.png",png_bytes(),"image/png")}).status_code==503
    assert routes._pipeline is None and routes._camera_pipeline is None


@pytest.mark.parametrize("value",[0,1.1,float("nan")])
def test_invalid_fallback_confidence_is_rejected(value):
    with pytest.raises(ValueError,match="FALLBACK_DETECTION_CONF"):
        Settings(fallback_detection_conf=value)


def test_optional_fallback_settings_are_read_from_environment(monkeypatch):
    monkeypatch.setenv("FALLBACK_DETECTOR_PATH","models/local/secondary.pt")
    monkeypatch.setenv("FALLBACK_DETECTION_CONF","0.05")
    settings=Settings.from_env()
    assert settings.fallback_detector_path.name=="secondary.pt"
    assert settings.fallback_detection_conf==.05


@pytest.mark.parametrize("endpoint",["/predict","/demo/predict"])
def test_optional_fallback_is_forwarded_to_both_pipeline_profiles(api,tmp_path,monkeypatch,endpoint):
    client,settings=api
    fallback=tmp_path/"secondary.pt";fallback.touch()
    camera=tmp_path/"camera.pt";camera.touch()
    settings.fallback_detector_model_path=str(fallback)
    settings.camera_detector_model_path=str(camera)
    captured={}
    class StubPipeline:
        def __init__(self,**kwargs):captured.update(kwargs)
        def run(self,path):return {"detections":[],"ocr_results":[]}
    monkeypatch.setattr(routes,"Pipeline",StubPipeline)
    response=client.post(endpoint,files={"file":("source.png",png_bytes(),"image/png")})
    assert response.status_code==200
    assert captured["fallback_detector_path"]==fallback
    assert captured["fallback_det_conf"]==.05
    assert captured["detector_path"]==(camera if endpoint=="/demo/predict" else settings.detector_path)
