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
    monkeypatch.setenv("ANPR_EXPORT_DIR", str(tmp_path / "exports"))
    monkeypatch.setenv("CAMERA_SETTINGS_DB", str(tmp_path / "camera-settings.sqlite3"))
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
    assert 'Operations workspace' in client.get('/').text
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


def test_recovery_requires_fallback_configuration():
    with pytest.raises(ValueError,match="requires FALLBACK"):
        Settings(recovery_detector_model_path="models/local/recovery.pt")


def test_missing_recovery_is_visible_without_model_loading(api,tmp_path):
    client,settings=api
    fallback=tmp_path/"fallback.pt";fallback.touch()
    settings.fallback_detector_model_path=str(fallback)
    settings.recovery_detector_model_path=str(tmp_path/"missing-recovery.pt")
    health=client.get("/health").json()
    assert health["recovery_detector_enabled"] is True and health["recovery_detector_available"] is False
    assert health["status"]=="model_missing"
    assert client.post("/demo/predict",files={"file":("source.png",png_bytes(),"image/png")}).status_code==503
    assert routes._pipeline is None and routes._camera_pipeline is None


@pytest.mark.parametrize("endpoint",["/predict","/demo/predict"])
def test_recovery_configuration_is_forwarded_to_both_profiles(api,tmp_path,monkeypatch,endpoint):
    client,settings=api
    fallback=tmp_path/"fallback.pt";fallback.touch();recovery=tmp_path/"recovery.pt";recovery.touch()
    camera=tmp_path/"camera.pt";camera.touch()
    settings.fallback_detector_model_path=str(fallback);settings.recovery_detector_model_path=str(recovery);settings.camera_detector_model_path=str(camera)
    settings.merge_fallback_fragments=True
    captured={}
    class StubPipeline:
        def __init__(self,**kwargs):captured.update(kwargs)
        def run(self,path):return {"detections":[],"ocr_results":[]}
    monkeypatch.setattr(routes,"Pipeline",StubPipeline)
    assert client.post(endpoint,files={"file":("source.png",png_bytes(),"image/png")}).status_code==200
    assert captured["recovery_detector_path"]==recovery
    assert captured["recovery_det_conf"]==.05
    assert captured["recovery_layout_mode"]=="rectify_unfold_rows_v5"

    assert captured["merge_fallback_fragments"] is True


@pytest.mark.parametrize("raw,expected",[("false",False),("0",False),("off",False),("true",True),("1",True)])
def test_fragment_env_bool_is_explicit(monkeypatch,raw,expected):
    monkeypatch.setenv("FALLBACK_MERGE_FRAGMENTS",raw)
    assert Settings.from_env().merge_fallback_fragments is expected


def test_fragment_env_rejects_typo(monkeypatch):
    monkeypatch.setenv("FALLBACK_MERGE_FRAGMENTS","flase")
    with pytest.raises(ValueError,match="boolean"):
        Settings.from_env()


def test_versioned_inference_contract(api, monkeypatch, tmp_path):
    client, settings = api
    settings.inference_api_key = "integration-test-key"
    settings.camera_api_key = "camera-test-key"
    camera = tmp_path / "camera.pt"
    camera.write_bytes(b"camera")
    settings.camera_detector_model_path = str(camera)
    calls = []

    class StubPipeline:
        def __init__(self, **kwargs):
            calls.append(kwargs["detector_path"])

        def run(self, path):
            return {
                "detections": [{"box": [1, 2, 6, 7], "det_conf": 0.8}],
                "ocr_results": [{
                    "crop_box": [1, 2, 6, 7], "text": "012345", "ocr_conf": 0.9,
                    "plate_parts": {"full_text": "012345", "plate_code": None,
                                    "plate_number": None, "split_basis": "unresolved",
                                    "requires_human_review": True},
                }],
                "validations": [],
                "decisions": [],
                "meta": {"detector": str(camera), "requires_human_review": True},
            }

    monkeypatch.setattr(routes, "Pipeline", StubPipeline)
    assert client.get("/v1/ready", headers={"X-Camera-Key": settings.camera_api_key}).status_code == 401
    headers = {"X-Inference-Key": settings.inference_api_key}
    ready = client.get("/v1/ready", headers=headers)
    assert ready.status_code == 200
    response = client.post("/v1/predict", headers=headers,
                           files={"file": ("x.png", png_bytes(), "image/png")})
    assert response.status_code == 200
    data = response.json()
    assert data["profile"] == ready.json()["profile"] == "gate-camera-v1"
    assert data["checkpoints"] == ready.json()["checkpoints"]
    assert len(data["checkpoints"]["primary"]) == 64
    assert data["ocr_results"][0]["plate_parts"]["full_text"] == "012345"
    assert data["ocr_results"][0]["plate_parts"]["plate_code"] is None
    assert str(camera) not in response.text
    assert calls == [camera]


def test_versioned_readiness_hides_missing_checkpoint_path(api):
    client, settings = api
    settings.inference_api_key = "integration-test-key"
    Path(settings.ocr_model_path).unlink()
    response = client.get("/v1/ready", headers={"X-Inference-Key": settings.inference_api_key})
    assert response.status_code == 503
    assert response.json() == {"detail": "inference profile unavailable"}


def test_gate_recovery_options_do_not_change_general_profile(api, monkeypatch, tmp_path):
    client, settings = api
    fallback, recovery = tmp_path / "fallback.pt", tmp_path / "recovery.pt"
    fallback.touch()
    recovery.touch()
    settings.fallback_detector_model_path = str(fallback)
    settings.camera_recovery_detector_model_path = str(recovery)
    settings.camera_merge_fallback_fragments = True
    options = []

    class StubPipeline:
        def __init__(self, **kwargs):
            options.append(kwargs)

        def run(self, path):
            return {"detections": [], "ocr_results": []}

    monkeypatch.setattr(routes, "Pipeline", StubPipeline)
    files = {"file": ("x.png", png_bytes(), "image/png")}
    assert client.post("/predict", files=files).status_code == 200
    assert client.post("/demo/predict", files=files).status_code == 200
    assert options[0]["recovery_detector_path"] is None
    assert options[0]["merge_fallback_fragments"] is False
    assert options[1]["recovery_detector_path"] == recovery
    assert options[1]["merge_fallback_fragments"] is True


def test_camera_listing_supplies_missing_plate_parts_without_editing_archive(api, monkeypatch):
    from app.camera import routes as camera_routes

    client, settings = api
    settings.camera_api_key = "history-test-key"
    monkeypatch.setattr(camera_routes, "get_settings", lambda: settings)
    old_readings = [{"text": "1200345"}, {"text": "100345"}]
    record = {
        "event_id": "archived-event", "captured_at": "2026-11-18T08:00:00+00:00",
        "camera_id": "gate-test", "original_filename": "capture.jpg",
        "result": {"ocr_results": old_readings},
    }
    monkeypatch.setattr(camera_routes.events, "list", lambda limit: [])
    monkeypatch.setattr(camera_routes, "recent_records", lambda limit: [record])
    response = client.get("/camera/events", headers={"X-Camera-Key": settings.camera_api_key})
    assert response.status_code == 200
    resolved, ambiguous = response.json()["events"][0]["result"]["ocr_results"]
    assert resolved["plate_parts"]["plate_code"] == "12"
    assert resolved["plate_parts"]["plate_number"] == "00345"
    assert ambiguous["plate_parts"]["plate_code"] is None
    assert ambiguous["plate_parts"]["plate_number"] is None
    assert old_readings == [{"text": "1200345"}, {"text": "100345"}]


def test_demo_saves_original_and_matching_xml(api, monkeypatch, tmp_path):
    from xml.etree import ElementTree

    client, _ = api

    class StubPipeline:
        def run(self, path):
            return {
                "detections": [{"box": [1, 2, 8, 9]}],
                "ocr_results": [{"text": "1200345", "plate_parts": {
                    "full_text": "1200345", "plate_code": "12", "plate_number": "00345",
                }}],
            }

    monkeypatch.setattr(routes, "_pipeline", StubPipeline())
    vehicle = routes.Vehicle(
        box=(0, 0, 15, 18), detection_class="car", det_conf=0.9,
        vehicle_type="SUV", color="Black", make="Mercedes-Benz",
    )
    monkeypatch.setattr(routes, "vehicle_evidence", lambda *args, **kwargs: ([vehicle], "ready"))
    content = png_bytes()
    response = client.post("/demo/predict", files={"file": ("same.png", content, "image/png")})
    assert response.is_success
    saved = response.json()["export"]
    directory = tmp_path / "exports" / saved["folder"]
    image = directory / saved["image_filename"]
    xml = directory / saved["xml_filename"]
    assert image.name == "ANPR_1200345_Black_Mercedes-Benz_Car.png"
    assert xml.name == "ANPR_1200345_Black_Mercedes-Benz_Car.xml"
    assert image.read_bytes() == content
    root = ElementTree.parse(xml).getroot()
    assert root.find("image").attrib["filename"] == image.name
    assert root.findtext("./plates/plate/plate_number") == "00345"
    assert root.findtext("source") == "upload"
    assert root.find("export").get("folder") == saved["folder"]
    assert root.findtext("./vehicles/vehicle/vehicle_type") == "Car"
    assert root.find("./vehicles/vehicle/vehicle_category") is None
    assert root.findtext("./vehicles/vehicle/color") == "Black"
    assert root.findtext("./vehicles/vehicle/make") == "Mercedes-Benz"
    assert set(directory.iterdir()) == {image, xml}


def test_issued_inference_key_persists_and_can_be_revoked(api, monkeypatch, tmp_path):
    from app.api import keys

    client, settings = api
    settings.inference_admin_key = "test-admin-key"
    monkeypatch.setenv("INFERENCE_KEY_DB", str(tmp_path / "keys.sqlite3"))
    monkeypatch.setattr(keys, "get_settings", lambda: settings)
    monkeypatch.setattr(routes, "load_inference_profile", lambda: ("gate-camera-v1", {}))
    admin = {"X-Inference-Admin-Key": settings.inference_admin_key}
    assert client.get("/api").is_success
    assert client.post("/v1/keys", json={"label": "garage"}).status_code == 401
    assert client.get("/v1/ready", headers={"X-Inference-Key": settings.inference_admin_key}).status_code == 401
    created = client.post("/v1/keys", headers=admin, json={"label": "garage"})
    assert created.status_code == 201
    assert created.headers["cache-control"] == "no-store"
    key = created.json()
    auth = {"X-Inference-Key": key["key"]}
    assert client.get("/v1/ready", headers=auth).is_success
    assert client.get("/v1/keys", headers={"X-Inference-Admin-Key": key["key"]}).status_code == 401
    assert client.get("/v1/keys", headers=admin).json() == [
        {name: key[name] for name in ("id", "label", "created_at")}
    ]
    assert key["key"].encode() not in (tmp_path / "keys.sqlite3").read_bytes()
    assert client.delete(f"/v1/keys/{key['id']}", headers=admin).status_code == 204
    assert client.get("/v1/ready", headers=auth).status_code == 401


def test_export_without_a_plate_uses_anpr_and_capture_id(monkeypatch, tmp_path):
    from app.capture_exports import save_export

    monkeypatch.setenv("ANPR_EXPORT_DIR", str(tmp_path))
    result = save_export(png_bytes(), ".png", {"ocr_results": []}, source="upload")
    base = f"ANPR_{result['id']}_Unknown_Unknown_Unknown"
    assert result["image_filename"] == base + ".png"
    assert result["xml_filename"] == base + ".xml"


def test_exports_share_the_amman_date_folder_without_overwriting(monkeypatch, tmp_path):
    from datetime import datetime, timezone
    from app import capture_exports

    class FixedTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 18, 22, 15, tzinfo=timezone.utc).astimezone(tz)

    monkeypatch.setattr(capture_exports, "datetime", FixedTime)
    monkeypatch.setenv("ANPR_EXPORT_DIR", str(tmp_path))
    result = {"ocr_results": [{"text": "1200345"}]}
    first = capture_exports.save_export(b"first", ".jpg", result, source="camera")
    second = capture_exports.save_export(b"second", ".png", result, source="upload")
    assert first["folder"] == second["folder"] == "2026-11-19"
    folder = tmp_path / first["folder"]
    assert (folder / first["image_filename"]).read_bytes() == b"first"
    assert (folder / second["image_filename"]).read_bytes() == b"second"
    assert first["xml_filename"] != second["xml_filename"]
    assert len(list(folder.iterdir())) == 4


def test_camera_settings_persist_alias_and_replace_camera_key(api, monkeypatch):
    from app.camera import routes as camera_routes
    from app.camera import settings as camera_settings

    client, settings = api
    settings.site_admin_key = "site-admin-test"
    settings.camera_api_key = "original-camera-key"
    monkeypatch.setattr(camera_settings, "get_settings", lambda: settings)
    monkeypatch.setattr(camera_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(camera_routes, "recent_records", lambda limit: [])
    admin = {"X-Site-Admin-Key": settings.site_admin_key}
    initial_headers = {"X-Camera-Key": settings.camera_api_key}
    assert client.get("/settings").is_success
    assert client.get("/ops/camera-settings").status_code == 401
    assert client.get("/camera/bridge/config", headers=initial_headers).json()["bridge_path"] == "/camera/events"
    new_key = "changed-camera-test-credential"
    response = client.put("/ops/camera-settings", headers=admin,
                          json={"public_host": "receiver.example.test", "public_port": 8443,
                                "transport": "HTTPS", "bridge_alias": "front-gate",
                                "camera_key": new_key})
    assert response.is_success
    assert new_key not in response.text
    saved = client.get("/ops/camera-settings", headers=admin).json()
    assert saved["canonical_path"] == "/camera/events"
    assert saved["effective"]["bridge_url"] == "https://receiver.example.test:8443/camera/bridge/front-gate"
    assert client.get("/camera/events", headers=initial_headers).status_code == 401
    assert client.get("/camera/events", headers={"X-Camera-Key": new_key}).is_success
    assert new_key.encode() not in camera_settings._path().read_bytes()


def test_remembered_viewer_is_scoped_and_ends_after_camera_key_rotation(api, monkeypatch):
    from app.camera import routes as camera_routes
    from app.camera import settings as camera_settings

    client, settings = api
    settings.site_admin_key = "site-admin-test"
    settings.camera_api_key = "camera-viewer-test-secret"
    monkeypatch.setattr(camera_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(camera_settings, "get_settings", lambda: settings)
    monkeypatch.setattr(camera_routes, "recent_records", lambda limit: [])
    viewer = {"X-Camera-Viewer": "1"}
    camera = {"X-Camera-Key": settings.camera_api_key}
    admin = {"X-Site-Admin-Key": settings.site_admin_key}

    assert not client.get("/camera/viewer/session", headers=viewer).json()["connected"]
    assert client.post("/camera/viewer/session", headers=viewer).status_code == 401
    signed_in = client.post("/camera/viewer/session", headers=camera)
    assert signed_in.status_code == 204
    cookie = signed_in.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert settings.camera_api_key.lower() not in cookie
    assert client.get("/camera/viewer/session", headers=viewer).json()["connected"]
    assert client.get("/camera/events", headers=viewer).is_success
    assert client.get("/camera/events").status_code == 401
    assert client.post("/camera/events", headers=viewer,
                       files={"file": ("x.png", png_bytes(), "image/png")}).status_code == 401

    assert client.delete("/camera/viewer/session", headers=viewer).status_code == 204
    assert not client.get("/camera/viewer/session", headers=viewer).json()["connected"]
    assert client.post("/camera/viewer/session", headers=camera).status_code == 204
    assert client.put("/ops/camera-settings", headers=admin,
                      json={"camera_key": "rotated-camera-key-test"}).is_success
    assert not client.get("/camera/viewer/session", headers=viewer).json()["connected"]
    assert client.get("/camera/events", headers=viewer).status_code == 401


def test_camera_auto_connect_is_limited_to_direct_local_listener():
    from fastapi import Request
    from app.camera.viewer import is_local_viewer

    settings = Settings(camera_auto_connect_local=True)

    def request(peer, host="10.10.0.10", forwarded=False):
        headers = [(b"host", host.encode())]
        if forwarded:
            headers.append((b"x-forwarded-for", b"10.10.0.20"))
        return Request({
            "type": "http", "scheme": "http", "path": "/camera/viewer/session",
            "server": ("10.10.0.10", 8000), "client": (peer, 49152), "headers": headers,
        })

    assert is_local_viewer(request("10.10.0.10"), settings)
    assert not is_local_viewer(request("10.10.0.20"), settings)
    assert not is_local_viewer(request("10.10.0.10", forwarded=True), settings)
    assert not is_local_viewer(request("10.10.0.10", host="public.example"), settings)
    assert not is_local_viewer(request("10.10.0.10"), Settings())
