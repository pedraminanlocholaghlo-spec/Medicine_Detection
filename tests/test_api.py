import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from config import Config
from src.api import create_app
from src.database.dataset_loader import Dataset, MedicineLabel
from src.ocr.text_detector import TextRegion
from src.pipeline import MedicineRecognizer
from src.text_processing.normalizer import compact_key


class StubExtractor:
    def __init__(self, regions):
        self.regions = regions
        self.loaded = False
        self.last_raw_output = None

    def load(self):
        self.loaded = True
        self.model_load_ms = 10.0

    def extract(self, image):
        return self.regions


def image_bytes():
    ok, buffer = cv2.imencode(".jpg", np.full((400, 600, 3), 240, dtype=np.uint8))
    assert ok
    return buffer.tobytes()


def make_client(regions, names=("AMOXICILLIN",), **config_overrides):
    config = Config(log_requests=False)
    for key, value in config_overrides.items():
        setattr(config, key, value)
    dataset = Dataset(labels=[MedicineLabel(name, compact_key(name)) for name in names])
    recognizer = MedicineRecognizer(config, extractor=StubExtractor(regions), dataset=dataset)
    return TestClient(create_app(config, recognizer))


@pytest.fixture
def client():
    regions = [
        TextRegion("AM0XICILLIN", 0.96, [10, 10, 300, 50]),
        TextRegion("500 MG", 0.94, [10, 60, 150, 90]),
    ]
    with make_client(regions) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["labels_in_dataset"] == 1


def test_recognize_returns_suggested_contract(client):
    response = client.post(
        "/api/recognize", files={"file": ("box.jpg", image_bytes(), "image/jpeg")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SUGGESTED"
    assert body["suggested_medicine"] == "AMOXICILLIN"
    assert body["match_score"] >= 60
    assert body["detected_texts"] == ["AM0XICILLIN", "500 MG"]
    assert body["user_confirmation_required"] is True
    assert body["manual_input_required"] is False
    assert body["processing_time_ms"] >= 0
    assert body["request_id"]


def test_unknown_requires_manual_input():
    with make_client([TextRegion("XQZVWK", 0.9, [0, 0, 100, 30])]) as client:
        body = client.post(
            "/api/recognize", files={"file": ("box.png", image_bytes(), "image/png")}
        ).json()
        assert body["status"] == "UNKNOWN"
        assert body["suggested_medicine"] is None
        assert body["match_score"] is None
        assert body["manual_input_required"] is True
        assert body["user_confirmation_required"] is False


def test_unsupported_extension_returns_415(client):
    response = client.post(
        "/api/recognize", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 415


def test_invalid_image_returns_400(client):
    response = client.post(
        "/api/recognize", files={"file": ("box.jpg", b"not-an-image", "image/jpeg")}
    )
    assert response.status_code == 400


def test_oversized_upload_returns_413():
    with make_client([TextRegion("AMOXICILLIN", 0.9, [0, 0, 100, 30])], max_upload_mb=0.0001) as client:
        response = client.post(
            "/api/recognize", files={"file": ("box.jpg", image_bytes(), "image/jpeg")}
        )
        assert response.status_code == 413


def test_confirmation_yes(client):
    recognized = client.post(
        "/api/recognize", files={"file": ("box.jpg", image_bytes(), "image/jpeg")}
    ).json()
    response = client.post(
        "/api/confirm", json={"request_id": recognized["request_id"], "accepted": True}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "CONFIRMED"
    assert body["medicine_name"] == "AMOXICILLIN"
    assert body["dataset_updated"] is False


def test_rejection_then_manual_input(client):
    recognized = client.post(
        "/api/recognize", files={"file": ("box.jpg", image_bytes(), "image/jpeg")}
    ).json()
    rejected = client.post(
        "/api/confirm", json={"request_id": recognized["request_id"], "accepted": False}
    ).json()
    assert rejected["status"] == "REJECTED"
    assert rejected["manual_input_required"] is True

    manual = client.post(
        "/api/manual-input",
        json={"request_id": recognized["request_id"], "medicine_name": "amoxicillin clavulanate"},
    )
    assert manual.status_code == 200
    body = manual.json()
    assert body["status"] == "MANUAL_ACCEPTED"
    assert body["medicine_name"] == "AMOXICILLIN CLAVULANATE"
    assert body["dataset_updated"] is False


def test_empty_manual_name_is_rejected(client):
    recognized = client.post(
        "/api/recognize", files={"file": ("box.jpg", image_bytes(), "image/jpeg")}
    ).json()
    response = client.post(
        "/api/manual-input",
        json={"request_id": recognized["request_id"], "medicine_name": "   "},
    )
    assert response.status_code == 400


def test_unknown_request_id_returns_404(client):
    response = client.post("/api/confirm", json={"request_id": "does-not-exist", "accepted": True})
    assert response.status_code == 404
