import numpy as np
import pytest

from config import Config
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
        self.model_load_ms = 12.5

    def extract(self, image):
        return self.regions


def dataset(*names):
    return Dataset(labels=[MedicineLabel(name, compact_key(name)) for name in names])


def blank_image(height=400, width=600):
    return np.full((height, width, 3), 235, dtype=np.uint8)


def test_successful_recognition_contract():
    extractor = StubExtractor(
        [
            TextRegion("AM0XICILLIN", 0.96, [120, 100, 400, 150]),
            TextRegion("500 MG", 0.94, [120, 170, 250, 210]),
            TextRegion("CAPSULE", 0.91, [120, 230, 300, 270]),
        ]
    )
    recognizer = MedicineRecognizer(
        Config(log_requests=False), extractor=extractor, dataset=dataset("AMOXICILLIN")
    ).load()

    result = recognizer.recognize(blank_image())

    assert result["status"] == "SUGGESTED"
    assert result["suggested_medicine"] == "AMOXICILLIN"
    assert result["match_score"] >= 60
    assert result["matched_text"]
    assert result["detected_texts"] == ["AM0XICILLIN", "500 MG", "CAPSULE"]
    assert result["user_confirmation_required"] is True
    assert result["manual_input_required"] is False
    assert result["processing_time_ms"] >= 0
    for stage in (
        "image_loading",
        "preprocessing",
        "ocr",
        "normalization",
        "exact_matching",
        "fuzzy_matching",
        "ranking",
        "total",
    ):
        assert stage in result["timings_ms"]


def test_unknown_when_no_label_matches():
    recognizer = MedicineRecognizer(
        Config(log_requests=False),
        extractor=StubExtractor([TextRegion("UNKNOWN TEXT", 0.9, [0, 0, 200, 40])]),
        dataset=dataset("AMOXICILLIN"),
    ).load()
    result = recognizer.recognize(blank_image())
    assert result["status"] == "UNKNOWN"
    assert result["suggested_medicine"] is None
    assert result["match_score"] is None
    assert result["manual_input_required"] is True
    assert result["user_confirmation_required"] is False


def test_empty_ocr_output():
    recognizer = MedicineRecognizer(
        Config(log_requests=False), extractor=StubExtractor([]), dataset=dataset("AMOXICILLIN")
    ).load()
    result = recognizer.recognize(blank_image())
    assert result["status"] == "UNKNOWN"
    assert result["detected_texts"] == []


def test_invalid_image_bytes_are_rejected():
    recognizer = MedicineRecognizer(
        Config(log_requests=False), extractor=StubExtractor([]), dataset=dataset("AMOXICILLIN")
    ).load()
    with pytest.raises(ValueError):
        recognizer.recognize(b"this is not an image")


def test_tiny_image_is_rejected():
    recognizer = MedicineRecognizer(
        Config(log_requests=False), extractor=StubExtractor([]), dataset=dataset("AMOXICILLIN")
    ).load()
    with pytest.raises(ValueError):
        recognizer.recognize(np.zeros((50, 50, 3), dtype=np.uint8))


def test_debug_output_keeps_raw_ocr():
    recognizer = MedicineRecognizer(
        Config(log_requests=False),
        extractor=StubExtractor([TextRegion("AMOXICILLIN", 0.99, [0, 0, 100, 40])]),
        dataset=dataset("AMOXICILLIN"),
    ).load()
    result = recognizer.recognize(blank_image(), include_debug=True)
    assert result["raw_ocr"][0]["text"] == "AMOXICILLIN"
    assert result["raw_ocr"][0]["bounding_box"] == [0, 0, 100, 40]
    assert "preprocess_ops" in result
