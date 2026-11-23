"""PaddleOCR text detection + recognition (English only).

The engine is created once per process and reused for every image. The raw
engine output for the last call is retained on `last_raw_output` for debugging
and evaluation; nothing is discarded before matching.
"""
from __future__ import annotations

import logging
import time

from src.ocr.text_detector import TextRegion, polygon_to_box, reading_order

logger = logging.getLogger(__name__)


class PaddleOCRExtractor:
    def __init__(self, language: str = "en", min_confidence: float = 0.30, config=None):
        self.language = language
        self.min_confidence = min_confidence
        self.config = config
        self._engine = None
        self.model_load_ms: float | None = None
        self.last_raw_output: dict | None = None

    # ---------------------------------------------------------------- model
    def _build_engine(self):
        # Must be set before Paddle is imported to take effect.
        threads = getattr(self.config, "ocr_num_threads", 0) if self.config else 0
        if threads:
            import os

            os.environ.setdefault("OMP_NUM_THREADS", str(threads))

        from paddleocr import PaddleOCR

        kwargs = dict(
            lang=self.language,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=bool(getattr(self.config, "ocr_use_mkldnn", False)),
        )
        if self.config is not None:
            if getattr(self.config, "ocr_det_model", None):
                kwargs["text_detection_model_name"] = self.config.ocr_det_model
            if getattr(self.config, "ocr_rec_model", None):
                kwargs["text_recognition_model_name"] = self.config.ocr_rec_model
            if getattr(self.config, "ocr_rec_batch_size", 0):
                kwargs["text_recognition_batch_size"] = self.config.ocr_rec_batch_size
            side = getattr(self.config, "ocr_det_limit_side_len", None)
            if side:
                kwargs["text_det_limit_side_len"] = side
                kwargs["text_det_limit_type"] = getattr(self.config, "ocr_det_limit_type", "max")
        return PaddleOCR(**kwargs)

    def load(self) -> "PaddleOCRExtractor":
        """Load the models explicitly (called once at application startup)."""
        if self._engine is None:
            started = time.perf_counter()
            self._engine = self._build_engine()
            self.model_load_ms = (time.perf_counter() - started) * 1000
            logger.info("OCR model loaded in %.0f ms", self.model_load_ms)
        return self

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    # ---------------------------------------------------------------- run
    def extract(self, image) -> list[TextRegion]:
        if self._engine is None:
            self.load()
        result = self._engine.predict(image)
        regions, raw_pages = self._parse_result(result)
        self.last_raw_output = {"pages": raw_pages}
        regions = [r for r in regions if r.text.strip()]
        return reading_order(regions)

    def _parse_result(self, result) -> tuple[list[TextRegion], list[dict]]:
        regions: list[TextRegion] = []
        raw_pages: list[dict] = []
        for page in result or []:
            data = self._page_dict(page)
            raw_pages.append(_raw_page(data))
            texts = data.get("rec_texts") or []
            scores = data.get("rec_scores") or []
            boxes = data.get("rec_boxes")
            if boxes is None or len(boxes) == 0:
                boxes = data.get("rec_polys")
            if boxes is None or len(boxes) == 0:
                boxes = data.get("dt_polys") or []
            for index, text in enumerate(texts):
                score = float(scores[index]) if index < len(scores) else 0.0
                if score < self.min_confidence:
                    # Kept in raw output, excluded from the returned regions.
                    continue
                box = None
                if index < len(boxes):
                    try:
                        box = polygon_to_box(boxes[index])
                    except (TypeError, ValueError, IndexError):
                        box = None
                regions.append(TextRegion(text=str(text), ocr_confidence=score, bounding_box=box))
        return regions, raw_pages

    @staticmethod
    def _page_dict(page) -> dict:
        if isinstance(page, dict):
            return page.get("res", page)
        json_attr = getattr(page, "json", None)
        if isinstance(json_attr, dict):
            return json_attr.get("res", json_attr)
        if callable(json_attr):
            value = json_attr()
            if isinstance(value, dict):
                return value.get("res", value)
        return {}

    # ---------------------------------------------------------------- debug
    def save_raw_output(self, path) -> None:
        import json
        from pathlib import Path

        if self.last_raw_output is None:
            return
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.last_raw_output, ensure_ascii=False, indent=2), encoding="utf-8")


RAW_PAGE_KEYS = (
    "rec_texts",
    "rec_scores",
    "rec_boxes",
    "rec_polys",
    "dt_polys",
    "text_det_params",
    "text_type",
)


def _raw_page(data: dict) -> dict:
    """Keep the relevant raw recognition fields, not the whole engine object."""
    return {key: _json_safe(data[key]) for key in RAW_PAGE_KEYS if key in data}


def _json_safe(value):
    """Convert numpy arrays / scalars so raw output can be serialized."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
