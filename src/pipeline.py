"""End-to-end recognition pipeline: image -> OCR -> match -> decision."""
from __future__ import annotations

import time
from pathlib import Path

from config import Config
from src.database.dataset_loader import Dataset, DatasetLoader
from src.decision.decision_engine import decide
from src.evaluation.performance import StageTimer
from src.matching.candidate_ranker import rank_candidates
from src.matching.matcher import Matcher
from src.ocr.text_extractor import PaddleOCRExtractor
from src.preprocessing import image_processor as ip


class MedicineRecognizer:
    """Holds the OCR model and dataset for the lifetime of the process."""

    def __init__(self, config: Config | None = None, extractor=None, dataset: Dataset | None = None):
        self.config = config or Config()
        self.extractor = extractor or PaddleOCRExtractor(
            language=self.config.ocr_language,
            min_confidence=self.config.ocr_min_confidence,
            config=self.config,
        )
        self._dataset = dataset
        self._matcher: Matcher | None = None
        self.model_load_ms: float | None = None

    # ---------------------------------------------------------------- setup
    def load(self) -> "MedicineRecognizer":
        started = time.perf_counter()
        if self._dataset is None:
            self._dataset = DatasetLoader.from_config(self.config).load()
        if hasattr(self.extractor, "load"):
            self.extractor.load()
            self.model_load_ms = getattr(self.extractor, "model_load_ms", None)
        self._matcher = Matcher(self._dataset.labels, self.config)
        self.dataset_load_ms = (time.perf_counter() - started) * 1000
        return self

    @property
    def dataset(self) -> Dataset:
        if self._dataset is None:
            self.load()
        return self._dataset

    @property
    def matcher(self) -> Matcher:
        if self._matcher is None:
            self.load()
        return self._matcher

    # ---------------------------------------------------------------- run
    def recognize(self, image, include_debug: bool = False) -> dict:
        if self._matcher is None:
            self.load()

        timer = StageTimer()
        with timer.stage("image_loading"):
            array = self._read_image(image)

        with timer.stage("preprocessing"):
            processed, applied_ops = ip.preprocess(array, self.config)

        with timer.stage("ocr"):
            regions = self.extractor.extract(processed)

        outcome = self.matcher.match(regions)
        timer.add("normalization", outcome.timings_ms.get("normalization", 0.0))
        timer.add("exact_matching", outcome.timings_ms.get("exact_matching", 0.0))
        timer.add("fuzzy_matching", outcome.timings_ms.get("fuzzy_matching", 0.0))

        with timer.stage("ranking"):
            ranked = rank_candidates(outcome.matches, self.config)

        detected_texts = [region.text for region in regions]
        decision = decide(ranked, detected_texts, self.config.match_threshold, config=self.config)
        result = decision.to_dict()
        timings = timer.snapshot()
        result["processing_time_ms"] = timings["total"]
        result["timings_ms"] = timings
        result["detected_regions"] = [region.to_dict() for region in regions]
        result["ocr_model_load_ms"] = round(self.model_load_ms, 1) if self.model_load_ms else None
        result["queries_matched"] = outcome.queries
        result["labels_in_dataset"] = len(self.dataset)

        if include_debug:
            result["preprocess_ops"] = applied_ops
            result["raw_ocr"] = [region.to_dict() for region in regions]
            raw = getattr(self.extractor, "last_raw_output", None)
            result["raw_ocr_engine_output"] = raw
        return result

    def _read_image(self, image):
        if isinstance(image, (bytes, bytearray)):
            array = ip.decode_image_bytes(bytes(image))
        elif isinstance(image, (str, Path)):
            array = ip.load_image(image)
        else:
            array = image
        ip.validate_image(array, self.config.min_image_side)
        return array
