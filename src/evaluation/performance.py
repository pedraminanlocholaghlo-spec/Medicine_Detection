"""Stage timing and performance reporting helpers."""
from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

STAGES = (
    "image_loading",
    "preprocessing",
    "ocr",
    "normalization",
    "exact_matching",
    "fuzzy_matching",
    "ranking",
)


@dataclass
class StageTimer:
    """Accumulates wall-clock milliseconds per named stage."""

    timings_ms: dict = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - started) * 1000)

    def add(self, name: str, milliseconds: float) -> None:
        self.timings_ms[name] = self.timings_ms.get(name, 0.0) + milliseconds

    def snapshot(self) -> dict:
        timings = {name: round(value, 1) for name, value in self.timings_ms.items()}
        total = sum(self.timings_ms.get(name, 0.0) for name in STAGES)
        timings["total"] = round(total, 1)
        return timings


@dataclass
class PerformanceReport:
    total_processed: int = 0
    timings_ms: list[float] = field(default_factory=list)

    def add(self, total_ms: float) -> None:
        self.total_processed += 1
        self.timings_ms.append(float(total_ms))

    def summary(self) -> dict:
        if not self.timings_ms:
            return {"total_processed": 0}
        ordered = sorted(self.timings_ms)
        return {
            "total_processed": self.total_processed,
            "average_processing_time_ms": round(statistics.fmean(ordered), 1),
            "median_processing_time_ms": round(statistics.median(ordered), 1),
            "max_processing_time_ms": round(ordered[-1], 1),
            "min_processing_time_ms": round(ordered[0], 1),
            "p95_processing_time_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 1),
            "under_6_seconds": all(value < 6000 for value in ordered),
        }
