"""Local performance measurement (model initialization measured separately).

    python scripts/benchmark.py
    python scripts/benchmark.py --images dataset/images/test1.jpg --runs 3

No performance number in this project is reported without running this or
scripts/evaluate.py on the actual machine.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from src.evaluation.performance import STAGES, PerformanceReport  # noqa: E402
from src.pipeline import MedicineRecognizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the recognition pipeline")
    parser.add_argument("--images", nargs="*", help="Image paths (default: all images in dataset/images)")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per image (default 3)")
    parser.add_argument("--report", default=str(ROOT / "outputs" / "performance_report.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Config()
    images = [Path(p) for p in (args.images or [str(p) for p in config.image_files()])]
    if not images:
        print("No images found. Put JPG/PNG files in dataset/images or pass --images.", file=sys.stderr)
        return 2

    started = time.perf_counter()
    recognizer = MedicineRecognizer(config).load()
    startup_ms = (time.perf_counter() - started) * 1000
    print(f"Model + dataset startup: {startup_ms:.0f} ms (measured separately)\n")

    performance = PerformanceReport()
    stage_values: dict[str, list[float]] = {stage: [] for stage in STAGES}
    per_image = []

    for image in images:
        recognizer.recognize(image)  # warm-up, not measured
        runs = []
        for _ in range(args.runs):
            result = recognizer.recognize(image)
            total = result["processing_time_ms"]
            runs.append(total)
            performance.add(total)
            for stage in STAGES:
                stage_values[stage].append(result["timings_ms"].get(stage, 0.0))
            print(
                f"{image.name}: {total:.0f} ms  status={result['status']} "
                f"suggestion={result['suggested_medicine']} score={result['match_score']}"
            )
        per_image.append(
            {"image": image.name, "runs_ms": [round(v, 1) for v in runs],
             "average_ms": round(statistics.fmean(runs), 1), "max_ms": round(max(runs), 1)}
        )

    summary = performance.summary()
    stage_summary = {
        stage: round(statistics.fmean(values), 1) for stage, values in stage_values.items() if values
    }
    report = {
        "startup_ms": round(startup_ms, 1),
        "ocr_model_load_ms": round(recognizer.model_load_ms or 0, 1),
        "config": {
            "max_image_side": config.max_image_side,
            "ocr_det_model": config.ocr_det_model,
            "ocr_rec_model": config.ocr_rec_model,
            "ocr_rec_batch_size": config.ocr_rec_batch_size,
        },
        "summary": summary,
        "average_stage_ms": stage_summary,
        "per_image": per_image,
    }

    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- Performance summary (per-image pipeline, model already loaded) ---")
    for key, value in summary.items():
        print(f"{key:>32}: {value}")
    print(f"{'average_stage_ms':>32}: {json.dumps(stage_summary)}")
    print(f"\nReport written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
