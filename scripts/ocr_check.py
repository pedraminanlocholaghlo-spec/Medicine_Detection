"""Milestone 1 check: verify PaddleOCR loads and reads English text.

    python scripts/ocr_check.py dataset/images/test1.jpg
    python scripts/ocr_check.py dataset/images/test1.jpg --save-raw outputs/raw.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from src.ocr.text_extractor import PaddleOCRExtractor  # noqa: E402
from src.preprocessing import image_processor as ip  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that PaddleOCR works on this machine")
    parser.add_argument("image")
    parser.add_argument("--save-raw", help="Write the raw OCR engine output to this JSON path")
    args = parser.parse_args()

    config = Config()
    image = ip.load_image(args.image)
    processed, ops = ip.preprocess(image, config)

    started = time.perf_counter()
    extractor = PaddleOCRExtractor(config.ocr_language, config.ocr_min_confidence, config).load()
    load_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    regions = extractor.extract(processed)
    inference_ms = (time.perf_counter() - started) * 1000

    print(f"Model load: {load_ms:.0f} ms")
    print(f"OCR inference: {inference_ms:.0f} ms")
    print(f"Preprocessing: {ops}")
    print(f"Detected regions: {len(regions)}\n")
    for region in regions:
        print(f"  {region.ocr_confidence:.3f}  {region.bounding_box}  {region.text}")

    if args.save_raw:
        extractor.save_raw_output(args.save_raw)
        print(f"\nRaw engine output: {args.save_raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
