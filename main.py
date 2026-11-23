"""Command-line entry point.

    python main.py dataset/images/test1.jpg --debug
    python main.py --serve --port 8000
"""
from __future__ import annotations

import argparse
import json
import sys

from config import Config

_recognizer = None


def get_recognizer(config: Config | None = None):
    """Load the OCR model and dataset once, then reuse them."""
    global _recognizer
    if _recognizer is None:
        from src.pipeline import MedicineRecognizer

        _recognizer = MedicineRecognizer(config or Config()).load()
    return _recognizer


def recognize_medicine(image_path: str, include_debug: bool = False) -> dict:
    return get_recognizer().recognize(image_path, include_debug=include_debug)


def main() -> int:
    parser = argparse.ArgumentParser(description="Medicine name recognition (OCR + fuzzy matching)")
    parser.add_argument("image", nargs="?", help="Path to a JPG/PNG medicine image")
    parser.add_argument("--debug", action="store_true", help="Include raw OCR output and stage timings")
    parser.add_argument("--serve", action="store_true", help="Start the API server for the mobile app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.serve:
        import uvicorn

        uvicorn.run("src.api:app", host=args.host, port=args.port, reload=False)
        return 0

    if not args.image:
        parser.error("provide an image path or use --serve")

    try:
        result = recognize_medicine(args.image, include_debug=args.debug)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
