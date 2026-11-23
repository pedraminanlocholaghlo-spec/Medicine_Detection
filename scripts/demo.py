"""Interactive demo: give a medicine photo, see the result.

    python scripts/demo.py                      # loop: pick / paste a path, see result
    python scripts/demo.py img1.jpg img2.jpg    # process these, then continue the loop

The OCR model is loaded once, so every image after the first takes only ~2-4 s.
At the prompt you can:
    - drag an image file into the terminal (or paste its path) and press Enter
    - press Enter on an empty line to open a file chooser
    - type q to quit
This script only reads images; it never changes the dataset.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from src.decision.confirmation import ConfirmationService  # noqa: E402
from src.pipeline import MedicineRecognizer  # noqa: E402

LINE = "-" * 60


def pick_file() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Choose a medicine image",
            filetypes=[("Images", "*.jpg *.jpeg *.png")],
        )
        root.destroy()
        return path
    except Exception as error:  # no GUI available
        print(f"(file chooser unavailable: {error})")
        return ""


def show(result: dict) -> None:
    print(LINE)
    print(f"Status        : {result['status']}")
    if result["status"] == "SUGGESTED":
        print(f"Suggestion    : {result['suggested_medicine']}")
        print(f"Match score   : {result['match_score']} / 100 (threshold {result['threshold']})")
        print(f"Matched text  : {result['matched_text']}")
        if result["ambiguous"]:
            print(f"Also possible : {', '.join(result['alternatives'])}")
    else:
        print("Suggestion    : none (no label reached the threshold)")
        top = result["candidates"][:3]
        if top:
            print("Closest labels: " + ", ".join(f"{c['medicine_label']} ({c['match_score']})" for c in top))
    print(f"Time          : {result['processing_time_ms'] / 1000:.2f} s")
    print("Text read from the image:")
    if result["detected_texts"]:
        for text in result["detected_texts"]:
            print(f"   - {text}")
    else:
        print("   (nothing readable)")
    print(LINE)


def confirm_step(service: ConfirmationService, result: dict) -> None:
    """Mimics what the mobile app does after /api/recognize."""
    request_id = uuid.uuid4().hex[:12]
    service.remember(request_id, result)
    if result["status"] == "SUGGESTED":
        answer = input(f"Is it {result['suggested_medicine']}? [y/n/Enter=skip]: ").strip().lower()
        if answer == "y":
            print(f"CONFIRMED -> {service.confirm(request_id, True)['medicine_name']}")
            return
        if answer != "n":
            return
        print("REJECTED.")
    name = input("Type the medicine name manually (Enter=skip): ").strip()
    if name:
        try:
            print(f"MANUAL_ACCEPTED -> {service.manual_input(request_id, name)['medicine_name']}")
        except ValueError as error:
            print(f"Invalid name: {error}")


def process(recognizer: MedicineRecognizer, service: ConfirmationService, path: str) -> None:
    path = path.strip().strip('"').strip("'")
    print(f"\nImage: {path}")
    try:
        result = recognizer.recognize(path)
    except ValueError as error:
        print(f"Error: {error}")
        return
    show(result)
    confirm_step(service, result)


def main() -> int:
    print("Loading OCR model and dataset (a few seconds, once) ...")
    recognizer = MedicineRecognizer(Config()).load()
    service = ConfirmationService()
    print(f"Ready. {len(recognizer.dataset)} medicine labels loaded.\n")

    for path in sys.argv[1:]:
        process(recognizer, service, path)

    while True:
        text = input("\nImage path (Enter = choose file, q = quit): ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            return 0
        path = text or pick_file()
        if not path:
            continue
        process(recognizer, service, path)


if __name__ == "__main__":
    raise SystemExit(main())
