"""Evaluate the recognition pipeline on the annotated dataset.

    python scripts/evaluate.py                      # labels.csv (+ negatives.csv if present)
    python scripts/evaluate.py --subset tuning      # use splits.csv, tuning rows only
    python scripts/evaluate.py --subset test        # FINAL test rows: run once, do not tune on it
    python scripts/evaluate.py --limit 10

Two kinds of images are evaluated:
  * in_dataset : the medicine IS in labels.csv. Outcomes: correct / incorrect / unknown.
                 Also reports whether the true medicine is in the top-1 / top-3 / top-5
                 candidates, independent of the threshold.
  * negative   : the medicine is NOT in labels.csv (negatives.csv). The only correct
                 answer is UNKNOWN. Outcomes: correct_unknown / false_suggestion.

Create the frozen split first with scripts/make_split.py. This script never tunes
thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from src.database.dataset_loader import DatasetLoader, DatasetError  # noqa: E402
from src.evaluation.performance import PerformanceReport  # noqa: E402
from src.pipeline import MedicineRecognizer  # noqa: E402
from src.text_processing.normalizer import compact_key, normalize_text  # noqa: E402

ANNOTATIONS = ROOT / "dataset" / "annotations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate medicine recognition on labeled images"
    )
    parser.add_argument("--labels", default=str(ANNOTATIONS / "labels.csv"))
    parser.add_argument(
        "--negatives",
        default=str(ANNOTATIONS / "negatives.csv"),
        help="Images of medicines NOT in labels.csv (optional)",
    )
    parser.add_argument("--splits", default=str(ANNOTATIONS / "splits.csv"))
    parser.add_argument(
        "--subset",
        choices=("all", "tuning", "test"),
        default="all",
        help="'tuning'/'test' read splits.csv created by make_split.py",
    )
    parser.add_argument("--images", default=str(ROOT / "dataset" / "images"))
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate only the first N rows"
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Default: outputs/evaluation_report.json (all) or "
        "outputs/evaluation_report_<subset>.json",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            clean = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            if clean.get("image_name"):
                rows.append(clean)
        return rows


def build_items(args, dataset) -> list[dict]:
    """Return [{image_name, medicine_label, kind}] for the chosen subset."""
    if args.subset != "all":
        splits_path = Path(args.splits)
        if not splits_path.exists():
            raise FileNotFoundError(
                f"{splits_path} not found. Run: python scripts/make_split.py"
            )
        return [
            {
                "image_name": r["image_name"],
                "medicine_label": r.get("medicine_label", ""),
                "kind": r.get("kind", "in_dataset"),
            }
            for r in read_rows(splits_path)
            if r.get("split") == args.subset
        ]

    items = [dict(row, kind="in_dataset") for row in dataset.rows]
    negatives_path = Path(args.negatives)
    if negatives_path.exists():
        known = {label.key for label in dataset.labels}
        for row in read_rows(negatives_path):
            label = normalize_text(row.get("medicine_label", ""))
            if label and compact_key(label) in known:
                print(
                    f"Warning: negative {row['image_name']} ('{label}') is in labels.csv; skipped"
                )
                continue
            items.append(
                {
                    "image_name": row["image_name"],
                    "medicine_label": label,
                    "kind": "negative",
                }
            )
    return items


def rate(numerator: int, denominator: int, digits: int = 4):
    return round(numerator / denominator, digits) if denominator else None


def main() -> int:
    args = parse_args()
    config = Config(labels_path=Path(args.labels), images_dir=Path(args.images))

    try:
        dataset = DatasetLoader.from_config(config).load(check_images=True)
    except DatasetError as error:
        print(f"Dataset error: {error}", file=sys.stderr)
        return 2
    if not dataset.rows:
        print(
            "No annotated rows found. Fill dataset/annotations/labels.csv first.",
            file=sys.stderr,
        )
        return 2
    if dataset.missing_images:
        print(
            f"Warning: {len(dataset.missing_images)} image(s) referenced but missing: "
            f"{', '.join(dataset.missing_images[:5])}"
        )

    try:
        items = build_items(args, dataset)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 2

    items = [
        item for item in items if (Path(args.images) / item["image_name"]).exists()
    ]
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("No usable annotated images found.", file=sys.stderr)
        return 2

    if args.subset == "test":
        print(
            "NOTE: this is the FINAL test subset. Do not change thresholds or code based on it.\n"
        )

    print(f"Loading OCR model once ... ({len(dataset.labels)} unique labels)")
    recognizer = MedicineRecognizer(config).load()
    print(f"Model load time: {recognizer.model_load_ms:.0f} ms\n")

    performance = PerformanceReport()
    stage_totals: dict[str, list[float]] = {}
    results = []
    pos = Counter()  # in-dataset outcomes
    neg = Counter()  # negative outcomes
    ranks: list[int | None] = []  # rank of the true label among candidates (positives)
    errors = 0

    for index, item in enumerate(items, start=1):
        image_path = Path(args.images) / item["image_name"]
        truth = item["medicine_label"]
        kind = item["kind"]
        try:
            result = recognizer.recognize(image_path)
        except ValueError as error:
            print(f"[{index}/{len(items)}] {item['image_name']}: ERROR {error}")
            errors += 1
            continue

        performance.add(result["processing_time_ms"])
        for stage, value in result["timings_ms"].items():
            stage_totals.setdefault(stage, []).append(value)

        candidates = result.get("candidates", [])
        top1 = candidates[0] if candidates else None
        truth_rank = None

        if kind == "negative":
            outcome = (
                "false_suggestion"
                if result["status"] == "SUGGESTED"
                else "correct_unknown"
            )
            neg[outcome] += 1
        else:
            truth_key = compact_key(truth)
            candidate_keys = [compact_key(c["medicine_label"]) for c in candidates]
            if truth_key in candidate_keys:
                truth_rank = candidate_keys.index(truth_key) + 1
            ranks.append(truth_rank)

            if (
                result["status"] == "SUGGESTED"
                and compact_key(result["suggested_medicine"] or "") == truth_key
            ):
                outcome = "correct"
            elif result["status"] == "SUGGESTED":
                outcome = "incorrect"
            else:
                outcome = "unknown"
            pos[outcome] += 1

        results.append(
            {
                "image_name": item["image_name"],
                "kind": kind,
                "medicine_label": truth,
                "status": result["status"],
                "suggested_medicine": result["suggested_medicine"],
                "match_score": result["match_score"],
                "matched_text": result["matched_text"],
                "outcome": outcome,
                "truth_rank": truth_rank,
                # Best candidate even when status is UNKNOWN (useful for threshold analysis).
                "top1_label": top1["medicine_label"] if top1 else None,
                "top1_score": top1["match_score"] if top1 else None,
                "candidates": [
                    [c["medicine_label"], c["match_score"]] for c in candidates
                ],
                "processing_time_ms": result["processing_time_ms"],
            }
        )
        print(
            f"[{index}/{len(items)}] {item['image_name']} ({kind}): truth={truth or '-'} "
            f"pred={result['suggested_medicine']} score={result['match_score']} "
            f"{outcome} ({result['processing_time_ms']:.0f} ms)"
        )

    evaluated_pos = sum(pos.values())
    evaluated_neg = sum(neg.values())
    suggestions = pos["correct"] + pos["incorrect"]
    timing = performance.summary()

    # Same keys as before, computed on in-dataset images only.
    overall = {
        "total_processed": evaluated_pos,
        "correct_suggestions": pos["correct"],
        "incorrect_suggestions": pos["incorrect"],
        "unknown_results": pos["unknown"],
        "errors": errors,
        "suggestion_rate": rate(suggestions, evaluated_pos) or 0.0,
        "wrong_suggestion_rate": rate(pos["incorrect"], evaluated_pos) or 0.0,
        "accuracy_on_suggestions": rate(pos["correct"], suggestions),
        "average_processing_time_ms": timing.get("average_processing_time_ms"),
        "max_processing_time_ms": timing.get("max_processing_time_ms"),
        "p95_processing_time_ms": timing.get("p95_processing_time_ms"),
        "under_6_seconds": timing.get("under_6_seconds"),
        "model_load_ms": round(recognizer.model_load_ms or 0, 1),
    }
    top_k = {
        "evaluated": evaluated_pos,
        "top1_hit_rate": rate(sum(1 for r in ranks if r == 1), evaluated_pos),
        "top3_hit_rate": rate(
            sum(1 for r in ranks if r is not None and r <= 3), evaluated_pos
        ),
        "top5_hit_rate": rate(sum(1 for r in ranks if r is not None), evaluated_pos),
        "note": "Independent of the threshold: is the true medicine among the candidates at all?",
    }
    negatives = {
        "total_processed": evaluated_neg,
        "correct_unknown": neg["correct_unknown"],
        "false_suggestions": neg["false_suggestion"],
        "correct_unknown_rate": rate(neg["correct_unknown"], evaluated_neg),
        "false_suggestion_rate": rate(neg["false_suggestion"], evaluated_neg),
        "note": "Medicines NOT in labels.csv: the correct answer is UNKNOWN.",
    }
    stage_averages = {
        stage: round(statistics.fmean(values), 1)
        for stage, values in stage_totals.items()
    }

    report = {
        "labels_file": str(args.labels),
        "subset": args.subset,
        "unique_labels": len(dataset.labels),
        "duplicate_rows": dataset.duplicate_rows,
        "threshold": config.match_threshold,
        "overall": overall,
        "top_k": top_k,
        "negatives": negatives,
        "average_stage_times_ms": stage_averages,
        "results": results,
    }
    report_arg = args.report or str(
        ROOT
        / "outputs"
        / (
            "evaluation_report.json"
            if args.subset == "all"
            else f"evaluation_report_{args.subset}.json"
        )
    )
    report_path = Path(report_arg)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n--- Evaluation summary (subset={args.subset}) ---")
    print("[in-dataset]")
    for key, value in overall.items():
        print(f"{key:>32}: {value}")
    print("[top-k]")
    for key, value in top_k.items():
        if key != "note":
            print(f"{key:>32}: {value}")
    print("[negatives]")
    for key, value in negatives.items():
        if key != "note":
            print(f"{key:>32}: {value}")
    print(f"{'average_stage_times_ms':>32}: {json.dumps(stage_averages)}")
    print(f"\nReport written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
