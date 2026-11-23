"""Create a frozen tuning/test split, grouped by medicine.

    python scripts/make_split.py
    python scripts/make_split.py --test-fraction 0.3 --seed 42

Reads
    dataset/annotations/labels.csv      (image_name,medicine_label)  -> kind "in_dataset"
    dataset/annotations/negatives.csv   (image_name[,medicine_label]) -> kind "negative"
        negatives are photos of medicines that are NOT in labels.csv; the correct
        system answer for them is UNKNOWN.

Writes
    dataset/annotations/splits.csv      (image_name,medicine_label,kind,split)

All images of one medicine land on the same side (tuning OR test), so a test
photo never has a sibling photo of the same box in the tuning set. The split is
frozen: an existing splits.csv is never overwritten without --force, so the
test set cannot drift while you tune.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.text_processing.normalizer import compact_key, normalize_text  # noqa: E402


def read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            clean = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            if clean.get("image_name"):
                rows.append(clean)
        return rows


def group_key(row: dict, kind: str) -> str:
    """Rows sharing a key always go to the same split."""
    label = row.get("medicine_label", "")
    key = compact_key(normalize_text(label)) if label else ""
    # Negatives without a label are grouped per image.
    return f"{kind}:{key or row['image_name']}"


def assign_splits(rows: list[dict], test_fraction: float, seed: int) -> dict[str, str]:
    """Return {group_key: 'tuning'|'test'} so that ~test_fraction of rows are test."""
    groups: dict[str, int] = defaultdict(int)
    for row in rows:
        groups[row["group"]] += 1
    ordered = sorted(groups)  # sorted first so the shuffle depends only on the seed
    random.Random(seed).shuffle(ordered)

    target = round(len(rows) * test_fraction)
    assignment: dict[str, str] = {}
    test_rows = 0
    for key in ordered:
        if test_rows < target:
            assignment[key] = "test"
            test_rows += groups[key]
        else:
            assignment[key] = "tuning"
    return assignment


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a frozen, medicine-grouped tuning/test split")
    parser.add_argument("--labels", default=str(ROOT / "dataset" / "annotations" / "labels.csv"))
    parser.add_argument("--negatives", default=str(ROOT / "dataset" / "annotations" / "negatives.csv"))
    parser.add_argument("--output", default=str(ROOT / "dataset" / "annotations" / "splits.csv"))
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing split (breaks test-set freeze)")
    args = parser.parse_args()

    if not 0 < args.test_fraction < 1:
        print("--test-fraction must be between 0 and 1", file=sys.stderr)
        return 2

    output = Path(args.output)
    if output.exists() and not args.force:
        print(f"{output} already exists. The split is frozen; use --force only if you accept "
              "that earlier test results are no longer comparable.", file=sys.stderr)
        return 2

    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"Labels file not found: {labels_path}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    known_keys: set[str] = set()
    for row in read_rows(labels_path):
        label = normalize_text(row.get("medicine_label", ""))
        if not label:
            continue
        known_keys.add(compact_key(label))
        rows.append({"image_name": row["image_name"], "medicine_label": label, "kind": "in_dataset"})

    negatives_path = Path(args.negatives)
    negative_count = 0
    if negatives_path.exists():
        for row in read_rows(negatives_path):
            label = normalize_text(row.get("medicine_label", ""))
            if label and compact_key(label) in known_keys:
                print(f"Error: negative image {row['image_name']} has label '{label}' which IS in "
                      "labels.csv. Negatives must be medicines that are not in the dataset.",
                      file=sys.stderr)
                return 2
            rows.append({"image_name": row["image_name"], "medicine_label": label, "kind": "negative"})
            negative_count += 1
    else:
        print(f"Note: {negatives_path.name} not found; the split will contain no negative samples.")

    if not rows:
        print("No rows found.", file=sys.stderr)
        return 2

    names = [row["image_name"] for row in rows]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        print(f"Error: image(s) listed more than once across files: {', '.join(sorted(duplicates)[:5])}",
              file=sys.stderr)
        return 2

    # Positives and negatives are split independently so both kinds appear on both sides.
    final: list[dict] = []
    for kind in ("in_dataset", "negative"):
        subset = [dict(row, group=group_key(row, kind)) for row in rows if row["kind"] == kind]
        if not subset:
            continue
        assignment = assign_splits(subset, args.test_fraction, args.seed)
        for row in subset:
            final.append({
                "image_name": row["image_name"],
                "medicine_label": row["medicine_label"],
                "kind": kind,
                "split": assignment[row["group"]],
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name", "medicine_label", "kind", "split"])
        writer.writeheader()
        writer.writerows(final)

    print(f"Wrote {output} (seed={args.seed}, test_fraction={args.test_fraction})")
    for kind in ("in_dataset", "negative"):
        for split in ("tuning", "test"):
            items = [r for r in final if r["kind"] == kind and r["split"] == split]
            medicines = {group_key(r, kind) for r in items}
            print(f"  {kind:>10} / {split:<6}: {len(items):>4} images, {len(medicines):>4} medicines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
