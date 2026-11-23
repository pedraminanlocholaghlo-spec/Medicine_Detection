"""Manual dataset loading: `dataset/annotations/labels.csv`.

CSV contract (exactly two required columns):
    image_name,medicine_label

Several images may point to the same medicine; labels are deduplicated into a
unique list. No external medicine database is ever consulted.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from src.text_processing.normalizer import compact_key, normalize_text

REQUIRED_COLUMNS = ("image_name", "medicine_label")


class DatasetError(ValueError):
    """Raised when the annotation file is missing or malformed."""


@dataclass(frozen=True)
class MedicineLabel:
    label: str          # normalized display form, e.g. "AMOXICILLIN"
    key: str            # comparison key, e.g. "AMOXICILLIN"
    image_count: int = 1


@dataclass
class Dataset:
    labels: list[MedicineLabel]
    rows: list[dict] = field(default_factory=list)
    duplicate_rows: int = 0
    missing_images: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def label_names(self) -> list[str]:
        return [item.label for item in self.labels]

    def find(self, key: str) -> MedicineLabel | None:
        return next((item for item in self.labels if item.key == key), None)


class DatasetLoader:
    def __init__(self, labels_path, images_dir):
        self.labels_path = Path(labels_path)
        self.images_dir = Path(images_dir)

    @classmethod
    def from_config(cls, config) -> "DatasetLoader":
        return cls(config.labels_path, config.images_dir)

    def load(self, check_images: bool = False) -> Dataset:
        if not self.labels_path.exists():
            raise DatasetError(
                f"Annotation file not found: {self.labels_path}. "
                "Create it from dataset/annotations/labels.example.csv."
            )

        rows: list[dict] = []
        missing_images: list[str] = []
        seen: dict[str, MedicineLabel] = {}
        duplicate_rows = 0

        with open(self.labels_path, encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = [name.strip() for name in (reader.fieldnames or [])]
            missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
            if missing:
                raise DatasetError(
                    f"{self.labels_path.name} is missing required column(s): {', '.join(missing)}. "
                    f"Expected header: {','.join(REQUIRED_COLUMNS)}"
                )
            for row in reader:
                image_name = (row.get("image_name") or "").strip()
                raw_label = (row.get("medicine_label") or "").strip()
                if not image_name or not raw_label:
                    continue
                label = normalize_text(raw_label)
                key = compact_key(label)
                if not key:
                    continue
                rows.append({"image_name": image_name, "medicine_label": label})
                if check_images and not (self.images_dir / image_name).exists():
                    missing_images.append(image_name)
                existing = seen.get(key)
                if existing is None:
                    seen[key] = MedicineLabel(label=label, key=key, image_count=1)
                else:
                    duplicate_rows += 1
                    seen[key] = MedicineLabel(label=existing.label, key=key, image_count=existing.image_count + 1)

        labels = sorted(seen.values(), key=lambda item: item.label)
        return Dataset(labels=labels, rows=rows, duplicate_rows=duplicate_rows, missing_images=missing_images)
