"""Detected text regions and bounding-box grouping helpers.

The OCR engine performs detection and recognition; this module owns the
region data structure and the geometry rules used to decide which text
regions belong together (same row or directly stacked).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

Box = list  # [x1, y1, x2, y2]


@dataclass
class TextRegion:
    """One recognized text region, exactly as required by the specification."""

    text: str
    ocr_confidence: float
    bounding_box: Optional[Box]

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "ocr_confidence": round(float(self.ocr_confidence), 4),
            "bounding_box": self.bounding_box,
        }


def polygon_to_box(polygon) -> Box:
    """Convert a 4-point polygon (or an existing [x1,y1,x2,y2]) to a bbox."""
    points = list(polygon)
    if len(points) == 4 and all(not hasattr(p, "__len__") for p in points):
        return [int(round(float(p))) for p in points]
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))]


def reading_order(regions: Iterable[TextRegion]) -> list[TextRegion]:
    """Top-to-bottom, left-to-right ordering; stable for regions without boxes."""

    def sort_key(item):
        index, region = item
        box = region.bounding_box
        if not box:
            return (float("inf"), float("inf"), index)
        return (box[1], box[0], index)

    return [region for _, region in sorted(enumerate(regions), key=sort_key)]


def _height(box: Box) -> int:
    return max(1, box[3] - box[1])


def same_row(a: Box, b: Box, max_gap_ratio: float = 1.5) -> bool:
    """Regions on one line: vertical overlap and a small horizontal gap."""
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    if overlap < 0.5 * min(_height(a), _height(b)):
        return False
    gap = max(a[0], b[0]) - min(a[2], b[2])
    return gap <= max_gap_ratio * max(_height(a), _height(b))


def stacked(a: Box, b: Box, max_gap_ratio: float = 1.0) -> bool:
    """`b` directly below `a` (or the reverse is checked by the caller)."""
    gap = b[1] - a[3]
    if not (-0.2 * _height(a) <= gap <= max_gap_ratio * max(_height(a), _height(b))):
        return False
    overlap = min(a[2], b[2]) - max(a[0], b[0])
    return overlap >= 0.3 * min(a[2] - a[0], b[2] - b[0])


def adjacent_pairs(
    regions: Sequence[TextRegion], max_gap_ratio: float = 1.5
) -> list[tuple[int, int]]:
    """Index pairs that may form one medicine name split across regions.

    Only true neighbours are returned: no blind concatenation of the whole
    page. Regions without bounding boxes are paired only when consecutive in
    reading order.
    """
    pairs: list[tuple[int, int]] = []
    for i, a in enumerate(regions):
        for j in range(i + 1, len(regions)):
            b = regions[j]
            if a.bounding_box is None or b.bounding_box is None:
                if j == i + 1:
                    pairs.append((i, j))
                continue
            if same_row(a.bounding_box, b.bounding_box, max_gap_ratio):
                left, right = (i, j) if a.bounding_box[0] <= b.bounding_box[0] else (j, i)
                pairs.append((left, right))
            elif stacked(a.bounding_box, b.bounding_box, max_gap_ratio):
                pairs.append((i, j))
            elif stacked(b.bounding_box, a.bounding_box, max_gap_ratio):
                pairs.append((j, i))
    return pairs
