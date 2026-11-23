"""Candidate ranking: one entry per medicine label, highest score first."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.matching.matcher import Match


@dataclass
class Candidate:
    medicine_label: str
    match_score: float
    matched_text: str
    ocr_confidence: float
    matching_method: str

    def to_dict(self) -> dict:
        return {
            "medicine_label": self.medicine_label,
            "match_score": round(self.match_score, 1),
            "matched_text": self.matched_text,
            "ocr_confidence": round(self.ocr_confidence, 3),
            "matching_method": self.matching_method,
        }


@dataclass
class RankedCandidates:
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def best_score(self) -> float:
        return self.best.match_score if self.best else 0.0


def rank_candidates(matches: list[Match], config) -> RankedCandidates:
    """Deduplicate by label, keep the strongest evidence, then sort."""
    best: dict[str, Candidate] = {}
    for match in matches:
        candidate = Candidate(
            medicine_label=match.medicine_label,
            match_score=match.match_score,
            matched_text=match.matched_text,
            ocr_confidence=match.ocr_confidence,
            matching_method=match.matching_method,
        )
        current = best.get(match.medicine_label)
        if current is None or _is_stronger(candidate, current):
            best[match.medicine_label] = candidate

    ordered = sorted(
        best.values(),
        key=lambda item: (-item.match_score, item.matching_method != "EXACT", -item.ocr_confidence, item.medicine_label),
    )
    return RankedCandidates(candidates=ordered[: config.max_candidates])


def _is_stronger(new: Candidate, current: Candidate) -> bool:
    if new.match_score != current.match_score:
        return new.match_score > current.match_score
    if new.matching_method != current.matching_method:
        return new.matching_method == "EXACT"
    return new.ocr_confidence > current.ocr_confidence


def ambiguity(ranked: RankedCandidates, config) -> tuple[bool, list[str]]:
    """Report near-ties so the user can review alternatives."""
    candidates = ranked.candidates
    if len(candidates) < 2:
        return False, []
    top, second = candidates[0], candidates[1]
    if top.match_score - second.match_score > config.ambiguity_margin:
        return False, []
    alternatives = [
        item.medicine_label
        for item in candidates[1:]
        if top.match_score - item.match_score <= config.ambiguity_margin
    ]
    return True, alternatives
