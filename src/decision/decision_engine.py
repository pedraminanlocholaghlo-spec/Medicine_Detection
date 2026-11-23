"""Threshold decision: SUGGESTED vs UNKNOWN.

`match_score` is a RapidFuzz text-similarity score on a 0..100 scale. It is
NOT a calibrated probability of correct medical identification.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.matching.candidate_ranker import RankedCandidates, ambiguity


@dataclass
class Decision:
    status: str                       # "SUGGESTED" or "UNKNOWN"
    suggested_medicine: str | None
    match_score: float | None
    matched_text: str | None
    detected_texts: list[str]
    user_confirmation_required: bool
    manual_input_required: bool
    threshold: float
    ambiguous: bool = False
    alternatives: list[str] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    ocr_confidence: float | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "suggested_medicine": self.suggested_medicine,
            "match_score": round(self.match_score, 1) if self.match_score is not None else None,
            "matched_text": self.matched_text,
            "detected_texts": self.detected_texts,
            "user_confirmation_required": self.user_confirmation_required,
            "manual_input_required": self.manual_input_required,
            "threshold": self.threshold,
            "ambiguous": self.ambiguous,
            "alternatives": self.alternatives,
            "ocr_confidence": round(self.ocr_confidence, 3) if self.ocr_confidence is not None else None,
            "candidates": self.candidates,
        }


def decide(
    ranked: RankedCandidates,
    detected_texts: list[str],
    threshold: float,
    ambiguity_margin: float = 5.0,
    config=None,
) -> Decision:
    if config is not None:
        threshold = config.match_threshold
        ambiguity_margin = config.ambiguity_margin

    if config is not None:
        is_ambiguous, alternatives = ambiguity(ranked, config)
    else:
        is_ambiguous, alternatives = _ambiguity_from_margin(ranked, ambiguity_margin)

    best = ranked.best
    if best is None or best.match_score < threshold:
        return Decision(
            status="UNKNOWN",
            suggested_medicine=None,
            match_score=None,
            matched_text=None,
            detected_texts=list(detected_texts),
            user_confirmation_required=False,
            manual_input_required=True,
            threshold=threshold,
            ambiguous=is_ambiguous,
            alternatives=alternatives,
            candidates=[candidate.to_dict() for candidate in ranked.candidates],
        )

    return Decision(
        status="SUGGESTED",
        suggested_medicine=best.medicine_label,
        match_score=best.match_score,
        matched_text=best.matched_text,
        detected_texts=list(detected_texts),
        user_confirmation_required=True,
        manual_input_required=False,
        threshold=threshold,
        ambiguous=is_ambiguous,
        alternatives=alternatives,
        candidates=[candidate.to_dict() for candidate in ranked.candidates],
        ocr_confidence=best.ocr_confidence,
    )


def _ambiguity_from_margin(ranked: RankedCandidates, margin: float) -> tuple[bool, list[str]]:
    candidates = ranked.candidates
    if len(candidates) < 2:
        return False, []
    top = candidates[0].match_score
    alternatives = [
        item.medicine_label for item in candidates[1:] if top - item.match_score <= margin
    ]
    return bool(alternatives), alternatives
