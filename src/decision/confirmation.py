"""Confirmation and manual-input handling.

Manually entered names are acknowledged for the current request only. They are
never written to the dataset; dataset updates require a separate validation
process.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from src.text_processing.normalizer import normalize_text


class UnknownRequestError(KeyError):
    pass


@dataclass
class PendingResult:
    request_id: str
    status: str
    suggested_medicine: str | None
    match_score: float | None
    created_at: float


class ConfirmationService:
    """In-memory store of recent recognition results."""

    def __init__(self, max_pending: int = 500):
        self._pending: dict[str, PendingResult] = {}
        self._max_pending = max_pending

    def remember(self, request_id: str, result: dict) -> None:
        if len(self._pending) >= self._max_pending:
            oldest = min(self._pending, key=lambda key: self._pending[key].created_at)
            self._pending.pop(oldest, None)
        self._pending[request_id] = PendingResult(
            request_id=request_id,
            status=result.get("status", "UNKNOWN"),
            suggested_medicine=result.get("suggested_medicine"),
            match_score=result.get("match_score"),
            created_at=time.time(),
        )

    def confirm(self, request_id: str, accepted: bool, medicine_name: str | None = None) -> dict:
        pending = self._pending.get(request_id)
        if pending is None:
            raise UnknownRequestError(request_id)

        if accepted:
            if not pending.suggested_medicine:
                raise ValueError("No medicine was suggested for this request")
            return {
                "request_id": request_id,
                "status": "CONFIRMED",
                "medicine_name": pending.suggested_medicine,
                "source": "SUGGESTION",
                "match_score": pending.match_score,
                "user_confirmation_required": False,
                "manual_input_required": False,
                "dataset_updated": False,
            }

        if medicine_name is not None and medicine_name.strip():
            return self.manual_input(request_id, medicine_name)

        return {
            "request_id": request_id,
            "status": "REJECTED",
            "medicine_name": None,
            "source": "USER_REJECTED",
            "match_score": pending.match_score,
            "user_confirmation_required": False,
            "manual_input_required": True,
            "dataset_updated": False,
        }

    def manual_input(self, request_id: str, medicine_name: str) -> dict:
        if request_id not in self._pending:
            raise UnknownRequestError(request_id)
        cleaned = _validate_manual_name(medicine_name)
        return {
            "request_id": request_id,
            "status": "MANUAL_ACCEPTED",
            "medicine_name": cleaned,
            "source": "MANUAL_INPUT",
            "match_score": None,
            "user_confirmation_required": False,
            "manual_input_required": False,
            "dataset_updated": False,
        }


def _validate_manual_name(medicine_name: str) -> str:
    normalized = normalize_text(medicine_name)
    if len(normalized) < 2:
        raise ValueError("Medicine name must contain at least 2 letters")
    if len(normalized) > 120:
        raise ValueError("Medicine name is too long (maximum 120 characters)")
    if not re.search(r"[A-Z]", normalized):
        raise ValueError("Medicine name must contain English letters")
    return normalized
