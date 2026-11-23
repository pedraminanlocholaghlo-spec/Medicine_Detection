"""Text normalization for English OCR output.

Kept independent from matching: the matcher only consumes the functions
exported here, never re-implements normalization rules.

Normalization never modifies the stored raw OCR text. `ocr_corrected()`
produces an *additional* controlled variant (0 -> O and similar confusions)
that is searched alongside the original, never instead of it.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Characters that OCR commonly confuses inside alphabetic words.
_OCR_CONFUSIONS = {"0": "O", "1": "I", "5": "S", "8": "B"}

_NON_ALNUM = re.compile(r"[^A-Z0-9]+")


def normalize_text(text: str) -> str:
    """Uppercase, NFKC-normalize, strip punctuation and collapse whitespace."""
    value = unicodedata.normalize("NFKC", text or "")
    value = value.upper()
    value = value.replace("–", "-").replace("—", "-")
    value = re.sub(r"[^A-Z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def compact_key(text: str) -> str:
    """Comparison key: letters and digits only (spaces/hyphens removed)."""
    return _NON_ALNUM.sub("", normalize_text(text))


def is_meaningful_token(token: str) -> bool:
    """True for tokens longer than one character that contain a letter."""
    return len(token) >= 2 and any(ch.isalpha() for ch in token)


def tokenize(text: str, ignored: frozenset = frozenset()) -> list[str]:
    """Normalized tokens with dosage/packaging noise and pure numbers removed."""
    tokens = []
    for token in normalize_text(text).split():
        if token in ignored or not is_meaningful_token(token):
            continue
        digits = sum(ch.isdigit() for ch in token)
        if digits and digits / len(token) > 0.4:
            continue
        tokens.append(token)
    return tokens


def ocr_corrected(token: str) -> str:
    """Controlled OCR correction for a single token.

    Only applied to mostly alphabetic tokens of at least four characters with
    at most 25% digits, so "B12", "500" or "MG" are never rewritten.
    """
    if not re.fullmatch(r"[A-Z0-9]+", token):
        return token
    digits = sum(ch.isdigit() for ch in token)
    letters = len(token) - digits
    if digits == 0 or letters < 4 or digits / len(token) > 0.25:
        return token
    return "".join(_OCR_CONFUSIONS.get(ch, ch) for ch in token)


def corrected_tokens(tokens: list[str]) -> list[str]:
    return [ocr_corrected(token) for token in tokens]


@dataclass(frozen=True)
class NormalizedText:
    original: str
    normalized: str
    key: str

    @classmethod
    def from_raw(cls, raw: str) -> "NormalizedText":
        normalized = normalize_text(raw)
        return cls(original=raw, normalized=normalized, key=compact_key(normalized))
