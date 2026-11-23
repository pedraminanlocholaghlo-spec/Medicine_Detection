"""Two-stage medicine label matching: exact first, then RapidFuzz fuzzy.

Queries are built only from meaningful tokens and from *adjacent* OCR regions
(same row / directly stacked). The whole page is never blindly concatenated.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from rapidfuzz import fuzz, process

from src.database.dataset_loader import MedicineLabel
from src.ocr.text_detector import TextRegion, adjacent_pairs
from src.text_processing.normalizer import compact_key, corrected_tokens, normalize_text, tokenize


@dataclass(frozen=True)
class Query:
    text: str                 # human readable OCR-derived text
    key: str                  # compact comparison key
    ocr_confidence: float
    region_indices: tuple


@dataclass(frozen=True)
class Match:
    medicine_label: str
    match_score: float
    matched_text: str
    ocr_confidence: float
    matching_method: str      # "EXACT" or "FUZZY"
    query_key: str


@dataclass
class MatchOutcome:
    matches: list[Match]
    queries: int
    exact_hits: int
    timings_ms: dict


def _token_combinations(tokens: Sequence[str]) -> list[list[str]]:
    """Single tokens, adjacent pairs and the full (short) sequence."""
    combinations: list[list[str]] = [[token] for token in tokens]
    combinations.extend([tokens[i : i + 2] for i in range(len(tokens) - 1)])
    if 2 <= len(tokens) <= 6:
        combinations.append(list(tokens))
    return combinations


def build_queries(regions: Sequence[TextRegion], config) -> list[Query]:
    candidates: list[tuple[list[str], float, tuple]] = []
    tokens_per_region: list[list[str]] = []

    for index, region in enumerate(regions):
        tokens = tokenize(region.text, config.ignored_tokens)
        tokens_per_region.append(tokens)
        for combination in _token_combinations(tokens):
            candidates.append((combination, region.ocr_confidence, (index,)))

    if config.combine_adjacent:
        for left, right in adjacent_pairs(regions):
            combined = tokens_per_region[left] + tokens_per_region[right]
            if not combined:
                continue
            # Keep short, name-sized combinations only.
            for combination in _token_combinations(combined[-6:]):
                if len(combination) < 2:
                    continue
                candidates.append(
                    (
                        combination,
                        min(regions[left].ocr_confidence, regions[right].ocr_confidence),
                        (left, right),
                    )
                )

    best: dict[str, Query] = {}
    for tokens, confidence, indices in candidates:
        # Original tokens and the controlled OCR-corrected variant are both
        # searched; the original is never replaced.
        for variant in (list(tokens), corrected_tokens(tokens)):
            text = normalize_text(" ".join(variant))
            key = compact_key(text)
            if len(key) < config.min_query_length:
                continue
            current = best.get(key)
            if current is None or confidence > current.ocr_confidence:
                best[key] = Query(text=text, key=key, ocr_confidence=confidence, region_indices=indices)

    queries = sorted(best.values(), key=lambda query: -query.ocr_confidence)
    return queries[: config.max_queries]


class Matcher:
    def __init__(self, labels: Sequence[MedicineLabel], config):
        self.labels = list(labels)
        self.config = config
        self._by_key = {label.key: label for label in self.labels}
        self._keys = [label.key for label in self.labels]
        self._scorer = getattr(fuzz, config.fuzzy_scorer)

    def match(self, regions: Sequence[TextRegion]) -> MatchOutcome:
        timings = {"exact_matching": 0.0, "fuzzy_matching": 0.0}
        matches: list[Match] = []
        exact_hits = 0

        started = time.perf_counter()
        queries = build_queries(regions, self.config)
        timings["normalization"] = (time.perf_counter() - started) * 1000

        exact_started = time.perf_counter()
        fuzzy_queries: list[Query] = []
        for query in queries:
            label = self._by_key.get(query.key)
            if label is None:
                fuzzy_queries.append(query)
                continue
            exact_hits += 1
            matches.append(
                Match(
                    medicine_label=label.label,
                    match_score=100.0,
                    matched_text=query.text,
                    ocr_confidence=query.ocr_confidence,
                    matching_method="EXACT",
                    query_key=query.key,
                )
            )
        timings["exact_matching"] = (time.perf_counter() - exact_started) * 1000

        fuzzy_started = time.perf_counter()
        for query in fuzzy_queries:
            if len(query.key) < self.config.min_fuzzy_query_length:
                continue
            results = process.extract(
                query.key,
                self._keys,
                scorer=self._scorer,
                score_cutoff=self.config.fuzzy_candidate_cutoff,
                limit=self.config.top_k_per_query,
            )
            for key, score, _ in results:
                label = self._by_key[key]
                matches.append(
                    Match(
                        medicine_label=label.label,
                        match_score=float(score),
                        matched_text=query.text,
                        ocr_confidence=query.ocr_confidence,
                        matching_method="FUZZY",
                        query_key=query.key,
                    )
                )
        timings["fuzzy_matching"] = (time.perf_counter() - fuzzy_started) * 1000

        return MatchOutcome(
            matches=matches, queries=len(queries), exact_hits=exact_hits, timings_ms=timings
        )
