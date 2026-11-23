from config import Config
from src.database.dataset_loader import Dataset, MedicineLabel
from src.decision.decision_engine import decide
from src.matching.candidate_ranker import Candidate, RankedCandidates, ambiguity, rank_candidates
from src.matching.matcher import Matcher, Match
from src.ocr.text_detector import TextRegion
from src.text_processing.normalizer import compact_key


def make_config(**overrides):
    cfg = Config(log_requests=False)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_dataset(*names):
    labels = [
        MedicineLabel(label=name, key=compact_key(name), image_count=1) for name in names
    ]
    return Dataset(labels=labels)


def recognize(regions, names=("AMOXICILLIN", "AMPICILLIN"), cfg=None):
    cfg = cfg or make_config()
    matcher = Matcher(make_dataset(*names).labels, cfg)
    outcome = matcher.match(regions)
    ranked = rank_candidates(outcome.matches, cfg)
    return ranked, decide(ranked, [r.text for r in regions], cfg.match_threshold, config=cfg)


def region(text, box=(10, 10, 300, 50), conf=0.95):
    return TextRegion(text=text, ocr_confidence=conf, bounding_box=list(box))


def test_exact_match_scores_100():
    ranked, decision = recognize([region("AMOXICILLIN")])
    assert ranked.best.medicine_label == "AMOXICILLIN"
    assert ranked.best.match_score == 100.0
    assert ranked.best.matching_method == "EXACT"
    assert decision.status == "SUGGESTED"
    assert decision.user_confirmation_required is True


def test_fuzzy_match_from_ocr_typo():
    ranked, decision = recognize([region("AMOXICILIN")], names=("AMOXICILLIN",))
    assert ranked.best.medicine_label == "AMOXICILLIN"
    assert ranked.best.match_score >= 60
    assert ranked.best.matching_method == "FUZZY"
    assert decision.status == "SUGGESTED"


def test_controlled_ocr_correction_matches_exactly():
    ranked, decision = recognize([region("AM0XICILLIN")], names=("AMOXICILLIN",))
    assert ranked.best.matching_method == "EXACT"
    assert ranked.best.match_score == 100.0
    assert decision.status == "SUGGESTED"


def test_fuzzy_match_prefers_correct_drug():
    ranked, _ = recognize([region("AMOXICILIN")], names=("AMOXICILLIN", "AMPICILLIN"))
    assert ranked.best.medicine_label == "AMOXICILLIN"
    second = ranked.candidates[1]
    assert second.medicine_label == "AMPICILLIN"
    assert second.match_score < ranked.best.match_score


def test_split_name_across_lines_uses_boxes():
    regions = [
        region("AMOXI", box=(100, 100, 260, 140)),
        region("CILLIN", box=(105, 145, 270, 185)),
    ]
    ranked, decision = recognize(regions, names=("AMOXICILLIN",))
    assert ranked.best.medicine_label == "AMOXICILLIN"
    assert ranked.best.match_score == 100.0
    assert decision.status == "SUGGESTED"


def test_unrelated_regions_are_not_concatenated():
    regions = [
        region("AMOXI", box=(10, 10, 100, 40)),
        region("CILLIN", box=(600, 500, 700, 540)),
    ]
    cfg = make_config(match_threshold=80)
    ranked, decision = recognize(regions, names=("AMOXICILLIN",), cfg=cfg)
    assert ranked.best is None or ranked.best.match_score < 100
    assert decision.status == "UNKNOWN"


def test_multiple_regions_are_all_reported():
    regions = [
        region("AM0XICILLIN", box=(10, 10, 300, 50)),
        region("500 MG", box=(10, 60, 150, 90)),
        region("CAPSULE", box=(10, 100, 200, 130)),
        region("EXP 2027", box=(10, 140, 200, 170)),
    ]
    ranked, decision = recognize(regions, names=("AMOXICILLIN",))
    assert decision.detected_texts == ["AM0XICILLIN", "500 MG", "CAPSULE", "EXP 2027"]
    assert decision.suggested_medicine == "AMOXICILLIN"


def test_unknown_medicine_requires_manual_input():
    _, decision = recognize([region("XQZVWK")], names=("AMOXICILLIN", "IBUPROFEN"))
    assert decision.status == "UNKNOWN"
    assert decision.suggested_medicine is None
    assert decision.match_score is None
    assert decision.manual_input_required is True
    assert decision.user_confirmation_required is False


def test_empty_ocr_output():
    _, decision = recognize([], names=("AMOXICILLIN",))
    assert decision.status == "UNKNOWN"
    assert decision.detected_texts == []
    assert decision.manual_input_required is True


def test_threshold_is_configurable():
    cfg = make_config(match_threshold=99.9)
    ranked, decision = recognize([region("AMOXICILIN")], names=("AMOXICILLIN",), cfg=cfg)
    assert ranked.best.match_score < 99.9
    assert decision.status == "UNKNOWN"
    assert decision.threshold == 99.9


def test_default_threshold_accepts_clear_fuzzy_match():
    cfg = make_config(match_threshold=60.0)
    _, decision = recognize([region("AMOXICILIN")], names=("AMOXICILLIN",), cfg=cfg)
    assert decision.status == "SUGGESTED"


def test_duplicate_candidates_are_not_returned():
    matches = [
        Match("AMOXICILLIN", 100.0, "AMOXICILLIN", 0.9, "EXACT", "AMOXICILLIN"),
        Match("AMOXICILLIN", 93.0, "AMOXICILLIN", 0.8, "FUZZY", "AMOXICILLIN"),
    ]
    ranked = rank_candidates(matches, make_config())
    assert len(ranked.candidates) == 1
    assert ranked.candidates[0].match_score == 100.0


def test_similar_candidates_are_flagged_ambiguous():
    cfg = make_config()
    ranked = RankedCandidates(
        candidates=[
            Candidate("AMOXICILLIN", 100.0, "AMOXICILLIN", 0.95, "EXACT"),
            Candidate("AMOXICILLINA", 97.0, "AMOXICILLIN", 0.9, "FUZZY"),
        ]
    )
    is_ambiguous, alternatives = ambiguity(ranked, cfg)
    assert is_ambiguous is True
    assert alternatives == ["AMOXICILLINA"]

    decision = decide(ranked, ["AMOXICILLIN"], cfg.match_threshold, config=cfg)
    assert decision.status == "SUGGESTED"
    assert decision.ambiguous is True
    assert decision.user_confirmation_required is True


def test_ranking_is_sorted_and_capped():
    matches = [
        Match("IBUPROFEN", 70.0, "IBUPROFEN", 0.9, "FUZZY", "IBUPROFEN"),
        Match("AMOXICILLIN", 95.0, "AMOXICILLIN", 0.9, "FUZZY", "AMOXICILLIN"),
        Match("AMPICILLIN", 80.0, "AMPICILLIN", 0.9, "FUZZY", "AMPICILLIN"),
    ]
    ranked = rank_candidates(matches, make_config(max_candidates=2))
    assert [c.medicine_label for c in ranked.candidates] == ["AMOXICILLIN", "AMPICILLIN"]
    assert ranked.candidates[0].match_score >= ranked.candidates[1].match_score
