from src.text_processing.normalizer import (
    compact_key,
    corrected_tokens,
    normalize_text,
    ocr_corrected,
    tokenize,
)

IGNORED = frozenset({"MG", "ML", "EXP", "BATCH", "LOT", "CAPSULE", "CAPSULES", "TABLET"})


def test_uppercase_and_whitespace():
    assert normalize_text("  amoxicillin  500 mg ") == "AMOXICILLIN 500 MG"


def test_punctuation_removed():
    assert normalize_text("Amoxicillin, (500mg)!") == "AMOXICILLIN 500MG"


def test_compact_key_removes_separators():
    assert compact_key("Amoxicillin-500 MG") == "AMOXICILLIN500MG"
    assert compact_key("amoxicillin") == compact_key("AMOXI CILLIN")


def test_fixture_examples():
    assert normalize_text("AMOXICILLIN") == "AMOXICILLIN"
    assert compact_key(normalize_text("amoxicillin")) == "AMOXICILLIN"


def test_controlled_ocr_correction():
    assert ocr_corrected("AM0XICILLIN") == "AMOXICILLIN"
    # A second confusion map entry is still applied
    assert ocr_corrected("AMOX1CILLIN") == "AMOXICILLIN"


def test_ocr_correction_leaves_numeric_tokens_alone():
    assert ocr_corrected("500") == "500"
    assert ocr_corrected("B12") == "B12"
    assert ocr_corrected("MG") == "MG"
    assert ocr_corrected("500XG") == "500XG"  # >25% digits


def test_original_is_preserved_by_corrected_tokens():
    assert corrected_tokens(["AM0XICILLIN"]) == ["AMOXICILLIN"]


def test_tokenize_removes_ignored_noise():
    assert tokenize("AMOXICILLIN CAPSULES EXP 2027 500 MG", IGNORED) == ["AMOXICILLIN"]
    assert tokenize("BATCH 1234", IGNORED) == []


def test_tokenize_keeps_letters_in_mixed_tokens():
    assert tokenize("CIPRO 500XG", IGNORED) == ["CIPRO"]
