"""Central configuration for the Medicine Name Recognition system.

Every tunable value lives here so that the pipeline modules stay free of
hardcoded paths, thresholds and model names.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _ignored_tokens() -> frozenset:
    """Short non-name tokens that must not be matched against medicine labels."""
    return frozenset(
        {
            # dosage units / forms
            "MG", "ML", "MCG", "G", "IU", "KG", "TAB", "TABS", "TABLET", "TABLETS",
            "CAP", "CAPS", "CAPSULE", "CAPSULES", "SYRUP", "SUSPENSION", "CREAM",
            "OINTMENT", "DROPS", "GEL", "INJECTION", "SOLUTION", "SPRAY",
            # packaging / regulatory
            "EXP", "EXPIRY", "MFG", "MFD", "BATCH", "LOT", "BN", "REF", "CODE",
            "STORE", "KEEP", "BELOW", "ABOVE", "PROTECT", "LIGHT", "MOISTURE",
            "WARNING", "CHILDREN", "REACH", "OUT", "FROM", "PRICE", "PACK", "BOX",
            "STRIP", "BLISTER", "PHARMA", "PHARMACEUTICAL", "PHARMACEUTICALS",
            "LTD", "CO", "INC", "GMBH", "PLC", "MADE", "USE", "ORAL", "EACH",
            "CONTAINS", "CONTAIN", "ACTIVE", "INGREDIENT", "DOSE", "DOSAGE",
            "THE", "AND", "OF", "WITH", "FOR", "IN", "BY", "NO",
        }
    )


@dataclass
class Config:
    # ---------- paths ----------
    root: Path = ROOT
    labels_path: Path = ROOT / "dataset" / "annotations" / "labels.csv"
    images_dir: Path = ROOT / "dataset" / "images"
    outputs_dir: Path = ROOT / "outputs"

    # ---------- upload validation ----------
    allowed_extensions: tuple = (".jpg", ".jpeg", ".png")
    max_upload_mb: float = 10.0
    min_image_side: int = 120

    # ---------- preprocessing ----------
    max_image_side: int = 960
    enable_enhancement: bool = False
    enhancement_ops: tuple = ("clahe", "sharpen")

    # ---------- OCR ----------
    ocr_language: str = "en"
    ocr_min_confidence: float = 0.30
    # Lightweight CPU models: mobile detector + mobile recognizer.
    # Measured on the target machine; see README for numbers.
    # For higher accuracy at a cost in latency, switch the recognizer to
    # "PP-OCRv5_mobile_rec" or "PP-OCRv6_medium_rec" in config.py.
    ocr_det_model: str = "PP-OCRv5_mobile_det"
    ocr_rec_model: str = "en_PP-OCRv3_mobile_rec"
    ocr_rec_batch_size: int = 16
    # Detection runs on the same 960px image. Capping threads at 4 measured
    # faster and more stable than letting Paddle use all 16 cores.
    ocr_det_limit_side_len: int = 960
    ocr_det_limit_type: str = "max"
    ocr_num_threads: int = 4
    ocr_use_mkldnn: bool = False  # oneDNN crashes on this Paddle build

    # ---------- matching ----------
    match_threshold: float = 60.0
    fuzzy_scorer: str = "ratio"  # RapidFuzz fuzz.ratio (0..100)
    fuzzy_candidate_cutoff: float = 50.0
    min_query_length: int = 3
    min_fuzzy_query_length: int = 4
    top_k_per_query: int = 5
    max_candidates: int = 5
    max_queries: int = 300
    combine_adjacent: bool = True
    # Two candidates closer than this score are reported as ambiguous.
    ambiguity_margin: float = 5.0
    ignored_tokens: frozenset = field(default_factory=_ignored_tokens)

    # ---------- service ----------
    log_requests: bool = True
    service_title: str = "Medicine Name Recognition"

    def image_files(self) -> list:
        if not self.images_dir.exists():
            return []
        return sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in self.allowed_extensions
        )
