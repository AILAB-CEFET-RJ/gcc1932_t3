import re
import unicodedata
from collections import Counter


def _normalize(text: str) -> str:
    """Lowercase + remove diacritics for key comparison."""
    lowered = text.lower()
    nfkd = unicodedata.normalize("NFKD", lowered)
    return re.sub(r"\s+", " ", nfkd.encode("ascii", "ignore").decode("ascii"))


# Labels as they appear in the DDsmall corpus
VALID_LABELS = {"Person", "Location", "Organization"}

# Single generic words that should not become lexicon entries on their own
GENERIC_TERMS = {
    "rua", "morro", "comunidade", "elementos", "bandidos",
    "trafico", "area", "local", "lugar", "homem", "mulher",
    "beco", "travessa", "avenida", "estrada", "rodovia",
}

# Domain acronyms that are meaningful despite short length
KNOWN_ACRONYMS = {"pm", "cv", "tcp", "adf", "bope", 
                  "pmerj", "upa", "upp", "rj", "dp", 
                  "br", "pf" }

MIN_CHAR_LENGTH = 3
MIN_FREQUENCY = 2
DOMINANCE_THRESHOLD = 0.80


def extract_lexicon(train_records: list[dict]) -> dict[str, dict]:
    """
    Walk annotated training records and build a raw entity index.
    Key: normalized surface form (lowercase, no diacritics, collapsed whitespace).
    Value: {"text", "surface_forms", "class_counts", "freq", "doc_ids"}

    `surface_forms` collects every distinct spelling seen for the key (e.g. both
    "polícia" and "policia"), so projection can match all of them later.
    """
    index: dict[str, dict] = {}

    for rec in train_records:
        text = rec["text"]
        for span in rec.get("spans", []):
            surface = text[span["start"]:span["end"]].strip()
            label = span["label"]
            key = _normalize(surface)

            if key not in index:
                index[key] = {
                    "text": surface,
                    "surface_forms": set(),
                    "class_counts": Counter(),
                    "freq": 0,
                    "doc_ids": set(),
                }

            index[key]["surface_forms"].add(surface)
            index[key]["class_counts"][label] += 1
            index[key]["freq"] += 1
            index[key]["doc_ids"].add(rec["id"])

    return index


def filter_lexicon(index: dict) -> dict[str, dict]:
    """
    Apply §6.2 normalization and filtering rules.
    Returns a clean lexicon: {normalized_key -> {text, label, freq, doc_count}}
    """
    lexicon = {}

    for key, info in index.items():
        # Length filter allow short known acronyms
        if len(key) < MIN_CHAR_LENGTH and key not in KNOWN_ACRONYMS:
            continue

        # Generic single-word filter
        if key in GENERIC_TERMS:
            continue

        # Frequency filter: too few occurrences is not enough evidence of dominance
        if info["freq"] <= MIN_FREQUENCY:
            continue

        # Dominance filter: discard ambiguous entries
        total = sum(info["class_counts"].values())
        dominant_label, dominant_count = info["class_counts"].most_common(1)[0]
        if dominant_count / total < DOMINANCE_THRESHOLD:
            continue

        if dominant_label not in VALID_LABELS:
            continue

        lexicon[key] = {
            "text": info["text"],
            "label": dominant_label,
            "freq": info["freq"],
            "doc_count": len(info["doc_ids"]),
            "surface_forms": sorted(info["surface_forms"]),
        }

    return lexicon
