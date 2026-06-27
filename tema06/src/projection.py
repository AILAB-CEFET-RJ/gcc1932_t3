from typing import NamedTuple

import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher


class CompiledLexicon(NamedTuple):
    matcher: PhraseMatcher
    nlp: Language


def compile_lexicon(lexicon: dict[str, dict], nlp: Language | None = None) -> CompiledLexicon:
    """
    Build a spaCy PhraseMatcher from the filtered lexicon.
    Call this ONCE before the projection loop.

    Uses attr="LOWER" for case-insensitive matching (still accent-sensitive).
    To cover accent variants (e.g. "polícia" vs "policia"), every spelling in
    `surface_forms` becomes its own pattern under the same label.
    Token-based matching gives word boundaries for free.
    If nlp is None, a blank Portuguese tokenizer is created automatically.
    """
    if nlp is None:
        nlp = spacy.blank("pt")

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    by_label: dict[str, list] = {"Person": [], "Location": [], "Organization": []}
    for info in lexicon.values():
        variants = info.get("surface_forms") or [info["text"]]
        for variant in variants:
            by_label[info["label"]].append(nlp.make_doc(variant))

    for label, patterns in by_label.items():
        if patterns:
            matcher.add(label, patterns)

    return CompiledLexicon(matcher=matcher, nlp=nlp)


def project_lexicon(text: str, compiled: CompiledLexicon) -> list[dict]:
    """
    Find all non-overlapping entity mentions using the PhraseMatcher.
    Longer spans win over shorter ones when overlapping (longest-match-first).
    Returns spans sorted by start character position.
    """
    matcher, nlp = compiled
    doc = nlp.make_doc(text)
    matches = matcher(doc)

    # Convert to character-offset dicts
    all_spans = []
    for match_id, start_tok, end_tok in matches:
        span = doc[start_tok:end_tok]
        all_spans.append({
            "start": span.start_char,
            "end": span.end_char,
            "text": span.text,
            "label": nlp.vocab.strings[match_id],
            "source": "lexical_projection",
        })

    # Resolve overlaps: longer span wins; tie-break by earlier position
    all_spans.sort(key=lambda x: (x["end"] - x["start"]), reverse=True)
    occupied: list[tuple[int, int]] = []
    selected = []
    for span in all_spans:
        s, e = span["start"], span["end"]
        if any(s < oe and e > os for os, oe in occupied):
            continue
        occupied.append((s, e))
        selected.append(span)

    selected.sort(key=lambda x: x["start"])
    return selected


def annotate_record(record: dict, compiled: CompiledLexicon) -> dict | None:
    """
    Return a pseudo-annotated record dict, or None if no spans were found.
    Uses record["text"] (set by stream_ddlarge).
    """
    text = record.get("text", "")
    spans = project_lexicon(text, compiled)
    if not spans:
        return None
    return {
        "id": record.get("id"),
        "text": text,
        "spans": spans,
        "source": "pseudo_label",
    }
