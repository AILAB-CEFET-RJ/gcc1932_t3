import json
from pathlib import Path
from typing import Iterator


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Ajusta start/end para excluir whitespace nas bordas do span."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def load_ddsmall(split_path: Path) -> list[dict]:
    """
    Load a DDsmall split from its single JSON file.
    Each record: {"text": str, "spans": [{"start": int, "end": int, "label": str}]}
    Labels: "Person", "Location", "Organization"
    An "id" is injected as the zero-based index within the split.
    Span offsets são normalizados para excluir whitespace nas bordas.
    """
    with open(split_path, encoding="utf-8") as f:
        records = json.load(f)
    for i, rec in enumerate(records):
        rec.setdefault("id", i)
        text = rec.get("text", "")
        for s in rec.get("spans", []):
            s["start"], s["end"] = _trim_span(text, s["start"], s["end"])
    return records


def stream_ddlarge(ddlarge_path: Path) -> Iterator[dict]:
    """
    Stream DDlarge records from a JSONL file to avoid loading 184k records into memory.
    Each line: {"relato": str, "logradouroLocal": str, "bairroLocal": str,
                "cidadeLocal": str, "pontodeReferenciaLocal": str, "assunto": str,
                "_sanitization": {"row_index_1based": int, ...}}
    Yields records with a synthetic "id" and "text" field added for uniformity.
    """
    with open(ddlarge_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["id"] = rec.get("_sanitization", {}).get("row_index_1based")
            rec["text"] = rec["relato"]
            yield rec
