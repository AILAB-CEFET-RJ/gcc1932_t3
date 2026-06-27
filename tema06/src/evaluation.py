import difflib
from collections import defaultdict

from src.lexicon import _normalize

NER_LABELS = ["Person", "Location", "Organization"]

# Labels do pt_core_news_lg → labels do DDsmall
PRETRAINED_LABEL_MAP = {"PER": "Person", "LOC": "Location", "ORG": "Organization"}


def predict_records(nlp, records: list[dict], label_map: dict = None) -> list[dict]:
    """Aplica nlp em cada registro e devolve predições no formato {id, text, spans}."""
    preds = []
    for rec in records:
        doc = nlp(rec["text"])
        spans = []
        for ent in doc.ents:
            label = label_map.get(ent.label_, ent.label_) if label_map else ent.label_
            if label in NER_LABELS:
                spans.append({"start": ent.start_char, "end": ent.end_char, "label": label})
        preds.append({"id": rec["id"], "text": rec["text"], "spans": spans})
    return preds


def print_results(scores: dict, title: str = ""):
    if title:
        print(f"\n=== {title} ===")
    print(f"{'Label':<14} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 52)
    for label in NER_LABELS:
        if label in scores:
            s = scores[label]
            print(f"{label:<14} {s['precision']:>6.3f} {s['recall']:>6.3f} {s['f1']:>6.3f}"
                  f" {s['tp']:>5} {s['fp']:>5} {s['fn']:>5}")
    print("-" * 52)
    print(f"{'micro':<14} {scores['micro']['precision']:>6.3f}"
          f" {scores['micro']['recall']:>6.3f} {scores['micro']['f1']:>6.3f}")
    print(f"{'macro F1':<14} {scores['macro']['f1']:>6.3f}")


def _span_set(record: dict) -> set[tuple]:
    """Extract (start, end, label) tuples from a record's spans."""
    return {(s["start"], s["end"], s["label"]) for s in record.get("spans", [])}


def evaluate_ner(gold_records: list[dict], pred_records: list[dict]) -> dict:
    """
    Entity-level evaluation: correct only when span boundaries AND label match.
    Both lists must be aligned (same order and same ids).
    Returns per-label and overall precision/recall/F1.
    """
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

    gold_by_id = {r["id"]: r for r in gold_records}
    pred_by_id = {r["id"]: r for r in pred_records}

    for doc_id, gold in gold_by_id.items():
        gold_spans = _span_set(gold)
        pred_spans = _span_set(pred_by_id.get(doc_id, {}))

        for span in gold_spans & pred_spans:
            tp[span[2]] += 1
        for span in pred_spans - gold_spans:
            fp[span[2]] += 1
        for span in gold_spans - pred_spans:
            fn[span[2]] += 1

    all_labels = set(tp) | set(fp) | set(fn)
    results: dict = {}

    for label in sorted(all_labels):
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        results[label] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "tp": tp[label], "fp": fp[label], "fn": fn[label],
        }

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    macro_f1 = sum(v["f1"] for v in results.values()) / len(results) if results else 0.0

    results["micro"] = {"precision": round(micro_p, 4), "recall": round(micro_r, 4), "f1": round(micro_f1, 4)}
    results["macro"] = {"f1": round(macro_f1, 4)}

    return results


def load_and_evaluate(model_path, test_records: list[dict], label_map: dict = None) -> dict:
    """Carrega um pipeline spaCy (caminho ou nome de pacote) e avalia no conjunto de teste."""
    import spacy
    nlp = spacy.load(model_path)
    preds = predict_records(nlp, test_records, label_map)
    return evaluate_ner(test_records, preds)


def _spans_overlap(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and a["end"] > b["start"]


def categorize_errors(gold_records: list[dict], pred_records: list[dict], lexicon: dict = None) -> list[dict]:
    """
    Compara gold vs predito por documento e categoriza cada discrepância (§9.3):

    - erro_classe: spans se sobrepõem, classes diferentes
    - erro_fronteira: spans se sobrepõem, mesma classe, fronteiras diferentes
        subcategoria: composta_parcial (entidade composta reconhecida só em parte) |
                      span_maior | deslocado
    - falso_positivo: span predito sem nenhum gold sobreposto (falso positivo "puro")
    - falso_negativo: span gold sem nenhum predito sobreposto (falso negativo "puro")
        subcategoria: no_lexico_nao_reconhecida | variacao_ortografica | fora_do_lexico

    Matches exatos (start, end, label) são acertos e não aparecem na lista.
    `lexicon` (dict normalizado, como produzido por filter_lexicon) é opcional;
    quando fornecido, habilita a subcategorização de falso_negativo.
    """
    pred_by_id = {r["id"]: r for r in pred_records}
    errors = []

    for gold in gold_records:
        doc_id = gold["id"]
        text = gold["text"]
        pred = pred_by_id.get(doc_id, {"spans": []})

        gold_spans = gold.get("spans", [])
        pred_spans = pred.get("spans", [])
        gold_used, pred_used = set(), set()

        # 1) acertos exatos — removidos da análise
        for gi, g in enumerate(gold_spans):
            for pi, p in enumerate(pred_spans):
                if pi in pred_used:
                    continue
                if g["start"] == p["start"] and g["end"] == p["end"] and g["label"] == p["label"]:
                    gold_used.add(gi)
                    pred_used.add(pi)
                    break

        # 2) sobreposições não-exatas — erro_classe ou erro_fronteira
        for gi, g in enumerate(gold_spans):
            if gi in gold_used:
                continue
            for pi, p in enumerate(pred_spans):
                if pi in pred_used or not _spans_overlap(g, p):
                    continue
                if g["label"] != p["label"]:
                    category, sub = "erro_classe", None
                else:
                    if p["start"] >= g["start"] and p["end"] <= g["end"]:
                        sub = "composta_parcial"
                    elif p["start"] <= g["start"] and p["end"] >= g["end"]:
                        sub = "span_maior"
                    else:
                        sub = "deslocado"
                    category = "erro_fronteira"
                errors.append({
                    "doc_id": doc_id, "text": text, "category": category, "subcategory": sub,
                    "gold": {**g, "surface": text[g["start"]:g["end"]]},
                    "pred": {**p, "surface": text[p["start"]:p["end"]]},
                })
                gold_used.add(gi)
                pred_used.add(pi)
                break

        # 3) gold restante — falso_negativo (puro)
        for gi, g in enumerate(gold_spans):
            if gi in gold_used:
                continue
            surface = text[g["start"]:g["end"]]
            key = _normalize(surface)
            sub, closest = "fora_do_lexico", None
            if lexicon is not None:
                if key in lexicon:
                    sub = "no_lexico_nao_reconhecida"
                else:
                    matches = difflib.get_close_matches(key, lexicon.keys(), n=1, cutoff=0.75)
                    if matches:
                        sub, closest = "variacao_ortografica", matches[0]
            errors.append({
                "doc_id": doc_id, "text": text, "category": "falso_negativo", "subcategory": sub,
                "gold": {**g, "surface": surface}, "pred": None, "closest_lexicon_match": closest,
            })

        # 4) predito restante — falso_positivo (puro)
        for pi, p in enumerate(pred_spans):
            if pi in pred_used:
                continue
            errors.append({
                "doc_id": doc_id, "text": text, "category": "falso_positivo", "subcategory": None,
                "gold": None, "pred": {**p, "surface": text[p["start"]:p["end"]]},
            })

    return errors


def summarize_error_categories(errors: list[dict]) -> dict:
    """Conta erros por (categoria, subcategoria), ordenado do mais para o menos frequente."""
    counts = defaultdict(int)
    for e in errors:
        counts[(e["category"], e.get("subcategory"))] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def sample_errors(errors: list[dict], category: str, subcategory: str = None, n: int = 5) -> list[dict]:
    """Amostra até n erros de uma (categoria, subcategoria) para inspeção manual."""
    pool = [e for e in errors if e["category"] == category and e.get("subcategory") == subcategory]
    return pool[:n]
