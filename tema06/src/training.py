import random
from pathlib import Path

import numpy as np
import spacy
from spacy.training import Example

from src.evaluation import NER_LABELS


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def records_to_examples(records: list[dict], nlp) -> list[Example]:
    """Converte registros DDsmall para lista de spaCy Examples.
    Spans que não alinham a fronteiras de tokens são descartados antes da criação
    do Example, evitando o warning W030."""
    examples = []
    for rec in records:
        doc = nlp.make_doc(rec["text"])
        raw = [(s["start"], s["end"], s["label"]) for s in rec.get("spans", [])]
        # doc.char_span retorna None quando o offset cai no meio de um token
        entities = [
            (start, end, label)
            for start, end, label in raw
            if doc.char_span(start, end) is not None
        ]
        try:
            ex = Example.from_dict(doc, {"entities": entities})
            examples.append(ex)
        except Exception:
            pass
    return examples


def train_ner(
    train_examples: list[Example],
    dev_examples: list[Example],
    output_path: Path,
    n_epochs: int = 30,
    batch_size: int = 32,
    dropout: float = 0.3,
    patience: int = 5,
    seed: int = 42,
) -> float:
    """Treina NER com spacy.blank('pt') e salva o melhor checkpoint por dev F1.
    Retorna o melhor F1 micro obtido na validação."""
    _seed_everything(seed)
    nlp = spacy.blank("pt")
    ner = nlp.add_pipe("ner")
    for label in NER_LABELS:
        ner.add_label(label)

    nlp.initialize(lambda: train_examples)

    best_f1, no_improve = 0.0, 0
    for epoch in range(n_epochs):
        random.shuffle(train_examples)
        losses = {}
        for batch in spacy.util.minibatch(train_examples, size=batch_size):
            nlp.update(batch, losses=losses, drop=dropout)

        scores = nlp.evaluate(dev_examples)
        f1 = scores["ents_f"]
        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            nlp.to_disk(output_path)
            no_improve = 0
            marker = " ✓"
        else:
            no_improve += 1

        print(f"  Epoch {epoch+1:3d} | loss={losses['ner']:8.1f} | dev F1={f1:.4f}{marker}")

        if no_improve >= patience:
            print(f"  Early stopping na época {epoch + 1}")
            break

    print(f"  Melhor dev F1: {best_f1:.4f}  →  modelo salvo em {output_path}")
    return best_f1
