from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from .data import Example


TOKEN = re.compile(r"[^\W_]+(?:\u200c[^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    return TOKEN.findall(text)


@dataclass
class Retrieved:
    example: Example
    methods: list[str]
    scores: dict[str, float]


class BM25Index:
    def __init__(self, examples: list[Example], k1: float = 1.5, b: float = 0.75):
        self.examples, self.k1, self.b = examples, k1, b
        self.documents = [Counter(tokenize(item.source)) for item in examples]
        self.lengths = [sum(doc.values()) for doc in self.documents]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        document_frequency: Counter[str] = Counter()
        for doc in self.documents:
            document_frequency.update(doc.keys())
        n = len(examples)
        self.idf = {term: math.log1p((n - freq + 0.5) / (freq + 0.5)) for term, freq in document_frequency.items()}

    def search(self, query: str, k: int) -> list[tuple[Example, float]]:
        if k <= 0:
            return []
        query_terms = Counter(tokenize(query))
        scored: list[tuple[int, float]] = []
        for index, doc in enumerate(self.documents):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / max(self.average_length, 1e-9))
            for term, query_count in query_terms.items():
                frequency = doc.get(term, 0)
                if frequency:
                    score += query_count * self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / (frequency + norm)
            if score > 0:
                scored.append((index, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [(self.examples[index], score) for index, score in scored[:k]]


class SemanticIndex:
    def __init__(self, examples: list[Example], model_name: str, device: str):
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Semantic retrieval requires: pip install -r requirements-semantic.txt") from exc
        self.np = np
        self.examples = examples
        self.model = SentenceTransformer(model_name, device=device)
        self.embeddings = np.asarray(self.model.encode([f"passage: {item.source}" for item in examples], normalize_embeddings=True, show_progress_bar=True))

    def search(self, query: str, k: int) -> list[tuple[Example, float]]:
        if k <= 0:
            return []
        vector = self.np.asarray(self.model.encode([f"query: {query}"], normalize_embeddings=True))[0]
        scores = self.embeddings @ vector
        order = self.np.argsort(-scores, kind="stable")[: min(k, len(self.examples))]
        return [(self.examples[int(index)], float(scores[index])) for index in order]


class Retriever:
    def __init__(self, examples: list[Example], config: dict):
        self.examples, self.config = examples, config
        method = config["method"]
        self.bm25 = BM25Index(examples, float(config.get("bm25_k1", 1.5)), float(config.get("bm25_b", 0.75))) if method in {"bm25", "hybrid"} else None
        self.semantic = SemanticIndex(examples, config["semantic_model"], config.get("semantic_device", "cuda")) if method in {"semantic", "hybrid"} else None

    def retrieve(self, query: str) -> list[Retrieved]:
        selected: dict[str, Retrieved] = {}
        order: list[str] = []
        searches = []
        if self.bm25:
            searches.append(("bm25", self.bm25.search(query, int(self.config.get("bm25_samples", 0)))))
        if self.semantic:
            searches.append(("semantic", self.semantic.search(query, int(self.config.get("semantic_samples", 0)))))
        for method, results in searches:
            for example, score in results:
                if example.id not in selected:
                    selected[example.id] = Retrieved(example, [], {})
                    order.append(example.id)
                selected[example.id].methods.append(method)
                selected[example.id].scores[method] = score
        return [selected[item_id] for item_id in order]
