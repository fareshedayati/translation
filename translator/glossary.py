from __future__ import annotations

from dataclasses import dataclass

from .data import Example, GlossaryEntry, contains_glossary_term
from .retrieval import BM25Index, tokenize


@dataclass(frozen=True)
class PhraseExamples:
    entry: GlossaryEntry
    examples: tuple[Example, ...]


class GlossaryExampleSelector:
    """Select source-context-relevant parallel examples for matched phrases."""

    def __init__(self, examples: list[Example]):
        self.examples = examples
        self._candidates: dict[str, tuple[Example, ...]] = {}

    def candidates(self, phrase: str) -> tuple[Example, ...]:
        if phrase not in self._candidates:
            self._candidates[phrase] = tuple(
                example
                for example in self.examples
                if contains_glossary_term(example.source, phrase)
            )
        return self._candidates[phrase]

    def select_for_phrase(
        self, source_text: str, phrase: str, limit: int
    ) -> tuple[Example, ...]:
        candidates = self.candidates(phrase)
        if limit <= 0 or not candidates:
            return ()

        phrase_tokens = set(tokenize(phrase))
        query = " ".join(
            token for token in tokenize(source_text) if token not in phrase_tokens
        )
        if not query:
            return candidates[:limit]

        ranking_documents = [
            Example(
                str(index),
                " ".join(
                    token
                    for token in tokenize(example.source)
                    if token not in phrase_tokens
                ),
                "",
            )
            for index, example in enumerate(candidates)
        ]
        ranked = BM25Index(ranking_documents).search(query, len(ranking_documents))
        ranked_indexes = [int(example.id) for example, _ in ranked]
        ranked_set = set(ranked_indexes)
        # If context words do not distinguish every candidate, retain corpus order
        # as a deterministic fallback for the remaining slots.
        ranked_indexes.extend(
            index for index in range(len(candidates)) if index not in ranked_set
        )
        return tuple(candidates[index] for index in ranked_indexes[:limit])

    def select(
        self,
        source_text: str,
        entries: list[GlossaryEntry],
        limit: int,
    ) -> list[PhraseExamples]:
        return [
            PhraseExamples(
                entry,
                self.select_for_phrase(source_text, entry.source, limit),
            )
            for entry in entries
        ]
