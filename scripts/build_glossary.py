#!/usr/bin/env python3
"""Build an evidence-backed contextual JSONL glossary from parallel examples."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import unicodedata
from pathlib import Path


TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
CHARACTER_VARIANTS = str.maketrans("كيى", "کیی")
CONTEXT_STOPWORDS = {
    "از",
    "است",
    "این",
    "آن",
    "با",
    "بر",
    "به",
    "در",
    "این",
    "برای",
    "تا",
    "هر",
    "هم",
    "همچنین",
    "را",
    "که",
    "و",
    "یا",
    "ان",
    "الی",
    "اینکه",
    "عن",
    "علی",
    "فی",
    "کل",
    "لا",
    "ما",
    "من",
    "هذا",
    "هو",
    "the",
    "a",
    "an",
    "and",
    "in",
    "of",
    "to",
}


def normalized_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).translate(CHARACTER_VARIANTS)
    text = "".join(character for character in text if unicodedata.category(character) != "Mn" and character != "ـ")
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def clean_target(text: str) -> str:
    value = text.replace("\x92", "’").strip()
    while value.startswith('"""'):
        value = value[3:].lstrip()
    value = value.strip('"').strip()
    return re.sub(r"\.{2,}$", ".", value)


def shorten_example(text: str, limit: int = 500) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a JSONL glossary by retaining only candidate source/target pairs "
            "attested together in the supplied parallel examples."
        )
    )
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--candidate-glossary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-source-words", type=int, default=8)
    parser.add_argument("--examples-per-sense", type=int, default=3)
    parser.add_argument("--single-word-min-support", type=int, default=2)
    parser.add_argument("--multiword-min-support", type=int, default=1)
    parser.add_argument(
        "--min-lift",
        type=float,
        default=3.0,
        help="Minimum observed/expected source-target document co-occurrence ratio.",
    )
    parser.add_argument(
        "--max-relative-position-delta",
        type=float,
        default=0.45,
        help="Maximum normalized source/target phrase-position difference.",
    )
    return parser.parse_args()


def load_candidates(path: Path, max_source_words: int) -> dict[str, dict]:
    candidates: dict[str, dict] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"source", "target"} <= set(reader.fieldnames or []):
            raise ValueError("candidate glossary must have source and target columns")
        for row in reader:
            source = str(row.get("source", "")).strip()
            target = clean_target(str(row.get("target", "")))
            source_tokens = normalized_tokens(source)
            target_tokens = normalized_tokens(target)
            if (
                not source_tokens
                or not target_tokens
                or (len(target_tokens) == 1 and len(target_tokens[0]) < 2)
                or len(source_tokens) > max_source_words
                or len(target_tokens) > 20
                or len(target) > 240
                or target.count("(") != target.count(")")
                or target.count("[") != target.count("]")
            ):
                continue
            source_key = " ".join(source_tokens)
            target_key = " ".join(target_tokens)
            item = candidates.setdefault(source_key, {"source": source, "targets": {}})
            item["targets"].setdefault(target_key, target)
    return candidates


def surrounding_words(tokens: list[str], start: int, width: int, term_tokens: set[str]) -> list[str]:
    nearby = tokens[max(0, start - 6) : start] + tokens[start + width : start + width + 6]
    return [
        token
        for token in nearby
        if token not in CONTEXT_STOPWORDS and token not in term_tokens and len(token) > 1
    ]


def load_examples(path: Path) -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            source = str(row.get("source", row.get("input", ""))).strip()
            target = str(row.get("target", row.get("output", ""))).strip()
            if not source or not target:
                raise ValueError(f"{path}:{line_number}: example requires source/input and target/output")
            examples.append((source, target))
    return examples


def collect_evidence(
    examples: list[tuple[str, str]],
    candidates: dict[str, dict],
    max_source_words: int,
    examples_per_sense: int,
    max_relative_position_delta: float,
) -> tuple[dict[tuple[str, str], dict], collections.Counter, collections.Counter]:
    evidence: dict[tuple[str, str], dict] = {}
    source_documents: collections.Counter = collections.Counter()
    target_documents: collections.Counter = collections.Counter()
    target_keys = {
        target_key
        for candidate in candidates.values()
        for target_key in candidate["targets"]
    }
    max_target_words = max((len(key.split()) for key in target_keys), default=1)
    for source, target in examples:
        source_tokens = normalized_tokens(source)
        target_tokens = normalized_tokens(target)
        found_targets: dict[str, list[int]] = {}
        for start in range(len(target_tokens)):
            for width in range(1, min(max_target_words, len(target_tokens) - start) + 1):
                target_key = " ".join(target_tokens[start : start + width])
                if target_key in target_keys:
                    found_targets.setdefault(target_key, []).append(start)
        target_documents.update(found_targets.keys())
        found_positions: dict[str, list[int]] = {}
        for start in range(len(source_tokens)):
            for width in range(1, min(max_source_words, len(source_tokens) - start) + 1):
                source_key = " ".join(source_tokens[start : start + width])
                if source_key in candidates:
                    found_positions.setdefault(source_key, []).append(start)
        source_documents.update(found_positions.keys())
        for source_key, source_starts in found_positions.items():
            candidate = candidates[source_key]
            width = len(source_key.split())
            for target_key in candidate["targets"]:
                if target_key not in found_targets:
                    continue
                target_width = len(target_key.split())
                source_denominator = max(1, len(source_tokens) - width)
                target_denominator = max(1, len(target_tokens) - target_width)
                delta, start = min(
                    (
                        abs(
                            source_start / source_denominator
                            - target_start / target_denominator
                        ),
                        source_start,
                    )
                    for source_start in source_starts
                    for target_start in found_targets[target_key]
                )
                if delta > max_relative_position_delta:
                    continue
                context = surrounding_words(source_tokens, start, width, set(source_key.split()))
                item = evidence.setdefault(
                    (source_key, target_key),
                    {"support": 0, "examples": [], "context": collections.Counter()},
                )
                item["support"] += 1
                item["context"].update(context)
                short = shorten_example(source)
                if short not in item["examples"] and len(item["examples"]) < examples_per_sense:
                    item["examples"].append(short)
    return evidence, source_documents, target_documents


def when_text(context: collections.Counter, multiple_senses: bool) -> str:
    cues = [word for word, _ in context.most_common(4)]
    if cues:
        joined = "»، «".join(cues)
        base = f"Use in source contexts resembling the cited examples, especially near «{joined}»."
    else:
        base = "Use where the source term has the meaning and grammatical role shown in the cited examples."
    if multiple_senses:
        base += " Compare the cited examples with the other listed senses before choosing."
    return base


def build_records(
    candidates: dict[str, dict],
    evidence: dict[tuple[str, str], dict],
    source_documents: collections.Counter,
    target_documents: collections.Counter,
    example_count: int,
    single_word_min_support: int,
    multiword_min_support: int,
    min_lift: float,
) -> list[dict]:
    records: list[dict] = []
    for source_key, candidate in candidates.items():
        source_word_count = len(source_key.split())
        senses_with_evidence = []
        for target_key, target in candidate["targets"].items():
            item = evidence.get((source_key, target_key))
            if not item:
                continue
            minimum_support = single_word_min_support if source_word_count == 1 else multiword_min_support
            if source_word_count == 1 and len(source_key) < 3:
                continue
            if item["support"] < minimum_support:
                continue
            expected_denominator = source_documents[source_key] * target_documents[target_key]
            lift = (item["support"] * example_count / expected_denominator) if expected_denominator else 0.0
            if lift < min_lift:
                continue
            senses_with_evidence.append((target.casefold(), target, item, lift))
        if not senses_with_evidence:
            continue
        senses_with_evidence.sort(key=lambda value: (-value[2]["support"], -value[3], value[0]))
        multiple = len(senses_with_evidence) > 1
        senses = [
            {
                "target": target,
                "when": when_text(item["context"], multiple),
                "examples": item["examples"],
            }
            for _, target, item, _ in senses_with_evidence
        ]
        support = sum(item["support"] for _, _, item, _ in senses_with_evidence)
        records.append(
            {
                "source": candidate["source"],
                "senses": senses,
                "notes": (
                    f"Evidence-backed draft: {support} aligned source-target occurrence(s) "
                    "found in examples.jsonl. Review context before formal use."
                ),
            }
        )
    records.sort(key=lambda row: (" ".join(normalized_tokens(row["source"])), row["source"]))
    return records


def main() -> None:
    args = parse_args()
    if (
        args.max_source_words < 1
        or args.examples_per_sense < 1
        or args.single_word_min_support < 1
        or args.multiword_min_support < 1
        or args.min_lift <= 0
        or not 0 <= args.max_relative_position_delta <= 1
    ):
        raise ValueError("numeric limits must be positive")
    candidates = load_candidates(args.candidate_glossary, args.max_source_words)
    examples = load_examples(args.examples)
    evidence, source_documents, target_documents = collect_evidence(
        examples,
        candidates,
        args.max_source_words,
        args.examples_per_sense,
        args.max_relative_position_delta,
    )
    records = build_records(
        candidates,
        evidence,
        source_documents,
        target_documents,
        len(examples),
        args.single_word_min_support,
        args.multiword_min_support,
        args.min_lift,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    sense_count = sum(len(record["senses"]) for record in records)
    print(
        f"Wrote {len(records)} terms and {sense_count} evidence-backed senses "
        f"from {len(examples)} examples to {args.output}"
    )


if __name__ == "__main__":
    main()
