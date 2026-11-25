#!/usr/bin/env python3
"""Evaluate phrase-specific parallel examples on the saved Fable BM25 sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

import yaml


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def trim_failed_suffix(path: Path) -> int:
    """Back up and remove a trailing failed suffix so generation can resume."""
    if not path.exists():
        return 0
    records = rows(path)
    first_failed = next(
        (
            index
            for index, record in enumerate(records)
            if not str(record.get("prediction", "")).strip()
        ),
        len(records),
    )
    if first_failed == len(records):
        return 0
    if any(
        str(record.get("prediction", "")).strip()
        for record in records[first_failed:]
    ):
        raise ValueError(
            f"cannot safely resume {path}: failed records are not a contiguous suffix"
        )
    backup = path.with_suffix(path.suffix + ".pre_retry")
    if not backup.exists():
        shutil.copy2(path, backup)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records[:first_failed]:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)
    return len(records) - first_failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style-repo", type=Path, default=Path("../Style-Aware-MT"))
    parser.add_argument("--glossary", type=Path, default=Path("data/glossary.jsonl"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--generation-cap", type=float, default=15.0)
    parser.add_argument("--phrase-examples", type=int, default=15)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phrase_examples < 0:
        raise ValueError("--phrase-examples cannot be negative")
    project_root = Path(__file__).resolve().parents[1]
    style_root = args.style_repo.expanduser().resolve()
    glossary_path = (
        args.glossary.expanduser().resolve()
        if args.glossary.is_absolute()
        else (project_root / args.glossary).resolve()
    )
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(style_root) not in sys.path:
        sys.path.insert(0, str(style_root))

    from translator.data import load_examples, load_glossary
    from translator.glossary import GlossaryExampleSelector
    from src.infer.run import Glossary, _normalize_glossary_text, build_fewshot_user, order_exemplars
    from src.infer.tfidf_svd_sweep import generate
    from src.retrieval.bm25 import BM25Index

    config_path = style_root / "configs/commercial_fable5_sparse_knn.yaml"
    sample_path = style_root / "data/splits/test_sweep10_seed42.jsonl"
    index_path = style_root / "data/bm25_index"
    style_path = style_root / "prompts/style_instruction_shoghi_effendi.txt"
    examples_path = project_root / "data/examples.jsonl"
    out_dir = style_root / "outputs/fable_bm25_phrase_examples_sweep10"
    results_dir = style_root / "results/fable_bm25_phrase_examples_sweep10"
    condition = f"fable_bm25_k16_contextual_glossary_phrase_examples{args.phrase_examples}"

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["prompt"]["style_instruction_file"] = str(style_path)
    sample_rows = rows(sample_path)
    sources = [row["input"] for row in sample_rows]

    contextual_entries = load_glossary(glossary_path)
    contextual_by_key = {
        _normalize_glossary_text(entry.source): entry for entry in contextual_entries
    }
    phrase_example_selector = GlossaryExampleSelector(load_examples(examples_path))
    entries: dict[str, tuple[str, tuple[str, ...]]] = {}
    ignored_short = 0
    source_rows = 0
    max_phrase_tokens = 1
    with glossary_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            record = json.loads(raw)
            source = str(record.get("source", "")).strip()
            senses = record.get("senses", [])
            key = _normalize_glossary_text(source)
            if not source or not key or not isinstance(senses, list):
                raise ValueError(f"{glossary_path}:{line_number}: invalid glossary entry")
            source_rows += 1
            if len(key.replace(" ", "")) < 2:
                ignored_short += 1
                continue
            targets: list[str] = []
            seen_targets: set[str] = set()
            for sense in senses:
                target = str(sense.get("target", "")).strip() if isinstance(sense, dict) else ""
                target_key = target.casefold()
                if target and target_key not in seen_targets:
                    targets.append(target)
                    seen_targets.add(target_key)
            if targets:
                entries[key] = (source, tuple(targets))
                max_phrase_tokens = max(max_phrase_tokens, len(key.split()))

    glossary = Glossary(entries, max_phrase_tokens, source_rows, ignored_short)

    bm25 = BM25Index(index_path)
    rankings, scores = bm25.rankings(sources, 16)
    selected = bm25.selected(rankings, 16)
    ordering = cfg.get("prompt", {}).get("ordering", "most_similar_last")
    rng = random.Random(cfg.get("prompt", {}).get("ordering_seed", 42))
    ordered = [order_exemplars(items, ordering, rng) for items in selected]
    # Match the established benchmark's deterministic workaround for the one
    # row that repeatedly produced an empty safety refusal.
    if len(ordered) > 108:
        ordered[108] = selected[108]

    matches = [
        glossary.matches(source, max_pairs=1_000_000, max_equivalences=15)
        for source in sources
    ]
    phrase_examples = []
    for source, source_matches in zip(sources, matches):
        source_selections = []
        for display, _ in source_matches:
            entry = contextual_by_key.get(_normalize_glossary_text(display))
            phrase = entry.source if entry else display
            source_selections.append(
                (
                    display,
                    phrase_example_selector.select_for_phrase(
                        source, phrase, args.phrase_examples
                    ),
                )
            )
        phrase_examples.append(source_selections)

    messages = []
    marker = "Now translate the following text into English in the same style:\n"
    for source, examples, source_selections in zip(sources, ordered, phrase_examples):
        message = build_fewshot_user(
            source,
            examples,
            glossary,
            glossary_scope="query",
            max_glossary_terms=1_000_000,
            max_glossary_equivalences=15,
        )
        evidence = ["Phrase-specific parallel examples ranked for this passage:"]
        for display, selected_examples in source_selections:
            if not selected_examples:
                continue
            evidence.append(f"Matched phrase: {display}")
            for number, example in enumerate(selected_examples, 1):
                evidence.append(
                    f"  {number}. Source: {example.source}\n"
                    f"     English: {example.target}"
                )
        if len(evidence) > 1:
            message = message.replace(marker, "\n".join(evidence) + "\n\n" + marker, 1)
        messages.append(message)

    system_chars = len(style_path.read_text(encoding="utf-8"))
    prompt_chars = sum(system_chars + len(message) for message in messages)
    estimated_input_tokens = round(prompt_chars / 4)
    estimated_output_tokens = round(sum(len(row["output"]) for row in sample_rows) / 4)
    input_rate, output_rate = cfg["generator"].get("pricing", [10.0, 50.0])
    estimated_cost = (
        estimated_input_tokens * input_rate + estimated_output_tokens * output_rate
    ) / 1_000_000
    manifest = {
        "condition": condition,
        "model": cfg["generator"]["model"],
        "sample": str(sample_path),
        "sample_sha256": sha256(sample_path),
        "n": len(sample_rows),
        "bm25": {**bm25.meta, "k": 16, "ordering": ordering},
        "glossary": {
            "file": str(glossary_path),
            "sha256": sha256(glossary_path),
            "loaded_terms": len(glossary.entries),
            "source_rows": source_rows,
            "ignored_short_terms": ignored_short,
            "max_equivalences": 15,
            "matched_terms": sum(map(len, matches)),
            "rows_with_matches": sum(bool(items) for items in matches),
            "phrase_examples_per_match": args.phrase_examples,
            "parallel_examples_file": str(examples_path),
            "parallel_examples_sha256": sha256(examples_path),
            "selected_phrase_examples": sum(
                len(examples)
                for selections in phrase_examples
                for _, examples in selections
            ),
            "selection_method": "source-side CPU BM25 among examples containing the complete phrase",
        },
        "style_instruction": str(style_path),
        "prompt_characters": prompt_chars,
        "estimated_input_tokens_char_div_4": estimated_input_tokens,
        "estimated_output_tokens_reference_div_4": estimated_output_tokens,
        "estimated_generation_cost_usd": round(estimated_cost, 4),
        "generation_cap_usd": args.generation_cap,
        "safety_refusal_workaround_row_zero_based": 108,
    }
    write_json(results_dir / "manifest.json", manifest)
    write_json(
        results_dir / "retrieval.json",
        {
            "queries": [
                {
                    "row": index,
                    "bm25_rows": rankings[index].tolist(),
                    "bm25_scores": scores[index].tolist(),
                    "glossary_matches": [
                        {
                            "source": display,
                            "equivalences": list(targets),
                            "parallel_example_ids": [
                                example.id for example in selected_examples
                            ],
                        }
                        for (display, targets), (_, selected_examples) in zip(
                            matches[index], phrase_examples[index]
                        )
                    ],
                }
                for index in range(len(sample_rows))
            ]
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.prepare_only:
        return

    output_path = out_dir / f"{condition}_{sample_path.stem}.jsonl"
    retry_count = trim_failed_suffix(output_path)
    if retry_count:
        print(f"[{condition}] retrying {retry_count} trailing failed rows")

    generate(
        cfg,
        sample_rows,
        messages,
        k=16,
        split=sample_path.stem,
        out_dir=out_dir,
        workers=args.workers,
        cost_cap=args.generation_cap,
        condition=condition,
        retrieval=f"okapi_bm25_k16_contextual_glossary_phrase_examples{args.phrase_examples}",
    )


if __name__ == "__main__":
    main()
