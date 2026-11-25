from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .data import load_examples, load_glossary, load_inputs, match_glossary
from .glossary import GlossaryExampleSelector
from .prompting import system_prompt, user_prompt
from .providers import make_provider
from .retrieval import Retriever


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Generic retrieval-assisted SOURCE-to-TARGET translation")
    result.add_argument("--config", default="config.example.yaml", help="YAML configuration file")
    result.add_argument("--source-language")
    result.add_argument("--target-language")
    result.add_argument("--input")
    result.add_argument("--examples")
    result.add_argument("--glossary")
    result.add_argument("--output")
    result.add_argument("--provider", choices=["openai", "anthropic"])
    result.add_argument("--model")
    result.add_argument("--retrieval", choices=["bm25", "semantic", "hybrid", "none"])
    result.add_argument("--bm25-samples", type=int)
    result.add_argument("--semantic-samples", type=int)
    result.add_argument("--semantic-device", help="Usually cuda; cpu is supported but slower")
    result.add_argument("--example-ordering", choices=["most_similar_first", "most_similar_last"])
    result.add_argument("--max-glossary-senses", type=int)
    result.add_argument(
        "--glossary-examples-per-phrase",
        type=int,
        help="Relevant parallel examples supplied for each matched glossary phrase (default: 0/off)",
    )
    result.add_argument("--paragraphs-per-request", type=int, help="Consecutive paragraphs per API request (default: 1)")
    result.add_argument("--workers", type=int, help="Concurrent API requests (default: 4)")
    result.add_argument("--max-output-tokens", type=int)
    result.add_argument("--reasoning-effort")
    result.add_argument("--dry-run", action="store_true", help="Build retrieval traces and prompts without calling an API")
    return result


def _path(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def _required(config: dict, section: str, key: str) -> Any:
    try:
        return config[section][key]
    except KeyError as exc:
        raise ValueError(f"missing configuration value: {section}.{key}") from exc


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent
    load_dotenv(base / ".env")

    languages = config.setdefault("languages", {})
    files = config.setdefault("files", {})
    provider_cfg = config.setdefault("provider", {})
    retrieval_cfg = config.setdefault("retrieval", {})
    processing = config.setdefault("processing", {})
    generation = config.setdefault("generation", {})
    overrides = [
        (languages, "source", args.source_language),
        (languages, "target", args.target_language),
        (files, "input", args.input),
        (files, "examples", args.examples),
        (files, "glossary", args.glossary),
        (files, "output", args.output),
        (provider_cfg, "name", args.provider),
        (provider_cfg, "model", args.model),
        (retrieval_cfg, "method", args.retrieval),
        (retrieval_cfg, "bm25_samples", args.bm25_samples),
        (retrieval_cfg, "semantic_samples", args.semantic_samples),
        (retrieval_cfg, "semantic_device", args.semantic_device),
        (retrieval_cfg, "ordering", args.example_ordering),
        (retrieval_cfg, "max_glossary_senses", args.max_glossary_senses),
        (
            retrieval_cfg,
            "glossary_examples_per_phrase",
            args.glossary_examples_per_phrase,
        ),
        (processing, "paragraphs_per_request", args.paragraphs_per_request),
        (processing, "workers", args.workers),
        (generation, "max_output_tokens", args.max_output_tokens),
        (generation, "reasoning_effort", args.reasoning_effort),
    ]
    for section, key, value in overrides:
        if value is not None:
            section[key] = value

    method = retrieval_cfg.get("method", "bm25")
    if method not in {"none", "bm25", "semantic", "hybrid"}:
        raise ValueError("retrieval.method must be none, bm25, semantic, or hybrid")
    bm25_samples, semantic_samples = int(retrieval_cfg.get("bm25_samples", 16)), int(retrieval_cfg.get("semantic_samples", 0))
    if bm25_samples < 0 or semantic_samples < 0:
        raise ValueError("retrieval sample counts cannot be negative")
    if method == "bm25" and bm25_samples < 1:
        raise ValueError("bm25 retrieval requires bm25_samples >= 1")
    if method == "semantic" and semantic_samples < 1:
        raise ValueError("semantic retrieval requires semantic_samples >= 1")
    if method == "hybrid" and (bm25_samples < 1 or semantic_samples < 1):
        raise ValueError("hybrid retrieval requires both sample counts >= 1")
    ordering = retrieval_cfg.get("ordering", "most_similar_last")
    if ordering not in {"most_similar_first", "most_similar_last"}:
        raise ValueError("retrieval.ordering must be most_similar_first or most_similar_last")
    max_glossary_senses = int(retrieval_cfg.get("max_glossary_senses", 15))
    if max_glossary_senses < 1:
        raise ValueError("retrieval.max_glossary_senses must be at least 1")
    glossary_examples_per_phrase = int(
        retrieval_cfg.get("glossary_examples_per_phrase", 0)
    )
    if glossary_examples_per_phrase < 0:
        raise ValueError("retrieval.glossary_examples_per_phrase cannot be negative")
    paragraphs_per_request = int(processing.get("paragraphs_per_request", 1))
    workers = int(processing.get("workers", 4))
    if paragraphs_per_request < 1:
        raise ValueError("processing.paragraphs_per_request must be at least 1")
    if workers < 1:
        raise ValueError("processing.workers must be at least 1")
    max_output_tokens = int(generation.get("max_output_tokens", 1024))
    if max_output_tokens < 1:
        raise ValueError("generation.max_output_tokens must be at least 1")

    input_path = _path(base, args.input or _required(config, "files", "input"))
    examples_path = _path(base, args.examples or files.get("examples"))
    glossary_path = _path(base, args.glossary or files.get("glossary"))
    style_path = _path(base, files.get("style_instruction"))
    output_path = _path(base, args.output or _required(config, "files", "output"))
    needs_phrase_examples = bool(glossary_path) and glossary_examples_per_phrase > 0
    if examples_path is None and (method != "none" or needs_phrase_examples):
        raise ValueError(
            "files.examples is required for example retrieval or phrase-specific glossary examples"
        )
    examples = load_examples(examples_path) if examples_path else []
    glossary = load_glossary(glossary_path)
    records = load_inputs(input_path)
    if not records:
        raise ValueError(f"{input_path}: no input passages found")
    retriever = Retriever(examples, retrieval_cfg) if method != "none" else None
    glossary_example_selector = GlossaryExampleSelector(examples)
    source_language, target_language = _required(config, "languages", "source"), _required(config, "languages", "target")
    style = style_path.read_text(encoding="utf-8") if style_path else ""
    system = system_prompt(source_language, target_language, style)

    api = None
    if not args.dry_run:
        key_name = provider_cfg.get("api_key_env", "OPENAI_API_KEY" if provider_cfg.get("name") == "openai" else "ANTHROPIC_API_KEY")
        api = make_provider(provider_cfg.get("name", "anthropic"), os.getenv(key_name, ""), _required(config, "provider", "model"), max_output_tokens, str(generation.get("reasoning_effort", "low")))

    batches = [records[start : start + paragraphs_per_request] for start in range(0, len(records), paragraphs_per_request)]
    prepared: list[dict[str, Any]] = []
    for batch_number, batch in enumerate(batches, 1):
        if len(batch) == 1:
            source = batch[0].source
            result_id = batch[0].id
        else:
            source = "\n\n".join(
                f"[PARAGRAPH {number} | ID {record.id}]\n{record.source}"
                for number, record in enumerate(batch, 1)
            )
            result_id = f"batch-{batch_number:06d}"
        retrieved = retriever.retrieve(source) if retriever else []
        if ordering == "most_similar_last":
            retrieved.reverse()
        matched = match_glossary(source, glossary)
        phrase_examples = glossary_example_selector.select(
            source,
            matched,
            glossary_examples_per_phrase,
        )
        examples_by_glossary_entry = {
            item.entry: item.examples for item in phrase_examples
        }
        prompt = user_prompt(
            source,
            retrieved,
            matched,
            source_language,
            target_language,
            paragraph_count=len(batch),
            max_glossary_senses=max_glossary_senses,
            phrase_examples=phrase_examples,
        )
        prepared.append(
            {
                "id": result_id,
                "source": source,
                "source_ids": [record.id for record in batch],
                "source_paragraphs": [record.source for record in batch],
                "translation": None,
                "source_language": source_language,
                "target_language": target_language,
                "retrieval_method": method,
                "retrieved_examples": [{"id": item.example.id, "source_file": str(examples_path) if examples_path else None, "source_line": item.example.line, "methods": item.methods, "scores": item.scores} for item in retrieved],
                "glossary_matches": [
                    {
                        "source": entry.source,
                        "parallel_examples": [
                            {
                                "id": example.id,
                                "source_file": str(examples_path) if examples_path else None,
                                "source_line": example.line,
                            }
                            for example in examples_by_glossary_entry.get(entry, ())
                        ],
                        "glossary_file": str(glossary_path),
                        "source_line": entry.line,
                    }
                    for entry in matched
                ],
                "metadata": [record.metadata for record in batch],
                "paragraph_count": len(batch),
                "_prompt": prompt,
            }
        )

    if args.dry_run:
        translations: list[str | None] = [None] * len(prepared)
    elif workers == 1:
        translations = [api.translate(system, item["_prompt"]) for item in prepared]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            translations = list(executor.map(lambda item: api.translate(system, item["_prompt"]), prepared))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for result, translation in zip(prepared, translations):
            result["translation"] = translation
            prompt = result.pop("_prompt")
            if args.dry_run:
                result["system_prompt"], result["user_prompt"] = system, prompt
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"{result['id']}: {'prepared' if args.dry_run else 'translated'}")
    print(f"Wrote {len(prepared)} request record(s) covering {len(records)} paragraph(s) to {output_path}")
