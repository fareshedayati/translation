#!/usr/bin/env python3
"""Merge a contextual JSONL glossary with a legacy source/target TSV glossary."""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
from typing import Any

try:
    from .build_glossary import clean_target, normalized_tokens
except ImportError:
    from build_glossary import clean_target, normalized_tokens


LEGACY_WHEN = "Legacy TSV candidate; use only when supported by context and examples."


def normalized_key(text: str) -> str:
    return " ".join(normalized_tokens(text))


def load_jsonl_entries(path: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            source = str(row.get("source", "")).strip()
            senses = row.get("senses", [])
            if not source or not isinstance(senses, list) or not senses:
                raise ValueError(f"{path}:{line_number}: source and non-empty senses are required")
            source_key = normalized_key(source)
            if not source_key:
                raise ValueError(f"{path}:{line_number}: source has no searchable text")
            entry = entries.setdefault(
                source_key,
                {
                    "source": source,
                    "senses": [],
                    "sense_keys": set(),
                    "notes": [],
                    "json_sense_count": 0,
                },
            )
            note = str(row.get("notes", "")).strip()
            if note and note not in entry["notes"]:
                entry["notes"].append(note)
            for sense in senses:
                if not isinstance(sense, dict):
                    raise ValueError(f"{path}:{line_number}: each sense must be an object")
                target = str(sense.get("target", "")).strip()
                when = str(sense.get("when", "")).strip()
                examples = sense.get("examples", [])
                if not target or not when or not isinstance(examples, list):
                    raise ValueError(
                        f"{path}:{line_number}: every sense needs target, when, and an examples list"
                    )
                target_key = normalized_key(target)
                if not target_key or target_key in entry["sense_keys"]:
                    continue
                entry["sense_keys"].add(target_key)
                entry["senses"].append(
                    {
                        "target": target,
                        "when": when,
                        "examples": [str(example).strip() for example in examples if str(example).strip()],
                    }
                )
                entry["json_sense_count"] += 1
    return entries


def load_tsv_candidates(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    groups: dict[str, dict[str, Any]] = {}
    valid_rows = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"source", "target"} <= set(reader.fieldnames or []):
            raise ValueError(f"{path}: TSV headers must include source and target")
        for line_number, row in enumerate(reader, 2):
            source = str(row.get("source", "")).strip()
            target = clean_target(str(row.get("target", "")))
            source_key = normalized_key(source)
            target_key = normalized_key(target)
            if not source_key or not target_key:
                continue
            valid_rows += 1
            group = groups.setdefault(
                source_key,
                {"source": source, "row_count": 0, "targets": {}},
            )
            group["row_count"] += 1
            candidate = group["targets"].setdefault(
                target_key,
                {"target": target, "count": 0, "first_line": line_number},
            )
            candidate["count"] += 1
    return groups, valid_rows


def merge_entries(
    json_entries: dict[str, dict[str, Any]],
    tsv_groups: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats: collections.Counter = collections.Counter()
    stats["json_terms"] = len(json_entries)
    stats["tsv_terms"] = len(tsv_groups)

    for source_key, group in tsv_groups.items():
        entry = json_entries.get(source_key)
        if entry is None:
            entry = {
                "source": group["source"],
                "senses": [],
                "sense_keys": set(),
                "notes": [],
                "json_sense_count": 0,
            }
            json_entries[source_key] = entry
            stats["legacy_only_terms"] += 1
        else:
            stats["overlapping_terms"] += 1

        added = 0
        already_covered = 0
        candidates = sorted(
            group["targets"].items(),
            key=lambda item: (-item[1]["count"], item[1]["first_line"], item[0]),
        )
        for target_key, candidate in candidates:
            if target_key in entry["sense_keys"]:
                already_covered += 1
                stats["tsv_senses_already_covered"] += 1
                continue
            entry["sense_keys"].add(target_key)
            entry["senses"].append(
                {
                    "target": candidate["target"],
                    "when": LEGACY_WHEN,
                    "examples": [],
                }
            )
            added += 1
            stats["legacy_senses_added"] += 1

        if entry["json_sense_count"]:
            summary = (
                f"Legacy TSV merge: {added} additional candidate sense(s); "
                f"{already_covered} candidate(s) already covered by evidence-backed senses. "
                "Legacy-only senses have no attached example evidence."
            )
        else:
            summary = (
                f"Legacy TSV import: {added} distinct candidate sense(s) from "
                f"{group['row_count']} valid row(s). No attached example evidence; contextual review required."
            )
        entry["notes"].append(summary)

    records: list[dict[str, Any]] = []
    for source_key, entry in sorted(json_entries.items(), key=lambda item: (item[0], item[1]["source"])):
        records.append(
            {
                "source": entry["source"],
                "senses": entry["senses"],
                "notes": " ".join(entry["notes"]),
            }
        )
    stats["merged_terms"] = len(records)
    stats["merged_senses"] = sum(len(record["senses"]) for record in records)
    return records, dict(stats)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a contextual JSONL glossary with a legacy source/target TSV glossary."
    )
    parser.add_argument("--jsonl", type=Path, required=True, help="Evidence-backed JSONL glossary")
    parser.add_argument("--tsv", type=Path, required=True, help="Legacy source/target TSV glossary")
    parser.add_argument("--output", type=Path, required=True, help="Merged JSONL destination")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_entries = load_jsonl_entries(args.jsonl)
    tsv_groups, valid_rows = load_tsv_candidates(args.tsv)
    records, stats = merge_entries(json_entries, tsv_groups)
    stats["valid_tsv_rows"] = valid_rows
    write_jsonl(args.output, records)
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    print(f"Wrote merged glossary to {args.output}")


if __name__ == "__main__":
    main()
