from __future__ import annotations

import json
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Example:
    id: str
    source: str
    target: str
    line: int = 0


@dataclass(frozen=True)
class GlossarySense:
    target: str
    when: str
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class GlossaryEntry:
    source: str
    senses: tuple[GlossarySense, ...]
    notes: str = ""
    line: int = 0


@dataclass(frozen=True)
class InputRecord:
    id: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: each JSONL row must be an object")
            rows.append((line_no, value))
    return rows


def load_examples(path: Path) -> list[Example]:
    examples: list[Example] = []
    seen: set[str] = set()
    for line_no, row in _read_jsonl(path):
        source, target = str(row.get("source", "")).strip(), str(row.get("target", "")).strip()
        example_id = str(row.get("id", f"line-{line_no}")).strip()
        if not source or not target:
            raise ValueError(f"{path}:{line_no}: example requires non-empty source and target")
        if example_id in seen:
            raise ValueError(f"{path}:{line_no}: duplicate example id {example_id!r}")
        seen.add(example_id)
        examples.append(Example(example_id, source, target, line_no))
    if not examples:
        raise ValueError(f"{path}: no examples found")
    return examples


def load_glossary(path: Path | None) -> list[GlossaryEntry]:
    if path is None:
        return []
    entries: list[GlossaryEntry] = []
    for line_no, row in _read_jsonl(path):
        source = str(row.get("source", "")).strip()
        raw_senses = row.get("senses", [])
        if not source or not isinstance(raw_senses, list) or not raw_senses:
            raise ValueError(f"{path}:{line_no}: glossary entry requires source and a non-empty senses list")
        senses: list[GlossarySense] = []
        for sense in raw_senses:
            if not isinstance(sense, dict) or not str(sense.get("target", "")).strip() or not str(sense.get("when", "")).strip():
                raise ValueError(f"{path}:{line_no}: every sense requires non-empty target and when")
            raw_examples = sense.get("examples", [])
            if not isinstance(raw_examples, list):
                raise ValueError(f"{path}:{line_no}: sense examples must be a list")
            senses.append(GlossarySense(str(sense["target"]).strip(), str(sense["when"]).strip(), tuple(str(item).strip() for item in raw_examples if str(item).strip())))
        entries.append(GlossaryEntry(source, tuple(senses), str(row.get("notes", "")).strip(), line_no))
    return entries


def load_inputs(path: Path) -> list[InputRecord]:
    if path.suffix.lower() == ".jsonl":
        records: list[InputRecord] = []
        for line_no, row in _read_jsonl(path):
            source = str(row.get("source", "")).strip()
            if not source:
                raise ValueError(f"{path}:{line_no}: input requires a non-empty source")
            record_id = str(row.get("id", f"line-{line_no}")).strip()
            records.append(InputRecord(record_id, source, {k: v for k, v in row.items() if k not in {"id", "source"}}))
        return records
    if path.suffix.lower() == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("DOCX input requires python-docx; install requirements.txt") from exc
        document = Document(path)
        paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        return [
            InputRecord(
                f"paragraph-{index:04d}",
                paragraph,
                {"source_file": str(path), "docx_paragraph": index},
            )
            for index, paragraph in enumerate(paragraphs, 1)
        ]
    if path.suffix.lower() == ".doc":
        if shutil.which("antiword"):
            command = ["antiword", str(path)]
        elif shutil.which("textutil"):
            command = ["textutil", "-convert", "txt", "-stdout", str(path)]
        else:
            raise RuntimeError(
                "DOC input requires antiword (Windows/Linux/macOS) or textutil (macOS). "
                "Install antiword or save the document as DOCX."
            )
        completed = subprocess.run(command, capture_output=True, check=False)
        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Could not read DOC input: {message or 'document conversion failed'}")
        text = completed.stdout.decode("utf-8", errors="replace")
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        return [
            InputRecord(
                f"paragraph-{index:04d}",
                paragraph,
                {"source_file": str(path), "doc_paragraph": index},
            )
            for index, paragraph in enumerate(paragraphs, 1)
        ]
    if path.suffix.lower() != ".txt":
        raise ValueError(f"{path}: input must be .docx, .doc, .txt, or .jsonl")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")) if part.strip()]
    return [InputRecord(f"paragraph-{index:04d}", paragraph) for index, paragraph in enumerate(paragraphs, 1)]


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def match_glossary(source_text: str, entries: list[GlossaryEntry]) -> list[GlossaryEntry]:
    normalized = normalize(source_text)
    def contains(entry: GlossaryEntry) -> bool:
        term = normalize(entry.source)
        # Space-delimited writing systems need word boundaries. For scripts normally
        # written without spaces, substring matching is the useful fallback.
        if any(character.isspace() for character in normalized):
            return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized) is not None
        return term in normalized

    # Longest terms first keeps the most informative phrases near the top of the prompt.
    return sorted((entry for entry in entries if contains(entry)), key=lambda item: len(normalize(item.source)), reverse=True)
