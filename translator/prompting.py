from __future__ import annotations

from .data import GlossaryEntry
from .retrieval import Retrieved


def system_prompt(source_language: str, target_language: str, style: str) -> str:
    extra = f"\n\nAdditional style requirements:\n{style.strip()}" if style.strip() else ""
    return (
        f"You are a professional translator from {source_language} to {target_language}. "
        "Translate accurately and completely. Preserve meaning, names, dates, structure, tone, "
        "and degree of formality. Retrieved examples demonstrate preferred usage but are not text "
        "to copy. The lexicon is context-sensitive: inspect every supplied sense and choose only "
        "the rendering whose conditions match the current passage. If no sense fits, translate from "
        "context rather than forcing an incorrect lexicon rendering. Return only the translation, "
        "without commentary, labels, or Markdown, except that requested paragraph markers must be "
        "copied exactly when several paragraphs are submitted together."
        + extra
    )


def user_prompt(source: str, examples: list[Retrieved], glossary: list[GlossaryEntry], source_language: str, target_language: str, *, paragraph_count: int = 1, max_glossary_senses: int = 15) -> str:
    blocks: list[str] = []
    if examples:
        blocks.append("RETRIEVED PARALLEL EXAMPLES")
        for number, item in enumerate(examples, 1):
            blocks.append(f"Example {number} [id={item.example.id}]\n{source_language}: {item.example.source}\n{target_language}: {item.example.target}")
    if glossary:
        blocks.append("CONTEXT-SENSITIVE LEXICON MATCHES")
        for entry in glossary:
            lines = [f"Term: {entry.source}"]
            for sense in entry.senses[:max_glossary_senses]:
                evidence = f" Examples: {' | '.join(sense.examples)}" if sense.examples else ""
                lines.append(f"- {sense.target} — use when: {sense.when}.{evidence}")
            if entry.notes:
                lines.append(f"Notes: {entry.notes}")
            blocks.append("\n".join(lines))
    if paragraph_count > 1:
        blocks.append(
            f"TRANSLATE THESE {paragraph_count} PARAGRAPHS\n"
            "Translate every paragraph independently and in order. Do not merge, omit, split, or "
            "reorder paragraphs. In the answer, copy each [PARAGRAPH ...] marker exactly and put "
            "only that paragraph's translation beneath it.\n\n"
            f"{source_language}:\n{source}"
        )
    else:
        blocks.append(f"TRANSLATE THIS PASSAGE\n{source_language}:\n{source}")
    return "\n\n".join(blocks)
