import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from translator.data import (
    Example,
    GlossaryEntry,
    GlossarySense,
    load_examples,
    load_glossary,
    load_inputs,
    match_glossary,
)
from translator.retrieval import BM25Index
from translator.glossary import GlossaryExampleSelector
from scripts.build_glossary import build_records, collect_evidence
from scripts.merge_glossaries import merge_entries


class ToolkitTests(unittest.TestCase):
    def test_bm25_prefers_matching_document(self):
        examples = [Example("river", "walk by the river bank", "rive"), Example("money", "deposit cash at the bank", "banque")]
        result = BM25Index(examples).search("cash deposit", 1)
        self.assertEqual(result[0][0].id, "money")

    def test_glossary_preserves_contextual_senses(self):
        entry = GlossaryEntry("bank", (GlossarySense("banque", "finance"), GlossarySense("rive", "river")))
        matches = match_glossary("They stood on the bank.", [entry])
        self.assertEqual(len(matches[0].senses), 2)

    def test_glossary_does_not_match_inside_longer_word(self):
        entry = GlossaryEntry("bank", (GlossarySense("banque", "finance"),))
        self.assertEqual(match_glossary("The bankruptcy filing.", [entry]), [])

    def test_glossary_selects_relevant_parallel_examples_for_phrase(self):
        examples = [
            Example("money", "deposit cash at the bank", "déposer à la banque"),
            Example("river", "walk beside the river bank", "marcher sur la rive"),
            Example("noise", "a bankruptcy filing", "une faillite"),
        ]
        selected = GlossaryExampleSelector(examples).select_for_phrase(
            "They rested beside the river bank.", "bank", 2
        )
        self.assertEqual([example.id for example in selected], ["river", "money"])

    def test_glossary_phrase_examples_require_the_complete_term(self):
        examples = [
            Example("exact", "the river bank", "la rive"),
            Example("longer", "the bankruptcy filing", "la faillite"),
        ]
        candidates = GlossaryExampleSelector(examples).candidates("bank")
        self.assertEqual([example.id for example in candidates], ["exact"])

    def test_glossary_phrase_examples_fall_back_to_corpus_order(self):
        examples = [
            Example("first", "first bank", "première banque"),
            Example("second", "second bank", "deuxième banque"),
        ]
        selected = GlossaryExampleSelector(examples).select_for_phrase("bank", "bank", 1)
        self.assertEqual([example.id for example in selected], ["first"])

    def test_glossary_phrase_examples_respect_per_phrase_limit(self):
        examples = [
            Example(str(index), f"context {index} bank", f"target {index}")
            for index in range(20)
        ]
        selected = GlossaryExampleSelector(examples).select_for_phrase(
            "bank", "bank", 15
        )
        self.assertEqual(len(selected), 15)

    def test_text_input_splits_on_blank_lines(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "input.txt"
            path.write_text("First paragraph.\n\nSecond\nparagraph.\n", encoding="utf-8")
            records = load_inputs(path)
        self.assertEqual([item.source for item in records], ["First paragraph.", "Second\nparagraph."])

    def test_docx_input_reads_nonempty_paragraphs(self):
        try:
            from docx import Document
        except ImportError:
            self.skipTest("python-docx is not installed in this test environment")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "input.docx"
            document = Document()
            document.add_paragraph("First paragraph.")
            document.add_paragraph("")
            document.add_paragraph("Second paragraph.")
            document.save(path)
            records = load_inputs(path)
        self.assertEqual([item.source for item in records], ["First paragraph.", "Second paragraph."])

    def test_examples_accept_input_output_aliases(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "examples.jsonl"
            path.write_text('{"input":"درود","output":"Greetings"}\n', encoding="utf-8")
            examples = load_examples(path)
        self.assertEqual((examples[0].source, examples[0].target), ("درود", "Greetings"))

    def test_tsv_glossary_groups_multiple_translations(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "glossary.tsv"
            path.write_text("source\ttarget\nعالم\tworld\nعالم\trealm\nعالم\tworld\n", encoding="utf-8")
            glossary = load_glossary(path)
        self.assertEqual(glossary[0].source, "عالم")
        self.assertEqual([sense.target for sense in glossary[0].senses], ["world", "realm"])

    def test_glossary_builder_filters_chance_cooccurrence(self):
        candidates = {
            "نور": {
                "source": "نور",
                "targets": {"light": "light", "radiance": "radiance", "the": "the"},
            }
        }
        examples = [
            ("نور دل", "light of the heart"),
            ("نور جهان", "light of the world"),
            ("نور الهی", "divine radiance"),
            ("نور تابان", "the shining radiance"),
        ] + [(f"متن دیگر {number}", "the other passage") for number in range(8)]
        evidence, source_documents, target_documents = collect_evidence(
            examples, candidates, max_source_words=2, examples_per_sense=2, max_relative_position_delta=1
        )
        records = build_records(
            candidates,
            evidence,
            source_documents,
            target_documents,
            len(examples),
            single_word_min_support=2,
            multiword_min_support=1,
            min_lift=3,
        )
        self.assertEqual([sense["target"] for sense in records[0]["senses"]], ["light", "radiance"])

    def test_glossary_merge_preserves_evidence_and_adds_legacy_senses(self):
        json_entries = {
            "نور": {
                "source": "نور",
                "senses": [
                    {
                        "target": "light",
                        "when": "Use for illumination.",
                        "examples": ["نور جهان"],
                    }
                ],
                "sense_keys": {"light"},
                "notes": ["Evidence-backed."],
                "json_sense_count": 1,
            }
        }
        tsv_groups = {
            "نور": {
                "source": "نور",
                "row_count": 3,
                "targets": {
                    "light": {"target": "light", "count": 2, "first_line": 2},
                    "radiance": {"target": "radiance", "count": 1, "first_line": 4},
                },
            }
        }
        records, stats = merge_entries(json_entries, tsv_groups)
        self.assertEqual([sense["target"] for sense in records[0]["senses"]], ["light", "radiance"])
        self.assertEqual(records[0]["senses"][0]["examples"], ["نور جهان"])
        self.assertEqual(records[0]["senses"][1]["examples"], [])
        self.assertEqual(stats["tsv_senses_already_covered"], 1)
        self.assertEqual(stats["legacy_senses_added"], 1)


if __name__ == "__main__":
    unittest.main()
