import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from translator.data import Example, GlossaryEntry, GlossarySense, load_inputs, match_glossary
from translator.retrieval import BM25Index


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


if __name__ == "__main__":
    unittest.main()
