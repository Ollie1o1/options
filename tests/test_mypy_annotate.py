"""mypy's output, turned into annotations that survive a log nobody opens.

The typecheck job is continue-on-error. Its findings are only worth producing
if they land somewhere visible, so the parsing has to be right about the shapes
mypy actually emits — with and without a column, notes attached to errors, and
messages containing the characters workflow commands treat as syntax.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.mypy_annotate import _escape, annotations_for  # noqa: E402


class ParsingTest(unittest.TestCase):
    def test_an_error_becomes_an_error_annotation(self):
        out = annotations_for(
            ['src/a.py:12: error: Name "x" is not defined  [name-defined]'])
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].startswith("::error "))
        self.assertIn("file=src/a.py", out[0])
        self.assertIn("line=12", out[0])

    def test_the_message_survives(self):
        out = annotations_for(
            ['src/a.py:12: error: Name "x" is not defined  [name-defined]'])
        self.assertIn('Name "x" is not defined', out[0])

    def test_a_column_is_carried_when_mypy_gives_one(self):
        out = annotations_for(["src/a.py:12:5: error: bad thing  [misc]"])
        self.assertIn("col=5", out[0])

    def test_a_missing_column_is_simply_absent(self):
        out = annotations_for(["src/a.py:12: error: bad thing  [misc]"])
        self.assertNotIn("col=", out[0])

    def test_notes_do_not_become_their_own_annotation(self):
        # mypy emits `note:` lines elaborating the error above them. One
        # problem must not be counted twice.
        out = annotations_for([
            "src/a.py:12: error: bad thing  [misc]",
            "src/a.py:12: note: did you mean something else?",
        ])
        self.assertEqual(len(out), 1)

    def test_the_summary_line_is_ignored(self):
        out = annotations_for([
            "src/a.py:12: error: bad thing  [misc]",
            "Found 1 error in 1 file (checked 289 source files)",
        ])
        self.assertEqual(len(out), 1)

    def test_clean_output_produces_nothing(self):
        self.assertEqual(
            annotations_for(["Success: no issues found in 289 source files"]), [])

    def test_empty_input_produces_nothing(self):
        self.assertEqual(annotations_for([]), [])


class EscapingTest(unittest.TestCase):
    """`::` and `,` are workflow-command syntax; a raw newline ends the
    command. An unescaped message silently truncates or corrupts the
    annotation, which is worse than not annotating."""

    def test_a_percent_is_escaped_first(self):
        out = annotations_for(["src/a.py:1: error: 100% wrong  [misc]"])
        self.assertIn("100%25 wrong", out[0])

    def test_a_newline_cannot_end_the_command_early(self):
        # mypy's output arrives one diagnostic per line, so this is defence in
        # depth rather than a shape seen in practice — but a raw newline in a
        # message truncates the annotation silently, and silent truncation is
        # exactly what this reporter exists to avoid.
        self.assertEqual(_escape("line one\nline two"), "line one%0Aline two")

    def test_escaping_a_percent_does_not_re_escape_its_own_output(self):
        # "%" must be substituted first: the other rules introduce "%", so a
        # naive order turns "\n" into "%0A" and then into "%250A".
        self.assertEqual(_escape("\n"), "%0A")
        self.assertEqual(_escape("50%\n"), "50%25%0A")


class BreakdownTest(unittest.TestCase):
    """With hundreds of errors, the capped annotation list says nothing about
    the shape of the problem. A histogram by error code and by file is what
    tells you whether this is one bad pattern repeated or 349 separate ones."""

    _SAMPLE = [
        'src/a.py:1: error: Need type annotation for "x"  [var-annotated]',
        'src/a.py:2: error: Need type annotation for "y"  [var-annotated]',
        "src/a.py:3: error: bad operand  [operator]",
        "src/b.py:9: error: bad operand  [operator]",
        "src/b.py:9: note: elaboration that is not its own error",
        "Found 4 errors in 2 files (checked 289 source files)",
    ]

    def test_counts_by_error_code(self):
        from scripts.mypy_annotate import breakdown

        b = breakdown(self._SAMPLE)
        self.assertEqual(b["codes"]["var-annotated"], 2)
        self.assertEqual(b["codes"]["operator"], 2)

    def test_counts_by_file(self):
        from scripts.mypy_annotate import breakdown

        b = breakdown(self._SAMPLE)
        self.assertEqual(b["files"]["src/a.py"], 3)
        self.assertEqual(b["files"]["src/b.py"], 1)

    def test_notes_are_not_counted(self):
        from scripts.mypy_annotate import breakdown

        self.assertEqual(breakdown(self._SAMPLE)["total"], 4)

    def test_an_error_without_a_code_is_still_counted(self):
        from scripts.mypy_annotate import breakdown

        b = breakdown(["src/a.py:1: error: something odd"])
        self.assertEqual(b["total"], 1)
        self.assertEqual(b["codes"]["(none)"], 1)

    def test_the_rendered_breakdown_names_the_worst_code(self):
        from scripts.mypy_annotate import render_breakdown

        text = render_breakdown(breakdown_=None, lines=self._SAMPLE)
        self.assertIn("var-annotated", text)
        self.assertIn("src/a.py", text)


class CapTest(unittest.TestCase):
    def test_the_annotation_count_is_capped(self):
        lines = [f"src/a.py:{i}: error: bad  [misc]" for i in range(1, 80)]
        out = annotations_for(lines, max_annotations=10)
        self.assertEqual(sum(1 for o in out if o.startswith("::error ")), 10)

    def test_the_true_total_is_still_reported(self):
        lines = [f"src/a.py:{i}: error: bad  [misc]" for i in range(1, 80)]
        out = annotations_for(lines, max_annotations=10)
        self.assertTrue(any("79 errors total" in o for o in out))

    def test_no_summary_line_when_nothing_was_dropped(self):
        out = annotations_for(["src/a.py:1: error: bad  [misc]"],
                              max_annotations=10)
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
