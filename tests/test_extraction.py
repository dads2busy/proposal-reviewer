"""Text extraction: heading style and silent-failure handling.

pandoc's markdown writer emits SETEXT headings (`===` underlines) up to 2.10 and
ATX (`#`) from 3.0. parse_sections keys on `#`, so a setext conversion drops every
`#`-derived section — and the surviving ones silently absorb the dropped text and
get scored against it. The flag that forces ATX was renamed across versions and
each version rejects the other's spelling, hence the fallback."""
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from proposal_reviewer import extract_text


def _run(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class TestPandocHeadings(unittest.TestCase):
    def test_modern_flag_is_tried_first(self):
        with mock.patch("subprocess.run", return_value=_run(stdout="# 1. Intro\n")) as r:
            self.assertEqual(extract_text(Path("p.docx")), "# 1. Intro\n")
        self.assertIn("--markdown-headings=atx", r.call_args_list[0].args[0])

    def test_falls_back_when_the_modern_flag_is_rejected(self):
        # pandoc 2.0.6 exits non-zero on --markdown-headings=atx.
        results = [_run(2, stderr="Unknown option --markdown-headings"),
                   _run(stdout="# 1. Intro\n")]
        with mock.patch("subprocess.run", side_effect=results) as r:
            self.assertEqual(extract_text(Path("p.docx")), "# 1. Intro\n")
        self.assertIn("--atx-headers", r.call_args_list[1].args[0])

    def test_a_real_conversion_failure_exits_rather_than_scoring_nothing(self):
        # Both spellings failing means pandoc is broken or the file is unreadable.
        # Returning "" here would produce an all-zero review with no error.
        with mock.patch("subprocess.run", return_value=_run(1, stderr="boom")):
            with self.assertRaises(SystemExit) as e:
                extract_text(Path("p.docx"))
        self.assertIn("boom", str(e.exception))

    def test_empty_output_from_a_successful_run_is_still_an_error(self):
        with mock.patch("subprocess.run", return_value=_run(0, stdout="   \n")):
            with self.assertRaises(SystemExit):
                extract_text(Path("p.docx"))


class TestPdftotext(unittest.TestCase):
    def test_failure_exits_rather_than_returning_empty_text(self):
        with mock.patch("subprocess.run", return_value=_run(1, stderr="no such file")):
            with self.assertRaises(SystemExit) as e:
                extract_text(Path("p.pdf"))
        self.assertIn("no such file", str(e.exception))

    def test_success_returns_the_layout_text(self):
        with mock.patch("subprocess.run", return_value=_run(stdout="1.1 Background\n")) as r:
            self.assertEqual(extract_text(Path("p.pdf")), "1.1 Background\n")
        self.assertIn("-layout", r.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
