"""PDF engine selection: which engines and fonts render() is willing to try."""
import unittest
from pathlib import Path
from unittest import mock

import proposal_reviewer
from proposal_reviewer import UNICODE_FONTS, render


def _calls(mock_run):
    return [c.args[0] for c in mock_run.call_args_list]


class TestRenderFontFallback(unittest.TestCase):
    def test_a_linux_unicode_font_is_among_the_candidates(self):
        # "Arial Unicode MS" alone is macOS-only, which left Linux with no PDF at
        # all even with a full TeX Live installed.
        self.assertIn("DejaVu Sans", UNICODE_FONTS)

    def test_tries_each_font_until_one_renders(self):
        def run(cmd, **kw):
            ok = any("mainfont=DejaVu Sans" == a for a in cmd)
            return mock.Mock(returncode=0 if ok else 1, stdout="", stderr="")
        with mock.patch("subprocess.run", side_effect=run) as r:
            out = render(Path("/tmp/x.md"))
        self.assertIn(".pdf", out)
        self.assertTrue(any("mainfont=DejaVu Sans" in c for c in _calls(r)))

    def test_reports_html_only_when_every_engine_fails(self):
        with mock.patch("subprocess.run", return_value=mock.Mock(returncode=1, stdout="", stderr="")):
            out = render(Path("/tmp/x.md"))
        self.assertIn("no PDF engine", out)
        self.assertNotIn(".pdf,", out)

    def test_a_missing_engine_does_not_abort_the_remaining_attempts(self):
        seen = []
        def run(cmd, **kw):
            seen.append(cmd)
            if "--pdf-engine=wkhtmltopdf" in cmd:
                raise FileNotFoundError("wkhtmltopdf")
            return mock.Mock(returncode=1, stdout="", stderr="")
        with mock.patch("subprocess.run", side_effect=run):
            render(Path("/tmp/x.md"))
        self.assertTrue(any("--pdf-engine=xelatex" in c for c in seen))


if __name__ == "__main__":
    unittest.main()
