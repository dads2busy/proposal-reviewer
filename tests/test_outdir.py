"""Where reports land: the --outdir override and its default."""
import tempfile
import unittest
from pathlib import Path

from proposal_reviewer import resolve_outdir


class TestResolveOutdir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_defaults_to_the_proposal_directory(self):
        proposal = self.root / "sub" / "p.docx"
        proposal.parent.mkdir()
        self.assertEqual(resolve_outdir(proposal, None), proposal.parent)

    def test_explicit_directory_wins(self):
        proposal = self.root / "p.docx"
        out = self.root / "reports"
        out.mkdir()
        self.assertEqual(resolve_outdir(proposal, str(out)), out)

    def test_missing_directory_is_created(self):
        proposal = self.root / "p.docx"
        out = self.root / "a" / "b"
        self.assertEqual(resolve_outdir(proposal, str(out)), out)
        self.assertTrue(out.is_dir())

    def test_tilde_is_expanded(self):
        proposal = self.root / "p.docx"
        got = resolve_outdir(proposal, "~")
        self.assertEqual(got, Path.home())

    def test_an_unusable_directory_exits_before_any_model_call(self):
        # Resolving up front is the point: 33 model calls then a write failure is
        # the expensive way to learn the path was wrong.
        proposal = self.root / "p.docx"
        blocker = self.root / "afile"
        blocker.write_text("not a directory")
        with self.assertRaises(SystemExit) as e:
            resolve_outdir(proposal, str(blocker))
        self.assertIn("not a directory", str(e.exception))

    def test_a_directory_without_write_permission_exits(self):
        proposal = self.root / "p.docx"
        out = self.root / "readonly"
        out.mkdir(mode=0o500)
        self.addCleanup(out.chmod, 0o700)
        with self.assertRaises(SystemExit):
            resolve_outdir(proposal, str(out))


if __name__ == "__main__":
    unittest.main()
