import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from providers import load_dotenv


class TestLoadDotenv(unittest.TestCase):
    def _write(self, text):
        p = Path(self.tmp) / ".env"
        p.write_text(text)
        return p

    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name

    def tearDown(self):
        self._td.cleanup()

    def test_sets_absent_keys(self):
        p = self._write("FOO=bar\n")
        env = {}
        n = load_dotenv(p, env)
        self.assertEqual(n, 1)
        self.assertEqual(env["FOO"], "bar")

    def test_does_not_overwrite_existing(self):
        p = self._write("FOO=fromfile\n")
        env = {"FOO": "fromenv"}
        load_dotenv(p, env)
        self.assertEqual(env["FOO"], "fromenv")

    def test_ignores_comments_and_blanks(self):
        p = self._write("# a comment\n\nFOO=bar\n   \n")
        env = {}
        self.assertEqual(load_dotenv(p, env), 1)
        self.assertEqual(env, {"FOO": "bar"})

    def test_strips_surrounding_quotes(self):
        p = self._write('A="one"\nB=\'two\'\n')
        env = {}
        load_dotenv(p, env)
        self.assertEqual(env["A"], "one")
        self.assertEqual(env["B"], "two")

    def test_value_may_contain_equals(self):
        p = self._write("K=a=b=c\n")
        env = {}
        load_dotenv(p, env)
        self.assertEqual(env["K"], "a=b=c")

    def test_missing_file_is_not_an_error(self):
        env = {}
        self.assertEqual(load_dotenv(Path(self.tmp) / "nope", env), 0)


if __name__ == "__main__":
    unittest.main()
