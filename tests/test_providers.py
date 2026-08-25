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


from providers import (split_model_ref, known_provider_names, provider_spec,
                       ProviderError, BUILTIN_PROVIDERS)


class TestModelRefs(unittest.TestCase):
    CFG = {"providers": {"uva": {"kind": "openai",
                                 "base_url": "https://example.invalid/api/",
                                 "api_key_env": "SOME_KEY"}}}

    def test_known_names_include_builtin_ollama(self):
        self.assertIn("ollama", known_provider_names(self.CFG))
        self.assertIn("uva", known_provider_names(self.CFG))

    def test_declared_prefix_is_a_provider(self):
        known = known_provider_names(self.CFG)
        self.assertEqual(split_model_ref("uva:Kimi K2.5", known),
                         ("uva", "Kimi K2.5"))

    def test_bare_name_with_colon_is_an_ollama_model(self):
        known = known_provider_names(self.CFG)
        self.assertEqual(split_model_ref("qwen2.5:32b", known),
                         ("ollama", "qwen2.5:32b"))

    def test_explicit_ollama_prefix_keeps_rest_intact(self):
        known = known_provider_names(self.CFG)
        self.assertEqual(split_model_ref("ollama:qwen2.5:32b", known),
                         ("ollama", "qwen2.5:32b"))

    def test_plain_name_is_ollama(self):
        known = known_provider_names(self.CFG)
        self.assertEqual(split_model_ref("gemma3", known), ("ollama", "gemma3"))

    def test_unknown_provider_spec_raises(self):
        with self.assertRaises(ProviderError):
            provider_spec(self.CFG, "nope")

    def test_builtin_ollama_spec_available_without_declaration(self):
        spec = provider_spec({}, "ollama")
        self.assertEqual(spec["kind"], "ollama")

    def test_literal_api_key_in_config_is_rejected(self):
        cfg = {"providers": {"bad": {"kind": "openai",
                                     "base_url": "https://example.invalid/",
                                     "api_key": "sk-literal"}}}
        with self.assertRaises(ProviderError) as ctx:
            provider_spec(cfg, "bad")
        self.assertIn(".env", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
