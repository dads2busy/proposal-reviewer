import os
import sys
import unittest
import urllib.error
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


import json as _json
from providers import make_chat_fn


def _reply(text):
    return _json.dumps({"choices": [{"message": {"content": text}}]}).encode()


class TestMakeChatFn(unittest.TestCase):
    CFG = {"providers": {"uva": {"kind": "openai",
                                 "base_url": "https://example.invalid/api/",
                                 "api_key_env": "TEST_KEY"}}}

    def test_openai_posts_expected_payload_and_header(self):
        seen = {}

        def transport(url, body, headers, timeout):
            seen["url"] = url
            seen["body"] = _json.loads(body)
            seen["headers"] = headers
            return _reply("hello")

        fn = make_chat_fn(self.CFG, "uva", transport=transport,
                          environ={"TEST_KEY": "sk-test"})
        out = fn("Kimi K2.5", "sys", "user", num_ctx=32768)

        self.assertEqual(out, "hello")
        self.assertEqual(seen["url"], "https://example.invalid/api/chat/completions")
        self.assertEqual(seen["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(seen["body"]["model"], "Kimi K2.5")
        self.assertEqual(seen["body"]["messages"][0]["role"], "system")
        self.assertNotIn("options", seen["body"])   # num_ctx is Ollama-only

    def test_openai_temperature_is_configurable(self):
        seen = {}

        def transport(url, body, headers, timeout):
            seen.update(_json.loads(body))
            return _reply("ok")

        fn = make_chat_fn(self.CFG, "uva", temperature=0.6, transport=transport,
                          environ={"TEST_KEY": "sk-test"})
        fn("Kimi K2.5", "sys", "user")
        self.assertEqual(seen["temperature"], 0.6)

    def test_ollama_includes_num_ctx(self):
        seen = {}

        def transport(url, body, headers, timeout):
            seen.update(_json.loads(body))
            return _reply("ok")

        fn = make_chat_fn({}, "ollama", transport=transport, environ={})
        fn("gemma3:27b", "sys", "user", num_ctx=16384)
        self.assertEqual(seen["options"]["num_ctx"], 16384)

    def test_missing_key_raises_before_network(self):
        def transport(*a, **k):
            raise AssertionError("should not reach the network")

        with self.assertRaises(ProviderError) as ctx:
            make_chat_fn(self.CFG, "uva", transport=transport, environ={})
        self.assertIn("TEST_KEY", str(ctx.exception))

    def test_401_message_mentions_hpc_node(self):
        def transport(url, body, headers, timeout):
            raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)

        fn = make_chat_fn(self.CFG, "uva", transport=transport,
                          environ={"TEST_KEY": "sk-test"})
        with self.assertRaises(ProviderError) as ctx:
            fn("Kimi K2.5", "sys", "user")
        self.assertIn("HPC node", str(ctx.exception))

    def test_429_retries_then_succeeds(self):
        calls = []

        def transport(url, body, headers, timeout):
            calls.append(1)
            if len(calls) < 3:
                raise urllib.error.HTTPError(url, 429, "Too Many", {}, None)
            return _reply("finally")

        fn = make_chat_fn(self.CFG, "uva", transport=transport,
                          environ={"TEST_KEY": "sk-test"}, sleep=lambda s: None)
        self.assertEqual(fn("Kimi K2.5", "sys", "user"), "finally")
        self.assertEqual(len(calls), 3)

    def test_connection_error_names_the_host(self):
        def transport(url, body, headers, timeout):
            raise urllib.error.URLError("refused")

        fn = make_chat_fn(self.CFG, "uva", transport=transport,
                          environ={"TEST_KEY": "sk-test"}, sleep=lambda s: None)
        with self.assertRaises(ProviderError) as ctx:
            fn("Kimi K2.5", "sys", "user")
        self.assertIn("example.invalid", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
