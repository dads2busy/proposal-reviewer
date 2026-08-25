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


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from proposal_reviewer import resolve_chat_fns


class TestResolveChatFns(unittest.TestCase):
    CFG = {"providers": {"uva": {"kind": "openai",
                                 "base_url": "https://example.invalid/api/",
                                 "api_key_env": "TEST_KEY"}}}

    def test_strips_provider_prefix_from_model_name(self):
        slots = resolve_chat_fns(self.CFG, ["uva:Kimi K2.5", "ollama:qwen2.5:32b"],
                                 environ={"TEST_KEY": "sk-test"})
        self.assertEqual([s[1] for s in slots], ["Kimi K2.5", "qwen2.5:32b"])

    def test_labels_come_from_slot_labels(self):
        slots = resolve_chat_fns(self.CFG, ["uva:Kimi K2.5", "uva:Kimi K2.5"],
                                 environ={"TEST_KEY": "sk-test"})
        self.assertEqual([s[0] for s in slots],
                         ["uva:Kimi K2.5 (sample 1)", "uva:Kimi K2.5 (sample 2)"])

    def test_each_slot_gets_a_callable(self):
        slots = resolve_chat_fns(self.CFG, ["uva:Kimi K2.5", "ollama:gemma3:27b"],
                                 environ={"TEST_KEY": "sk-test"})
        for _, _, fn in slots:
            self.assertTrue(callable(fn))

    def test_undeclared_prefix_is_treated_as_an_ollama_model(self):
        # With no matching provider declared, the whole string is a model name.
        # This is what keeps `qwen2.5:32b` working.
        slots = resolve_chat_fns({}, ["nosuch:model", "gemma3"], environ={})
        self.assertEqual([s[1] for s in slots], ["nosuch:model", "gemma3"])

    def test_openai_provider_without_key_raises(self):
        with self.assertRaises(ProviderError):
            resolve_chat_fns(self.CFG, ["uva:Kimi K2.5", "gemma3"], environ={})


from proposal_reviewer import synthesize_recommendations


class TestSynthesizeRecommendationsNumCtx(unittest.TestCase):
    """Regression: synthesize_recommendations must forward num_ctx to the chat_fn
    so an Ollama synthesis call keeps sending options.num_ctx once main() passes a
    make_chat_fn-produced chat_fn (whose num_ctx defaults to None) instead of the
    old _ollama_chat (whose num_ctx defaulted to DEFAULT_NUM_CTX)."""

    def test_forwards_num_ctx_to_chat_fn(self):
        cfg = {"criteria": [{"name": "C", "max": 10, "section": "1", "look_for": "x"}]}
        results = {
            "a": [{"awarded": 2, "max": 10, "gaps": ["needs more detail"]}],
            "b": [{"awarded": 3, "max": 10, "gaps": ["needs more detail"]}],
        }
        seen = {}

        def fake_chat_fn(model, system, user, timeout=900, num_ctx=None):
            seen["num_ctx"] = num_ctx
            return _json.dumps([{"title": "t", "points_at_stake": 5,
                                 "criteria": ["C"], "do": "x", "why": "y"}])

        synthesize_recommendations(cfg, ["a", "b"], results,
                                   chat_fn=fake_chat_fn, num_ctx=32768)
        self.assertEqual(seen["num_ctx"], 32768)


class TestTemperatureThreading(unittest.TestCase):
    """The second sample is only worth paying for if it actually samples wider,
    so the temperature per slot has to reach the wire."""

    CFG = {"providers": {"uva": {"kind": "openai",
                                 "base_url": "https://example.invalid/api/",
                                 "api_key_env": "TEST_KEY"}}}

    def test_each_slots_temperature_reaches_the_payload(self):
        seen = []

        def transport(url, body, headers, timeout):
            seen.append(_json.loads(body.decode())["temperature"])
            return _json.dumps({"choices": [{"message": {"content": "x"}}]}).encode()

        slots = resolve_chat_fns(self.CFG, ["uva:Kimi K2.5", "uva:Kimi K2.5"],
                                 environ={"TEST_KEY": "sk-test"},
                                 temperatures=(0.2, 0.6), transport=transport)
        for _, bare, fn in slots:
            fn(bare, "sys", "user")
        self.assertEqual(seen, [0.2, 0.6])


from proposal_reviewer import resolve_synthesis_slot


class TestResolveSynthesisSlot(unittest.TestCase):
    """`synthesis_model` is a documented config key; main() must honor it rather
    than always reusing slot 0."""

    CFG = {"providers": {"uva": {"kind": "openai",
                                 "base_url": "https://example.invalid/api/",
                                 "api_key_env": "TEST_KEY"}}}

    def _slots(self):
        return resolve_chat_fns(self.CFG, ["uva:Kimi K2.5", "uva:Kimi K2.5"],
                                environ={"TEST_KEY": "sk-test"})

    def test_defaults_to_the_first_ensemble_slot(self):
        slots = self._slots()
        model, fn = resolve_synthesis_slot(self.CFG, slots,
                                           environ={"TEST_KEY": "sk-test"})
        self.assertEqual(model, "Kimi K2.5")
        self.assertIs(fn, slots[0][2])

    def test_config_synthesis_model_overrides_slot_zero(self):
        cfg = dict(self.CFG, synthesis_model="gemma3:27b")
        slots = self._slots()
        model, fn = resolve_synthesis_slot(cfg, slots,
                                           environ={"TEST_KEY": "sk-test"})
        self.assertEqual(model, "gemma3:27b")
        self.assertIsNot(fn, slots[0][2])

    def test_synthesis_model_is_posted_to_its_own_provider(self):
        # A bare model name must not be sent to slot 0's remote endpoint.
        cfg = dict(self.CFG, synthesis_model="gemma3:27b")
        seen = {}

        def transport(url, body, headers, timeout):
            seen["url"] = url
            seen["auth"] = "Authorization" in headers
            return _json.dumps({"choices": [{"message": {"content": "x"}}]}).encode()

        slots = self._slots()
        model, fn = resolve_synthesis_slot(cfg, slots,
                                           environ={"TEST_KEY": "sk-test"},
                                           transport=transport)
        fn(model, "sys", "user")
        self.assertIn("localhost:11434", seen["url"])
        self.assertFalse(seen["auth"])

    def test_unknown_synthesis_provider_raises(self):
        cfg = dict(self.CFG, synthesis_model="nosuch:model")
        # 'nosuch' is not a known provider, so this is an Ollama model name and
        # must resolve cleanly rather than exploding.
        model, _ = resolve_synthesis_slot(cfg, self._slots(),
                                          environ={"TEST_KEY": "sk-test"})
        self.assertEqual(model, "nosuch:model")


if __name__ == "__main__":
    unittest.main()
