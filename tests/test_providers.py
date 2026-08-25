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
        p.chmod(0o600)          # the normal, secure state; see TestEnvPermissions
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


from proposal_reviewer import check_providers


class TestCheckProviders(unittest.TestCase):
    """--check-provider answers one question: can THIS machine reach its backends?
    It must diagnose every slot, so it reports failures instead of raising."""

    CFG = {"providers": {"uva": {"kind": "openai",
                                 "base_url": "https://example.invalid/api/",
                                 "api_key_env": "TEST_KEY"}}}

    def test_reports_ok_when_call_succeeds(self):
        def transport(url, body, headers, timeout):
            return _reply("pong")

        rows = check_providers(self.CFG, ["uva:Kimi K2.5", "uva:Kimi K2.5"],
                               environ={"TEST_KEY": "sk-test"},
                               transport=transport)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(ok for _, ok, _ in rows))

    def test_reports_failure_detail_without_raising(self):
        def transport(url, body, headers, timeout):
            raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)

        rows = check_providers(self.CFG, ["uva:Kimi K2.5", "uva:Kimi K2.5"],
                               environ={"TEST_KEY": "sk-test"},
                               transport=transport)
        self.assertFalse(rows[0][1])
        self.assertIn("HPC node", rows[0][2])

    def test_missing_key_is_reported_not_raised(self):
        rows = check_providers(self.CFG, ["uva:Kimi K2.5", "uva:Kimi K2.5"],
                               environ={}, transport=lambda *a: _reply("x"))
        self.assertFalse(rows[0][1])
        self.assertIn("TEST_KEY", rows[0][2])

    def test_a_broken_slot_does_not_hide_a_working_one(self):
        # Each slot is diagnosed independently: slot 0 is unresolvable, slot 1
        # is fine, and the report has to show both.
        def transport(url, body, headers, timeout):
            return _reply("pong")

        rows = check_providers(self.CFG, ["uva:Kimi K2.5", "gemma3:27b"],
                               environ={}, transport=transport)
        self.assertFalse(rows[0][1])
        self.assertTrue(rows[1][1])

    def test_rows_carry_the_slot_labels(self):
        rows = check_providers(self.CFG, ["uva:Kimi K2.5", "uva:Kimi K2.5"],
                               environ={"TEST_KEY": "sk-test"},
                               transport=lambda *a: _reply("pong"))
        self.assertEqual([r[0] for r in rows],
                         ["uva:Kimi K2.5 (sample 1)", "uva:Kimi K2.5 (sample 2)"])


import io
import contextlib


class TestEnvPermissions(unittest.TestCase):
    """A key leaked through file permissions leaked as thoroughly as one committed
    to git. .env holds a real secret, so loading it hardens it."""

    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name

    def tearDown(self):
        self._td.cleanup()

    def _env(self, mode):
        p = Path(self.tmp) / ".env"
        p.write_text("FOO=bar\n")
        p.chmod(mode)
        return p

    def _load(self, p):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            load_dotenv(p, {})
        return err.getvalue()

    def test_group_readable_env_is_tightened_to_600(self):
        p = self._env(0o640)
        self._load(p)
        self.assertEqual(p.stat().st_mode & 0o777, 0o600)

    def test_world_readable_env_is_tightened_to_600(self):
        p = self._env(0o644)
        self._load(p)
        self.assertEqual(p.stat().st_mode & 0o777, 0o600)

    def test_loose_permissions_are_reported(self):
        p = self._env(0o644)
        msg = self._load(p)
        self.assertIn(str(p), msg)
        self.assertIn("644", msg)

    def test_values_still_load_after_hardening(self):
        p = self._env(0o644)
        env = {}
        with contextlib.redirect_stderr(io.StringIO()):
            load_dotenv(p, env)
        self.assertEqual(env["FOO"], "bar")

    def test_private_env_is_left_alone_and_silent(self):
        p = self._env(0o600)
        self.assertEqual(self._load(p), "")
        self.assertEqual(p.stat().st_mode & 0o777, 0o600)

    def test_missing_file_needs_no_permission_work(self):
        missing = Path(self.tmp) / "nope"
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(load_dotenv(missing, {}), 0)
        self.assertEqual(err.getvalue(), "")


def _sse(*chunks):
    """Encode chunks the way Open WebUI streams them."""
    lines = []
    for c in chunks:
        lines.append("data: " + _json.dumps(c))
        lines.append("")
    lines += ["data: [DONE]", ""]
    return "\n".join(lines).encode()


def _delta(text):
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


class TestStreamingReply(unittest.TestCase):
    """UVA's Open WebUI answers text/event-stream and IGNORES `stream: false`, so a
    client that only json.loads() the body gets a JSONDecodeError on every call."""

    CFG = {"providers": {"uva": {"kind": "openai",
                                 "base_url": "https://example.invalid/api/",
                                 "api_key_env": "TEST_KEY"}}}

    def _fn(self, raw):
        return make_chat_fn(self.CFG, "uva", transport=lambda *a: raw,
                            environ={"TEST_KEY": "sk-test"}, sleep=lambda s: None)

    def test_reassembles_a_streamed_message(self):
        raw = _sse(_delta(""), _delta("Hello"), _delta(", world"))
        self.assertEqual(self._fn(raw)("m", "sys", "user"), "Hello, world")

    def test_done_sentinel_is_not_treated_as_content(self):
        out = self._fn(_sse(_delta("ok")))("m", "sys", "user")
        self.assertNotIn("DONE", out)
        self.assertEqual(out, "ok")

    def test_chunks_without_content_are_skipped(self):
        raw = _sse({"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                   _delta("body"))
        self.assertEqual(self._fn(raw)("m", "sys", "user"), "body")

    def test_plain_json_completion_still_works(self):
        self.assertEqual(self._fn(_reply("unstreamed"))("m", "sys", "user"),
                         "unstreamed")

    def test_unparseable_body_becomes_an_actionable_error(self):
        with self.assertRaises(ProviderError) as ctx:
            self._fn(b"<html>NetBadge login</html>")("m", "sys", "user")
        self.assertIn("example.invalid", str(ctx.exception))


class TestStreamingRobustness(unittest.TestCase):
    """The SSE reassembler sees whatever the proxy and the upstream model produce.
    A body it cannot turn into text must raise, never return an empty string: an
    empty reply reaches the scorer as a genuine 0 with an empty justification."""

    CFG = {"providers": {"uva": {"kind": "openai",
                                 "base_url": "https://example.invalid/api/",
                                 "api_key_env": "TEST_KEY"}}}

    def _fn(self, raw):
        return make_chat_fn(self.CFG, "uva", transport=lambda *a: raw,
                            environ={"TEST_KEY": "sk-test"}, sleep=lambda s: None)

    def test_stream_opening_with_a_comment_frame_still_parses(self):
        # Reverse proxies inject ': ping' to hold a long generation open.
        raw = b": ping\n\n" + _sse(_delta("hello"))
        self.assertEqual(self._fn(raw)("m", "sys", "user"), "hello")

    def test_stream_opening_with_an_event_line_still_parses(self):
        raw = b"event: message\n" + _sse(_delta("hello"))
        self.assertEqual(self._fn(raw)("m", "sys", "user"), "hello")

    def test_error_delivered_as_a_200_stream_raises(self):
        raw = _sse({"error": {"message": "model overloaded"}})
        with self.assertRaises(ProviderError) as ctx:
            self._fn(raw)("m", "sys", "user")
        self.assertIn("example.invalid", str(ctx.exception))

    def test_stream_with_only_a_done_sentinel_raises(self):
        with self.assertRaises(ProviderError):
            self._fn(b"data: [DONE]\n\n")("m", "sys", "user")

    def test_null_content_raises_instead_of_returning_none(self):
        raw = _json.dumps({"choices": [{"message": {"content": None}}]}).encode()
        with self.assertRaises(ProviderError):
            self._fn(raw)("m", "sys", "user")

    def test_non_text_content_raises(self):
        raw = _json.dumps({"choices": [{"message": {"content": [1, 2]}}]}).encode()
        with self.assertRaises(ProviderError):
            self._fn(raw)("m", "sys", "user")

    def test_role_only_frames_do_not_count_as_content(self):
        raw = _sse({"choices": [{"delta": {"role": "assistant"}}]},
                   {"choices": [{"finish_reason": "stop", "delta": {}}]})
        with self.assertRaises(ProviderError):
            self._fn(raw)("m", "sys", "user")

    def test_whitespace_only_reply_raises(self):
        with self.assertRaises(ProviderError):
            self._fn(_sse(_delta("   ")))("m", "sys", "user")


if __name__ == "__main__":
    unittest.main()
