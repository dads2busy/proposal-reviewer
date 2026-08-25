# UVA RC GenAI Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the engine score proposals using UVA's OpenAI-compatible RC GenAI service (Kimi K2.5) alongside local Ollama, with `.env`-based secrets and reports that don't overclaim when both ensemble slots hold the same model.

**Architecture:** A new `providers.py` module owns everything network- and secret-related: `.env` loading, `provider:model` resolution, and building a `chat_fn` for a given provider. `proposal_reviewer.py` keeps its existing `chat_fn` seam and gains a factory call at three sites. Report wording changes are confined to `build_combined`.

**Tech Stack:** Python 3, stdlib only (`urllib`, `json`, `os`, `re`) plus PyYAML. Tests are `unittest`, run with `python -m unittest`. **pytest is not installed — never use it.**

## Global Constraints

- **Stdlib-only except PyYAML.** Do not add `python-dotenv`, `requests`, or `openai`.
- **No commercial API providers.** UVA-hosted and localhost only.
- **Never accept a literal API key** from a config file or CLI argument — environment only.
- Base URL: `https://open-webui.rc.virginia.edu/api/` (trailing slash), endpoint `chat/completions`.
- Model id is `Kimi K2.5` — **with a space**, not a hyphen.
- **The UVA API answers only from an HPC node.** All tests here run offline against fakes. Live verification is `--check-provider` on Rivanna.
- Existing configs use bare model names and must keep working with zero edits.
- Tests live in `tests/`, use `unittest.TestCase`, and import from `proposal_reviewer` or `providers`.

---

### Task 1: `.env` loading

**Files:**
- Create: `providers.py`
- Create: `.env.example`
- Test: `tests/test_providers.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `load_dotenv(path=None, environ=None) -> int` — reads `KEY=value` lines, sets only keys absent from `environ`, returns the count set. Defaults to `.env` beside `providers.py` and `os.environ`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_providers.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_providers -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'providers'`

- [ ] **Step 3: Write minimal implementation**

```python
# providers.py
"""Model backends for proposal_reviewer.

Resolves "provider:model" strings to a chat callable, and loads secrets from
.env. Kept separate from the engine so the eventual web service can reuse it.
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_dotenv(path=None, environ=None):
    """Load KEY=value pairs from a .env file into `environ`.

    Only sets keys that are absent, so a real environment variable (SLURM job
    script, systemd unit, Kubernetes secret) always beats a stale .env file.
    Returns the number of keys set. A missing file is not an error."""
    path = Path(path) if path else DEFAULT_ENV_PATH
    environ = os.environ if environ is None else environ
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in environ:
            environ[key] = value
            n += 1
    return n
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_providers -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Create `.env.example`**

```bash
cat > .env.example <<'EOF'
# Copy to .env and fill in. .env is gitignored — never commit a real key.
#
# UVA RC GenAI key: log in to https://open-webui.rc.virginia.edu/ with NetBadge,
# then Settings -> Account -> API Keys.
#
# NOTE: this API only answers from an HPC node (Rivanna/Afton). A VPN-connected
# laptop gets HTTP 401 even with a valid key.
UVARC_GenAI_API=
EOF
```

- [ ] **Step 6: Commit**

```bash
git add providers.py .env.example tests/test_providers.py
git commit -m "feat: load secrets from .env with real-environment precedence"
```

---

### Task 2: Resolve `provider:model` references

**Files:**
- Modify: `providers.py`
- Test: `tests/test_providers.py`

**Interfaces:**
- Consumes: `load_dotenv` from Task 1.
- Produces:
  - `BUILTIN_PROVIDERS = {"ollama": {"kind": "ollama", "base_url": "http://localhost:11434/v1/"}}`
  - `known_provider_names(cfg) -> set[str]`
  - `split_model_ref(ref, known) -> (provider_name, model_name)`
  - `provider_spec(cfg, name) -> dict` — raises `ProviderError` on unknown name or literal `api_key`.
  - `class ProviderError(Exception)`

**Why the prefix must be checked against known names:** `qwen2.5:32b` is a legitimate bare Ollama model containing a colon. Splitting unconditionally on the first colon would resolve it to provider `qwen2.5`. So a prefix only counts as a provider when it names a declared or builtin one.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_providers.py, above the __main__ block
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_providers -v`
Expected: FAIL with `ImportError: cannot import name 'split_model_ref'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to providers.py

BUILTIN_PROVIDERS = {
    "ollama": {"kind": "ollama", "base_url": "http://localhost:11434/v1/"},
}


class ProviderError(Exception):
    """Configuration or connectivity problem with a model backend."""


def known_provider_names(cfg):
    """Provider names this config can resolve: declared ones plus builtins."""
    return set(BUILTIN_PROVIDERS) | set((cfg or {}).get("providers") or {})


def split_model_ref(ref, known):
    """'uva:Kimi K2.5' -> ('uva', 'Kimi K2.5').

    A prefix only counts as a provider when it names a known one — otherwise
    the whole string is an Ollama model, so 'qwen2.5:32b' stays intact."""
    prefix, sep, rest = ref.partition(":")
    if sep and prefix in known:
        return prefix, rest
    return "ollama", ref


def provider_spec(cfg, name):
    """Merged spec for a provider. Raises ProviderError on unknown names or on
    a literal key in the config."""
    declared = ((cfg or {}).get("providers") or {}).get(name)
    if declared is None and name not in BUILTIN_PROVIDERS:
        available = ", ".join(sorted(known_provider_names(cfg)))
        raise ProviderError(
            f"Unknown provider {name!r}. Declare it under `providers:` in the "
            f"config, or use one of: {available}")
    spec = dict(BUILTIN_PROVIDERS.get(name, {}))
    spec.update(declared or {})
    if "api_key" in spec:
        raise ProviderError(
            f"Provider {name!r} sets `api_key` directly. Configs are committed "
            f"to git, so keys must not live in them. Use `api_key_env: VARNAME` "
            f"and put the value in .env instead.")
    return spec
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_providers -v`
Expected: PASS (14 tests)

- [ ] **Step 5: Commit**

```bash
git add providers.py tests/test_providers.py
git commit -m "feat: resolve provider:model refs against known provider names"
```

---

### Task 3: OpenAI-compatible chat function

**Files:**
- Modify: `providers.py`
- Test: `tests/test_providers.py`

**Interfaces:**
- Consumes: `provider_spec`, `ProviderError` from Task 2.
- Produces: `make_chat_fn(cfg, provider_name, temperature=0.2, transport=None) -> callable`.
  The returned callable has signature `(model, system_prompt, user_prompt, timeout=900, num_ctx=None) -> str`, matching what `_chat_json` calls.
  `transport(url, payload_bytes, headers, timeout) -> bytes` is injectable for testing; it defaults to `urllib`.

**Behavioral requirements:**
- `kind: "ollama"` includes `options.num_ctx` when `num_ctx` is not None; `kind: "openai"` never includes it (context length is server-side).
- `openai` sends `Authorization: Bearer <value of api_key_env>`; a missing/empty env var raises `ProviderError` before any network call.
- HTTP 401 → `ProviderError` naming the HPC-node requirement.
- HTTP 429 → retry up to 3 attempts with 1s/2s backoff, then `ProviderError`.
- Connection failure → `ProviderError` naming the host.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_providers.py, above the __main__ block
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
```

Add `import urllib.error` to the test file's imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_providers -v`
Expected: FAIL with `ImportError: cannot import name 'make_chat_fn'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to providers.py

MAX_HTTP_ATTEMPTS = 3


def _urllib_transport(url, body, headers, timeout):
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def make_chat_fn(cfg, provider_name, temperature=0.2, transport=None,
                 environ=None, sleep=None):
    """Build a chat callable for one provider.

    Returned signature is (model, system_prompt, user_prompt, timeout=900,
    num_ctx=None) -> str, matching what _chat_json expects."""
    spec = provider_spec(cfg, provider_name)
    transport = transport or _urllib_transport
    environ = os.environ if environ is None else environ
    sleep = sleep or time.sleep
    kind = spec.get("kind", "ollama")
    base = spec.get("base_url", "")
    if not base.endswith("/"):
        base += "/"
    url = base + "chat/completions"
    host = re.sub(r"^https?://([^/]+).*$", r"\1", base)

    headers = {"Content-Type": "application/json"}
    if kind == "openai":
        var = spec.get("api_key_env")
        if not var:
            raise ProviderError(
                f"Provider {provider_name!r} is kind 'openai' but sets no "
                f"`api_key_env`.")
        key = (environ.get(var) or "").strip()
        if not key:
            raise ProviderError(
                f"Provider {provider_name!r} needs ${var}, which is unset or "
                f"empty. Put it in .env (see .env.example) or export it.")
        headers["Authorization"] = f"Bearer {key}"

    def chat(model, system_prompt, user_prompt, timeout=900, num_ctx=None):
        payload = {"model": model, "temperature": temperature,
                   "messages": [{"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}]}
        if kind == "ollama" and num_ctx is not None:
            payload["options"] = {"num_ctx": num_ctx}
        body = json.dumps(payload).encode()

        for attempt in range(MAX_HTTP_ATTEMPTS):
            try:
                raw = transport(url, body, dict(headers), timeout)
                return json.loads(raw)["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    raise ProviderError(
                        f"{provider_name}: key rejected by {host} (HTTP 401). "
                        f"Check the value in .env — and confirm you are on an "
                        f"HPC node: this API returns 401 from a VPN-connected "
                        f"laptop even with a valid key.") from e
                if e.code == 429 and attempt < MAX_HTTP_ATTEMPTS - 1:
                    sleep(2 ** attempt)
                    continue
                if e.code == 429:
                    raise ProviderError(
                        f"{provider_name}: rate limited by {host} after "
                        f"{MAX_HTTP_ATTEMPTS} attempts (HTTP 429).") from e
                if e.code >= 500 and attempt < MAX_HTTP_ATTEMPTS - 1:
                    sleep(2 ** attempt)
                    continue
                raise ProviderError(
                    f"{provider_name}: {host} returned HTTP {e.code}.") from e
            except urllib.error.URLError as e:
                raise ProviderError(
                    f"{provider_name}: cannot reach {host} ({e.reason}). "
                    f"On an HPC node? Is Ollama running?") from e
        raise ProviderError(f"{provider_name}: exhausted retries against {host}.")

    return chat
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_providers -v`
Expected: PASS (21 tests)

- [ ] **Step 5: Commit**

```bash
git add providers.py tests/test_providers.py
git commit -m "feat: OpenAI-compatible chat backend with actionable error mapping"
```

---

### Task 4: Slot labels — fix the duplicate-model collision

**Files:**
- Modify: `proposal_reviewer.py:523-569` (`main`)
- Create test: `tests/test_slots.py`

**Interfaces:**
- Produces: `slot_labels(models) -> [str, str]` in `proposal_reviewer.py`.

**Why this exists:** `main()` keys `all_results` by model name, and `build_combined` does `ra, rb = all_results[A], all_results[B]`. With `models: ["uva:Kimi K2.5", "uva:Kimi K2.5"]` the second run overwrites the first, both per-model report files collide on one filename, and the combined report compares a result against itself. Labelling the two slots distinctly fixes all three at once and leaves `build_combined` and its existing tests untouched.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_slots.py
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proposal_reviewer import slot_labels


class TestSlotLabels(unittest.TestCase):
    def test_distinct_models_keep_their_names(self):
        self.assertEqual(slot_labels(["gemma3:27b", "qwen2.5:32b"]),
                         ["gemma3:27b", "qwen2.5:32b"])

    def test_identical_models_get_sample_suffixes(self):
        self.assertEqual(slot_labels(["uva:Kimi K2.5", "uva:Kimi K2.5"]),
                         ["uva:Kimi K2.5 (sample 1)", "uva:Kimi K2.5 (sample 2)"])

    def test_labels_are_always_distinct(self):
        for pair in (["a", "a"], ["a", "b"]):
            labels = slot_labels(pair)
            self.assertEqual(len(set(labels)), 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_slots -v`
Expected: FAIL with `ImportError: cannot import name 'slot_labels'`

- [ ] **Step 3: Write minimal implementation**

Add to `proposal_reviewer.py` just above `def main():`

```python
def slot_labels(models):
    """Distinct display keys for the two ensemble slots.

    Results are keyed by label, so two samples of the same model must not
    collide — otherwise the second overwrites the first and the combined
    report compares a result against itself."""
    a, b = models
    if a == b:
        return [f"{a} (sample 1)", f"{b} (sample 2)"]
    return [a, b]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_slots -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add proposal_reviewer.py tests/test_slots.py
git commit -m "fix: distinct slot labels so duplicate models don't collide"
```

---

### Task 5: Wire the provider factory into the engine

**Files:**
- Modify: `proposal_reviewer.py:30` (drop `OLLAMA_URL`), `:284-295` (`_ollama_chat`), `:174-198` (`review_with_model`), `:360-383` (`synthesize_recommendations`), `:523-569` (`main`)
- Test: `tests/test_providers.py`

**Interfaces:**
- Consumes: `make_chat_fn`, `split_model_ref`, `known_provider_names`, `load_dotenv`, `ProviderError` from Tasks 1–3; `slot_labels` from Task 4.
- Produces: `resolve_chat_fns(cfg, models) -> [(label, bare_model, chat_fn), ...]` in `proposal_reviewer.py`, one entry per slot, in order.

**Note on temperature:** slot 0 gets 0.2 (the current value). Slot 1 gets 0.2 as well **for now** — the raise to 0.6 lands in Task 6, where the single-family condition that justifies it is implemented.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_providers.py, above the __main__ block
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_providers -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_chat_fns'`

- [ ] **Step 3: Replace `_ollama_chat` and add the resolver**

In `proposal_reviewer.py`, delete the `OLLAMA_URL` constant at line 30 and replace the body of `_ollama_chat` (lines 284–295) with a delegation, so existing default arguments keep working:

```python
def _ollama_chat(model, system_prompt, user_prompt, timeout=900, num_ctx=DEFAULT_NUM_CTX):
    """Single local Ollama chat call; returns the message content string.
    Retained as the default chat_fn for callers that don't resolve a provider."""
    return providers.make_chat_fn({}, "ollama")(
        model, system_prompt, user_prompt, timeout=timeout, num_ctx=num_ctx)
```

Add near the top imports:

```python
import providers
from providers import ProviderError
```

Add above `def main():`

```python
def resolve_chat_fns(cfg, models, environ=None, temperatures=(0.2, 0.2)):
    """One (label, bare_model, chat_fn) per ensemble slot, in order.

    The label is the display/results key; the bare model is what gets sent to
    the backend with any provider prefix stripped."""
    known = providers.known_provider_names(cfg)
    labels = slot_labels(models)
    slots = []
    for i, ref in enumerate(models):
        pname, bare = providers.split_model_ref(ref, known)
        fn = providers.make_chat_fn(cfg, pname, temperature=temperatures[i],
                                    environ=environ)
        slots.append((labels[i], bare, fn))
    return slots
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_providers -v`
Expected: PASS

- [ ] **Step 5: Rewrite `main()`'s review loop**

Replace lines 546–553 of `proposal_reviewer.py` (the `all_results = {}` loop) with:

```python
    providers.load_dotenv()
    try:
        slots = resolve_chat_fns(cfg, models)
    except ProviderError as e:
        sys.exit(str(e))

    all_results = {}
    for label, bare, chat_fn in slots:
        print(f"Reviewing with {label} ...", file=sys.stderr)
        try:
            res, report = review_with_model(bare, cfg, sections, full_text,
                                            chat_fn=chat_fn)
        except ProviderError as e:
            sys.exit(str(e))
        all_results[label] = res
        safe = label.replace(":", "_").replace("/", "_").replace(" ", "_")
        (outdir / stamped(f"compliance_report_{safe}", ts, "md")).write_text(report)

    labels = [s[0] for s in slots]
```

Then change the two calls below it to use `labels` instead of `models`:

```python
    recs = synthesize_recommendations(cfg, labels, all_results,
                                      chat_fn=slots[0][2])
    combined = build_combined(cfg, labels, all_results, kw_pct, kw_missing, recs)
```

Note `synthesize_recommendations` currently defaults `chat_fn=_ollama_chat` and takes `model = cfg.get("synthesis_model") or models[0]`. Since `labels[0]` may now carry a `(sample 1)` suffix, change that line inside `synthesize_recommendations` to accept an explicit model:

```python
def synthesize_recommendations(cfg, models, all_results, chat_fn=_ollama_chat,
                               synth_model=None):
    ...
    model = synth_model or cfg.get("synthesis_model") or models[0]
```

and pass `synth_model=slots[0][1]` from `main()`.

- [ ] **Step 6: Run the full suite**

Run: `python -m unittest discover -s tests -v`
Expected: PASS — all pre-existing tests still green.

- [ ] **Step 7: Verify no config changes are needed**

Run: `python proposal_reviewer.py README.md --config configs/cdc-ip-26-0102.yaml --dry-run`
Expected: dry-run output as before, no provider errors. (README.md is a stand-in document; we only need the config to load and sections to map.)

- [ ] **Step 8: Commit**

```bash
git add proposal_reviewer.py tests/test_providers.py
git commit -m "feat: route model calls through the provider factory"
```

---

### Task 6: Detect single-family runs and vary the second sample

**Files:**
- Modify: `proposal_reviewer.py` (`resolve_chat_fns` call in `main`)
- Test: `tests/test_slots.py`

**Interfaces:**
- Produces: `is_single_family(cfg, models) -> bool` in `proposal_reviewer.py`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_slots.py, above the __main__ block
from proposal_reviewer import is_single_family

CFG = {"providers": {"uva": {"kind": "openai",
                             "base_url": "https://example.invalid/api/",
                             "api_key_env": "TEST_KEY"}}}


class TestSingleFamily(unittest.TestCase):
    def test_same_provider_and_model_is_single_family(self):
        self.assertTrue(is_single_family(CFG, ["uva:Kimi K2.5", "uva:Kimi K2.5"]))

    def test_different_models_is_not(self):
        self.assertFalse(is_single_family(CFG, ["gemma3:27b", "qwen2.5:32b"]))

    def test_same_model_different_providers_is_not(self):
        cfg = {"providers": {"uva": CFG["providers"]["uva"],
                             "other": {"kind": "openai",
                                       "base_url": "https://other.invalid/api/",
                                       "api_key_env": "TEST_KEY"}}}
        self.assertFalse(is_single_family(cfg, ["uva:qwen", "other:qwen"]))

    def test_prefixed_and_bare_same_ollama_model_is_single_family(self):
        self.assertTrue(is_single_family({}, ["ollama:gemma3:27b", "gemma3:27b"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_slots -v`
Expected: FAIL with `ImportError: cannot import name 'is_single_family'`

- [ ] **Step 3: Write minimal implementation**

```python
def is_single_family(cfg, models):
    """True when both ensemble slots resolve to the same provider AND model.

    Two samples of one model are a reliability check, not two independent
    opinions — the combined report must not claim corroboration."""
    known = providers.known_provider_names(cfg)
    a = providers.split_model_ref(models[0], known)
    b = providers.split_model_ref(models[1], known)
    return a == b
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_slots -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Raise the second sample's temperature in `main()`**

Replace the `resolve_chat_fns(cfg, models)` call added in Task 5 with:

```python
    single = is_single_family(cfg, models)
    temps = (0.2, 0.6) if single else (0.2, 0.2)
    try:
        slots = resolve_chat_fns(cfg, models, temperatures=temps)
    except ProviderError as e:
        sys.exit(str(e))
```

At 0.2 two samples of one model risk being near-identical, which makes the second call pure cost. Sampling wider is what makes "is this score stable?" a real question.

- [ ] **Step 6: Run the full suite**

Run: `python -m unittest discover -s tests -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add proposal_reviewer.py tests/test_slots.py
git commit -m "feat: detect single-family runs and widen the second sample"
```

---

### Task 7: Honest report wording under single-family

**Files:**
- Modify: `proposal_reviewer.py:413-468` (`build_combined`), and its call site in `main`
- Test: `tests/test_recommendations.py` (extend `TestBuildCombinedIntegration`)

**Interfaces:**
- Consumes: `is_single_family` from Task 6.
- Produces: `build_combined(cfg, models, all_results, kw_pct, kw_missing, recs, single_family=False)` — new keyword-only-by-default trailing parameter, so every existing caller and test keeps working.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_recommendations.py inside TestBuildCombinedIntegration
    def test_default_wording_claims_two_models(self):
        cfg, models, results, recs = self._cfg_results_recs()
        md = build_combined(cfg, models, results, 100.0, [], recs)
        self.assertIn("both models dock points", md)
        self.assertIn("models diverge", md)
        self.assertNotIn("no independent second opinion", md)

    def test_single_family_relabels_and_warns(self):
        cfg, models, results, recs = self._cfg_results_recs()
        md = build_combined(cfg, models, results, 100.0, [], recs,
                            single_family=True)
        self.assertIn("consistent across both samples", md)
        self.assertIn("unstable score", md)
        self.assertIn("consistently scored", md)
        self.assertIn("no independent second opinion", md)
        self.assertNotIn("both models dock points", md)

    def test_single_family_keeps_section_numbering(self):
        cfg, models, results, recs = self._cfg_results_recs()
        md = build_combined(cfg, models, results, 100.0, [], recs,
                            single_family=True)
        for heading in ("## 1. Overall scores", "## 3. Per-criterion",
                        "## 4. Priority gaps", "## 5. Strengths",
                        "## 6. Missing RFA phrases", "## 7. Final"):
            self.assertIn(heading, md)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_recommendations -v`
Expected: FAIL — `build_combined() got an unexpected keyword argument 'single_family'`

- [ ] **Step 3: Write minimal implementation**

Change the signature at line 413 and add a wording table at the top of the function body:

```python
def build_combined(cfg, models, all_results, kw_pct, kw_missing, recs,
                   single_family=False):
    A, B = models
    ra, rb = all_results[A], all_results[B]
    ga, gb = section_subtotals(cfg, ra), section_subtotals(cfg, rb)
    ta, tb = sum(r["awarded"] for r in ra), sum(r["awarded"] for r in rb)
    total = cfg.get("total_points", sum(c["max"] for c in cfg["criteria"]))
    section_totals = cfg.get("section_totals") or {g: ga[g][1] for g in ga}

    if single_family:
        provenance = "UVA RC GenAI — single model, two samples"
        per_crit_hd = "Per-criterion scores (★ = unstable score, model is guessing)"
        gaps_hd = "Priority gaps — consistent across both samples (ranked by points at stake)"
        strengths_hd = "Strengths — consistently scored ≥80% of max"
        banner = ("\n> **No independent second opinion.** Both columns are the same "
                  "model sampled twice, so agreement between them reflects "
                  "consistency, not corroboration. Shared blind spots stay "
                  "invisible here.\n")
    else:
        provenance = "offline (local models)"
        per_crit_hd = "Per-criterion scores (★ = models diverge ≥2 pts)"
        gaps_hd = "Priority gaps — both models dock points (ranked by points at stake)"
        strengths_hd = "Strengths — both models score ≥80% of max"
        banner = ""
```

Then replace the four literal strings further down:

```python
    L = [f"# Combined Compliance Assessment: {A} vs {B}",
         f"\n_{cfg.get('name','')} — generated {datetime.date.today()}, {provenance}._\n",
         banner,
         "---\n## 1. Overall scores\n", f"| Section | {A} | {B} |", "|---|---|---|"]
```

```python
    L.append(f"\n## 3. {per_crit_hd}\n")
```

```python
    L.append(f"\n## 4. {gaps_hd}\n")
```

```python
    L.append(f"\n## 5. {strengths_hd}\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_recommendations -v`
Expected: PASS

- [ ] **Step 5: Pass the flag from `main()`**

```python
    combined = build_combined(cfg, labels, all_results, kw_pct, kw_missing, recs,
                              single_family=single)
```

- [ ] **Step 6: Run the full suite**

Run: `python -m unittest discover -s tests -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add proposal_reviewer.py tests/test_recommendations.py
git commit -m "feat: relabel combined report when both slots are one model"
```

---

### Task 8: `--check-provider`

**Files:**
- Modify: `proposal_reviewer.py` (`main`, argparse)
- Test: `tests/test_providers.py`

**Interfaces:**
- Consumes: `resolve_chat_fns`, `ProviderError`.
- Produces: `check_providers(cfg, models, environ=None) -> [(label, ok: bool, detail: str)]`.

This is the command that tells you whether a given machine is inside the security zone. It must not require a proposal file.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_providers.py, above the __main__ block
from proposal_reviewer import check_providers


class TestCheckProviders(unittest.TestCase):
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_providers -v`
Expected: FAIL with `ImportError: cannot import name 'check_providers'`

- [ ] **Step 3: Write minimal implementation**

`resolve_chat_fns` needs a `transport` passthrough for this to be testable. Add the parameter and forward it:

```python
def resolve_chat_fns(cfg, models, environ=None, temperatures=(0.2, 0.2),
                     transport=None):
    known = providers.known_provider_names(cfg)
    labels = slot_labels(models)
    slots = []
    for i, ref in enumerate(models):
        pname, bare = providers.split_model_ref(ref, known)
        fn = providers.make_chat_fn(cfg, pname, temperature=temperatures[i],
                                    environ=environ, transport=transport)
        slots.append((labels[i], bare, fn))
    return slots
```

Then:

```python
def check_providers(cfg, models, environ=None, transport=None):
    """One cheap call per slot. Returns (label, ok, detail) rows; never raises.

    This is how you confirm a machine can reach its backends — in particular
    whether it is inside the HPC security zone the UVA API requires."""
    rows = []
    known = providers.known_provider_names(cfg)
    labels = slot_labels(models)
    for label, ref in zip(labels, models):
        pname, bare = providers.split_model_ref(ref, known)
        try:
            fn = providers.make_chat_fn(cfg, pname, environ=environ,
                                        transport=transport)
            reply = fn(bare, "Reply with the single word: ok.", "ping",
                       timeout=60)
            rows.append((label, True, f"{pname} -> {bare}: "
                                      f"{reply.strip()[:40] or '(empty reply)'}"))
        except ProviderError as e:
            rows.append((label, False, str(e)))
        except Exception as e:                      # noqa: BLE001 - report, don't crash
            rows.append((label, False, f"{type(e).__name__}: {e}"))
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_providers -v`
Expected: PASS

- [ ] **Step 5: Wire the flag**

In `main()`'s argparse block, make `proposal` optional and add the flag:

```python
    ap.add_argument("proposal", nargs="?")
    ap.add_argument("--check-provider", action="store_true",
                    help="Test connectivity and auth for each configured model, then exit")
```

Immediately after `models` validation (before the proposal is read):

```python
    if args.check_provider:
        providers.load_dotenv()
        rows = check_providers(cfg, models)
        for label, ok, detail in rows:
            print(f"{'OK  ' if ok else 'FAIL'}  {label}: {detail}")
        sys.exit(0 if all(ok for _, ok, _ in rows) else 1)

    if not args.proposal:
        sys.exit("PROPOSAL is required (omit it only with --check-provider).")
```

- [ ] **Step 6: Verify the flag works against Ollama locally**

Run: `python proposal_reviewer.py --config configs/cdc-ip-26-0102.yaml --check-provider`
Expected: with Ollama running, two `OK` lines and exit 0. With Ollama stopped, two `FAIL` lines naming `localhost:11434` and exit 1. **Confirm both.**

- [ ] **Step 7: Commit**

```bash
git add proposal_reviewer.py tests/test_providers.py
git commit -m "feat: --check-provider for connectivity and auth diagnosis"
```

---

### Task 9: Config, docs, and the invariant rewrite

**Files:**
- Create: `configs/uva-kimi-example.yaml`
- Modify: `CLAUDE.md`, `README.md`

- [ ] **Step 1: Write the example config**

Copy an existing config's criteria wholesale so the file is runnable, and add the provider block. Start from `configs/arpa-h-igor.yaml`:

```bash
sed 's/^models: .*/models: ["uva:Kimi K2.5", "uva:Kimi K2.5"]/' \
    configs/arpa-h-igor.yaml > configs/uva-kimi-example.yaml
```

Then insert above the `models:` line:

```yaml
providers:
  uva:
    kind: openai
    base_url: https://open-webui.rc.virginia.edu/api/
    api_key_env: UVARC_GenAI_API
```

- [ ] **Step 2: Verify it loads**

Run: `python proposal_reviewer.py README.md --config configs/uva-kimi-example.yaml --dry-run`
Expected: dry-run output with `Models: ['uva:Kimi K2.5', 'uva:Kimi K2.5']`, no provider errors (dry-run makes no calls).

- [ ] **Step 3: Rewrite the `CLAUDE.md` invariant**

Replace the "Core invariant: everything runs locally" section with:

```markdown
## Core invariant: drafts never reach a commercial provider

An unpublished proposal draft must never be sent to a commercial third party.
Two backends satisfy this: a local Ollama, and UVA's RC GenAI service, which is
UVA-hosted and stays inside the institution. **Do not add OpenAI, Anthropic,
Google, or any other commercial API provider to the engine.**

The earlier form of this rule said "everything runs on localhost." That was a
proxy for the requirement above, and it ruled out UVA-hosted inference for no
good reason.

**UVA RC GenAI answers only from an HPC node.** A VPN-connected laptop gets
HTTP 401 from Apache even with a valid key — the proxy rejects the request
before Open WebUI evaluates the token. Browser access works from anywhere via
NetBadge, so browser success does not imply API access works. Run
`--check-provider` on the target machine to confirm.
```

Add to the Conventions section:

```markdown
- Two models per config, ideally different families. When both slots hold the
  same model, the engine sets `single_family` and rewrites the combined report's
  language — two samples are a reliability check, not corroboration. Never
  reword those sections back to claiming two independent opinions.
- Secrets live in `.env` (gitignored); configs name the variable via
  `api_key_env` and never hold a value. The engine rejects a literal `api_key:`.
```

- [ ] **Step 4: Update `README.md`**

Add after the Install section:

```markdown
## Model backends

Model names may carry a provider prefix. A bare name means local Ollama, so
existing configs need no changes.

    models: ["gemma3:27b", "qwen2.5:32b"]          # both local Ollama
    models: ["uva:Kimi K2.5", "ollama:qwen2.5:32b"]  # hybrid

Non-local providers are declared in the config and take their key from the
environment — never from the file:

    providers:
      uva:
        kind: openai
        base_url: https://open-webui.rc.virginia.edu/api/
        api_key_env: UVARC_GenAI_API

Copy `.env.example` to `.env` and fill in the key. `.env` is gitignored.

**UVA RC GenAI only answers from an HPC node.** From a VPN-connected laptop it
returns HTTP 401 even with a valid key. Check any machine with:

    python proposal_reviewer.py --config CONFIG --check-provider

UVA hosts a single model, so a UVA-only config runs the same model twice. The
engine detects this and relabels the combined report: two samples measure
whether a score is *stable*, not whether two independent reviewers agree.
```

- [ ] **Step 5: Run the full suite one last time**

Run: `python -m unittest discover -s tests -v`
Expected: PASS, all tests.

- [ ] **Step 6: Commit**

```bash
git add configs/uva-kimi-example.yaml CLAUDE.md README.md
git commit -m "docs: provider config, .env secrets, and the revised invariant"
```

---

## Live verification (Rivanna only)

None of the tasks above touch the network. After Task 9, on a Rivanna node:

- [ ] Create `.env` there with a real key (it does not travel through git), `chmod 600`.
- [ ] Run `python proposal_reviewer.py --config configs/uva-kimi-example.yaml --check-provider`. Expected: two `OK` lines.
- [ ] Run the same command from a VPN-connected laptop. Expected: two `FAIL` lines whose text names the HPC-node requirement. This confirms the error message earns its keep.
- [ ] Run a full review against a real proposal and read the combined report's banner and section headings to confirm the single-family wording reads correctly in context.
