# JSON-Retry + Timestamped Outputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retry model calls whose reply fails to parse as JSON (with a corrective nudge), and give every run a distinct, timestamped set of output files so runs never overwrite each other.

**Architecture:** A new `_chat_json` helper centralizes "call model → strip fences → json.loads → validate", retrying with a corrective nudge on parse/validate failure and raising `JSONRetryError` after the attempt budget. `ask_model` and `synthesize_recommendations` route through it, keeping their existing degraded fallbacks for the exhausted case. `main` computes one timestamp per run and suffixes every output filename via a small `stamped` helper.

**Tech Stack:** Python 3 stdlib (`urllib`, `json`, `re`, `datetime`), PyYAML. Tests use stdlib `unittest`.

## Global Constraints

- **Offline only:** model calls go to the local Ollama endpoint `OLLAMA_URL`. No cloud/API calls. Tests inject `chat_fn` and never hit the network.
- **Stdlib-only except PyYAML.** Tests use stdlib `unittest`, NOT pytest.
- **Retry covers parse/validate failures ONLY** — network errors from `chat_fn`/`_ollama_chat` must propagate, not be retried (a 900s timeout must not be retried 3×).
- **`MAX_JSON_ATTEMPTS = 3`** total attempts (1 initial + 2 retries). Corrective nudge appended on retries.
- **Timestamp format:** `YYYYMMDD-HHMMSS`, one shared timestamp per run, suffixed on every output file.
- **Preserve existing degraded behavior:** exhausted reviewer call → the 0-score "UNPARSED MODEL OUTPUT" dict; exhausted/failed synthesis → `deterministic_recommendations`.
- Repo is under git on `main`. Each task ends by running the suite, then committing with the trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- `python3` is the interpreter (no `python`). Test command (repo root): `python3 -m unittest tests.test_recommendations -v`

## File Structure

- `proposal_reviewer.py` — add `JSONRetryError`, `MAX_JSON_ATTEMPTS`, `_JSON_NUDGE`, `_chat_json` (after `_ollama_chat`); refactor `ask_model`; refactor the try-block in `synthesize_recommendations`; add `stamped` helper; rewrite the output-writing block in `main`.
- `tests/test_recommendations.py` — append new test classes (existing 25 tests must keep passing).

**Ordering note (critical):** `ask_model` (~line 108) is defined BEFORE `_ollama_chat`/`_chat_json` (~line 230+). A default like `chat_fn=_ollama_chat` on `ask_model` would raise `NameError` at definition time. So `ask_model` uses `chat_fn=None` and resolves `_ollama_chat` at call time inside the body. `_chat_json` and `synthesize_recommendations` are defined AFTER `_ollama_chat`, so their `chat_fn=_ollama_chat` defaults are fine.

---

### Task 1: `_chat_json` retry helper

**Files:**
- Modify: `proposal_reviewer.py` (insert immediately after `_ollama_chat`, before `def _digest_to_prompt`)
- Test: `tests/test_recommendations.py`

**Interfaces:**
- Consumes: `_ollama_chat` (existing), module globals `re`, `json`, `sys`.
- Produces:
  - `class JSONRetryError(Exception)` with attribute `.raw` (str, last unparseable content).
  - `MAX_JSON_ATTEMPTS = 3`
  - `_chat_json(model, system, user, validate, attempts=MAX_JSON_ATTEMPTS, chat_fn=_ollama_chat)` → returns `validate(parsed)`; raises `JSONRetryError(raw)` when no attempt yields valid JSON. `validate(parsed)` returns the cleaned result or raises to trigger a retry. Network exceptions from `chat_fn` propagate (not retried).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_recommendations.py` (before the `if __name__` block):

```python
from proposal_reviewer import _chat_json, JSONRetryError, MAX_JSON_ATTEMPTS


class TestChatJSONRetry(unittest.TestCase):
    def test_returns_validated_on_first_success(self):
        calls = []
        def fake(m, s, u):
            calls.append(u)
            return '{"awarded": 5}'
        out = _chat_json("m", "sys", "user", validate=lambda d: d, chat_fn=fake)
        self.assertEqual(out, {"awarded": 5})
        self.assertEqual(len(calls), 1)

    def test_retries_then_succeeds_with_nudge(self):
        calls = []
        seq = ["not json", '{"ok": true}']
        def fake(m, s, u):
            calls.append(u)
            return seq[len(calls) - 1]
        out = _chat_json("m", "sys", "USERPROMPT", validate=lambda d: d, chat_fn=fake)
        self.assertEqual(out, {"ok": True})
        self.assertEqual(len(calls), 2)
        self.assertIn("USERPROMPT", calls[1])          # original prompt retained
        self.assertIn("not valid JSON", calls[1])      # corrective nudge appended on retry

    def test_raises_after_exhausting_attempts(self):
        calls = []
        def fake(m, s, u):
            calls.append(u)
            return "still not json"
        with self.assertRaises(JSONRetryError) as ctx:
            _chat_json("m", "sys", "user", validate=lambda d: d, chat_fn=fake)
        self.assertEqual(len(calls), MAX_JSON_ATTEMPTS)
        self.assertEqual(ctx.exception.raw, "still not json")

    def test_validate_rejection_triggers_retry(self):
        calls = []
        seq = ['{"x": 1}', '[1, 2]']
        def fake(m, s, u):
            calls.append(u)
            return seq[len(calls) - 1]
        def validate(parsed):
            if not isinstance(parsed, list):
                raise ValueError("need a list")
            return parsed
        out = _chat_json("m", "sys", "user", validate=validate, chat_fn=fake)
        self.assertEqual(out, [1, 2])
        self.assertEqual(len(calls), 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_recommendations -v`
Expected: FAIL — `ImportError: cannot import name '_chat_json'`

- [ ] **Step 3: Write minimal implementation**

Insert into `proposal_reviewer.py` immediately after `_ollama_chat` (after its `return ...["content"]` line, before `def _digest_to_prompt`):

```python
class JSONRetryError(Exception):
    """Raised when a model never returns parseable, valid JSON within the attempt budget.
    `.raw` holds the last unparseable content for the caller's fallback message."""
    def __init__(self, raw):
        self.raw = raw
        super().__init__("model did not return valid JSON")


MAX_JSON_ATTEMPTS = 3
_JSON_NUDGE = ("\n\nIMPORTANT: your previous reply was NOT valid JSON. "
               "Respond with ONLY the JSON, no prose and no code fences.")


def _chat_json(model, system, user, validate, attempts=MAX_JSON_ATTEMPTS, chat_fn=_ollama_chat):
    """Call the model and parse its reply as JSON, retrying with a corrective nudge on
    parse/validate failure. `validate(parsed)` returns the cleaned result or raises to
    trigger a retry. Network errors from chat_fn propagate (NOT retried here).
    Raises JSONRetryError(last_raw) if no attempt yields valid JSON."""
    raw = ""
    for i in range(attempts):
        prompt = user if i == 0 else user + _JSON_NUDGE
        raw = chat_fn(model, system, prompt)
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            return validate(json.loads(cleaned))
        except Exception:
            print(f"  {model}: unparseable JSON (attempt {i + 1}/{attempts})", file=sys.stderr)
            continue
    raise JSONRetryError(raw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_recommendations -v`
Expected: PASS (`TestChatJSONRetry` 4 tests + existing 25)

- [ ] **Step 5: Commit**

```bash
git add proposal_reviewer.py tests/test_recommendations.py
git commit -m "Add _chat_json: JSON-parse retry with corrective nudge

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Route `ask_model` through `_chat_json`

**Files:**
- Modify: `proposal_reviewer.py` — `ask_model` (currently ~lines 108-130)
- Test: `tests/test_recommendations.py`

**Interfaces:**
- Consumes: `_chat_json`, `JSONRetryError` (Task 1); `_ollama_chat` (existing).
- Produces: `ask_model(model, system_prompt, crit, sect_title, body, rfa_context="", chat_fn=None)` — unchanged return contract: a dict with `awarded` (clamped 0..max), `max`, and on exhaustion the 0-score dict with `justification` starting `"UNPARSED MODEL OUTPUT:\n"`. `chat_fn=None` resolves to `_ollama_chat` at call time.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_recommendations.py` (before `if __name__`):

```python
from proposal_reviewer import ask_model


class TestAskModelRetry(unittest.TestCase):
    CRIT = {"name": "Test", "look_for": "stuff", "max": 10}

    def test_parses_and_clamps_on_success(self):
        fake = lambda m, s, u: '{"awarded": 15, "strengths": [], "gaps": []}'
        res = ask_model("m", "sys", self.CRIT, "Title", "body text", chat_fn=fake)
        self.assertEqual(res["awarded"], 10)   # clamped to max
        self.assertEqual(res["max"], 10)

    def test_retries_bad_then_good(self):
        calls = []
        seq = ["garbage", '{"awarded": 7}']
        def fake(m, s, u):
            calls.append(u)
            return seq[len(calls) - 1]
        res = ask_model("m", "sys", self.CRIT, "Title", "body", chat_fn=fake)
        self.assertEqual(res["awarded"], 7)
        self.assertEqual(len(calls), 2)

    def test_unparsed_fallback_after_exhaustion(self):
        fake = lambda m, s, u: "never json"
        res = ask_model("m", "sys", self.CRIT, "Title", "body", chat_fn=fake)
        self.assertEqual(res["awarded"], 0)
        self.assertEqual(res["max"], 10)
        self.assertIn("UNPARSED MODEL OUTPUT", res["justification"])
        self.assertIn("never json", res["justification"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_recommendations.TestAskModelRetry -v`
Expected: FAIL — `TypeError: ask_model() got an unexpected keyword argument 'chat_fn'`

- [ ] **Step 3: Replace the `ask_model` body**

Replace the entire current function (from `def ask_model(...)` through its `except Exception:` fallback return) with:

```python
def ask_model(model, system_prompt, crit, sect_title, body, rfa_context="", chat_fn=None):
    if chat_fn is None:
        chat_fn = _ollama_chat
    user = (f"CRITERION: {crit['name']}  (max {crit['max']} points)\n"
            f"WHAT REVIEWERS LOOK FOR: {crit['look_for']}\n")
    if rfa_context:
        user += f"\nRELEVANT RFA CONTEXT:\n{rfa_context}\n"
    user += f"\nPROPOSAL SECTION — {sect_title}:\n{body[:14000]}"

    def coerce(d):
        d["awarded"] = max(0, min(int(d.get("awarded", 0) or 0), crit["max"]))
        d["max"] = crit["max"]
        return d

    try:
        return _chat_json(model, system_prompt, user, coerce, chat_fn=chat_fn)
    except JSONRetryError as e:
        return {"awarded": 0, "max": crit["max"], "strengths": [], "gaps": [],
                "justification": "UNPARSED MODEL OUTPUT:\n" + e.raw}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_recommendations -v`
Expected: PASS (`TestAskModelRetry` 3 tests + all earlier). The 14000-char body truncation, prompt format, temp 0.2 and num_ctx 16384 are preserved (the latter two now come from `_ollama_chat`).

- [ ] **Step 5: Commit**

```bash
git add proposal_reviewer.py tests/test_recommendations.py
git commit -m "Route ask_model through _chat_json (retry on bad JSON)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Route `synthesize_recommendations` through `_chat_json`

**Files:**
- Modify: `proposal_reviewer.py` — the `try/except` block in `synthesize_recommendations` (currently ~lines 283-292)
- Test: `tests/test_recommendations.py`

**Interfaces:**
- Consumes: `_chat_json`, `JSONRetryError` (Task 1); `deterministic_recommendations` (existing).
- Produces: unchanged signature `synthesize_recommendations(cfg, models, all_results, chat_fn=_ollama_chat)` and contract: returns up to `n` recs, or `deterministic_recommendations(digest, n)` on `JSONRetryError`/network error, or `[]` when no digest.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_recommendations.py` (before `if __name__`). This uses the existing `_sample_cfg_and_results()` helper already defined in this file:

```python
class TestSynthesisRetry(unittest.TestCase):
    def test_retries_then_succeeds_instead_of_fallback(self):
        cfg, models, results = _sample_cfg_and_results()
        seq = ["garbage",
               '[{"title":"Fix","points_at_stake":10,"criteria":[],"do":"d","why":"w"}]']
        calls = []
        def fake(m, s, u):
            calls.append(u)
            return seq[len(calls) - 1]
        recs = synthesize_recommendations(cfg, models, results, chat_fn=fake)
        self.assertEqual(recs[0]["title"], "Fix")   # model output, not deterministic fallback
        self.assertEqual(len(calls), 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_recommendations.TestSynthesisRetry -v`
Expected: FAIL — the current single-attempt code falls back to `deterministic_recommendations` on the first "garbage", so `recs[0]["title"]` is a criterion name (e.g. "Background"), not "Fix"; and `len(calls)` is 1.

- [ ] **Step 3: Replace the try/except block**

In `synthesize_recommendations`, replace this block:

```python
    try:
        content = chat_fn(model, system, user)
        cleaned = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
        recs = json.loads(cleaned)
        if not isinstance(recs, list) or not recs:
            raise ValueError("empty or non-list synthesis output")
        return recs[:n]
    except Exception as e:
        print(f"  synthesis failed ({e}); using deterministic ranking", file=sys.stderr)
        return deterministic_recommendations(digest, n)
```

with:

```python
    def _valid_recs(parsed):
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("empty or non-list synthesis output")
        return parsed

    try:
        recs = _chat_json(model, system, user, _valid_recs, chat_fn=chat_fn)
        return recs[:n]
    except Exception as e:
        print(f"  synthesis failed ({e}); using deterministic ranking", file=sys.stderr)
        return deterministic_recommendations(digest, n)
```

(The broad `except Exception` still catches `JSONRetryError` AND network errors, both routing to the deterministic fallback — preserving the existing guarantee.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_recommendations -v`
Expected: PASS. `TestSynthesisRetry` passes, and the existing `TestSynthesis` tests still pass: `test_falls_back_on_garbage` (3 garbage calls → `JSONRetryError` → fallback), `test_falls_back_on_exception` (chat_fn raises → propagates → fallback), `test_uses_model_output_when_valid`, `test_strips_code_fences`, `test_respects_top_n`, `test_empty_when_no_gaps`.

- [ ] **Step 5: Commit**

```bash
git add proposal_reviewer.py tests/test_recommendations.py
git commit -m "Route synthesis through _chat_json (retry before deterministic fallback)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Timestamped output filenames

**Files:**
- Modify: `proposal_reviewer.py` — add `stamped` helper (insert just before `def main`); rewrite the output-writing block inside `main` (currently ~lines 442-460)
- Test: `tests/test_recommendations.py`

**Interfaces:**
- Consumes: `datetime` (already imported at top of file).
- Produces: `stamped(base, ts, ext) -> str` → `f"{base}_{ts}.{ext}"`. `main` computes `ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")` once and suffixes all output files; `render()` is unchanged (the timestamp lives in the stem, so `with_suffix` still yields `..._<ts>.html` / `..._<ts>.pdf`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_recommendations.py` (before `if __name__`):

```python
from pathlib import Path as _Path
from proposal_reviewer import stamped


class TestStamped(unittest.TestCase):
    def test_composes_timestamped_name(self):
        self.assertEqual(
            stamped("compliance_report_combined", "20260622-211500", "md"),
            "compliance_report_combined_20260622-211500.md")

    def test_with_suffix_swaps_only_extension(self):
        # render() derives html/pdf from the combined .md path via with_suffix;
        # the timestamp must survive because it lives in the stem, not the extension.
        p = _Path(stamped("compliance_report_combined", "20260622-211500", "md"))
        self.assertEqual(p.with_suffix(".pdf").name,
                         "compliance_report_combined_20260622-211500.pdf")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_recommendations.TestStamped -v`
Expected: FAIL — `ImportError: cannot import name 'stamped'`

- [ ] **Step 3: Add the `stamped` helper**

Insert into `proposal_reviewer.py` immediately before `def main():`:

```python
def stamped(base, ts, ext):
    """Compose a timestamped output filename, e.g.
    stamped('compliance_report_combined', '20260622-211500', 'md')
    -> 'compliance_report_combined_20260622-211500.md'."""
    return f"{base}_{ts}.{ext}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_recommendations.TestStamped -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Wire timestamps into `main`**

Replace the current output block in `main` (from `outdir = path.parent` through the final `print(f"Wrote per-model reports, ...")`) with:

```python
    outdir = path.parent
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    all_results = {}
    for model in models:
        print(f"Reviewing with {model} ...", file=sys.stderr)
        res, report = review_with_model(model, cfg, sections, full_text)
        all_results[model] = res
        safe = model.replace(":", "_").replace("/", "_")
        (outdir / stamped(f"compliance_report_{safe}", ts, "md")).write_text(report)

    kw_text, kw_pct, kw_missing = keyword_scan(full_text, cfg.get("keywords", []))
    (outdir / stamped("compliance_report_keywords", ts, "md")).write_text(kw_text)

    print("Synthesizing recommendations ...", file=sys.stderr)
    recs = synthesize_recommendations(cfg, models, all_results)
    combined = build_combined(cfg, models, all_results, kw_pct, kw_missing, recs)
    cpath = outdir / stamped("compliance_report_combined", ts, "md")
    cpath.write_text(combined)
    rendered = render(cpath)
    print(f"Wrote per-model reports, keywords, and {cpath.name} ({rendered})", file=sys.stderr)
```

- [ ] **Step 6: Verify the module imports and CLI parses**

Run: `python3 proposal_reviewer.py --help`
Expected: argparse usage prints with no traceback (confirms no syntax/name errors in `main`).

- [ ] **Step 7: Run full suite**

Run: `python3 -m unittest tests.test_recommendations -v`
Expected: PASS (all tests).

- [ ] **Step 8: Commit**

```bash
git add proposal_reviewer.py tests/test_recommendations.py
git commit -m "Timestamp all output filenames so runs don't overwrite

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Retry strategy = corrective nudge, 3 attempts → Task 1 (`_chat_json`, `_JSON_NUDGE`, `MAX_JSON_ATTEMPTS`). ✓
- Retry scope = reviewer + synthesis → Task 2 (`ask_model`) + Task 3 (`synthesize_recommendations`). ✓
- Retry covers parse/validate only; network errors propagate → `_chat_json` calls `chat_fn` outside the try/except; tested implicitly by `synthesize`'s existing `test_falls_back_on_exception` (raise propagates → fallback). ✓
- Preserve degraded fallbacks → Task 2 returns the 0-score "UNPARSED" dict on `JSONRetryError`; Task 3 keeps the deterministic fallback. ✓
- Timestamp `YYYYMMDD-HHMMSS`, shared per run, all files → Task 4. ✓
- `render()` html/pdf get the timestamp via stem → Task 4 `test_with_suffix_swaps_only_extension`. ✓
- Offline/stdlib/unittest invariants → Global Constraints; all tests inject `chat_fn`. ✓
- Out-of-scope items (network retry, config counts, retention) → not implemented. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every test has real assertions. ✓

**Type consistency:** `_chat_json(model, system, user, validate, attempts, chat_fn)` is called identically in Tasks 2 and 3. `JSONRetryError.raw` is set in Task 1 and read in Task 2. `ask_model`'s `chat_fn=None`→`_ollama_chat` sentinel matches the ordering note. `stamped(base, ts, ext)` signature matches all four call sites in `main`. The reviewer-result dict shape (`awarded`/`max`/`strengths`/`gaps`/`justification`) is unchanged from the existing engine. ✓

**Ordering hazard checked:** `ask_model` (defined before `_ollama_chat`) uses `chat_fn=None` and resolves `_ollama_chat`/`_chat_json` at call time — no def-time NameError. `_chat_json` and `synthesize_recommendations` are defined after `_ollama_chat`, so their `chat_fn=_ollama_chat` defaults are valid. ✓
