# JSON-parse retry + timestamped output files

**Date:** 2026-06-22
**Status:** Approved design, pending implementation plan
**Component:** `proposal_reviewer.py` — `ask_model`, `synthesize_recommendations`, `main`

## Problem

Two reliability gaps surfaced while reviewing real drafts:

1. **A single malformed JSON reply silently zeros a criterion.** When a model wraps
   its JSON oddly or adds stray text, `ask_model`'s `json.loads` fails and the engine
   records `awarded: 0` with an "UNPARSED MODEL OUTPUT" note — no retry. Observed live:
   gemma3:27b returned `"awarded": 9` for criterion 3.1 (influenza/methodological
   expertise) but the reply didn't parse, so the report showed **0/10**, dragging the
   total down ~9 points. The model knew the answer; the engine threw it away.

2. **Each run overwrites the previous run's reports.** Output files use fixed names
   (`compliance_report_<model>.md`, `..._combined.md/.html/.pdf`, `..._keywords.md`),
   all written to the proposal's directory. Re-running a draft destroys the prior
   report, so drafts can't be compared across runs.

## Goal

- Retry a model call when its reply fails to parse as JSON, nudging the model to
  return valid JSON, before falling back to the existing degraded result.
- Give every run a distinct, sortable set of output files via a shared timestamp,
  so runs never overwrite each other.

## Decisions (from brainstorming)

1. **Retry strategy:** corrective nudge. Up to `MAX_JSON_ATTEMPTS = 3` total attempts
   (1 initial + 2 retries). On each retry, append a corrective instruction telling the
   model its previous reply was not valid JSON and to respond with ONLY the JSON.
2. **Retry scope:** both the reviewer call (`ask_model`) and the synthesis call
   (`synthesize_recommendations`).
3. **Retry covers parse/validate failures only**, NOT network errors. A network
   timeout from `_ollama_chat` propagates unchanged (retrying a 900 s timeout three
   times would mean ~45 min per criterion). This matches today's network behavior.
4. **Timestamp format:** `YYYYMMDD-HHMMSS`, one shared timestamp per run, suffixed on
   every output file.

## Core invariants preserved

- **Offline only:** no new network behavior; same local Ollama endpoint.
- **Stdlib-only except PyYAML:** retry/timestamp use `datetime`, `json`, `re`, `urllib`
  (all already imported). Tests use stdlib `unittest`.
- **Two-model ensemble + deterministic combine** unchanged. Retry makes the existing
  fallbacks (0-score reviewer result; deterministic synthesis list) rarer, not gone.

## Design

### A. `_chat_json` — call + parse with retry

A new helper centralizes "call model → strip code fences → `json.loads` → validate",
retrying on failure with a corrective nudge.

```python
class JSONRetryError(Exception):
    """Raised when a model never returns parseable/valid JSON within the attempt budget.
    Carries `.raw` (the last unparseable content) for the caller's fallback message."""

MAX_JSON_ATTEMPTS = 3

def _chat_json(model, system, user, validate, attempts=MAX_JSON_ATTEMPTS, chat_fn=_ollama_chat):
    ...
```

- `validate(parsed)` is a caller-supplied callback that returns the cleaned result or
  raises to trigger a retry (lets each caller enforce its own JSON shape).
- Loop up to `attempts` times: attempt 1 uses `user` as-is; retries append a corrective
  nudge, e.g. `"\n\nIMPORTANT: your previous reply was NOT valid JSON. Respond with ONLY
  the JSON, no prose and no code fences."`
- Each attempt: `content = chat_fn(model, system, user_with_nudge)`; strip fences with
  the existing regex (`^```(?:json)?|```$`, MULTILINE); `json.loads`; `validate`.
- On parse/validate exception: print a one-line stderr note
  (`f"  {model}: unparseable JSON, retry {i}/{attempts}"`) and continue.
- After all attempts fail: `raise JSONRetryError(last_raw_content)`.
- **Network exceptions from `chat_fn` are NOT caught here** — they propagate to the
  caller (preserving current per-caller network handling).
- `chat_fn` is injectable so the retry loop is unit-testable with no network.

### B. Wire into the two callers

**`ask_model`** — refactor to reuse `_ollama_chat` (DRY; identical payload: temp 0.2,
`num_ctx` 16384) via `_chat_json`:

```python
def _coerce(d):
    d["awarded"] = max(0, min(int(d.get("awarded", 0) or 0), crit["max"]))
    d["max"] = crit["max"]
    return d
try:
    return _chat_json(model, system_prompt, user, _coerce)
except JSONRetryError as e:
    return {"awarded": 0, "max": crit["max"], "strengths": [], "gaps": [],
            "justification": "UNPARSED MODEL OUTPUT:\n" + e.raw}
```

Behavior preserved: a genuinely unparseable criterion still yields the 0-score result —
but only after 3 attempts. Network errors still propagate as before.

**`synthesize_recommendations`** — use `_chat_json` with a list validator; on
`JSONRetryError` **or** network error, fall back to `deterministic_recommendations`:

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

The existing `chat_fn` parameter on `synthesize_recommendations` is threaded into
`_chat_json` so current tests keep working.

### C. Timestamped output files

In `main`, compute one timestamp per run and suffix every output file with it:

```python
ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def stamped(base, ts, ext):
    """e.g. stamped('compliance_report_combined','20260622-211500','md')
       -> 'compliance_report_combined_20260622-211500.md'"""
    return f"{base}_{ts}.{ext}"
```

Applied to all outputs, sharing the one `ts`:
- per model: `compliance_report_<safe-model>_<ts>.md`
- keywords:  `compliance_report_keywords_<ts>.md`
- combined:  `compliance_report_combined_<ts>.md`

`render()` derives the HTML/PDF from the combined `.md` path via `with_suffix`. Because
`<ts>` is part of the stem (not the extension), `with_suffix(".html")` /
`with_suffix(".pdf")` correctly yield `..._combined_<ts>.html` / `..._combined_<ts>.pdf`
with no change to `render()`.

The final stderr summary prints the timestamped combined filename. The `.gitignore`
glob `compliance_report_*` still matches, so reports remain untracked.

## Testing (stdlib `unittest`, appended to `tests/test_recommendations.py`)

- `_chat_json`:
  - good JSON on first attempt → returns the validated object; `chat_fn` called once.
  - garbage-then-good (fake `chat_fn` returns bad JSON, then valid) → returns validated
    result; assert the second call's `user` argument contains the corrective nudge.
  - always-garbage → raises `JSONRetryError` after exactly `MAX_JSON_ATTEMPTS` calls;
    `e.raw` holds the last content.
- `ask_model` (inject `chat_fn`): bad-then-good → coerced score dict (awarded clamped to
  0..max); always-bad → the 0-score "UNPARSED MODEL OUTPUT" dict, no exception escapes.
- `synthesize_recommendations`: existing fallback tests still pass; add garbage-then-valid
  → now succeeds (returns model recs) instead of falling back.
- `stamped`: exact-string assertion for a known base/ts/ext.

## Out of scope

- Retrying network timeouts / connection errors.
- Configurable retry counts or per-config retry settings.
- Retention/cleanup of old timestamped reports.
- Changing scoring, section parsing, or the combined-report structure.
