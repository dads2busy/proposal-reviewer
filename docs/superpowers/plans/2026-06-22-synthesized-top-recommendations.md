# Synthesized "Top recommendations" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a synthesized, ranked "Top recommendations" section to the combined compliance report that merges both reviewer models' gaps into a deduplicated, importance-ordered action list.

**Architecture:** A post-combine step builds a deterministic per-criterion digest from the existing two-model results, sends it to ONE local Ollama call that returns a merged top-N JSON list, and renders it as the new §2 of the combined report. The network call is isolated behind an injectable function so all logic is unit-testable offline; on any failure it falls back to a deterministic points-ranked list so the report never breaks.

**Tech Stack:** Python 3 stdlib (`urllib`, `json`, `re`), PyYAML (already used). Tests use stdlib `unittest`.

## Global Constraints

- **Offline only:** all model calls go to the local Ollama endpoint `OLLAMA_URL` (`http://localhost:11434/v1/chat/completions`). No cloud/API calls — ever.
- **Stdlib-only except PyYAML:** do not add dependencies. Tests use `unittest`, NOT pytest.
- **Exactly two reviewer models:** the synthesis step REUSES one configured model; it is not a third reviewer and must not change the two reviewer passes or the combine step's two-model assumption.
- **Not a git repo:** this project is not under version control. Each task's final step is a **checkpoint** that runs the full test suite (no `git commit`). If the user wants versioning, that's a separate decision.
- **All edits in `proposal_reviewer.py`** (engine) plus new `tests/` and doc updates. Adding a proposal/config must never require engine edits — but adding this *engine feature* legitimately does.
- **Test run command (from repo root):** `python3 -m unittest tests.test_recommendations -v`

## File Structure

- `proposal_reviewer.py` — add helpers (`is_required_criterion`, `is_missing_result`), `build_recommendation_digest`, `_ollama_chat`, `_digest_to_prompt`, `deterministic_recommendations`, `synthesize_recommendations`, `render_recommendations`; modify `build_combined` signature + body; wire into `main`.
- `tests/__init__.py` — new, empty (makes `tests` a package).
- `tests/test_recommendations.py` — new, all unit tests (offline; no live Ollama).
- `.claude/rules/writing-configs.md` — document the two new optional config keys.
- `configs/cdc-ip-26-0102.yaml` — add commented example of the new keys.

---

### Task 1: Required/missing detection helpers

**Files:**
- Modify: `proposal_reviewer.py` (add after `keyword_scan`, before the `# ── combine ──` block, ~line 172)
- Create: `tests/__init__.py` (empty)
- Test: `tests/test_recommendations.py`

**Interfaces:**
- Produces: `is_required_criterion(crit: dict) -> bool` and `is_missing_result(res: dict) -> bool`.

- [ ] **Step 1: Create the empty test package file**

Create `tests/__init__.py` with no content (empty file).

- [ ] **Step 2: Write the failing test**

Create `tests/test_recommendations.py`:

```python
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proposal_reviewer import is_required_criterion, is_missing_result


class TestDetectionHelpers(unittest.TestCase):
    def test_required_detects_data_management_plan(self):
        self.assertTrue(is_required_criterion(
            {"name": "Eval: data management plan", "look_for": "DMP covering access"}))

    def test_required_detects_letter_of_support(self):
        self.assertTrue(is_required_criterion(
            {"name": "Capacity: collaborations", "look_for": "at least one letter of support"}))

    def test_required_detects_org_chart(self):
        self.assertTrue(is_required_criterion(
            {"name": "Project management", "look_for": "organizational chart with roles"}))

    def test_required_false_for_plain_criterion(self):
        self.assertFalse(is_required_criterion(
            {"name": "Background", "look_for": "influenza knowledge and clear problems"}))

    def test_missing_true_on_section_missing_justification(self):
        self.assertTrue(is_missing_result(
            {"justification": "Required section missing.", "gaps": []}))

    def test_missing_true_on_placeholder_gap(self):
        self.assertTrue(is_missing_result(
            {"justification": "", "gaps": ["The section is entirely placeholder text"]}))

    def test_missing_false_on_substantive_result(self):
        self.assertFalse(is_missing_result(
            {"justification": "Solid section", "gaps": ["one minor detail could be clearer"]}))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m unittest tests.test_recommendations -v`
Expected: FAIL — `ImportError: cannot import name 'is_required_criterion'`

- [ ] **Step 4: Write minimal implementation**

Add to `proposal_reviewer.py` (after `keyword_scan`, before `# ── combine ──`):

```python
# ── recommendation synthesis ────────────────────────────────────────
REQUIRED_PATTERNS = (
    "data management plan",
    "letter of support", "letters of support",
    "organizational chart", "org chart",
)
MISSING_MARKERS = ("not found", "missing", "placeholder", "still need")


def is_required_criterion(crit):
    """True if the criterion hinges on a required attachment the narrative can only reference."""
    text = (crit.get("name", "") + " " + crit.get("look_for", "")).lower()
    return any(p in text for p in REQUIRED_PATTERNS)


def is_missing_result(res):
    """True if a model's result indicates the scored section was absent or placeholder."""
    blob = (res.get("justification", "") + " " + " ".join(res.get("gaps", []) or [])).lower()
    return any(m in blob for m in MISSING_MARKERS)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m unittest tests.test_recommendations -v`
Expected: PASS (7 tests in `TestDetectionHelpers`)

- [ ] **Step 6: Checkpoint**

Run the full suite: `python3 -m unittest tests.test_recommendations -v` — all green. (No git commit; repo is not version-controlled.)

---

### Task 2: Recommendation digest builder

**Files:**
- Modify: `proposal_reviewer.py` (add after `is_missing_result`)
- Test: `tests/test_recommendations.py`

**Interfaces:**
- Consumes: `is_required_criterion`, `is_missing_result` (Task 1).
- Produces: `build_recommendation_digest(cfg: dict, models: list[str], all_results: dict) -> list[dict]`. Each dict: `{"section": str, "name": str, "max": int, "awarded": [int, int], "points_at_stake": int, "gaps": [str], "required": bool, "missing": bool}`. Includes only criteria where BOTH models scored below max (mirrors §3/§4 gap logic).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_recommendations.py` (before the `if __name__` block):

```python
from proposal_reviewer import build_recommendation_digest


def _sample_cfg_and_results():
    cfg = {"criteria": [
        {"section": "1.1", "name": "Background", "max": 5, "look_for": "influenza"},
        {"section": "2.5", "name": "Eval: data management plan", "max": 3, "look_for": "dmp"},
        {"section": "3.1", "name": "Strong capacity", "max": 10, "look_for": "expertise"},
    ]}
    A, B = "m1", "m2"
    results = {
        A: [
            {"awarded": 0, "gaps": ["entirely placeholder text"], "justification": ""},
            {"awarded": 0, "gaps": [], "justification": "Required section missing."},
            {"awarded": 10, "gaps": [], "justification": "great"},
        ],
        B: [
            {"awarded": 0, "gaps": ["no content provided"], "justification": ""},
            {"awarded": 0, "gaps": [], "justification": "missing"},
            {"awarded": 9, "gaps": ["one minor gap"], "justification": "great"},
        ],
    }
    return cfg, [A, B], results


class TestDigest(unittest.TestCase):
    def test_excludes_item_when_one_model_at_max(self):
        cfg, models, results = _sample_cfg_and_results()
        digest = build_recommendation_digest(cfg, models, results)
        self.assertNotIn("Strong capacity", [d["name"] for d in digest])

    def test_points_at_stake_sums_both_shortfalls(self):
        cfg, models, results = _sample_cfg_and_results()
        digest = build_recommendation_digest(cfg, models, results)
        bg = next(d for d in digest if d["name"] == "Background")
        self.assertEqual(bg["points_at_stake"], 10)  # (5-0)+(5-0)

    def test_required_and_missing_flags_set(self):
        cfg, models, results = _sample_cfg_and_results()
        digest = build_recommendation_digest(cfg, models, results)
        dmp = next(d for d in digest if "data management" in d["name"].lower())
        self.assertTrue(dmp["required"])
        self.assertTrue(dmp["missing"])

    def test_gaps_merged_from_both_models(self):
        cfg, models, results = _sample_cfg_and_results()
        digest = build_recommendation_digest(cfg, models, results)
        bg = next(d for d in digest if d["name"] == "Background")
        self.assertEqual(set(bg["gaps"]), {"entirely placeholder text", "no content provided"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_recommendations -v`
Expected: FAIL — `ImportError: cannot import name 'build_recommendation_digest'`

- [ ] **Step 3: Write minimal implementation**

Add to `proposal_reviewer.py` after `is_missing_result`:

```python
def build_recommendation_digest(cfg, models, all_results):
    """Per-criterion candidates for recommendations: items where BOTH models docked
    points (same selection rule as §3/§4). Order follows cfg['criteria']."""
    A, B = models
    ra, rb = all_results[A], all_results[B]
    digest = []
    for crit, a, b in zip(cfg["criteria"], ra, rb):
        mx = crit["max"]
        if a["awarded"] < mx and b["awarded"] < mx:
            missing = is_missing_result(a) or is_missing_result(b)
            digest.append({
                "section": str(crit.get("section", "")),
                "name": crit["name"],
                "max": mx,
                "awarded": [a["awarded"], b["awarded"]],
                "points_at_stake": (mx - a["awarded"]) + (mx - b["awarded"]),
                "gaps": (a.get("gaps") or []) + (b.get("gaps") or []),
                "required": is_required_criterion(crit) or missing,
                "missing": missing,
            })
    return digest
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_recommendations -v`
Expected: PASS (`TestDigest` 4 tests + earlier 7)

- [ ] **Step 5: Checkpoint**

Run the full suite: `python3 -m unittest tests.test_recommendations -v` — all green.

---

### Task 3: Synthesis call + deterministic fallback

**Files:**
- Modify: `proposal_reviewer.py` (add after `build_recommendation_digest`)
- Test: `tests/test_recommendations.py`

**Interfaces:**
- Consumes: `build_recommendation_digest` (Task 2), `DEFAULT_ROLE`, `OLLAMA_URL` (existing module globals).
- Produces:
  - `_ollama_chat(model, system_prompt, user_prompt, timeout=900) -> str` (raw message content; the only networked function).
  - `deterministic_recommendations(digest, n) -> list[dict]`.
  - `synthesize_recommendations(cfg, models, all_results, chat_fn=_ollama_chat) -> list[dict]`. Returns `[]` when no gaps; otherwise up to `n` recommendation dicts shaped `{"title": str, "points_at_stake": int, "criteria": [str], "do": str, "why": str}`. Falls back to `deterministic_recommendations` on any error/garbage. `chat_fn` is injectable for offline tests.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_recommendations.py` (before `if __name__`):

```python
from proposal_reviewer import synthesize_recommendations, deterministic_recommendations


class TestSynthesis(unittest.TestCase):
    def test_uses_model_output_when_valid(self):
        cfg, models, results = _sample_cfg_and_results()
        fake = lambda m, s, u: (
            '[{"title":"Write Background","points_at_stake":10,'
            '"criteria":["§1.1"],"do":"Add the section","why":"empty"}]')
        recs = synthesize_recommendations(cfg, models, results, chat_fn=fake)
        self.assertEqual(recs[0]["title"], "Write Background")

    def test_strips_code_fences(self):
        cfg, models, results = _sample_cfg_and_results()
        fake = lambda m, s, u: '```json\n[{"title":"X","points_at_stake":1,"criteria":[],"do":"d","why":"w"}]\n```'
        recs = synthesize_recommendations(cfg, models, results, chat_fn=fake)
        self.assertEqual(recs[0]["title"], "X")

    def test_respects_top_n(self):
        cfg, models, results = _sample_cfg_and_results()
        cfg["top_recommendations"] = 1
        fake = lambda m, s, u: (
            '[{"title":"a","points_at_stake":10,"criteria":[],"do":"x","why":"y"},'
            '{"title":"b","points_at_stake":5,"criteria":[],"do":"x","why":"y"}]')
        recs = synthesize_recommendations(cfg, models, results, chat_fn=fake)
        self.assertEqual(len(recs), 1)

    def test_falls_back_on_exception(self):
        cfg, models, results = _sample_cfg_and_results()
        def boom(m, s, u):
            raise TimeoutError("timed out")
        recs = synthesize_recommendations(cfg, models, results, chat_fn=boom)
        self.assertTrue(recs)  # deterministic list, non-empty
        # required/missing items rank first; Background & DMP both qualify
        self.assertIn(recs[0]["title"], ("Background", "Eval: data management plan"))

    def test_falls_back_on_garbage(self):
        cfg, models, results = _sample_cfg_and_results()
        recs = synthesize_recommendations(cfg, models, results, chat_fn=lambda m, s, u: "not json at all")
        self.assertTrue(recs)
        self.assertIn("points_at_stake", recs[0])

    def test_empty_when_no_gaps(self):
        cfg = {"criteria": [{"section": "1", "name": "x", "max": 5, "look_for": ""}]}
        results = {"m1": [{"awarded": 5, "gaps": []}], "m2": [{"awarded": 5, "gaps": []}]}
        recs = synthesize_recommendations(cfg, ["m1", "m2"], results, chat_fn=lambda *a: "[]")
        self.assertEqual(recs, [])

    def test_deterministic_orders_required_first(self):
        cfg, models, results = _sample_cfg_and_results()
        from proposal_reviewer import build_recommendation_digest
        digest = build_recommendation_digest(cfg, models, results)
        recs = deterministic_recommendations(digest, 7)
        # Background (10 pts, required) before DMP (6 pts, required)
        self.assertEqual(recs[0]["title"], "Background")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_recommendations -v`
Expected: FAIL — `ImportError: cannot import name 'synthesize_recommendations'`

- [ ] **Step 3: Write minimal implementation**

Add to `proposal_reviewer.py` after `build_recommendation_digest`:

```python
SYNTH_SYSTEM = (
    "You are {role}. You are given the gaps two independent reviewers found in a grant "
    "proposal, each with the points at stake. Produce the TOP {n} most important "
    "recommendations to improve the proposal. MERGE related or duplicate gaps from the two "
    "reviewers into a single recommendation. Rank by importance: points at stake is the "
    "primary signal, but a missing REQUIRED element (data management plan, letters of "
    "support, organizational chart) or a hard compliance miss may outrank a higher-point "
    "soft gap. Return FEWER than {n} items if there are fewer real gaps. "
    'Respond ONLY with a JSON array; each item: {{"title": <str>, "points_at_stake": <int>, '
    '"criteria": [<str>], "do": <str>, "why": <str>}}'
)


def _ollama_chat(model, system_prompt, user_prompt, timeout=900):
    """Single local Ollama chat call; returns the message content string."""
    payload = {"model": model, "temperature": 0.2,
               "messages": [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}],
               "options": {"num_ctx": 16384}}
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def _digest_to_prompt(digest):
    lines = []
    for d in digest:
        flag = "  [REQUIRED/MISSING]" if d["required"] else ""
        lines.append(f"- {d['name']} (§{d['section']}) — {d['points_at_stake']} pts at stake{flag}")
        for g in d["gaps"]:
            lines.append(f"    • {g}")
    return "\n".join(lines)


def deterministic_recommendations(digest, n):
    """Fallback ranking when the synthesis model is unavailable or returns garbage.
    Sort by (required, points_at_stake) descending."""
    ordered = sorted(digest, key=lambda d: (d["required"], d["points_at_stake"]), reverse=True)
    recs = []
    for d in ordered[:n]:
        why = ("Section is missing or placeholder; both reviewers scored it near zero."
               if d["missing"] else "Both reviewers docked points here.")
        recs.append({
            "title": d["name"],
            "points_at_stake": d["points_at_stake"],
            "criteria": [f"§{d['section']}"] if d["section"] else [],
            "do": d["gaps"][0] if d["gaps"] else "Address the reviewers' gaps for this criterion.",
            "why": why,
        })
    return recs


def synthesize_recommendations(cfg, models, all_results, chat_fn=_ollama_chat):
    """Ranked, deduplicated recommendations merging both models' gaps. Falls back to a
    deterministic ranking if the synthesis call fails or returns unusable output."""
    digest = build_recommendation_digest(cfg, models, all_results)
    if not digest:
        return []
    n = int(cfg.get("top_recommendations", 7))
    model = cfg.get("synthesis_model") or models[0]
    role = cfg.get("reviewer_role", DEFAULT_ROLE)
    system = SYNTH_SYSTEM.format(role=role, n=n)
    user = ("CRITERIA WITH GAPS (ranked candidates):\n" + _digest_to_prompt(digest)
            + f"\n\nReturn the top {n} merged recommendations as a JSON array.")
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

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_recommendations -v`
Expected: PASS (`TestSynthesis` 7 tests + earlier)

- [ ] **Step 5: Checkpoint**

Run the full suite: `python3 -m unittest tests.test_recommendations -v` — all green.

---

### Task 4: Render the recommendations section

**Files:**
- Modify: `proposal_reviewer.py` (add after `synthesize_recommendations`)
- Test: `tests/test_recommendations.py`

**Interfaces:**
- Produces: `render_recommendations(recs: list[dict]) -> str` — returns the full "## 2. Top recommendations ..." section markdown. Handles empty input.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_recommendations.py` (before `if __name__`):

```python
from proposal_reviewer import render_recommendations


class TestRender(unittest.TestCase):
    def test_header_and_item_format(self):
        md = render_recommendations([
            {"title": "Write Background", "points_at_stake": 10,
             "criteria": ["§1.1"], "do": "Add the section", "why": "currently empty"}])
        self.assertIn("## 2. Top recommendations (synthesized, ranked by importance)", md)
        self.assertIn("### 1. Write Background (§1.1) — 10 pts at stake", md)
        self.assertIn("**Do:** Add the section", md)
        self.assertIn("**Why:** currently empty", md)

    def test_numbers_multiple_items(self):
        md = render_recommendations([
            {"title": "A", "points_at_stake": 5, "criteria": [], "do": "x", "why": "y"},
            {"title": "B", "points_at_stake": 3, "criteria": [], "do": "x", "why": "y"}])
        self.assertIn("### 1. A", md)
        self.assertIn("### 2. B", md)

    def test_empty_recs_message(self):
        md = render_recommendations([])
        self.assertIn("## 2. Top recommendations", md)
        self.assertIn("No material gaps", md)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_recommendations -v`
Expected: FAIL — `ImportError: cannot import name 'render_recommendations'`

- [ ] **Step 3: Write minimal implementation**

Add to `proposal_reviewer.py` after `synthesize_recommendations`:

```python
def render_recommendations(recs):
    """Render the synthesized recommendations as the combined-report §2 body."""
    out = ["\n## 2. Top recommendations (synthesized, ranked by importance)\n"]
    if not recs:
        out.append("No material gaps — both reviewers scored every criterion at or near max.")
        return "\n".join(out)
    for i, r in enumerate(recs, 1):
        crits = ", ".join(r.get("criteria") or [])
        suffix = f" ({crits})" if crits else ""
        pts = r.get("points_at_stake", "")
        pts_str = f" — {pts} pts at stake" if pts != "" else ""
        out.append(f"\n### {i}. {r.get('title', '(untitled)')}{suffix}{pts_str}")
        out.append(f"**Do:** {str(r.get('do', '')).strip()}")
        out.append(f"**Why:** {str(r.get('why', '')).strip()}")
    return "\n".join(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_recommendations -v`
Expected: PASS (`TestRender` 3 tests + earlier)

- [ ] **Step 5: Checkpoint**

Run the full suite: `python3 -m unittest tests.test_recommendations -v` — all green.

---

### Task 5: Wire the section into the combined report

**Files:**
- Modify: `proposal_reviewer.py` — `build_combined` signature + body (~lines 185-237) and `main` (~line 305)
- Test: `tests/test_recommendations.py`

**Interfaces:**
- Consumes: `render_recommendations` (Task 4), `synthesize_recommendations` (Task 3).
- Produces: `build_combined(cfg, models, all_results, kw_pct, kw_missing, recs) -> str` (NEW trailing `recs` param). Inserts §2 after Overall scores; renumbers later sections to §3–§7.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_recommendations.py` (before `if __name__`):

```python
from proposal_reviewer import build_combined


class TestBuildCombinedIntegration(unittest.TestCase):
    def _cfg_results_recs(self):
        cfg = {"name": "T", "total_points": 5,
               "criteria": [{"section": "1.1", "name": "Background", "max": 5,
                             "look_for": "x", "group": "G"}],
               "section_groups": {"1": "G"}, "section_totals": {"G": 5}}
        A, B = "m1", "m2"
        results = {
            A: [{"awarded": 0, "strengths": [], "gaps": ["placeholder"]}],
            B: [{"awarded": 0, "strengths": [], "gaps": ["empty"]}],
        }
        recs = [{"title": "Fix Background", "points_at_stake": 10,
                 "criteria": ["§1.1"], "do": "Write it", "why": "empty"}]
        return cfg, [A, B], results, recs

    def test_recommendations_section_is_second(self):
        cfg, models, results, recs = self._cfg_results_recs()
        md = build_combined(cfg, models, results, 100.0, [], recs)
        self.assertLess(md.index("## 2. Top recommendations"),
                        md.index("## 3. Per-criterion"))

    def test_renders_recommendation_and_renumbered_final(self):
        cfg, models, results, recs = self._cfg_results_recs()
        md = build_combined(cfg, models, results, 100.0, [], recs)
        self.assertIn("### 1. Fix Background", md)
        self.assertIn("## 7. Final", md)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_recommendations -v`
Expected: FAIL — `TypeError: build_combined() missing 1 required positional argument: 'recs'`

- [ ] **Step 3: Update `build_combined` signature**

Change line 185 from:

```python
def build_combined(cfg, models, all_results, kw_pct, kw_missing):
```

to:

```python
def build_combined(cfg, models, all_results, kw_pct, kw_missing, recs):
```

- [ ] **Step 4: Insert the §2 render call after Overall scores**

Find (the end of section 1, ~line 199):

```python
    L.append(f"| **TOTAL** | **{ta}/{total}** | **{tb}/{total}** |\n")

    L.append("\n## 2. Per-criterion scores (★ = models diverge ≥2 pts)\n")
```

Replace with:

```python
    L.append(f"| **TOTAL** | **{ta}/{total}** | **{tb}/{total}** |\n")

    L.append(render_recommendations(recs))

    L.append("\n## 3. Per-criterion scores (★ = models diverge ≥2 pts)\n")
```

- [ ] **Step 5: Renumber the remaining section headers**

In `build_combined`, change these four header strings (only the leading numbers change):

- `"\n## 3. Priority gaps — both models dock points (ranked by points at stake)\n"` → `"\n## 4. Priority gaps — both models dock points (ranked by points at stake)\n"`
- `"\n## 4. Strengths — both models score ≥80% of max\n"` → `"\n## 5. Strengths — both models score ≥80% of max\n"`
- `"\n## 5. Missing RFA phrases (deterministic scan)\n"` → `"\n## 6. Missing RFA phrases (deterministic scan)\n"`
- `f"\n## 6. Final\n**{A}: {ta}/{total} · {B}: {tb}/{total} · Keywords: {kw_pct:.0f}%**\n"` → `f"\n## 7. Final\n**{A}: {ta}/{total} · {B}: {tb}/{total} · Keywords: {kw_pct:.0f}%**\n"`

- [ ] **Step 6: Run integration test to verify it passes**

Run: `python3 -m unittest tests.test_recommendations -v`
Expected: PASS (`TestBuildCombinedIntegration` 2 tests + earlier)

- [ ] **Step 7: Wire synthesis into `main`**

Find (~line 305):

```python
    combined = build_combined(cfg, models, all_results, kw_pct, kw_missing)
```

Replace with:

```python
    print("Synthesizing recommendations ...", file=sys.stderr)
    recs = synthesize_recommendations(cfg, models, all_results)
    combined = build_combined(cfg, models, all_results, kw_pct, kw_missing, recs)
```

- [ ] **Step 8: Verify the module still imports and CLI parses**

Run: `python3 proposal_reviewer.py --help`
Expected: argparse usage prints with no traceback (confirms no syntax/name errors).

- [ ] **Step 9: Checkpoint**

Run the full suite: `python3 -m unittest tests.test_recommendations -v` — all green.

---

### Task 6: Document the new config keys

**Files:**
- Modify: `.claude/rules/writing-configs.md` (Schema section)
- Modify: `configs/cdc-ip-26-0102.yaml` (add commented example near the top, after `models:`)

**Interfaces:** none (documentation only).

- [ ] **Step 1: Add the keys to the config schema doc**

In `.claude/rules/writing-configs.md`, in the `## Schema` list, after the `models:` bullet, add:

```markdown
- `top_recommendations` (optional): number of items in the combined report's synthesized
  "Top recommendations" section. Defaults to 7.
- `synthesis_model` (optional): which model writes the synthesized recommendations.
  Defaults to the first entry in `models` (no extra pull needed). It is a post-combine
  step that reuses a reviewer model, not a third reviewer.
```

- [ ] **Step 2: Add a commented example to the CDC config**

In `configs/cdc-ip-26-0102.yaml`, after the `models:` line block, add:

```yaml
# Combined-report synthesis (both optional):
# top_recommendations: 7        # size of the ranked "Top recommendations" section
# synthesis_model: gemma3:27b   # defaults to the first model above if unset
```

- [ ] **Step 3: Verify the config still loads**

Run: `python3 -c "import yaml; yaml.safe_load(open('configs/cdc-ip-26-0102.yaml')); print('ok')"`
Expected: `ok`

- [ ] **Step 4: Checkpoint**

Run the full suite: `python3 -m unittest tests.test_recommendations -v` — all green. Feature complete.

---

## Self-Review

**Spec coverage:**
- Local model synthesis pass → Task 3 (`synthesize_recommendations` + `_ollama_chat`). ✓
- Ranking by points-at-stake + judged severity → SYNTH_SYSTEM prompt + `required` flag in digest (Task 2); deterministic fallback sorts by `(required, points_at_stake)` (Task 3). ✓
- Configurable count default 7 → `cfg.get("top_recommendations", 7)` (Task 3); documented (Task 6). ✓
- `synthesis_model` default models[0] → Task 3; documented (Task 6). ✓
- Required/missing flag (placeholder sections + named attachments) → Task 1 helpers, used in Task 2. ✓
- Placement at top (new §2), renumber → Task 5. ✓
- Output format (title/points/criteria/do/why) → Task 4 render + Task 3 schema. ✓
- Failure handling never crashes report → Task 3 try/except → deterministic fallback; tests `test_falls_back_on_exception`, `test_falls_back_on_garbage`. ✓
- Offline / stdlib-only / two-model invariants → Global Constraints; synthesis reuses a configured model. ✓
- Testing (digest, renderer, fallback) → Tasks 1-5 tests. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every test shows real assertions. ✓

**Type consistency:** Recommendation dict shape `{title, points_at_stake, criteria, do, why}` is identical across `synthesize_recommendations`, `deterministic_recommendations`, `render_recommendations`, and all tests. `build_recommendation_digest` dict shape `{section, name, max, awarded, points_at_stake, gaps, required, missing}` is consumed consistently by `deterministic_recommendations` and `_digest_to_prompt`. `build_combined` new `recs` param matches `render_recommendations` input. ✓
