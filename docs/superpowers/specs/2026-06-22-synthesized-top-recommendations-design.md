# Synthesized "Top recommendations" section for the combined report

**Date:** 2026-06-22
**Status:** Approved design, pending implementation plan
**Component:** `proposal_reviewer.py` — `build_combined` / new `synthesize_recommendations`

## Problem

The combined report (`compliance_report_combined.md`) presents the two reviewer
models' findings but never *synthesizes* them. Today:

- **§3 Priority gaps** ranks criteria by points-at-stake but dumps each model's raw
  gap bullets separately under each criterion — no merging or deduplication.
- **§6 Final** lists only the top-5 criterion *names* by points.

Neither produces a single, deduplicated, prioritized list of actionable
recommendations that takes *both* models' findings into account and decides what
matters most. That synthesized list is what this feature adds.

## Goal

Add a synthesized **"Top recommendations"** section to the combined report: a ranked,
deduplicated list of the most important actions to improve the proposal, merging
both models' gaps into unified items, sorted by importance (points-at-stake adjusted
for severity).

## Decisions (from brainstorming)

1. **Method:** a local model synthesis pass (one extra Ollama call), not a purely
   deterministic merge. The synthesis model actually rewrites/merges overlapping
   findings into unified recommendations.
2. **Ranking basis:** points-at-stake is the primary signal, but the model may adjust
   for severity — a missing *required* element (data management plan, letters of
   support, org chart) or a hard NOFO compliance miss can outrank a higher-point
   "soft" gap. The model explains the ranking.
3. **List size:** configurable via `top_recommendations` (default 7). The model may
   return fewer if there are fewer real gaps.
4. **Placement:** at the top of the combined report, immediately after Overall scores
   (new §2); subsequent sections renumber.

## Core invariants preserved

- **Offline only:** the synthesis call goes to the same local Ollama
  OpenAI-compatible endpoint (`localhost:11434`). No cloud/API calls.
- **Exactly two models:** synthesis is a *post-combine step* that reuses one of the
  two configured reviewer models — it is not a third reviewer. The two-model ensemble
  and the combine step's two-model assumption are unchanged.
- **Stdlib-only except PyYAML:** synthesis reuses the existing `urllib` + JSON-cleanup
  pattern from `ask_model`. No new dependency.

## Design

### New function: `synthesize_recommendations(cfg, models, all_results)`

A post-combine step that runs after both reviewer passes complete, called from (or
just before) `build_combined`.

**Input digest.** Built from the existing per-criterion results — no re-reading the
proposal. For each criterion:

- `name`, `max`
- both models' `awarded` scores
- `points_at_stake` — the `lost` value §3 already computes:
  `(max - a_awarded) + (max - b_awarded)`, counted only when *both* models docked
  points (mirrors §3's gap selection)
- both models' `gaps` bullets
- `required` / `missing` flag — set true when:
  - the section was a placeholder or absent (model returned the
    "SECTION MISSING" / placeholder path, or awarded 0 with a "missing"/"placeholder"
    gap), OR
  - the criterion's `look_for` / `name` names a required attachment (data management
    plan, letter(s) of support, organizational chart). A small keyword check over the
    criterion text drives this flag.

Plus the overall scores (per-model totals and section subtotals) for context.

**Model call.** One local Ollama call (same `urllib` request shape, `temperature` 0.2,
`num_ctx` 16384 — the digest is small and fits comfortably). System prompt instructs
the model to:

- return the top-N (`top_recommendations`, default 7) recommendations
- merge related gaps across *both* models into a single item (dedup)
- rank by points-at-stake, allowed to adjust for severity (required/compliance-critical
  outranks equal-or-higher-point soft gaps)
- respond ONLY with a JSON array; each item:
  `{ "title": str, "points_at_stake": int, "criteria": [str], "do": str, "why": str }`

**Output rendering.** The engine renders the JSON array into the new section. Per item:

```
### 1. <title> — <points_at_stake> pts at stake
**Do:** <do>
**Why:** <why>
```

(`criteria` may be rendered as a small "(§1.1, §1.4)" suffix on the title line.)

### Config keys (both optional)

```yaml
top_recommendations: 7      # number of items; default 7 if unset
synthesis_model: gemma3:27b # model that writes the synthesis; default = models[0]
```

- `top_recommendations`: default 7.
- `synthesis_model`: default = first configured model (`models[0]`); no extra
  `ollama pull` needed. Overridable per opportunity.

### Placement

New section inserted into `build_combined`'s output immediately after
"## 1. Overall scores":

```
## 2. Top recommendations (synthesized, ranked by importance)
```

All later sections renumber (Per-criterion → §3, Priority gaps → §4, Strengths → §5,
Missing phrases → §6, Final → §7).

### Failure handling

The synthesis call must never crash the report (Ollama can time out or return
unparseable output):

- Wrap the call + JSON parse in try/except (same cleanup logic as `ask_model`).
- On timeout / connection error / unparseable JSON: log a warning to stderr and
  **fall back to a deterministic ranked list** — reuse the existing points-at-stake
  ranking (the data behind §6) rendered as the Top recommendations section, so the
  report always contains *a* ranked list.
- The rest of the combined report is unaffected either way.

## Testing

- **Digest builder:** feed two fake result sets → assert `points_at_stake` math and
  the `required`/`missing` flag logic (placeholder section, named-attachment criterion).
- **Renderer:** feed a canned synthesis JSON array → assert the rendered section
  matches the expected markdown (titles, Do/Why, points).
- **Fallback:** simulate a failed/garbage synthesis response → assert the deterministic
  ranked list renders and no exception escapes `build_combined`.

## Out of scope

- Changing the two reviewer passes or their prompts.
- Changing how sections/keywords are scored.
- Any cloud/API usage.
- Per-recommendation cross-references to specific line numbers in the proposal.
