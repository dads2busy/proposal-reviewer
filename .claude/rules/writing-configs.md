---
description: How to author a proposal config (rubric + keywords). Load when working in configs/.
paths:
  - "configs/**"
---

# Writing a proposal config

A config is one YAML file per funding opportunity. The engine reads it; you never edit
`proposal_reviewer.py` to add a proposal.

## Schema

- `name`: human label shown in report headers.
- `total_points`: grand total (defaults to the sum of criterion maxes if omitted).
- `reviewer_role`: persona injected into the system prompt
  (e.g. "an experienced CDC peer reviewer scoring a cooperative-agreement application").
- `models`: exactly two Ollama model tags, **different families**.
- `top_recommendations` (optional): number of items in the combined report's synthesized
  "Top recommendations" section. Defaults to 7.
- `synthesis_model` (optional): which model writes the synthesized recommendations.
  Defaults to the first entry in `models` (no extra pull needed). It is a post-combine
  step that reuses a reviewer model, not a third reviewer.
- `stop_marker` (optional): truncate extracted text at the first occurrence of this string;
  use it to drop references/appendices the last section would otherwise absorb.
- `rfa_context` (optional): text appended to every criterion prompt for shared framing.
- `section_groups`: map a section-id prefix → the scored section it rolls up into.
- `section_totals`: published subtotal per scored section (drives the overall-scores table).
- `criteria`: list of scored items, each:
  - `section`: numbered narrative id (`"1.2"`) OR `"*"` for whole-document.
  - `name`, `max`, `look_for` (what reviewers reward).
  - `group` (optional): overrides prefix-based grouping; **required when `section: "*"`**.
- `keywords`: deterministic regex scan. Each entry: `pattern` (lowercased regex), `category`,
  optional `label`. Use these for exact, checkable requirements models gloss over —
  named platforms/hubs, required attachments, specific mandated phrases.

## Procedure to add a proposal

1. Copy the closest existing config in `configs/` as a starting point.
2. Transcribe the RFA/NOFO scoring table into `criteria` with the exact point caps.
3. Set each `section` id to match the draft's actual numbered headings.
4. Add `keywords` for the small, exact requirements (the ones worth few points but easy to miss).
5. Run `--dry-run` against the draft; confirm every criterion resolves to a found section
   with a sensible word count.
6. Adjust `stop_marker` / section ids until the mapping is clean, then do a full run.

## Patterns by example

- `configs/cdc-ip-26-0102.yaml` — section-targeted point rubric (100 pts).
- `configs/doe-genesis-16a.yaml` — whole-document scoring with `section: "*"` + `rfa_context`.
