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
- `models`: exactly two model refs, **different families** where the backend offers a
  choice. A bare tag (`qwen2.5:32b`) is a local Ollama model; a `provider:model` ref
  (`uva:Kimi K2.5`) resolves against `providers` below. When both slots name the same
  model the engine treats the run as single-family: it widens the second sample's
  temperature and rewrites the combined report so it claims consistency, not agreement.
- `providers` (optional): non-Ollama backends, keyed by the prefix used in `models`.

      providers:
        uva:
          kind: openai                                    # OpenAI-compatible chat API
          base_url: https://open-webui.rc.virginia.edu/api/
          api_key_env: UVARC_GenAI_API                    # NAME of an env var, never a key

  `api_key_env` names an environment variable read from the real environment or `.env`;
  a literal `api_key:` is rejected, because configs are committed and keys must not be.
  Only self-hosted or institutional endpoints belong here — never a commercial provider.
  Confirm a machine can reach one with `--check-provider`.
- `top_recommendations` (optional): number of items in the combined report's synthesized
  "Top recommendations" section. Defaults to 7.
- `synthesis_model` (optional): which model writes the synthesized recommendations.
  Defaults to the first entry in `models` (no extra pull needed). It is a post-combine
  step that reuses a reviewer model, not a third reviewer. It may carry its own
  `provider:` prefix and is resolved against its own backend, so naming a local model
  here while the ensemble runs remotely (or the reverse) works.
- `stop_marker` (optional): truncate extracted text at the first occurrence of this string;
  use it to drop references/appendices the last section would otherwise absorb.
- `heading_patterns` (optional): `[{pattern: <regex>, id: "4.1"}, ...]`. Assigns section IDs
  to headings that carry no usable number of their own, so a proposal whose outline is
  unnumbered ("TA1. Digital Twins") can still be scored section by section instead of
  collapsing to `section: "*"`. Checked before the numbered passes; TOC lines (containing
  `....`) are skipped. Make the patterns fussy — a loose `^TA4\.` also matches subtask
  labels and padded table cells, and only the FIRST match wins, so a stray hit early in the
  document silently anchors the section in the wrong place. Verify with `--dry-run`.
- `max_section_chars` (optional): per-criterion cap on section text sent to the model.
  Defaults to 14000. Raise it when `--dry-run` flags `⚠ TRUNCATED`.
- `num_ctx` (optional): Ollama context window, default 16384 tokens. Raise it *with*
  `max_section_chars` — at roughly 4 chars/token a section that fits the cap but
  overflows the window is dropped by the model instead of by us, which looks like a
  confidently wrong score rather than an error. `--dry-run` prints both and flags
  `⚠ EXCEEDS num_ctx`. 27-32B models run fine at 32768 on a 64GB machine.
- `rfa_context` (optional): text appended to every criterion prompt for shared framing.
- `required_elements` (optional): attachments this opportunity mandates that the narrative
  can only reference (e.g. `"letters of support"`). A criterion whose name/`look_for`
  mentions one is ranked as a hard compliance item, and the list is named in the synthesis
  prompt. **Declare only what the solicitation actually requires** — these phrases are fed
  to the model as things a proposal must contain, so an inherited or aspirational entry
  becomes a confidently-worded recommendation to add something nobody asked for. Omit the
  key entirely when the opportunity mandates nothing; a missing section is still flagged.
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
   with a sensible word count, and that no row is flagged `⚠ TRUNCATED`.
6. Adjust `stop_marker` / `heading_patterns` / section ids until the mapping is clean, then
   do a full run.

## Patterns by example

- `configs/cdc-ip-26-0102.yaml` — section-targeted point rubric (100 pts).
- `configs/doe-genesis-16a.yaml` — whole-document scoring with `section: "*"` + `rfa_context`.
- `configs/arpa-h-igor.yaml` — `heading_patterns` for an unnumbered outline, plus a raised
  `max_section_chars`; also shows how to weight a rubric the sponsor publishes without
  points, and how to omit a criterion scored against a document you aren't reviewing.
