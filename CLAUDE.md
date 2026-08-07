# proposal-reviewer

Offline, config-driven compliance reviewer for grant proposals. A single generic engine
(`proposal_reviewer.py`) scores any proposal against a rubric defined in a YAML config
under `configs/`, using a **two-model local ensemble** (Ollama) plus a deterministic
keyword pass, and merges the results into a combined report. See @README.md for the
user-facing overview.

## Core invariant: everything runs locally

The whole point is zero cloud/API usage — reviews run on local models via Ollama so they
can be re-run cheaply while a draft is iterated. Do not add cloud API calls to the engine.
The model endpoint is the local Ollama OpenAI-compatible server at `localhost:11434`.

## Architecture — one engine, many configs

- `proposal_reviewer.py` — generic; knows nothing about any specific proposal.
- `configs/*.yaml` — one per funding opportunity: criteria, point caps, section map,
  keywords, model picks. **Adding a proposal = adding a config, never editing the engine.**
- Pipeline: extract text → parse numbered sections → for each criterion ask both models
  for points/strengths/gaps (JSON) → deterministic keyword scan → combine → render md/html/pdf.

## Two scoring modes (set per criterion via `section`)

- Section-targeted: `section: "1.2"` feeds that numbered narrative section. (CDC config.)
- Whole-document: `section: "*"` feeds the entire document. (DOE config.)

## Conventions

- Exactly TWO models per config, and they should be **different families** (e.g. llama + qwen):
  the combine step assumes two, and independence between families is the signal we trust.
- Reports are written next to the PROPOSAL file, never into this repo. `.gitignore` keeps
  any stray `compliance_report_*` out of version control.
- Commit configs and the engine; never commit a proposal or its generated reports here.
- Engine is stdlib-only except PyYAML.

## Workflow rules

- Before any full run on a new or edited config, run `--dry-run` FIRST. It validates section
  mapping, word counts, and keyword coverage without calling models. Fix mapping there.
- Full run: `python proposal_reviewer.py PROPOSAL --config configs/NAME.yaml`
- Authoring a new config: see `.claude/rules/writing-configs.md`.

## Known gotchas

- The LAST numbered section absorbs all trailing matter (references, appendices). If a
  proposal has those, set `stop_marker: "References"` (or similar) in the config to trim.
- The engine only sees narrative text. Criteria depending on attachments (org charts,
  letters of support, separate data-management-plan files) can only be judged by whether
  the narrative *references* them — flag, don't trust, those scores.
- The heading parser keys on numbered headings (`1`, `1.2`, ...); unnumbered subheadings
  fold into the preceding numbered section. A proposal with no numbered headings collapses
  to one "full document" — give it `heading_patterns` in the config to name the sections
  (see `configs/arpa-h-igor.yaml`), or fall back to `section: "*"` criteria.
- Section text is capped per criterion (`max_section_chars`, default 14000). `section: "*"`
  on a long proposal therefore shows the models only the opening pages — `--dry-run` flags
  this as `⚠ TRUNCATED`. Prefer real sections over `"*"` whenever the document is long.

## Lineage

Generalized from `~/git/DoE_Genesis_16A/assess_compliance.py`, which did the same job with
two cloud models (Claude + GPT) for a single proposal. This repo is the offline, reusable form.
