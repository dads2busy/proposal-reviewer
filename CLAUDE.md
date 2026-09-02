# proposal-reviewer

Config-driven compliance reviewer for grant proposals. A single generic engine
(`proposal_reviewer.py`) scores any proposal against a rubric defined in a YAML config
under `configs/`, using a **two-model ensemble** — local Ollama, UVA's self-hosted RC
GenAI, or one of each — plus a deterministic keyword pass, and merges the results into
a combined report. See @README.md for the user-facing overview.

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

That service also **always streams** (`text/event-stream`) and ignores
`stream: false`, so `providers._parse_reply` reassembles the `delta.content`
frames. Do not "simplify" it back to a single `json.loads` — that breaks every
UVA call while leaving Ollama green.

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
- When both slots hold the same model, the engine sets `single_family` and rewrites the
  combined report's language — two samples are a reliability check, not corroboration, and
  the second sample's temperature rises to 0.6 so it actually samples wider. Never reword
  those sections back to claiming two independent opinions.
- Secrets live in `.env` (gitignored, and `chmod 600`'d on read); configs name the variable
  via `api_key_env` and never hold a value. The engine rejects a literal `api_key:`.
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
- Markdown fed from `pandoc` must carry ATX (`#`) headings. Pandoc's markdown writer
  emits SETEXT (`===` underlines) up to 2.10 and ATX from 3.0, and the heading parser
  keys on `#` — a setext conversion collapses the whole document to one section and
  every `section:` misses, silently. Convert with `--markdown-headings=atx` (>=2.11) or
  `--atx-headers` (<=2.11); each version rejects the other spelling, so try one and fall
  back. `--dry-run` catches it: the parsed-section list comes back nearly empty.
- Section text is capped per criterion (`max_section_chars`, default 14000). `section: "*"`
  on a long proposal therefore shows the models only the opening pages — `--dry-run` flags
  this as `⚠ TRUNCATED`. Prefer real sections over `"*"` whenever the document is long.

## Lineage

Generalized from `~/git/DoE_Genesis_16A/assess_compliance.py`, which did the same job with
two cloud models (Claude + GPT) for a single proposal. This repo is the offline, reusable form.
