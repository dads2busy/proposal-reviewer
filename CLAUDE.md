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

**Ollama is called on its native `/api/chat`, never the OpenAI-compat `/v1/`
path.** The compat endpoint accepts an `options` object and silently discards
it: asking `/v1/` for `num_ctx: 4096` loaded a 32768 context, while `/api/`
asked identically loaded 4096. Every run before this was fixed ignored the
config's `num_ctx` entirely. The two endpoints differ in request *and* reply
shape — native puts `temperature` inside `options`, needs `stream: false`, and
answers `{"message": {"content": ...}}` rather than `{"choices": [...]}`, which
is why `_parse_ollama_reply` exists alongside `_parse_reply`. Unifying them
re-breaks one backend or the other.

**Local calls cap `num_thread` to `providers._usable_cpus()`.** llama.cpp sizes
its pool from the physical machine, so a Slurm job holding 8 of a node's 40
cores spawns 40 threads onto 8 and thrashes. The GPU looks healthy while this
happens — fully offloaded, no throttle flag, just starved at a third
utilization and 70W of 300W.

The cap buys **generation** speed, not prompt speed. On a V100 with
`gemma3:27b` fully offloaded, feeding a 14000-char section (2337 tokens, the
size `max_section_chars` actually sends): prompt eval was ~10200 tok/s both
capped and uncapped, because prefill batches onto the GPU; generation ran 33.1
tok/s capped against 22.6 uncapped. Sweeping `num_thread` on an 8-core job,
1 through 16 threads are flat and only the oversubscribed default costs
anything — so **do not ask Slurm for more cores on account of this.** Two to
four is plenty for a GPU-resident model; the earlier 4x prompt-eval figure was
measured on a 318-token prompt and does not survive at real section sizes.

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
- Reports are written next to the PROPOSAL file by default; `--outdir DIR` overrides it
  (resolved before any model call, so a bad path fails fast). Never point it into this
  repo — `.gitignore` keeps any stray `compliance_report_*` out of version control.
- Commit configs and the engine; never commit a proposal or its generated reports here.
- Engine is stdlib-only except PyYAML.

## Workflow rules

- Before any full run on a new or edited config, run `--dry-run` FIRST. It validates section
  mapping, word counts, and keyword coverage without calling models. Fix mapping there.
- Full run: `python proposal_reviewer.py PROPOSAL --config configs/NAME.yaml`
- A full run calls `preflight()` first: one ping per slot (the `--check-provider`
  check) and an exit if either fails. It exists because slot A reviews every
  criterion before slot B's first call, so a dead B otherwise surfaced only after A's
  whole pass. `--no-check` skips it; `--dry-run` must stay model-free, so never move
  the check ahead of it.
- The ping timeout is 600s for Ollama slots, 60s for remote. Do not shorten the local
  one: the ping is the call that loads a 17-24 GB model off GPFS, so a short timeout
  reports a healthy node as down. It costs nothing when Ollama is not running —
  connection refused returns instantly. The local ping also passes the config's
  `num_ctx`, so the first real call doesn't reload the model at a different context.
- **The engine never launches Ollama.** Auto-start was considered and rejected: run
  before `ijob`, it would put a 24 GB model on a shared login node's CPU, and the
  refusal is what stops that. Instead, an unreachable `localhost` Ollama fails with
  `Ollama is not running. On a GPU node, start it with: ollama serve > /tmp/ollama-$USER.log 2>&1 &`.
  The hint in `make_chat_fn` is per backend kind — Ollama on another host gets
  `Is Ollama running on <host>?`, a remote API gets `On an HPC node?` — so don't
  collapse it back into one shared message.
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
