# proposal-reviewer

Config-driven compliance review for grant proposals. Runs a two-model ensemble —
on a local [Ollama](https://ollama.com), on UVA's self-hosted RC GenAI service, or
one of each — plus a deterministic keyword pass against a proposal, scores it
against a rubric defined in a YAML config, and merges everything into a combined
report (`.md` / `.html` / `.pdf`).

No draft ever reaches a commercial provider — the whole point is to review
proposals repeatedly, for free, while you iterate on a draft. Models run either on
your own machine via Ollama or on UVA's self-hosted RC GenAI service; see
[Model backends](#model-backends).

## Why config-driven

The engine (`proposal_reviewer.py`) is proposal-agnostic. Everything specific to a
given funding opportunity — criteria, point caps, section map, keywords, model
picks — lives in a YAML file under `configs/`. One engine, one config per proposal.

## Install

    brew install ollama pandoc          # pdftotext comes via poppler: brew install poppler
    pip install pyyaml
    ollama pull gemma3:27b              # two DIFFERENT families; both ~27-32B run
    ollama pull qwen2.5:32b             # comfortably on a 64GB machine
    # PDF rendering needs a Unicode-capable engine, one of:
    pip install weasyprint              # or a TeX with xelatex (brew install --cask mactex-no-gui)

Requires Ollama >= 0.19 (MLX backend on Apple Silicon).

## Model backends

Model names may carry a provider prefix. A bare name means local Ollama, so
existing configs need no changes.

    models: ["gemma3:27b", "qwen2.5:32b"]            # both local Ollama
    models: ["uva:Kimi K2.5", "ollama:qwen2.5:32b"]  # hybrid

Non-local providers are declared in the config and take their key from the
environment — never from the file:

    providers:
      uva:
        kind: openai
        base_url: https://open-webui.rc.virginia.edu/api/
        api_key_env: UVARC_GenAI_API

Copy `.env.example` to `.env` and fill in the key. `.env` is gitignored, and the
engine tightens it to mode 600 on read.

**UVA RC GenAI only answers from an HPC node.** From a VPN-connected laptop it
returns HTTP 401 even with a valid key, because Apache rejects the request before
the token is ever evaluated — browser access via NetBadge works from anywhere, so
it proves nothing about API access. Check any machine with:

    python proposal_reviewer.py --config CONFIG --check-provider

UVA hosts a single model, so a UVA-only config runs the same model twice. The
engine detects this and relabels the combined report: two samples measure whether
a score is *stable*, not whether two independent reviewers agree. It also widens
the second sample's temperature so the extra call is worth making.

See `configs/uva-kimi-example.yaml` for a complete example.

## Use

    # Validate a config against a proposal without running models:
    python proposal_reviewer.py PROPOSAL --config configs/cdc-ip-26-0102.yaml --dry-run

    # Full review:
    python proposal_reviewer.py PROPOSAL --config configs/cdc-ip-26-0102.yaml

    # Override models ad hoc:
    python proposal_reviewer.py PROPOSAL --config CONFIG --models llama3.3:70b qwen3.6:27b

    # Confirm this machine can reach and authenticate against each backend:
    python proposal_reviewer.py --config CONFIG --check-provider

`PROPOSAL` can be `.docx`, `.pdf`, `.md`, `.tex`, or `.txt`. Reports are written
next to the proposal file.

## Writing a config

See `configs/cdc-ip-26-0102.yaml` (section-targeted scoring) and
`configs/doe-genesis-16a.yaml` (whole-document scoring) for the two patterns, and
`configs/arpa-h-igor.yaml` for a proposal whose outline carries no section numbers.

Each criterion targets a numbered narrative section (e.g. `section: "1.2"`) or the
whole document (`section: "*"`). If the draft's headings aren't numbered, map them
to IDs with `heading_patterns` rather than falling back to `"*"`:

    heading_patterns:
      - {pattern: '^TA1\.\s[A-Z][a-z]', id: "4.1"}

Section text is capped per criterion at `max_section_chars` (default 14000) to stay
inside the model context window, which `num_ctx` sets (default 16384 tokens). Raise
the two together — `--dry-run` flags any section that would be cut and warns when the
cap exceeds what the window can hold. The model awards `0..max` points per criterion;
section subtotals roll up via `section_groups` / `section_totals`. The `keywords`
list is a deterministic regex scan for exact, checkable requirements that models
sometimes gloss (specific platforms, named hubs, required attachments).

## Output

Every run stamps its files with a shared `YYYYMMDD-HHMMSS` suffix (`<ts>`) so
repeated runs don't overwrite each other:

- `compliance_report_<model>_<ts>.md` — one per model
- `compliance_report_keywords_<ts>.md` — deterministic phrase coverage
- `compliance_report_combined_<ts>.md` / `.html` / `.pdf` — merged: synthesized
  top recommendations, per-section scores per model, per-criterion divergence flags,
  gaps where both models converge, shared strengths, and missing-phrase scan
