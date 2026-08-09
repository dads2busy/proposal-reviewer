# proposal-reviewer

Offline, config-driven compliance review for grant proposals. Runs a two-model
**local** ensemble (via [Ollama](https://ollama.com)) plus a deterministic keyword
pass against a proposal, scores it against a rubric defined in a YAML config, and
merges everything into a combined report (`.md` / `.html` / `.pdf`).

Nothing leaves the machine and it consumes no cloud/API usage — the whole point is
to run proposal reviews locally and repeatedly while you iterate on a draft.

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

## Use

    # Validate a config against a proposal without running models:
    python proposal_reviewer.py PROPOSAL --config configs/cdc-ip-26-0102.yaml --dry-run

    # Full review:
    python proposal_reviewer.py PROPOSAL --config configs/cdc-ip-26-0102.yaml

    # Override models ad hoc:
    python proposal_reviewer.py PROPOSAL --config CONFIG --models llama3.3:70b qwen3.6:27b

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
