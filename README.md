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

## Setup

Two supported environments. Pick the one you are on — the engine is identical, only
the model backend and the PDF toolchain differ.

### A. Your own machine (macOS, local Ollama)

    brew install ollama pandoc          # pdftotext comes via poppler: brew install poppler
    pip install pyyaml
    ollama pull gemma3:27b              # two DIFFERENT families; both ~27-32B run
    ollama pull qwen2.5:32b             # comfortably on a 64GB machine
    # PDF rendering needs a Unicode-capable engine, one of:
    pip install weasyprint              # or a TeX with xelatex (brew install --cask mactex-no-gui)

Requires Ollama >= 0.19 (MLX backend on Apple Silicon).

### B. UVA HPC (Rivanna / Afton)

No Ollama here and no root needed — models come from UVA RC GenAI, and everything
else installs into your home directory. Do this once:

    # 1. Clone
    git clone git@github.com:dads2busy/proposal-reviewer.git ~/git/proposal-reviewer
    cd ~/git/proposal-reviewer

    # 2. pandoc >= 3. The system pandoc (/usr/bin/pandoc) is 2.0.6 and too old for
    #    TWO independent reasons: it writes setext headings the section parser cannot
    #    see, and its LaTeX output no longer compiles under TeX Live 2025.
    mkdir -p ~/.local/bin
    curl -sSL https://github.com/jgm/pandoc/releases/download/3.1.11/pandoc-3.1.11-linux-amd64.tar.gz \
        | tar xz -C /tmp
    cp /tmp/pandoc-3.1.11/bin/pandoc ~/.local/bin/

    # 3. ~/.local/bin must come BEFORE /usr/bin or you silently get 2.0.6 back
    grep -q '.local/bin' ~/.bashrc || echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
    source ~/.bashrc
    pandoc --version | head -1          # must report 3.x

    # 4. Python dependency (pdftotext is already at /usr/bin/pdftotext)
    pip install --user pyyaml

    # 5. API key — see .env.example for where to get it
    cp .env.example .env && chmod 600 .env

Then confirm the machine can authenticate:

    python proposal_reviewer.py --config configs/uva-kimi-example.yaml --check-provider

Both login and compute nodes reach the API, so this works before you claim a node.

#### PDF output on HPC

    module load texlive/2025           # provides xelatex

That is the whole requirement, but it must be loaded in the shell that runs the
review, since rendering happens at the end of the run. Without it you still get
`.md` and `.html`, and the run prints `no PDF engine found`.

**Do not use weasyprint on Rivanna/Afton.** `pip install weasyprint` succeeds and
then every render fails, because RHEL 8's libpango predates a symbol it needs
(`undefined symbol: pango_context_set_round_glyph_positions`). TeX Live is the
working route here.

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

    # Collect reports somewhere else (useful when the proposal sits on a
    # read-only share, or you want a run history separate from the draft):
    python proposal_reviewer.py PROPOSAL --config CONFIG --outdir ~/reviews/spirale

`PROPOSAL` can be `.docx`, `.pdf`, `.md`, `.tex`, or `.txt`. Reports are written
next to the proposal file unless `--outdir DIR` says otherwise; the directory is
created if missing, and an unusable one is rejected before any model is called
rather than after a full run.

## Running a review on HPC, step by step

A full review is ~30 model calls against a long document and takes 15 minutes or
more. That is too long for a login node and too long to trust an SSH connection, so
run it under tmux on a compute node.

**Order matters: tmux on the login node, `ijob` inside it.** Reversed, a dropped
connection kills the job.

    # 1. On the login node, start a named tmux session
    tmux new -s review

    # 2. Inside tmux, claim a compute node. Find your allocation with `allocations`.
    ijob -A <your_allocation> -p standard -c 1 -t 2:00:00

    # One core is enough: with a UVA backend the engine only waits on HTTP, so
    # nothing compute-heavy runs locally. Ask for more only if you point the
    # config at models running on the node itself.

    # 3. On the compute node, set up the shell
    module load texlive/2025
    cd ~/git/proposal-reviewer

    # 4. Validate the config against the proposal first — no model calls, seconds
    python proposal_reviewer.py /path/to/PROPOSAL.docx \
        --config configs/arpa-h-igor.yaml --dry-run

    # 5. Run it
    python proposal_reviewer.py /path/to/PROPOSAL.docx \
        --config configs/arpa-h-igor.yaml \
        --models "uva:Kimi K2.5" "uva:Kimi K2.5"

Detach with `Ctrl-b d` and log out; the run continues. Reattach later with:

    tmux attach -t review

Check on it without attaching using `squeue -u $USER` to confirm the job is alive.

Step 4 is not optional in practice. `--dry-run` reports which sections each
criterion actually matched and whether any got truncated, which is where config
mistakes surface — after step 5 you have paid for the whole run to find out.

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

The `.md` and `.html` are always written. The `.pdf` needs a Unicode-capable engine
on PATH — the run tries wkhtmltopdf, weasyprint, then xelatex/lualatex against
several Unicode fonts, and reports `no PDF engine found` rather than failing if none
works. On HPC that means `module load texlive/2025` before the run.

Files land beside the proposal unless `--outdir DIR` says otherwise.
