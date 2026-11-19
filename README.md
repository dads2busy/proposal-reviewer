# proposal-reviewer

Config-driven compliance review for grant proposals. Two models score a proposal
against a rubric defined in a YAML config, a deterministic keyword pass checks for
required phrases, and the results are merged into a combined report (`.md`, `.html`,
`.pdf`). The models run on a local [Ollama](https://ollama.com), on UVA's
self-hosted RC GenAI service, or one on each.

No draft ever reaches a commercial provider. The point is to review a proposal
repeatedly, at no cost, while the draft is still changing. See
[Model backends](#model-backends) for where the models can run.

## Why config-driven

The engine (`proposal_reviewer.py`) knows nothing about any particular proposal.
Everything specific to a funding opportunity (criteria, point caps, section map,
keywords, model choices) lives in a YAML file under `configs/`. Reviewing a new
proposal means adding a config, not editing the engine.

## How a review runs

```mermaid
flowchart TD
    A["Extract text<br/>(.docx / .pdf / .md / .tex / .txt)"] --> B["Parse numbered sections<br/>(trim at stop_marker)"]
    B --> C{"--dry-run?"}
    C -- yes --> C1["Report section mapping,<br/>truncation, keyword coverage"] --> X1(["Stop: no model calls"])
    C -- no --> D{"Ping both models<br/>(skip with --no-check)"}
    D -- a slot fails --> X2(["Stop: nothing reviewed"])
    D -- both OK --> E["Model A scores every criterion"]
    E --> F["Model B scores every criterion"]
    F --> G["Keyword scan<br/>(deterministic regex)"]
    G --> H["Synthesize top recommendations<br/>(falls back to a fixed ranking)"]
    H --> I["Build combined report"]
    I --> J["Write .md, render .html and .pdf"]
```

Each criterion sees only its own section (`section: "1.2"`) or the whole document
(`section: "*"`), capped at `max_section_chars`. A model's report is written once it
has scored every criterion, so model A's report exists before model B starts; the
combined report comes last. The synthesis is written by model A unless the config
names a `synthesis_model`. If that call fails, the report falls back to a fixed
ranking of the gaps.

## Setup

There are two supported environments. The engine is the same in both; only the
model backend and the PDF toolchain differ.

### A. Your own machine (macOS, local Ollama)

    brew install ollama pandoc poppler  # poppler provides pdftotext
    pip install pyyaml
    ollama pull gemma3:27b              # two DIFFERENT families; both ~27-32B models
    ollama pull qwen2.5:32b             # run comfortably on a 64 GB machine
    # PDF rendering needs a Unicode-capable engine, one of:
    pip install weasyprint              # or a TeX with xelatex (brew install --cask mactex-no-gui)

Requires Ollama 0.19 or later (MLX backend on Apple Silicon). The UVA backend does
not work from a laptop, even on the VPN; see [Model backends](#model-backends).

### B. UVA HPC (Rivanna / Afton)

No root is needed; everything installs into your home directory. Do this once:

    # 1. Clone
    git clone git@github.com:dads2busy/proposal-reviewer.git ~/git/proposal-reviewer
    cd ~/git/proposal-reviewer

    # 2. pandoc >= 3. The system pandoc (/usr/bin/pandoc) is 2.0.6, which is too old
    #    for two separate reasons: it writes setext headings the section parser
    #    cannot see, and its LaTeX output no longer compiles under TeX Live 2025.
    mkdir -p ~/.local/bin
    curl -sSL https://github.com/jgm/pandoc/releases/download/3.1.11/pandoc-3.1.11-linux-amd64.tar.gz \
        | tar xz -C /tmp
    cp /tmp/pandoc-3.1.11/bin/pandoc ~/.local/bin/

    # 3. ~/.local/bin must come BEFORE /usr/bin, or you silently get 2.0.6 back
    grep -q '.local/bin' ~/.bashrc || echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
    source ~/.bashrc
    pandoc --version | head -1          # must report 3.x

    # 4. Python dependency (pdftotext is already at /usr/bin/pdftotext)
    pip install --user pyyaml

    # 5. API key for UVA RC GenAI (.env.example says where to get it)
    cp .env.example .env && chmod 600 .env

Then confirm that the node can authenticate:

    python proposal_reviewer.py --config configs/uva-kimi-example.yaml --check-provider

Run this on the node that will do the review, and again whenever you move to a
different one. Whether the API answers depends on the node, not the key (see
[Model backends](#model-backends)).

#### Local Ollama on HPC (optional)

Ollama runs on a GPU compute node without root. Install the Linux release archive
(`ollama-linux-amd64`, from the
[Ollama releases page](https://github.com/ollama/ollama/releases)) into `~/.local`,
so that you have `~/.local/bin/ollama` and its CUDA libraries in
`~/.local/lib/ollama/`. The binary finds them relative to itself. Then pull the
models once:

    ollama serve &                      # needed even to pull
    ollama pull gemma3:27b              # 17 GB
    ollama pull qwen2.5:32b             # 24 GB; both land in ~/.ollama

**Nothing starts the server for you.** The engine does not launch Ollama. Start
`ollama serve` in every job, on the compute node, before the review; it stops when
the allocation ends. If you forget, a full run stops at once with
`Ollama is not running` and the command to start it, before reviewing anything.

Which GPU partition to use depends on the backends:

- **Ollama only:** any GPU partition. `bii-gpu` with `--gres=gpu:v100:1` is tested.
- **Hybrid (UVA + Ollama):** the general `gpu` partition with `--gres=gpu:1`. The
  UVA API has refused `bii-gpu` in every check so far. On 2026-11-19 Slurm placed a
  `gpu` job on an RTX A6000 (48 GB), where `qwen2.5:32b` fits with room to spare.
  `--gres=gpu:1` takes whatever card is free, so a smaller card may push part of the
  model onto the CPU.

Two things on a V100 look alarming but are harmless:

- The serve log says `skipping CUDA device: compute capability not in compiled
  architectures`. Only the `cuda_v13` build rejects the V100; Ollama falls back to
  `cuda_v12` and offloads every layer.
- The two models do not fit in 32 GB of VRAM together. The engine runs one model
  over every criterion before switching, so a run pays for one model swap, not one
  per criterion.

#### PDF output on HPC

    module load texlive/2025           # provides xelatex

Load it in the shell that runs the review, since rendering happens at the end of
the run. Without it you still get `.md` and `.html`, and the run prints
`no PDF engine found`.

**Do not use weasyprint on Rivanna/Afton.** `pip install weasyprint` succeeds, but
every render then fails because RHEL 8's libpango lacks a symbol it needs
(`undefined symbol: pango_context_set_round_glyph_positions`). Use TeX Live.

## Model backends

A model name may carry a provider prefix. A bare name means local Ollama, so older
configs need no changes:

    models: ["gemma3:27b", "qwen2.5:32b"]            # both local Ollama
    models: ["uva:Kimi K2.5", "ollama:qwen2.5:32b"]  # hybrid
    models: ["uva:Kimi K2.5", "uva:Kimi K2.5"]       # UVA only

"Local" means the machine running the review: your laptop, or an HPC GPU node with
`ollama serve` running. Ollama is called on its native `/api/chat` endpoint, so the
config's `num_ctx` takes effect, and its thread count is capped to the cores the job
holds.

Other providers are declared in the config and take their key from the environment,
never from the file:

    providers:
      uva:
        kind: openai
        base_url: https://open-webui.rc.virginia.edu/api/
        api_key_env: UVARC_GenAI_API

Copy `.env.example` to `.env` and fill in the key. `.env` is gitignored, and the
engine tightens it to mode 600 when it reads it. A literal `api_key:` in a config
is rejected.

An Ollama on another machine can be declared the same way, with `kind: ollama` and
a `base_url` such as `http://<gpu-node>:11434/api/`. That server must be started
with `OLLAMA_HOST=0.0.0.0:11434` to accept outside connections, which also opens it
to anyone on the cluster network while it runs. We have not yet run a review this
way.

### UVA RC GenAI access

**Being on an HPC node is necessary but not sufficient.** A VPN-connected laptop
gets HTTP 401 even with a valid key, because Apache rejects the request before the
key is checked. Browser access via NetBadge works from anywhere, so it says nothing
about API access. Some HPC nodes are refused the same way. The sign is a 401 that is
identical with a valid key and with none, and carries no `WWW-Authenticate` header.
A rejected key would instead come back as JSON from Open WebUI. So when this
happens, do not start editing `.env`.

What we have observed so far:

| Where | UVA API | Last checked |
|---|---|---|
| Login node | allowed | 2026-11-19 |
| `standard` partition | allowed | 2026-09-02 |
| `gpu` partition | allowed | 2026-11-19 |
| `bii-gpu` partition | **401** | 2026-11-19 |
| VPN-connected laptop | **401** | 2026-08-25 |

This is not a published policy and may change, so check the node you are on:

    python proposal_reviewer.py --config CONFIG --check-provider

Even from an allowed node, the service sometimes drops a reply partway through or
stalls until the read times out. The engine retries these, as it does a 5xx or a
429, up to three attempts with a short backoff, and prints a line to stderr for each
retry:

    uva: IncompleteRead from open-webui.rc.virginia.edu; retrying in 1s (attempt 2/3)

A run on 2026-11-19 logged nine such retries and recovered from all of them. A call
that fails all three attempts stops the run with
`dropped the reply 3 times in a row`. A timeout against Ollama is not retried,
because that call has already waited its full timeout.

UVA hosts a single model, so a UVA-only config runs the same model twice. The engine
detects this and changes the wording of the combined report: two samples show
whether a score is stable, not whether two independent reviewers agree. It also
raises the second sample's temperature to 0.6 so the second call samples more
widely. For two independent opinions, use a hybrid run.

See `configs/uva-kimi-example.yaml` for a complete example.

## Use

    # Validate a config against a proposal without calling any model:
    python proposal_reviewer.py PROPOSAL --config configs/cdc-ip-26-0102.yaml --dry-run

    # Full review, with the models named in the config:
    python proposal_reviewer.py PROPOSAL --config configs/cdc-ip-26-0102.yaml

    # Override the config's models for one run:
    python proposal_reviewer.py PROPOSAL --config CONFIG --models "uva:Kimi K2.5" qwen2.5:32b

    # Confirm this node can reach and authenticate against each backend:
    python proposal_reviewer.py --config CONFIG --check-provider

    # Skip the backend check that a full run makes first:
    python proposal_reviewer.py PROPOSAL --config CONFIG --no-check

    # Write reports somewhere other than beside the proposal:
    python proposal_reviewer.py PROPOSAL --config CONFIG --outdir ~/reviews/spirale

`PROPOSAL` can be `.docx`, `.pdf`, `.md`, `.tex`, or `.txt`. `--outdir` is useful
when the proposal sits on a read-only share or you want run history kept apart from
the draft. The directory is created if missing, and an unusable one is rejected
before any model is called.

A full run starts by pinging both models (the `--check-provider` check) and stops if
either fails. Model A reviews every criterion before model B makes any call, so
without this check a dead second backend would show up only after model A's whole
pass. A local model gets up to ten minutes to answer the ping, since that call is
what loads it into memory; a remote one gets one minute. `--dry-run` makes no model
calls and skips the check.

**Proposals in several files.** The engine reviews one file. When a submission
spreads the scored material across several (a narrative, a schedule spreadsheet, a
task document), merge them into one Markdown file with `#` headings first, and
review that. The IGoR proposal does this with a `build_review_input.py` kept beside
the proposal, not in this repo.

## Running a review on HPC, step by step

A full review is about 35 model calls. A hybrid run of the IGoR proposal on
2026-11-19 took 58 minutes: 46 for Kimi to score 16 criteria, 6 for `qwen2.5:32b`
on an A6000, and 6 for the synthesis on Kimi. That is too long for a login node and
too long to trust to one SSH connection, so run it under tmux.

**Order matters: start tmux on the login node, then `ijob` inside it.** The other way
round, a dropped connection kills the job.

    # 1. On the login node, note its name, then start a named tmux session
    hostname                            # e.g. udc-ba38-32c0; you need it to reattach
    tmux new -s review

    # 2. Inside tmux, claim a compute node. Find your allocation with `allocations`.
    #    UVA only: no GPU, one core
    ijob -A <your_allocation> -p standard -c 1 -t 2:00:00
    #    Local Ollama only: any GPU partition
    ijob -A <your_allocation> -p bii-gpu --gres=gpu:v100:1 -c 4 -t 3:00:00
    #    Hybrid (UVA + local Ollama): the general gpu partition
    ijob -A <your_allocation> -p gpu --gres=gpu:1 -c 4 -t 3:00:00

    # A UVA-only run just waits on HTTP, so one core is enough. With local Ollama,
    # 2-4 cores is plenty for a GPU-resident model; more buys nothing.

    # 3. On the compute node, set up the shell
    module load texlive/2025
    cd ~/git/proposal-reviewer
    ollama serve > /tmp/ollama-$USER.log 2>&1 &   # local Ollama only

    # 4. Validate the config against the proposal: no model calls, a few seconds
    python proposal_reviewer.py /path/to/PROPOSAL.docx \
        --config configs/arpa-h-igor.yaml --dry-run

    # 5. Confirm THIS node can reach the backends; step 4 does not test that.
    #    Step 6 checks too, but checking here tells you before the long run.
    python proposal_reviewer.py --config configs/arpa-h-igor.yaml \
        --check-provider --models "uva:Kimi K2.5" qwen2.5:32b

    # 6. Run it, saving the progress messages to a log in your home directory.
    #    Hybrid:
    python proposal_reviewer.py /path/to/PROPOSAL.docx \
        --config configs/arpa-h-igor.yaml \
        --models "uva:Kimi K2.5" qwen2.5:32b 2>&1 | tee ~/review-$(date +%Y%m%d).log
    #    UVA only:     --models "uva:Kimi K2.5" "uva:Kimi K2.5"
    #    Ollama only:  --models gemma3:27b qwen2.5:32b

    # 7. Optional: type `exit` now. The shell runs it when the review finishes,
    #    which ends the ijob and releases the node instead of holding it until
    #    the time limit. The engine never reads the keyboard, so the typed
    #    command waits for the shell.
    exit

Step 4 is not optional in practice. `--dry-run` shows which section each criterion
matched and whether any was truncated, which is where config mistakes show up. After
step 6 you have paid for the whole run to find out.

### Detaching and coming back

Detach with `Ctrl-b d` and log out; the run continues. Reattach with:

    tmux attach -t review

The session lives on the login node where you started it, and
`login.hpc.virginia.edu` may put you on a different one when you reconnect. If
`tmux attach` finds no session, go to the node you noted in step 1. From your
laptop, `-J` is the reliable way, because a hop from one login node to another may
be refused without a forwarded key:

    ssh -J login.hpc.virginia.edu udc-ba38-32c0   # the hostname from step 1
    tmux attach -t review

To check without attaching, `tail ~/review-*.log` shows the progress messages, and
`squeue -u $USER` shows whether the job is still alive. The run is finished when the
log ends with `Wrote per-model reports, keywords, and compliance_report_combined_...`.

### Getting the reports

Reports are written beside the proposal (or in `--outdir`). To read the PDF on your
laptop:

    scp "login.hpc.virginia.edu:~/path/to/compliance_report_combined_<ts>.pdf" .

Open OnDemand's file browser works too.

## Writing a config

See `configs/cdc-ip-26-0102.yaml` (section-targeted scoring) and
`configs/doe-genesis-16a.yaml` (whole-document scoring) for the two patterns, and
`configs/arpa-h-igor.yaml` for a proposal whose outline has no section numbers. The
full list of config keys, and a step-by-step for authoring a new config, is in
`.claude/rules/writing-configs.md`.

Each criterion targets a numbered narrative section (e.g. `section: "1.2"`) or the
whole document (`section: "*"`). If the draft's headings are not numbered, map them
to IDs with `heading_patterns` rather than falling back to `"*"`:

    heading_patterns:
      - {pattern: '^TA1\.\s[A-Z][a-z]', id: "4.1"}

Section text is capped per criterion at `max_section_chars` (default 14000) to stay
inside the context window that `num_ctx` sets (default 16384 tokens). Raise the two
together. `--dry-run` flags any section that would be cut and warns when the cap
exceeds what the window can hold. The model awards `0..max` points per criterion,
and section subtotals roll up via `section_groups` and `section_totals`. The
`keywords` list is a deterministic regex scan for exact, checkable requirements that
models sometimes miss (specific platforms, named hubs, required attachments).

## Output

Every run stamps its files with a shared `YYYYMMDD-HHMMSS` suffix (`<ts>`), so
repeated runs do not overwrite each other:

- `compliance_report_<model>_<ts>.md`: one per model
- `compliance_report_keywords_<ts>.md`: deterministic phrase coverage
- `compliance_report_combined_<ts>.md` / `.html` / `.pdf`: the merged report, with
  synthesized top recommendations, per-section scores for each model, criteria where
  the models diverge, gaps both models found, shared strengths, and the
  missing-phrase scan

The `.md` and `.html` are always written. The `.pdf` needs a Unicode-capable engine
on PATH. The run tries wkhtmltopdf, then weasyprint, then xelatex and lualatex with
several Unicode fonts, and prints `no PDF engine found` rather than failing if none
works. On HPC that means `module load texlive/2025` before the run. Pandoc's warning
about a missing `<title>` element during rendering is harmless.

Files land beside the proposal unless `--outdir DIR` says otherwise.

## Other tools

`page_reducer.py` suggests cuts for a proposal that is over its page limit. It sends
a `.docx` to a local Ollama model and writes a copy with suggested deletions struck
through and rewrites in red, to accept or reject in Word. It needs `python-docx`
(`pip install python-docx`) and runs on local Ollama only:

    python page_reducer.py PROPOSAL.docx --model qwen3.6:27b --target-reduction 0.20

## Development

The engine depends only on the Python standard library and PyYAML. The tests use
`unittest` and fake transports, so they make no network calls:

    python -m unittest discover -s tests

Project conventions for working on the engine are in `CLAUDE.md`.
