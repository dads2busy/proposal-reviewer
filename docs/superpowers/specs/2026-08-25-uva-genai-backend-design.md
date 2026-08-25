# UVA RC GenAI backend — design

_2026-08-25_

## Problem

The engine can only talk to a local Ollama at `localhost:11434`. UVA Research Computing
now hosts an OpenAI-compatible LLM service (RC GenAI) running Kimi K2.5, inside UVA.
Reviews should be able to run against it: it is far more capable than a 27B local model,
it runs on hardware that outclasses a laptop, and — critically — it keeps unpublished
proposal drafts inside the institution.

This spec covers the **engine** only. The eventual web interface for research staff and
its deployment are a separate spec; see Non-goals.

## Revising the core invariant

`CLAUDE.md` currently states: _"The whole point is zero cloud/API usage... Do not add
cloud API calls to the engine."_

That was always a proxy for the real requirement, which is: **an unpublished proposal
draft must never reach a commercial third party.** "Runs on localhost" satisfied it;
so does "runs on UVA-hosted hardware inside UVA's network." The invariant gets rewritten
to state the requirement directly, and to keep prohibiting commercial API providers.

## Verified constraints

Established by testing on 2026-08-25, not assumed:

| Fact | Evidence |
|---|---|
| Base URL `https://open-webui.rc.virginia.edu/api/`, endpoint `/chat/completions` | live docs + successful call |
| Auth `Authorization: Bearer $UVARC_GenAI_API` | successful call |
| Model id is `Kimi K2.5` — **with a space** | live docs; the archived guide is stale |
| Rate limit 60 req/min | live docs |
| **API works only from an HPC node** | Rivanna login node → 200; VPN laptop → 401 with the same key |

The last row is the load-bearing one. From outside the HPC security zone, Apache returns
a 401 that is byte-identical with and without the `Authorization` header — the proxy
rejects the request before Open WebUI evaluates the token. Browser access works from
anywhere via Shibboleth session cookie, so **browser success does not imply API access
works.**

Consequence: live runs happen on Rivanna. Local development uses a fake `chat_fn` or a
local Ollama.

## Non-goals

- Web interface for research staff — separate spec.
- Deployment target (RC Kubernetes / VM / Open OnDemand) — separate spec. Two questions
  are open for RC: whether the K8s cluster reaches the RC GenAI API, and whether an RC VM
  can host a persistent NetBadge-authenticated app.
- Streaming responses. The engine needs whole JSON replies, not incremental tokens.
- Any commercial API provider.

## Design

### 1. Secrets

Three layers, each with one job:

- **`.env`** (repo root, already gitignored) holds the secret. The engine `chmod 600`s it
  on first read and warns if it is group- or world-readable.
- **`.env.example`** is committed, contains `UVARC_GenAI_API=` with no value, and documents
  where to get a key.
- **The config** names the variable, never the value: `api_key_env: UVARC_GenAI_API`.

At startup the engine loads `.env` from the repo root into `os.environ`, filling only
variables that are not already set — so a real environment variable (a SLURM job script,
a systemd unit, a Kubernetes secret) always wins over a stale `.env`. Strict `KEY=value`
parsing: blank lines and `#` comments ignored, surrounding quotes stripped, no shell
expansion, no `export` handling. Roughly 15 lines of stdlib; `python-dotenv` is not worth
weakening the stdlib-only rule for.

**A literal key in a config is a hard error.** If a config contains `api_key:` rather than
`api_key_env:`, the engine exits with a message pointing at `.env`. Configs are committed;
keys in them leak. RC's container policy independently forbids baking credentials into
images, so this is a deployment requirement, not only hygiene.

### 2. Provider layer

Model strings become `provider:model`. A bare name means `ollama`, so every existing
config keeps working untouched.

```yaml
providers:
  uva:
    kind: openai
    base_url: https://open-webui.rc.virginia.edu/api/
    api_key_env: UVARC_GenAI_API

models: ["uva:Kimi K2.5", "ollama:qwen2.5:32b"]
```

`kind: openai` POSTs `{model, messages, temperature}` to `{base_url}chat/completions` with
a Bearer header and reads `choices[0].message.content`. Same shape `_ollama_chat` already
builds, minus the Ollama-only `options.num_ctx`, which is dropped for `openai` providers
(context length is server-side there). Still stdlib `urllib`; no new dependencies.

`_ollama_chat` becomes one case of a `chat_fn` factory. The engine already threads
`chat_fn` through every call site, so this is a change at one seam rather than throughout.

Model strings split on the **first** colon only, so `ollama:qwen2.5:32b` resolves to
provider `ollama`, model `qwen2.5:32b`.

### 3. Errors

Failures map to actionable messages rather than tracebacks:

| Condition | Message |
|---|---|
| 401 | key rejected — check `$UVARC_GenAI_API`, **and confirm you are on an HPC node** (the API 401s from VPN) |
| connection refused / DNS failure | cannot reach `<host>` — on an HPC node? is Ollama running? |
| 429 | back off and retry with exponential delay, up to 3 attempts |
| 5xx | retry once, then fail with the status and body excerpt |

The 401 message naming the HPC-node requirement is deliberate: that single fact cost hours
to discover and is the most likely failure a new user hits.

### 4. `--check-provider`

A flag that makes one cheap call per configured provider and reports reachability, auth,
and (where the endpoint permits) the model list. This is the first thing to run when
something breaks, the way to confirm a given node is inside the security zone, and later
the web service's health check.

### 5. Honest reporting when both slots are one model

UVA hosts exactly one model. The combine step assumes two models from **different
families** and treats their independence as the signal worth trusting. Two samples of Kimi
are not independent: they share training data, biases, and blind spots. If the model is
systematically generous about budget justification, both samples are generous, and "both
models converge on this gap" degrades to "the model agrees with itself."

Two samples still buy something real — a **reliability** check, showing whether a score is
stable or the model is guessing. That is worth having. It is just not corroboration, and
the report must not claim otherwise.

The engine detects that both ensemble entries resolve to the same provider **and** model,
sets a `single_family` flag, and rewrites the combined report's language:

| Default wording | Under `single_family` |
|---|---|
| "Priority gaps — both models dock points" | "Priority gaps — consistent across both samples" |
| "★ = models diverge ≥2 pts" | "★ = unstable score, model is guessing" |
| "Strengths — both models score ≥80% of max" | "Strengths — consistently scored ≥80%" |
| "_offline (local models)_" | "_UVA RC GenAI — single model, two samples_" |

Plus a banner near the top stating plainly that the run had no independent second opinion,
and that agreement between samples reflects consistency rather than corroboration.

The second sample runs at a higher temperature (0.2 → 0.6). At 0.2 the two calls risk being
near-identical, making the second pure cost with no signal. Sampling wider is what makes
"is this score stable?" a real question.

The scoring pipeline is otherwise unchanged. When the two entries differ — a real
two-family ensemble, once RC adds a second model or a local Ollama is available alongside —
none of this activates and reports read exactly as they do today.

### 6. Rate limiting

60 req/min. A review is roughly `2 models × N criteria + 1 synthesis` sequential calls —
about 30 for a typical config, well inside the limit. No client-side throttle;
the 429 handler covers a shared-server surprise. Revisit if the web service runs
concurrent reviews.

## Testing

- **Provider resolution:** `provider:model` splitting, bare-name defaulting to ollama,
  first-colon-only splitting, unknown provider error.
- **`.env` loading:** precedence (real env wins), comments, quotes, blank lines, missing file.
- **Config rejection:** literal `api_key:` exits with the pointed error.
- **Payload shape:** `openai` provider omits `options.num_ctx`; ollama includes it. Asserted
  against a fake transport, not the network.
- **Error mapping:** injected 401 / 429 / connection error produce the right messages, and
  429 retries.
- **`single_family` detection:** same provider+model → flag set; differing → not set;
  report wording switches accordingly; second sample gets the raised temperature.

All of it runs offline against a fake `chat_fn` — the seam already exists and the current
tests already use it. Live verification is a single `--check-provider` run on Rivanna.

## Migration

Fully backward compatible. Existing configs name bare models, which resolve to `ollama`
with the current URL and behavior. No config edits required; no report changes for
existing two-family setups.

## Open questions

Neither blocks this work:

1. Will RC add a second model family, or can a local Ollama run alongside on an HPC node?
   Either would restore genuine ensemble independence and make `single_family` unnecessary.
2. Where does the eventual web service live, and does that host reach the RC GenAI API?
