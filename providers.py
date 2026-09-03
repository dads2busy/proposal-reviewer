"""Model backends for proposal_reviewer.

Resolves "provider:model" strings to a chat callable, and loads secrets from
.env. Kept separate from the engine so the eventual web service can reuse it.
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_ENV_PATH = Path(__file__).resolve().parent / ".env"


def _usable_cpus():
    """Cores this process may actually run on — not cores the machine has.

    llama.cpp sizes its thread pool from the physical machine, so a Slurm job
    holding 4 of a node's 40 cores spawns 40 threads onto 4 and thrashes. On a
    V100 that measured 7.4 tok/s generation and 50.9 tok/s prompt eval against
    33.8 and 189.9 with the pool capped — the GPU sat at a third utilization and
    70W of 300W, starved rather than throttled, the whole time.

    sched_getaffinity is the only call that sees the cgroup, and it is
    Linux-only; macOS falls back to the machine count, which is the right answer
    there because a laptop is not partitioning cores away from us."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:                     # macOS, Windows
        return os.cpu_count() or 1


def _harden(path):
    """chmod 600 a .env others can read, and say so on the way.

    A key leaked through file permissions leaked as thoroughly as one committed
    to git, and a shared HPC filesystem makes a group-readable key a real
    exposure rather than a theoretical one."""
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        return
    if not mode & 0o077:
        return
    print(f"warning: {path} is readable by others (mode {mode:03o}); "
          f"tightening to 600.", file=sys.stderr)
    try:
        path.chmod(0o600)
    except OSError as e:
        print(f"warning: could not chmod {path}: {e}", file=sys.stderr)


def load_dotenv(path=None, environ=None):
    """Load KEY=value pairs from a .env file into `environ`.

    Only sets keys that are absent, so a real environment variable (SLURM job
    script, systemd unit, Kubernetes secret) always beats a stale .env file.
    Returns the number of keys set. A missing file is not an error."""
    path = Path(path) if path else DEFAULT_ENV_PATH
    environ = os.environ if environ is None else environ
    if not path.exists():
        return 0
    _harden(path)
    n = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in environ:
            environ[key] = value
            n += 1
    return n


BUILTIN_PROVIDERS = {
    # Ollama's NATIVE api, not its OpenAI-compat /v1/ path. The compat endpoint
    # accepts an `options` object and silently discards it, so num_ctx never
    # reached the model: asking /v1/ for num_ctx 4096 loaded a 32768 context,
    # while /api/ asked the same way loaded 4096.
    "ollama": {"kind": "ollama", "base_url": "http://localhost:11434/api/"},
}


class ProviderError(Exception):
    """Configuration or connectivity problem with a model backend."""


def known_provider_names(cfg):
    """Provider names this config can resolve: declared ones plus builtins."""
    return set(BUILTIN_PROVIDERS) | set((cfg or {}).get("providers") or {})


def split_model_ref(ref, known):
    """'uva:Kimi K2.5' -> ('uva', 'Kimi K2.5').

    A prefix only counts as a provider when it names a known one — otherwise
    the whole string is an Ollama model, so 'qwen2.5:32b' stays intact."""
    prefix, sep, rest = ref.partition(":")
    if sep and prefix in known:
        return prefix, rest
    return "ollama", ref


def provider_spec(cfg, name):
    """Merged spec for a provider. Raises ProviderError on unknown names or on
    a literal key in the config."""
    declared = ((cfg or {}).get("providers") or {}).get(name)
    if declared is None and name not in BUILTIN_PROVIDERS:
        available = ", ".join(sorted(known_provider_names(cfg)))
        raise ProviderError(
            f"Unknown provider {name!r}. Declare it under `providers:` in the "
            f"config, or use one of: {available}")
    spec = dict(BUILTIN_PROVIDERS.get(name, {}))
    spec.update(declared or {})
    if "api_key" in spec:
        raise ProviderError(
            f"Provider {name!r} sets `api_key` directly. Configs are committed "
            f"to git, so keys must not live in them. Use `api_key_env: VARNAME` "
            f"and put the value in .env instead.")
    return spec


MAX_HTTP_ATTEMPTS = 3


def _urllib_transport(url, body, headers, timeout):
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _parse_reply(raw):
    """Message content from either a single JSON completion or an SSE stream.

    UVA's Open WebUI always answers `text/event-stream` and ignores `stream: false`,
    so the client reassembles the deltas itself. Ollama returns one JSON object.

    Raises ValueError on any body that carries no usable text. That matters more
    than it looks: an upstream error arriving as HTTP 200 inside a stream frame
    would otherwise return "", which the scorer records as a genuine 0 with an
    empty justification and reports as a real gap."""
    text = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw

    # Sniff anywhere, not just at the start — a proxy holding a long generation
    # open prefixes the stream with ': keep-alive' comment or 'event:' frames.
    frames = [ln.strip()[len("data:"):].strip()
              for ln in text.splitlines() if ln.strip().startswith("data:")]

    if not frames:
        try:
            content = json.loads(text)["choices"][0]["message"]["content"]
        except json.JSONDecodeError as e:
            raise ValueError(
                "the body is neither JSON nor an SSE stream; an HTML body "
                "usually means a login page rather than an API response") from e
        if not isinstance(content, str):
            raise ValueError(f"message content was "
                             f"{type(content).__name__}, not text")
        return content

    parts = []
    for payload in frames:
        if not payload or payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue                      # keep-alive or comment frame
        for choice in chunk.get("choices") or []:
            piece = (choice.get("delta") or {}).get("content")
            if piece is None:
                piece = (choice.get("message") or {}).get("content")
            if piece is None:
                continue                  # role-only or finish_reason frame
            if not isinstance(piece, str):
                raise ValueError(f"a stream frame carried "
                                 f"{type(piece).__name__} content, not text")
            parts.append(piece)

    joined = "".join(parts)
    if not joined.strip():
        raise ValueError(
            "the stream carried no message content — an upstream error is "
            "often delivered this way, with HTTP 200")
    return joined


def _parse_ollama_reply(raw):
    """Message content from Ollama's native /api/chat response.

    A different shape from the OpenAI-compat one: {"message": {"content": ...}}
    rather than {"choices": [...]}. Kept separate from _parse_reply so the SSE
    reassembly that UVA depends on is not touched by local-backend changes."""
    text = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            "the body is not a single JSON object; a stream here means the "
            "request went out without `stream: false`") from e
    if doc.get("error"):
        raise ValueError(f"Ollama reported an error: {doc['error']}")
    content = (doc.get("message") or {}).get("content")
    if content is None:
        raise ValueError("the reply carried no message content")
    if not isinstance(content, str):
        raise ValueError(f"message content was "
                         f"{type(content).__name__}, not text")
    if not content.strip():
        # Same reasoning as the SSE path: silence scored as a real 0 with an
        # empty justification is worse than a loud failure.
        raise ValueError("the reply carried empty message content")
    return content


def make_chat_fn(cfg, provider_name, temperature=0.2, transport=None,
                 environ=None, sleep=None):
    """Build a chat callable for one provider.

    Returned signature is (model, system_prompt, user_prompt, timeout=900,
    num_ctx=None) -> str, matching what _chat_json expects."""
    spec = provider_spec(cfg, provider_name)
    transport = transport or _urllib_transport
    environ = os.environ if environ is None else environ
    sleep = sleep or time.sleep
    kind = spec.get("kind", "ollama")
    base = spec.get("base_url", "")
    if not base.endswith("/"):
        base += "/"
    if kind == "ollama":
        # A hand-written config may still point at the compat path; silently
        # dropping options is exactly the failure this endpoint change fixes,
        # so redirect rather than honour it.
        base = re.sub(r"/v1/$", "/api/", base)
        url = base + "chat"
    else:
        url = base + "chat/completions"
    parse = _parse_ollama_reply if kind == "ollama" else _parse_reply
    host = re.sub(r"^https?://([^/]+).*$", r"\1", base)

    headers = {"Content-Type": "application/json"}
    if kind == "openai":
        var = spec.get("api_key_env")
        if not var:
            raise ProviderError(
                f"Provider {provider_name!r} is kind 'openai' but sets no "
                f"`api_key_env`.")
        key = (environ.get(var) or "").strip()
        if not key:
            raise ProviderError(
                f"Provider {provider_name!r} needs ${var}, which is unset or "
                f"empty. Put it in .env (see .env.example) or export it.")
        headers["Authorization"] = f"Bearer {key}"

    def chat(model, system_prompt, user_prompt, timeout=900, num_ctx=None):
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        if kind == "ollama":
            # Native shape: sampling and runtime settings both live in
            # `options`, and the endpoint streams unless told not to.
            # num_thread rides every local call while num_ctx goes only when
            # asked for — an oversubscribed pool is wrong at any context size.
            options = {"temperature": temperature,
                       "num_thread": _usable_cpus()}
            if num_ctx is not None:
                options["num_ctx"] = num_ctx
            payload = {"model": model, "messages": messages,
                       "stream": False, "options": options}
        else:
            payload = {"model": model, "temperature": temperature,
                       "messages": messages}
        body = json.dumps(payload).encode()

        for attempt in range(MAX_HTTP_ATTEMPTS):
            try:
                raw = transport(url, body, dict(headers), timeout)
                try:
                    return parse(raw)
                except ValueError as e:   # includes json.JSONDecodeError
                    excerpt = (raw[:120] if isinstance(raw, (bytes, bytearray))
                               else str(raw)[:120])
                    raise ProviderError(
                        f"{provider_name}: {host} returned a reply this client "
                        f"could not use — {e}. First bytes: {excerpt!r}") from e
                except (KeyError, IndexError, TypeError, AttributeError,
                        UnicodeDecodeError) as e:
                    excerpt = (raw[:120] if isinstance(raw, (bytes, bytearray))
                               else str(raw)[:120])
                    raise ProviderError(
                        f"{provider_name}: {host} returned an unrecognized "
                        f"response shape ({type(e).__name__}). "
                        f"First bytes: {excerpt!r}") from e
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    raise ProviderError(
                        f"{provider_name}: key rejected by {host} (HTTP 401). "
                        f"Check the value in .env — and confirm you are on an "
                        f"HPC node: this API returns 401 from a VPN-connected "
                        f"laptop even with a valid key.") from e
                if e.code == 429 and attempt < MAX_HTTP_ATTEMPTS - 1:
                    sleep(2 ** attempt)
                    continue
                if e.code == 429:
                    raise ProviderError(
                        f"{provider_name}: rate limited by {host} after "
                        f"{MAX_HTTP_ATTEMPTS} attempts (HTTP 429).") from e
                if e.code >= 500 and attempt < MAX_HTTP_ATTEMPTS - 1:
                    sleep(2 ** attempt)
                    continue
                raise ProviderError(
                    f"{provider_name}: {host} returned HTTP {e.code}.") from e
            except urllib.error.URLError as e:
                raise ProviderError(
                    f"{provider_name}: cannot reach {host} ({e.reason}). "
                    f"On an HPC node? Is Ollama running?") from e
        raise ProviderError(f"{provider_name}: exhausted retries against {host}.")

    return chat
