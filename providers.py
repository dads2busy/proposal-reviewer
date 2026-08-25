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


def load_dotenv(path=None, environ=None):
    """Load KEY=value pairs from a .env file into `environ`.

    Only sets keys that are absent, so a real environment variable (SLURM job
    script, systemd unit, Kubernetes secret) always beats a stale .env file.
    Returns the number of keys set. A missing file is not an error."""
    path = Path(path) if path else DEFAULT_ENV_PATH
    environ = os.environ if environ is None else environ
    if not path.exists():
        return 0
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
    "ollama": {"kind": "ollama", "base_url": "http://localhost:11434/v1/"},
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
    url = base + "chat/completions"
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
        payload = {"model": model, "temperature": temperature,
                   "messages": [{"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}]}
        if kind == "ollama" and num_ctx is not None:
            payload["options"] = {"num_ctx": num_ctx}
        body = json.dumps(payload).encode()

        for attempt in range(MAX_HTTP_ATTEMPTS):
            try:
                raw = transport(url, body, dict(headers), timeout)
                return json.loads(raw)["choices"][0]["message"]["content"]
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
