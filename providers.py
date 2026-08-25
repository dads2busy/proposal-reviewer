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
