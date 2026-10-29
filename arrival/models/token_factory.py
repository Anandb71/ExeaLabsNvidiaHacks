"""Nebius Token Factory client (OpenAI-compatible). Unused by the demo path.
Credentials come from arrival.config: the environment, else an untracked .env in the cwd or repo root (read only
when a value is unset, never overriding set variables)."""

from __future__ import annotations

import json
import urllib.request

from arrival import config as settings
from arrival.config import load_dotenv  # noqa: F401  (kept importable from here)

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"  # confirm against https://docs.tokenfactory.nebius.com/

ROLES = {
    "omni": "NEMOTRON_OMNI_MODEL",  # read the scan
    "ultra": "NEMOTRON_ULTRA_MODEL",  # assemble twin, fill gaps
    "super": "NEMOTRON_SUPER_MODEL",  # propose tasks
    "lightning": "NEMOTRON_LIGHTNING_MODEL",  # wordings, labels
}


class TokenFactoryNotConfigured(RuntimeError):
    pass


def config() -> dict:
    return {
        "api_key": settings.get("TOKEN_FACTORY_API_KEY"),
        "base_url": settings.get("TOKEN_FACTORY_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        "models": {role: settings.get(var) for role, var in ROLES.items()},
    }


def complete(
    messages: list[dict], role: str = "super", model: str | None = None, timeout: float = 60.0, **params
) -> str:
    """One chat completion from Nebius Token Factory: the model for `role` (or `model`), `messages` and any extra
    request parameters. Returns the reply's text. Raises TokenFactoryNotConfigured without a key or model id, and
    ValueError when the reply has no text (no choices, null content or an error object)."""
    cfg = config()
    if not cfg["api_key"]:
        raise TokenFactoryNotConfigured("TOKEN_FACTORY_API_KEY is not set. See docs/08-access.md.")
    model = model or cfg["models"].get(role)
    if not model:
        raise TokenFactoryNotConfigured(f"No model id for role {role!r}; set {ROLES.get(role, 'a model env var')}.")
    body = json.dumps({"model": model, "messages": messages, **params}).encode()
    req = urllib.request.Request(
        cfg["base_url"] + "/chat/completions",
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        reply = json.load(resp)
    # a reply can come back with no text: empty `choices`, `content: null` when the token limit ran out during
    # reasoning, or an `error` object; say so instead of an IndexError/KeyError or None reaching the caller's parser
    choices = reply.get("choices") if isinstance(reply, dict) else None
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = first.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ValueError(
            f"Token Factory reply from {model} has no message text "
            f"(finish_reason={first.get('finish_reason')!r}): {json.dumps(reply)[:200]}"
        )
    return content
