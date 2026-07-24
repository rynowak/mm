"""OpenRouter image-edit client: seed image + instruction -> edited image.

Used to build identity-preserving bufo training data with a teacher model. The whole point
is EDIT mode (reference-conditioned) — feed a real bufo and ask for the same character in a
new pose/expression/angle — because free generation drifts identity (the v6 regression).

Runs locally (no GPU). Reads OPENROUTER_API_KEY from env.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request

OR_URL = "https://openrouter.ai/api/v1/chat/completions"


def _data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def _extract_image(resp: dict) -> str | None:
    """Return a base64 data URL for the first image found in an OpenRouter response."""
    try:
        msg = resp["choices"][0]["message"]
    except (KeyError, IndexError):
        return None
    for img in msg.get("images") or []:
        url = (img.get("image_url") or {}).get("url") or img.get("url")
        if isinstance(url, str) and url.startswith("data:image"):
            return url
    content = msg.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                url = (part.get("image_url") or {}).get("url") or part.get("url")
                if isinstance(url, str) and url.startswith("data:image"):
                    return url
    return None


def build_body(seed_path: str | list[str], instruction: str, model: str) -> dict:
    """Build the exact OpenRouter chat-completions request body (text + one or more images)."""
    seeds = [seed_path] if isinstance(seed_path, str) else list(seed_path)
    content: list[dict] = [{"type": "text", "text": instruction}]
    content += [{"type": "image_url", "image_url": {"url": _data_url(p)}} for p in seeds]
    return {"model": model, "messages": [{"role": "user", "content": content}], "modalities": ["image", "text"]}


def edit_image(
    seed_path: str | list[str],
    instruction: str,
    model: str,
    out_path: str,
    key: str | None = None,
    timeout: int = 240,
) -> str:
    """Edit seed image(s) per instruction with `model`; write result to out_path.

    seed_path may be a single path or a list of reference images of the SAME character
    (multi-image grounding markedly improves identity consistency). Returns out_path.
    """
    key = key or os.environ["OPENROUTER_API_KEY"]
    body = build_body(seed_path, instruction, model)
    req = urllib.request.Request(
        OR_URL,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/rynowak/mm",
            "X-Title": "bufo-teacher",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {e.read().decode()[:500]}") from e
    url = _extract_image(resp)
    if not url:
        raise RuntimeError("no image in response: " + json.dumps(resp)[:600])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(url.split(",", 1)[1]))
    return out_path
