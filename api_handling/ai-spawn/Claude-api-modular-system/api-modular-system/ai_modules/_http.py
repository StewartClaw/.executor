"""Tiny dependency-free JSON-over-HTTP helper shared by all providers."""

import json
import urllib.error
import urllib.request


class ProviderError(RuntimeError):
    """Raised when a provider request fails or returns an unusable response."""

    def __init__(self, provider, message, status=None, body=None):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.status = status
        self.body = body


def post_json(provider, url, payload, headers=None, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise ProviderError(provider, f"HTTP {e.code}: {body[:500]}", e.code, body) from e
    except urllib.error.URLError as e:
        raise ProviderError(provider, f"connection failed: {e.reason}") from e
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ProviderError(provider, f"response was not JSON: {raw[:500]}", body=raw) from e
