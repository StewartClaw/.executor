import json
import math
import os
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from ..errors import ConfigurationError, ProviderError


def post_json(url, headers, payload, timeout):
    request = Request(url, data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        # Never surface raw response bodies, URLs or credentials in error text.
        raise ProviderError(f"Provider HTTP request failed (status {exc.code})") from None
    except (URLError, TimeoutError, OSError):
        raise ProviderError("Provider connection failed or timed out") from None
    except (ValueError, UnicodeError):
        raise ProviderError("Provider returned invalid JSON") from None


class HTTPProvider:
    def __init__(self, *, model, api_key=None, env_key=None, base_url, timeout=60,
                 max_output_tokens=1024, transport=None):
        if not isinstance(model, str) or not model.strip():
            raise ConfigurationError("model must be nonempty text")
        if not isinstance(base_url, str):
            raise ConfigurationError("base_url must be an HTTP(S) URL")
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment or parsed.username:
            raise ConfigurationError("base_url must be an HTTP(S) URL without credentials, query, or fragment")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ConfigurationError("timeout must be positive and finite")
        if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int) or max_output_tokens < 1:
            raise ConfigurationError("max_output_tokens must be a positive integer")
        key = api_key if api_key is not None else (os.environ.get(env_key) if env_key else None)
        if env_key and (not isinstance(key, str) or not key.strip()):
            raise ConfigurationError(f"Provide api_key or set {env_key}")
        if transport is not None and not callable(transport):
            raise ConfigurationError("transport must be callable")
        self.model = model
        self._api_key = key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self._transport = transport or post_json

    def _post(self, path, headers, payload):
        try:
            return self._transport(self.base_url + path, headers, payload, self.timeout)
        except ProviderError:
            raise
        except Exception:
            raise ProviderError("Provider transport failed") from None

    def generate(self, data, instruction):
        try:
            result = self._generate(data, instruction)
        except ProviderError:
            raise
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ProviderError("Provider returned an unexpected response shape") from None
        if not isinstance(result, str) or not result.strip():
            raise ProviderError("Provider returned no usable text (possibly blocked or incomplete)")
        return result.strip()
