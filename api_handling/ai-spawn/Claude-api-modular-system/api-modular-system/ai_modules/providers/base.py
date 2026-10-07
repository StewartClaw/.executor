import os
from abc import ABC, abstractmethod

from .._http import ProviderError, post_json


class Provider(ABC):
    """Base class every provider module implements.

    A provider turns (system, prompt) into a plain text response. Subclasses set
    `name`, `default_model`, optionally `api_key_env`, and implement `generate`.
    """

    name = "base"
    default_model = None
    api_key_env = None  # environment variable read when no api_key is passed
    requires_api_key = True

    def __init__(self, model=None, api_key=None, base_url=None, timeout=120, **options):
        self.model = model or self.default_model
        if not self.model:
            raise ValueError(f"{self.name}: a model name is required")
        self.api_key = api_key or (os.environ.get(self.api_key_env) if self.api_key_env else None)
        if self.requires_api_key and not self.api_key:
            raise ValueError(
                f"{self.name}: no API key. Pass api_key=... or set {self.api_key_env}."
            )
        self.base_url = (base_url or self.default_base_url()).rstrip("/")
        self.timeout = timeout
        self.options = options  # provider-specific extras, e.g. extra headers

    def default_base_url(self):
        return ""

    def _post(self, url, payload, headers=None):
        return post_json(self.name, url, payload, headers, self.timeout)

    def _fail(self, message, body=None):
        return ProviderError(self.name, message, body=body)

    @abstractmethod
    def generate(self, prompt, system=None, max_tokens=1024, temperature=None):
        """Return the model's text response to `prompt`."""
