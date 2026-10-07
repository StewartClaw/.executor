import os

from .openai import OpenAIProvider


def _dig(data, path):
    """Follow a dotted path like 'choices.0.message.content' into nested JSON."""
    for key in path.split("."):
        data = data[int(key)] if isinstance(data, list) else data[key]
    return data


class CustomProvider(OpenAIProvider):
    """A manually configured endpoint.

    By default it speaks the OpenAI chat-completions format, which most hosted and
    self-hosted servers accept (LM Studio, vLLM, Groq, Together, OpenRouter, Azure, ...).
    For anything else, override the request and response handling:

        base_url      required, e.g. "http://localhost:1234/v1"
        path          endpoint path appended to base_url (default "/chat/completions")
        api_key       optional; sent as a Bearer token unless auth_header is set
        auth_header   header name for the key, e.g. "x-api-key" (sent without "Bearer ")
        headers       dict of extra headers
        build_request callable(prompt, system, model, max_tokens, temperature) -> JSON payload
        response_path dotted path to the text in the JSON response, e.g. "output.text"
    """

    name = "custom"
    requires_api_key = False

    def __init__(self, model=None, api_key=None, base_url=None, **options):
        base_url = base_url or os.environ.get("CUSTOM_AI_BASE_URL")
        if not base_url:
            raise ValueError("custom: base_url is required")
        super().__init__(
            model=model or os.environ.get("CUSTOM_AI_MODEL") or "default",
            api_key=api_key or os.environ.get("CUSTOM_AI_API_KEY"),
            base_url=base_url,
            **options,
        )

    def headers(self):
        h = dict(self.options.get("headers") or {})
        if self.api_key:
            auth = self.options.get("auth_header")
            if auth:
                h[auth] = self.api_key
            else:
                h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def generate(self, prompt, system=None, max_tokens=1024, temperature=None):
        builder = self.options.get("build_request")
        if builder:
            payload = builder(prompt, system, self.model, max_tokens, temperature)
        else:
            payload = self.build_payload(prompt, system, max_tokens, temperature)
        path = self.options.get("path", "/chat/completions")
        data = self._post(f"{self.base_url}{path}", payload, self.headers())
        response_path = self.options.get("response_path", "choices.0.message.content")
        try:
            text = _dig(data, response_path)
        except (KeyError, IndexError, TypeError, ValueError):
            raise self._fail(f"could not find text at '{response_path}' in response", data)
        if not isinstance(text, str) or not text:
            raise self._fail("response contained no text", data)
        return text


def setup(ai_model, model=None, api_key=None, **options):
    """Plug a manually configured custom endpoint into `ai_model` (an AIModel) and return it.

        from ai_modules import AIModel
        from ai_modules.providers import custom

        ai = custom.setup(AIModel(), base_url="http://localhost:1234/v1")
    """
    ai_model.attach(CustomProvider(model=model, api_key=api_key, **options))
    return ai_model
