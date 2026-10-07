from .base import HTTPProvider
from ..errors import ProviderError


class AnthropicProvider(HTTPProvider):
    def _generate(self, data, instruction):
        response = self._post("/messages", {"x-api-key": self._api_key, "anthropic-version": "2023-06-01"},
                              {"model": self.model, "system": instruction,
                               "messages": [{"role": "user", "content": data}],
                               "max_tokens": self.max_output_tokens})
        if response.get("stop_reason") == "max_tokens":
            raise ProviderError("Anthropic response reached its output limit")
        return "\n".join(b["text"] for b in response["content"] if b.get("type") == "text")


def setup_anthropic(*, model, api_key=None, base_url="https://api.anthropic.com/v1", **kwargs):
    return AnthropicProvider(model=model, api_key=api_key, env_key="ANTHROPIC_API_KEY", base_url=base_url, **kwargs)
