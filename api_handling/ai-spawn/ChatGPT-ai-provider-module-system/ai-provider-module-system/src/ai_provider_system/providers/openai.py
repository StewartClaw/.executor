from .base import HTTPProvider
from ..errors import ProviderError


class OpenAIProvider(HTTPProvider):
    def _generate(self, data, instruction):
        response = self._post("/responses", {"Authorization": f"Bearer {self._api_key}"},
                              {"model": self.model, "instructions": instruction, "input": data,
                               "max_output_tokens": self.max_output_tokens, "store": False})
        if response.get("status") in ("failed", "incomplete"):
            raise ProviderError("OpenAI response failed or reached its output limit")
        return "\n".join(block["text"] for item in response.get("output", [])
                         if item.get("type") == "message" for block in item.get("content", [])
                         if block.get("type") == "output_text")


def setup_openai(*, model, api_key=None, base_url="https://api.openai.com/v1", **kwargs):
    return OpenAIProvider(model=model, api_key=api_key, env_key="OPENAI_API_KEY", base_url=base_url, **kwargs)
