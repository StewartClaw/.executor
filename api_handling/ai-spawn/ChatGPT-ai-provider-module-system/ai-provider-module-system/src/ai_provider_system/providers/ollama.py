from .base import HTTPProvider
from ..errors import ProviderError


class OllamaProvider(HTTPProvider):
    def _generate(self, data, instruction):
        response = self._post("/api/chat", {}, {"model": self.model,
                              "messages": [{"role": "system", "content": instruction},
                                           {"role": "user", "content": data}],
                              "stream": False, "options": {"num_predict": self.max_output_tokens}})
        if response.get("error") or response.get("done_reason") == "length":
            raise ProviderError("Ollama response failed or reached its output limit")
        return response["message"]["content"]


def setup_ollama(*, model, base_url="http://localhost:11434", **kwargs):
    return OllamaProvider(model=model, base_url=base_url, **kwargs)
