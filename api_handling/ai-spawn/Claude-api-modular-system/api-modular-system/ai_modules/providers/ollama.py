import os

from .base import Provider


class OllamaProvider(Provider):
    """Local models served by Ollama (https://ollama.com). No API key needed."""

    name = "ollama"
    default_model = "llama3.1"
    requires_api_key = False

    def default_base_url(self):
        return os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def generate(self, prompt, system=None, max_tokens=1024, temperature=None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        options = {"num_predict": max_tokens}
        if temperature is not None:
            options["temperature"] = temperature
        # Ollama's default context window is small; raise it for large data via num_ctx.
        if "num_ctx" in self.options:
            options["num_ctx"] = self.options["num_ctx"]
        payload = {"model": self.model, "messages": messages, "stream": False, "options": options}
        data = self._post(f"{self.base_url}/api/chat", payload)
        text = (data.get("message") or {}).get("content")
        if not text:
            raise self._fail("response contained no text", data)
        return text


def setup(ai_model, model=None, api_key=None, **options):
    """Plug a local Ollama model into `ai_model` (an AIModel) and return it.

        from ai_modules import AIModel
        from ai_modules.providers import ollama

        ai = ollama.setup(AIModel())
    """
    ai_model.attach(OllamaProvider(model=model, api_key=api_key, **options))
    return ai_model
