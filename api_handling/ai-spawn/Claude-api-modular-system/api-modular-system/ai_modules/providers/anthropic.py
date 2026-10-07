from .base import Provider


class AnthropicProvider(Provider):
    name = "anthropic"
    default_model = "claude-sonnet-5-5"
    api_key_env = "ANTHROPIC_API_KEY"

    def default_base_url(self):
        return "https://api.anthropic.com"

    def generate(self, prompt, system=None, max_tokens=1024, temperature=None):
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        data = self._post(f"{self.base_url}/v1/messages", payload, headers)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        if not text:
            raise self._fail("response contained no text", data)
        return text


def setup(ai_model, model=None, api_key=None, **options):
    """Plug Anthropic (Claude) into `ai_model` (an AIModel) and return it.

        from ai_modules import AIModel
        from ai_modules.providers import anthropic

        ai = anthropic.setup(AIModel())
    """
    ai_model.attach(AnthropicProvider(model=model, api_key=api_key, **options))
    return ai_model
