from .base import Provider


class OpenAIProvider(Provider):
    name = "openai"
    default_model = "gpt-5"
    api_key_env = "OPENAI_API_KEY"

    def default_base_url(self):
        return "https://api.openai.com/v1"

    def build_payload(self, prompt, system, max_tokens, temperature):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages, "max_completion_tokens": max_tokens}
        if temperature is not None:
            payload["temperature"] = temperature
        return payload

    def headers(self):
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def generate(self, prompt, system=None, max_tokens=1024, temperature=None):
        payload = self.build_payload(prompt, system, max_tokens, temperature)
        data = self._post(f"{self.base_url}/chat/completions", payload, self.headers())
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise self._fail("unexpected response shape", data)
        if not text:
            raise self._fail("response contained no text", data)
        return text


def setup(ai_model, model=None, api_key=None, **options):
    """Plug OpenAI into `ai_model` (an AIModel) and return it.

        from ai_modules import AIModel
        from ai_modules.providers import openai

        ai = openai.setup(AIModel())
    """
    ai_model.attach(OpenAIProvider(model=model, api_key=api_key, **options))
    return ai_model
