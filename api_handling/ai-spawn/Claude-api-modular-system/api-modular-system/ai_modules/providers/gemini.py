from .base import Provider


class GeminiProvider(Provider):
    name = "gemini"
    default_model = "gemini-2.5-flash"
    api_key_env = "GEMINI_API_KEY"

    def default_base_url(self):
        return "https://generativelanguage.googleapis.com/v1beta"

    def generate(self, prompt, system=None, max_tokens=1024, temperature=None):
        config = {"maxOutputTokens": max_tokens}
        if temperature is not None:
            config["temperature"] = temperature
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": config,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"{self.base_url}/models/{self.model}:generateContent"
        data = self._post(url, payload, {"x-goog-api-key": self.api_key})
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError):
            raise self._fail("unexpected response shape (possibly blocked by safety filters)", data)
        text = "".join(p.get("text", "") for p in parts)
        if not text:
            raise self._fail("response contained no text", data)
        return text


def setup(ai_model, model=None, api_key=None, **options):
    """Plug Google Gemini into `ai_model` (an AIModel) and return it.

        from ai_modules import AIModel
        from ai_modules.providers import gemini

        ai = gemini.setup(AIModel())
    """
    ai_model.attach(GeminiProvider(model=model, api_key=api_key, **options))
    return ai_model
