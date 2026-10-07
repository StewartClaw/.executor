from urllib.parse import quote
from .base import HTTPProvider
from ..errors import ProviderError


class GeminiProvider(HTTPProvider):
    def _generate(self, data, instruction):
        name = self.model.removeprefix("models/")
        response = self._post(f"/models/{quote(name, safe='')}:generateContent", {"x-goog-api-key": self._api_key},
                              {"systemInstruction": {"parts": [{"text": instruction}]},
                               "contents": [{"role": "user", "parts": [{"text": data}]}],
                               "generationConfig": {"maxOutputTokens": self.max_output_tokens}})
        candidates = response.get("candidates", [])
        if not candidates:
            raise ProviderError("Gemini returned no candidate (possibly blocked)")
        if candidates[0].get("finishReason") not in (None, "STOP"):
            raise ProviderError("Gemini response was blocked or incomplete")
        return "\n".join(p["text"] for p in candidates[0]["content"]["parts"]
                         if "text" in p and not p.get("thought"))


def setup_gemini(*, model, api_key=None, base_url="https://generativelanguage.googleapis.com/v1beta", **kwargs):
    return GeminiProvider(model=model, api_key=api_key, env_key="GEMINI_API_KEY", base_url=base_url, **kwargs)
