"""Run with: python -m unittest discover tests   (no network or API keys needed)"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_modules import AIModel, Provider, ProviderError, register_provider  # noqa: E402
from ai_modules.model import split_text  # noqa: E402


def fake_post(response):
    """Patch post_json so each call is recorded and returns `response`."""
    calls = []

    def _post(provider, url, payload, headers=None, timeout=120):
        calls.append({"provider": provider, "url": url, "payload": payload, "headers": headers or {}})
        return response(payload) if callable(response) else response

    return mock.patch("ai_modules.providers.base.post_json", _post), calls


class ProviderTests(unittest.TestCase):
    def test_anthropic(self):
        patch, calls = fake_post({"content": [{"type": "text", "text": "Sales rose."}]})
        with patch:
            out = AIModel("anthropic", api_key="k").summarize("a,b\n1,2", "Summarize")
        self.assertEqual(out, "Sales rose.")
        c = calls[0]
        self.assertEqual(c["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(c["headers"]["x-api-key"], "k")
        self.assertIn("anthropic-version", c["headers"])
        self.assertIn("<data>\na,b\n1,2\n</data>", c["payload"]["messages"][0]["content"])
        self.assertIn("system", c["payload"])
        self.assertNotIn("temperature", c["payload"])

    def test_openai(self):
        patch, calls = fake_post({"choices": [{"message": {"content": "ok"}}]})
        with patch:
            out = AIModel("openai", api_key="k", temperature=0.2).generate("hi")
        self.assertEqual(out, "ok")
        c = calls[0]
        self.assertEqual(c["url"], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(c["headers"]["Authorization"], "Bearer k")
        self.assertEqual(c["payload"]["messages"][0]["role"], "system")
        self.assertEqual(c["payload"]["temperature"], 0.2)

    def test_gemini(self):
        patch, calls = fake_post({"candidates": [{"content": {"parts": [{"text": "g"}]}}]})
        with patch:
            out = AIModel("gemini", api_key="k", model="gemini-2.5-pro").generate("hi")
        self.assertEqual(out, "g")
        c = calls[0]
        self.assertTrue(c["url"].endswith("/models/gemini-2.5-pro:generateContent"))
        self.assertEqual(c["headers"]["x-goog-api-key"], "k")
        self.assertIn("systemInstruction", c["payload"])

    def test_gemini_blocked_response(self):
        patch, _ = fake_post({"promptFeedback": {"blockReason": "SAFETY"}})
        with patch, self.assertRaises(ProviderError):
            AIModel("gemini", api_key="k").generate("hi")

    def test_ollama_needs_no_key(self):
        patch, calls = fake_post({"message": {"content": "local"}})
        with patch:
            out = AIModel("ollama", num_ctx=32768).generate("hi")
        self.assertEqual(out, "local")
        c = calls[0]
        self.assertEqual(c["url"], "http://localhost:11434/api/chat")
        self.assertFalse(c["payload"]["stream"])
        self.assertEqual(c["payload"]["options"]["num_ctx"], 32768)

    def test_custom_openai_compatible(self):
        patch, calls = fake_post({"choices": [{"message": {"content": "c"}}]})
        with patch:
            out = AIModel("custom", base_url="http://localhost:1234/v1/", model="m").generate("hi")
        self.assertEqual(out, "c")
        self.assertEqual(calls[0]["url"], "http://localhost:1234/v1/chat/completions")
        self.assertNotIn("Authorization", calls[0]["headers"])

    def test_custom_fully_configured(self):
        patch, calls = fake_post({"result": {"outputs": [{"text": "x"}]}})
        with patch:
            model = AIModel(
                "custom", base_url="https://api.example.com", api_key="k",
                path="/generate", auth_header="x-api-key", headers={"X-Team": "data"},
                build_request=lambda prompt, system, model, max_tokens, temp: {"input": prompt},
                response_path="result.outputs.0.text",
            )
            out = model.generate("hi")
        self.assertEqual(out, "x")
        c = calls[0]
        self.assertEqual(c["url"], "https://api.example.com/generate")
        self.assertEqual(c["payload"], {"input": "hi"})
        self.assertEqual(c["headers"], {"X-Team": "data", "x-api-key": "k"})

    def test_custom_bad_response_path(self):
        patch, _ = fake_post({"other": 1})
        with patch, self.assertRaises(ProviderError):
            AIModel("custom", base_url="http://x", response_path="a.b").generate("hi")


class AIModelTests(unittest.TestCase):
    def test_unknown_provider(self):
        with self.assertRaises(ValueError):
            AIModel("nope")

    def test_missing_key(self):
        with mock.patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            AIModel("anthropic")

    def test_key_from_env(self):
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "env"}):
            self.assertEqual(AIModel("openai").provider.api_key, "env")

    def test_from_config(self):
        m = AIModel.from_config({"provider": "ollama", "model": "qwen3", "max_tokens": 50})
        self.assertEqual((m.provider_name, m.model, m.max_tokens), ("ollama", "qwen3", 50))

    def test_dict_data_sent_as_json(self):
        patch, calls = fake_post({"message": {"content": "ok"}})
        with patch:
            AIModel("ollama").summarize({"revenue": [1, 2]}, "Describe")
        self.assertIn('"revenue"', calls[0]["payload"]["messages"][1]["content"])

    def test_chunked_summary(self):
        patch, calls = fake_post(lambda p: {"message": {"content": "note"}})
        data = "".join(f"row {i}\n" for i in range(100))
        with patch:
            out = AIModel("ollama").summarize(data, "Summarize", chunk_chars=200)
        self.assertEqual(out, "note")
        n_chunks = len(split_text(data, 200))
        self.assertGreater(n_chunks, 1)
        self.assertEqual(len(calls), n_chunks + 1)
        self.assertIn("Notes on part 1", calls[-1]["payload"]["messages"][1]["content"])

    def test_split_text_keeps_everything(self):
        text = "short\n" + "x" * 50 + "\nend\n"
        chunks = split_text(text, 20)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(c) <= 20 for c in chunks))

    def test_module_setup_plugs_into_empty_model(self):
        from ai_modules.providers import anthropic, custom, gemini, ollama, openai

        cases = [
            (anthropic, {"api_key": "k"}), (openai, {"api_key": "k"}), (gemini, {"api_key": "k"}),
            (ollama, {}), (custom, {"base_url": "http://x"}),
        ]
        for module, kwargs in cases:
            ai = AIModel()
            self.assertIs(module.setup(ai, **kwargs), ai)
            self.assertEqual(ai.provider_name, module.__name__.rsplit(".", 1)[1])

    def test_setup_swaps_provider(self):
        from ai_modules.providers import gemini, ollama

        ai = ollama.setup(AIModel(max_tokens=99))
        gemini.setup(ai, api_key="k")
        self.assertEqual((ai.provider_name, ai.max_tokens), ("gemini", 99))

    def test_generate_without_provider(self):
        with self.assertRaises(RuntimeError):
            AIModel().generate("hi")

    def test_register_setup_function(self):
        class Upper(Provider):
            name = "upper"
            default_model = "u"
            requires_api_key = False

            def generate(self, prompt, system=None, max_tokens=1024, temperature=None):
                return prompt.upper()

        def setup(ai_model, model=None, api_key=None, **options):
            return ai_model.attach(Upper(model=model, **options))

        register_provider("upper", setup)
        self.assertEqual(AIModel("upper").generate("hi"), "HI")

    def test_register_provider(self):
        class Echo(Provider):
            name = "echo"
            default_model = "echo-1"
            requires_api_key = False

            def generate(self, prompt, system=None, max_tokens=1024, temperature=None):
                return prompt.upper()

        register_provider("echo", Echo)
        self.assertEqual(AIModel("echo").generate("hi"), "HI")
        with self.assertRaises(TypeError):
            register_provider("bad", object)


if __name__ == "__main__":
    unittest.main()
