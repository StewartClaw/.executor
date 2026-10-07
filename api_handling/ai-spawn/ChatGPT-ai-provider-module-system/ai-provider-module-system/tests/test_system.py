import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from ai_provider_system import (AIModel, setup_custom, setup_openai, setup_anthropic,
                                setup_gemini, setup_ollama, ConfigurationError,
                                ProviderError, SummarizationError)
from ai_provider_system.data import serialize_data, split_text
from ai_provider_system.providers.base import post_json


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.response


class SystemTests(unittest.TestCase):
    def test_01_text_and_instruction(self):
        calls = []
        model = AIModel(setup_custom(lambda d, i: calls.append((d, i)) or " summary "))
        self.assertEqual(model.summarize("hello", "Explain"), "summary")
        self.assertEqual(calls, [("hello", "Explain")])

    def test_02_dict(self):
        data = {"name": "caf\u00e9", "nested": {"value": 3}}
        self.assertEqual(json.loads(serialize_data(data)), data)
        self.assertIn("caf\u00e9", serialize_data(data))

    def test_03_list(self):
        self.assertEqual(json.loads(serialize_data([1, {"x": True}, None])), [1, {"x": True}, None])

    def test_04_dataframe(self):
        import pandas as pd
        frame = pd.DataFrame([[1, None], [2, 4]], columns=["same", "same"], index=["a", "b"])
        data = json.loads(serialize_data(frame))
        self.assertEqual(data["columns"], ["same", "same"])
        self.assertEqual(data["index"], ["a", "b"])
        self.assertEqual(data["data"], [[1, None], [2, 4]])

    def test_05_unsupported_data(self):
        for data in (123, None, (1, 2), object()):
            with self.assertRaises(TypeError):
                serialize_data(data)
        with self.assertRaises(TypeError):
            serialize_data({"object": object()})

    def test_06_empty_input(self):
        model = AIModel(setup_custom(lambda d, i: "summary"))
        with self.assertRaises(ValueError):
            model.summarize(" \n")
        self.assertEqual(model.summarize([]), "summary")

    def test_07_provider_configuration(self):
        model = AIModel()
        with self.assertRaises(ConfigurationError):
            model.summarize("x")
        with self.assertRaises(ConfigurationError):
            model.set_provider(object())
        self.assertIs(model.set_provider(setup_custom(lambda d, i: "ok")), model)
        self.assertEqual(model.summarize("x"), "ok")

    def test_08_invalid_limits_and_instructions(self):
        for value in (0, -1, True, 2.5):
            with self.assertRaises(ConfigurationError):
                AIModel(max_input_chars=value)
            with self.assertRaises(ConfigurationError):
                AIModel(max_rounds=value)
        model = AIModel(setup_custom(lambda d, i: "ok"), max_input_chars=20)
        for value in (None, "", "  "):
            with self.assertRaises(ConfigurationError):
                model.summarize("x", value)
        with self.assertRaises(ConfigurationError):
            model.summarize("x" * 100, "Summarize")

    def test_09_lossless_chunking(self):
        for text in ("abc" * 100, "a b\n" * 100, "\U0001f600" * 100, ""):
            parts = split_text(text, 17)
            self.assertEqual("".join(parts), text)
            self.assertTrue(all(0 < len(p) <= 17 for p in parts))
        with self.assertRaises(ValueError):
            split_text("abc", 0)

    def test_10_chunk_then_merge(self):
        calls = []
        model = AIModel(setup_custom(lambda d, i: calls.append((d, i)) or "fact"), max_input_chars=300)
        self.assertEqual(model.summarize("abcdefgh" * 100, "Summarize"), "fact")
        self.assertGreater(len(calls), 2)
        self.assertTrue(all(len(d) + len(i) <= 300 for d, i in calls))
        self.assertEqual("".join(d for d, _ in calls[:-1]), "abcdefgh" * 100)
        self.assertIn("Combine these partial summaries", calls[-1][1])

    def test_11_hierarchical_reduction(self):
        calls = []
        def callback(data, instruction):
            calls.append((data, instruction))
            return data[:10]
        model = AIModel(setup_custom(callback), max_input_chars=300)
        self.assertTrue(model.summarize("abcdefghij" * 2000, "Summarize"))
        self.assertTrue(all(len(d) + len(i) <= 300 for d, i in calls))
        first_round_count = len(split_text("abcdefghij" * 2000, len(calls[0][0])))
        # More than the original pieces plus final synthesis proves another reduction pass.
        self.assertGreater(len(calls), first_round_count + 1)

    def test_12_convergence_guard(self):
        model = AIModel(setup_custom(lambda d, i: d), max_input_chars=300)
        with self.assertRaises(SummarizationError):
            model.summarize("x" * 1000, "Summarize")
        model = AIModel(setup_custom(lambda d, i: "x" * 10), max_input_chars=300, max_rounds=1)
        with self.assertRaises(SummarizationError):
            model.summarize("x" * 20000, "Summarize")

    def test_13_openai_request(self):
        fake = FakeTransport({"status": "completed", "output": [
            {"type": "reasoning"}, {"type": "message", "content": [
                {"type": "output_text", "text": "one"}, {"type": "output_text", "text": "two"}]}]})
        provider = setup_openai(model="test-model", api_key="fake-key", transport=fake)
        self.assertEqual(provider.generate("data", "instruction"), "one\ntwo")
        url, headers, payload, timeout = fake.calls[0]
        self.assertEqual(url, "https://api.openai.com/v1/responses")
        self.assertEqual(headers["Authorization"], "Bearer fake-key")
        self.assertEqual(payload["instructions"], "instruction")
        self.assertEqual(payload["input"], "data")
        self.assertFalse(payload["store"])

    def test_14_anthropic_request(self):
        fake = FakeTransport({"content": [{"type": "thinking"}, {"type": "text", "text": "summary"}]})
        provider = setup_anthropic(model="test-model", api_key="fake-key", transport=fake, max_output_tokens=200)
        self.assertEqual(provider.generate("data", "instruction"), "summary")
        url, headers, payload, _ = fake.calls[0]
        self.assertTrue(url.endswith("/v1/messages"))
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(headers["x-api-key"], "fake-key")
        self.assertEqual(payload["max_tokens"], 200)
        self.assertEqual(payload["system"], "instruction")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "data"}])

    def test_15_gemini_request(self):
        fake = FakeTransport({"candidates": [{"finishReason": "STOP", "content": {"parts": [
            {"thought": True, "text": "private"}, {"text": "summary"}]}}]})
        provider = setup_gemini(model="models/test-model", api_key="fake-key", transport=fake)
        self.assertEqual(provider.generate("data", "instruction"), "summary")
        url, headers, payload, _ = fake.calls[0]
        self.assertTrue(url.endswith("/models/test-model:generateContent"))
        self.assertNotIn("fake-key", url)
        self.assertEqual(headers["x-goog-api-key"], "fake-key")
        self.assertEqual(payload["systemInstruction"]["parts"][0]["text"], "instruction")
        self.assertEqual(payload["contents"][0]["parts"][0]["text"], "data")

    def test_16_ollama_request(self):
        fake = FakeTransport({"message": {"content": "summary"}, "done": True})
        provider = setup_ollama(model="local-model", transport=fake, base_url="http://localhost:11434/")
        self.assertEqual(provider.generate("data", "instruction"), "summary")
        url, headers, payload, _ = fake.calls[0]
        self.assertEqual(url, "http://localhost:11434/api/chat")
        self.assertFalse(payload["stream"])
        self.assertEqual(headers, {})
        self.assertEqual(payload["messages"][0]["content"], "instruction")

    def test_17_keys_and_configuration(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True):
            self.assertEqual(setup_openai(model="test")._api_key, "env-key")
            self.assertEqual(setup_openai(model="test", api_key="explicit")._api_key, "explicit")
            with self.assertRaises(ConfigurationError):
                setup_anthropic(model="test")
        for kwargs in ({"model": ""}, {"model": "test", "timeout": 0},
                       {"model": "test", "timeout": float("nan")},
                       {"model": "test", "base_url": "file:///tmp"},
                       {"model": "test", "max_output_tokens": True}):
            with self.assertRaises(ConfigurationError):
                setup_ollama(**kwargs)

    def test_18_unusable_provider_responses(self):
        fixtures = [
            (setup_openai, {"output": []}),
            (setup_openai, {"status": "incomplete"}),
            (setup_anthropic, {"content": [{"type": "text", "text": ""}]}),
            (setup_anthropic, {"stop_reason": "max_tokens"}),
            (setup_gemini, {"candidates": []}),
            (setup_gemini, {"candidates": [{"finishReason": "SAFETY"}]}),
            (setup_ollama, {"message": {}}),
            (setup_ollama, {"done_reason": "length"}),
        ]
        for setup, response in fixtures:
            with self.subTest(provider=setup.__name__, response=response):
                kwargs = {} if setup == setup_ollama else {"api_key": "fake"}
                provider = setup(model="test", transport=FakeTransport(response), **kwargs)
                with self.assertRaises(ProviderError):
                    provider.generate("data", "instruction")

    def test_19_http_transport_and_errors(self):
        fake_response = io.BytesIO(b'{"message": "ok"}')
        with patch("ai_provider_system.providers.base.urlopen", return_value=fake_response) as request:
            self.assertEqual(post_json("https://example.test", {"x-test": "yes"}, {"data": "hello"}, 4), {"message": "ok"})
            sent = request.call_args.args[0]
            self.assertEqual(json.loads(sent.data), {"data": "hello"})
            self.assertEqual(sent.method, "POST")
            self.assertEqual(request.call_args.kwargs["timeout"], 4)
        errors = [HTTPError("https://example.test/secret", 429, "secret", {}, None),
                  URLError("secret"), TimeoutError("secret")]
        for error in errors:
            with patch("ai_provider_system.providers.base.urlopen", side_effect=error):
                with self.assertRaises(ProviderError) as caught:
                    post_json("https://example.test", {}, {}, 1)
                self.assertNotIn("secret", str(caught.exception))
        with patch("ai_provider_system.providers.base.urlopen", return_value=io.BytesIO(b"not-json")):
            with self.assertRaises(ProviderError):
                post_json("https://example.test", {}, {}, 1)

    def test_20_custom_failures_and_provider_replacement(self):
        with self.assertRaises(ConfigurationError):
            setup_custom(None)
        for callback in (lambda d, i: None, lambda d, i: "", lambda d, i: 3):
            with self.assertRaises(ProviderError):
                AIModel(setup_custom(callback)).summarize("data")
        def fail(data, instruction):
            raise RuntimeError("private key")
        with self.assertRaises(ProviderError) as caught:
            AIModel(setup_custom(fail)).summarize("data")
        self.assertNotIn("private key", str(caught.exception))
        model = AIModel(setup_custom(lambda d, i: "first"))
        model.set_provider(setup_custom(lambda d, i: "second"))
        self.assertEqual(model.summarize("data"), "second")


if __name__ == "__main__":
    unittest.main()
