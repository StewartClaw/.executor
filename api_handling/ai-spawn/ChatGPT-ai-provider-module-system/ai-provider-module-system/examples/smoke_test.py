"""Explicit live check: python examples/smoke_test.py openai --model YOUR_MODEL."""
import argparse
from ai_provider_system import AIModel, setup_openai, setup_anthropic, setup_gemini, setup_ollama


def main():
    parser = argparse.ArgumentParser(description="Make one live summarization request (cloud services may charge).")
    parser.add_argument("provider", choices=["openai", "anthropic", "gemini", "ollama"])
    parser.add_argument("--model", required=True, help="An available model ID for your provider")
    parser.add_argument("--base-url", help="Override the provider endpoint")
    args = parser.parse_args()
    setups = {"openai": setup_openai, "anthropic": setup_anthropic,
              "gemini": setup_gemini, "ollama": setup_ollama}
    kwargs = {"model": args.model}
    if args.base_url:
        kwargs["base_url"] = args.base_url
    model = AIModel(setups[args.provider](**kwargs))
    print(model.summarize({"orders": 42, "revenue": 1260, "period": "Monday"},
                          "Summarize this report in one sentence."))


if __name__ == "__main__":
    main()
