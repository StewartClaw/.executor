"""Summarize a data file from the command line.

    python examples/summarize.py anthropic sales.csv "Summarize monthly sales trends"
    python examples/summarize.py ollama report.json "List the three biggest risks" --model llama3.1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_modules import AIModel, ProviderError  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("provider", help="anthropic | openai | gemini | ollama | custom")
    p.add_argument("file", help="data file to summarize")
    p.add_argument("instruction", help="what to do with the data")
    p.add_argument("--model")
    p.add_argument("--base-url", help="for custom (or a non-default ollama host)")
    p.add_argument("--max-tokens", type=int, default=1500)
    p.add_argument("--chunk-chars", type=int, help="split data larger than this many characters")
    args = p.parse_args()

    options = {"base_url": args.base_url} if args.base_url else {}
    model = AIModel(args.provider, model=args.model, max_tokens=args.max_tokens, **options)
    data = Path(args.file).read_text(encoding="utf-8", errors="replace")
    try:
        print(model.summarize(data, args.instruction, chunk_chars=args.chunk_chars))
    except ProviderError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
