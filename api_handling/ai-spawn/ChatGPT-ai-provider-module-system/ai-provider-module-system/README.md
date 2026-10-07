# AI provider module system

A Python package that turns text, dictionaries, lists, or pandas DataFrames into a written summary. Create an `AIModel`, attach a provider returned by its setup function, and call `summarize(data, instruction)`.

Python 3.10+ is required. HTTP providers use the standard library: no vendor SDKs are needed. Pandas is optional.

## Install

From the unpacked project folder:

```sh
python -m pip install .
# Optional DataFrame support:
python -m pip install ".[pandas]"
```

You can also install the provided `.whl` directly with `python -m pip install PATH_TO_WHEEL`.

## Quick start

```python
from ai_provider_system import AIModel, setup_openai

# Set OPENAI_API_KEY in your environment first.
# Replace this model ID with one available to your account.
model = AIModel()
model.set_provider(setup_openai(model="YOUR_OPENAI_MODEL"))
summary = model.summarize(
    {"sales": [120, 160, 140], "region": "West"},
    "Write a short summary of the sales trend.",
)
print(summary)
```

You may also pass the provider directly to `AIModel(provider)`.

## Provider setup

Each provider lives in its own file under `src/ai_provider_system/providers/`.

| Setup function | Required configuration | Default endpoint |
| --- | --- | --- |
| `setup_openai` | `model`; `api_key` or `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| `setup_anthropic` | `model`; `api_key` or `ANTHROPIC_API_KEY` | `https://api.anthropic.com/v1` |
| `setup_gemini` | `model`; `api_key` or `GEMINI_API_KEY` | `https://generativelanguage.googleapis.com/v1beta` |
| `setup_ollama` | `model` installed in your running Ollama service | `http://localhost:11434` |
| `setup_custom` | A callback `(data: str, instruction: str) -> str` | Your implementation |

The four HTTP setup functions accept `base_url`, `timeout=60` (seconds), `max_output_tokens=1024`, and an optional injectable `transport(url, headers, payload, timeout) -> dict`. Model IDs are explicit because availability and context limits vary by service and account. `max_output_tokens` is a cap; reasoning models may need a larger value to produce visible text.

```python
from ai_provider_system import (
    AIModel, setup_anthropic, setup_gemini, setup_ollama, setup_custom,
)

claude = AIModel(setup_anthropic(model="YOUR_CLAUDE_MODEL"))
gemini = AIModel(setup_gemini(model="YOUR_GEMINI_MODEL"))
local = AIModel(setup_ollama(model="YOUR_INSTALLED_MODEL"))

def my_provider(data, instruction):
    return my_model.generate(prompt=instruction + "\n\n" + data)

custom = AIModel(setup_custom(my_provider))
```

`my_model` in that custom example is your own model client. A runnable offline callback example is in `examples/custom_demo.py`.

## Input and chunking

Text is passed as written. Dicts and lists must be JSON compatible and are encoded as readable Unicode JSON. DataFrames are encoded with pandas' `split` orientation, preserving columns, index, and row order, including duplicate column labels. Pandas converts missing cells to JSON null. Other types and nonserializable nested values raise errors. Empty text is rejected; empty lists, dicts, and DataFrames are valid structured inputs.

`AIModel(provider, max_input_chars=12000, max_rounds=8)` uses a character budget for the instruction plus data. Small inputs take one request. Large inputs are split losslessly, preferring whitespace, and every piece is summarized. Partial summaries are reduced recursively if necessary, then combined in a final request. Original instructions are retained in all steps. The process stops with `SummarizationError` if outputs do not shrink or the round limit is reached.

The character budget is configurable and is **not** an exact token limit. Leave room for provider framing and generated output, and choose a budget appropriate to your model and language. A large JSON value or DataFrame may be split inside a record, so chunks are treated as text pieces rather than independently valid documents. Summaries are lossy and should not replace exact numerical calculations. Multi-chunk inputs cause multiple sequential requests.

## Errors

- `ConfigurationError`: missing provider/key, invalid settings, or insufficient chunk budget.
- `ProviderError`: connection failure, HTTP error, malformed/empty output, blocked response, or detected truncation.
- `SummarizationError`: chunk reduction does not converge.
- `TypeError` / `ValueError`: unsupported or invalid input.

Requests have timeouts. There are no automatic retries or background calls. HTTP errors expose a status code without raw response bodies or credentials. Do not put API keys in source files. Inputs are sent to the configured provider; use Ollama or a custom local callback when local processing is required.

## Tests

Install the optional pandas extra so the real DataFrame test can run. From the project folder, PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

On macOS/Linux:

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

There are 20 tests covering all five providers, request bodies, serialization (including a real DataFrame), chunk reconstruction and reduction, configuration, response errors, and mocked HTTP behavior. All service responses are fake; the tests make no network calls.

## Live smoke test

After installing the package, set the relevant environment variable and run:

```powershell
$env:OPENAI_API_KEY = "YOUR_KEY"
python examples/smoke_test.py openai --model YOUR_OPENAI_MODEL
```

Other providers:

```sh
python examples/smoke_test.py anthropic --model YOUR_CLAUDE_MODEL
python examples/smoke_test.py gemini --model YOUR_GEMINI_MODEL
python examples/smoke_test.py ollama --model YOUR_INSTALLED_MODEL
```

These commands deliberately make a live request; cloud requests may incur charges. No live service validation was performed during this build.

## API references

Adapters follow the documented REST interfaces:

- [OpenAI Responses](https://developers.openai.com/api/reference/python/resources/responses/methods/create)
- [Anthropic Messages](https://platform.claude.com/docs/en/api/messages/create)
- [Gemini generateContent](https://ai.google.dev/api/generate-content)
- [Ollama chat](https://docs.ollama.com/api/chat)

## Build

```sh
python -m pip install build
python -m build
```

This project has no attached GitHub repository. The source, tests, examples, wheel, source distribution and project ZIP can be imported into a repository later.
