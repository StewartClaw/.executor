# AI Modules

One `AIModel` class that lets a program send data plus an instruction to Anthropic, OpenAI,
Gemini, a local Ollama model, or any manually configured endpoint, and get written text back.
Pure Python 3.9+ standard library: nothing to install.

## Quick start

```python
from ai_modules import AIModel

model = AIModel("anthropic")            # reads ANTHROPIC_API_KEY
summary = model.summarize(csv_text, "Summarize the monthly sales trends and flag anomalies.")
print(summary)
```

Switching providers changes only the first line:

```python
AIModel("openai", model="gpt-5")                    # OPENAI_API_KEY
AIModel("gemini", model="gemini-2.5-flash")         # GEMINI_API_KEY
AIModel("ollama", model="llama3.1", num_ctx=32768)  # local, no key; OLLAMA_HOST to change host
AIModel("custom", base_url="http://localhost:1234/v1", model="my-model")
```

### Plugging a module in with its setup function

Each provider lives in its own file under `ai_modules/providers/` (`anthropic.py`, `openai.py`,
`gemini.py`, `ollama.py`, `custom.py`). Every one of them has a `setup()` function that builds that
provider and plugs it into an `AIModel` object:

```python
from ai_modules import AIModel
from ai_modules.providers import anthropic, ollama

ai = AIModel(max_tokens=1500)                  # empty: no provider yet
anthropic.setup(ai, model="claude-sonnet-5-5") # plug Anthropic in
print(ai.summarize(data, "Summarize this"))

ollama.setup(ai, model="llama3.1")             # swap to a local model; other settings stay
```

`setup(ai_model, model=None, api_key=None, **options)` returns the same `AIModel`, so
`ai = gemini.setup(AIModel())` also works. `AIModel("anthropic")` is shorthand that calls
`anthropic.setup` for you.

Any provider also accepts `api_key="..."` directly, and `base_url=...` to point at a proxy.

| Provider    | Default model        | Key variable        |
|-------------|----------------------|---------------------|
| `anthropic` | `claude-sonnet-5-5`  | `ANTHROPIC_API_KEY` |
| `openai`    | `gpt-5`              | `OPENAI_API_KEY`    |
| `gemini`    | `gemini-2.5-flash`   | `GEMINI_API_KEY`    |
| `ollama`    | `llama3.1`           | none                |
| `custom`    | `CUSTOM_AI_MODEL`    | `CUSTOM_AI_API_KEY` (optional) |

Defaults are only fallbacks; pass `model=` for the model you actually want.

## Summarizing data

`model.summarize(data, instruction)` accepts a string, bytes, a dict or list (sent as JSON),
or a pandas DataFrame (sent as CSV). The data is wrapped in `<data>` tags followed by your
instruction, and a default analyst system prompt asks for prose grounded in the data.
Override it with `AIModel(..., system="...")`.

Data too large for the model's context window can be split: `chunk_chars=100_000` summarizes
each chunk with your instruction in mind, then combines the notes into one final answer.

Other settings on `AIModel`: `max_tokens` (default 1024, the response length cap),
`temperature` (left to the provider's default unless set). Use `model.generate(prompt)` for a
raw prompt without the data wrapper.

## Choosing the provider from settings

```python
model = AIModel.from_config("config.json")   # {"provider": "gemini", "model": "...", "max_tokens": 1500}
model = AIModel.from_env()                   # AI_PROVIDER=ollama AI_MODEL=qwen3
```

## The custom option

By default `custom` speaks the OpenAI chat-completions format, which covers LM Studio, vLLM,
Groq, Together, OpenRouter, Azure OpenAI and most other servers. For an API with its own
format, describe the request and where to find the answer:

```python
AIModel(
    "custom",
    base_url="https://api.example.com",
    path="/generate",                       # default /chat/completions
    api_key="...", auth_header="x-api-key", # default sends Authorization: Bearer <key>
    headers={"X-Team": "data"},
    build_request=lambda prompt, system, model, max_tokens, temperature: {
        "input": f"{system}\n\n{prompt}", "length": max_tokens,
    },
    response_path="result.outputs.0.text",  # dotted path into the JSON response
)
```

## Adding a new provider module

Create a new file (say `ai_modules/providers/mistral.py`) with a `Provider` subclass and a
`setup` function, then register it:

```python
from ai_modules.providers.base import Provider

class MistralProvider(Provider):
    name = "mistral"
    default_model = "mistral-large-latest"
    api_key_env = "MISTRAL_API_KEY"

    def default_base_url(self):
        return "https://api.mistral.ai/v1"

    def generate(self, prompt, system=None, max_tokens=1024, temperature=None):
        ...  # call self._post(url, payload, headers) and return the text

def setup(ai_model, model=None, api_key=None, **options):
    ai_model.attach(MistralProvider(model=model, api_key=api_key, **options))
    return ai_model
```

`mistral.setup(AIModel())` works right away. To also allow `AIModel("mistral")`, add it to the
`PROVIDERS` table in `providers/__init__.py`, or call `register_provider("mistral", mistral.setup)`.

## Errors

Every failed request (HTTP error, connection refused, unexpected response, safety block)
raises `ProviderError`, with `.provider`, `.status` and `.body` for inspection.

## Command line

```
python examples/summarize.py anthropic sales.csv "Summarize monthly sales trends"
python examples/summarize.py ollama report.json "List the three biggest risks" --model llama3.1
```

## Layout

```
ai_modules/
  model.py            AIModel: holds the plugged-in provider, summarize(), chunking
  _http.py            shared JSON-over-HTTP helper and ProviderError
  providers/
    base.py           Provider base class
    anthropic.py      AnthropicProvider + setup()
    openai.py         OpenAIProvider + setup()
    gemini.py         GeminiProvider + setup()
    ollama.py         OllamaProvider + setup()
    custom.py         CustomProvider + setup()
    __init__.py       name -> setup() table and register_provider()
examples/summarize.py
tests/                python -m unittest discover tests  (mocked, no keys needed)
```
