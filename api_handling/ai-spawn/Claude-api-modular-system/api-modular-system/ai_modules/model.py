import json
import os

from .providers import PROVIDERS, Provider

DEFAULT_SYSTEM = (
    "You are a careful data analyst. Read the data you are given and respond in clear, "
    "well-organized prose. Base every statement on the data; say so when the data "
    "does not support an answer."
)


class AIModel:
    """Central object the main program uses. A provider module gets plugged into it.

    Either name the provider and let AIModel call that module's setup function:

        model = AIModel("anthropic")                      # key from ANTHROPIC_API_KEY
        model = AIModel("openai", model="gpt-5", api_key="sk-...")

    or create it empty and plug a module in yourself:

        from ai_modules.providers import ollama
        model = ollama.setup(AIModel(), model="llama3.1")

    then:

        text = model.summarize(data, "Summarize monthly sales trends")
    """

    def __init__(self, provider=None, model=None, api_key=None, system=DEFAULT_SYSTEM,
                 max_tokens=1024, temperature=None, **options):
        self.provider = None
        self.system = system
        self.max_tokens = max_tokens
        self.temperature = temperature
        if provider is not None:
            key = provider.lower()
            if key not in PROVIDERS:
                raise ValueError(f"unknown provider '{provider}'. Available: {', '.join(sorted(PROVIDERS))}")
            PROVIDERS[key](self, model=model, api_key=api_key, **options)

    def attach(self, provider):
        """Plug a Provider instance in, replacing any previous one. Called by each module's setup()."""
        if not isinstance(provider, Provider):
            raise TypeError("attach() needs a Provider instance")
        self.provider = provider
        return self

    @property
    def provider_name(self):
        return self.provider.name if self.provider else None

    @property
    def model(self):
        return self.provider.model if self.provider else None

    def __repr__(self):
        return f"AIModel(provider={self.provider_name!r}, model={self.model!r})"

    # ---- construction from settings -------------------------------------------------

    @classmethod
    def from_config(cls, config):
        """Build from a dict or a path to a JSON file, e.g. {"provider": "gemini", "model": "..."}."""
        if isinstance(config, (str, os.PathLike)):
            with open(config, encoding="utf-8") as f:
                config = json.load(f)
        config = dict(config)
        return cls(config.pop("provider"), **config)

    @classmethod
    def from_env(cls):
        """Build from AI_PROVIDER / AI_MODEL environment variables (keys use each provider's own variable)."""
        provider = os.environ.get("AI_PROVIDER")
        if not provider:
            raise ValueError("AI_PROVIDER is not set")
        return cls(provider, model=os.environ.get("AI_MODEL") or None)

    # ---- calls ----------------------------------------------------------------------

    def generate(self, prompt, system=None, max_tokens=None, temperature=None):
        """Send a raw prompt and return the text response."""
        if self.provider is None:
            raise RuntimeError("no provider plugged in. Use AIModel('<name>') or <module>.setup(model).")
        return self.provider.generate(
            prompt,
            system=self.system if system is None else system,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature if temperature is None else temperature,
        )

    def summarize(self, data, instruction="Summarize the key points of this data.",
                  chunk_chars=None, **kwargs):
        """Send `data` with an `instruction` and return the written response.

        `data` can be a string, bytes, a dict/list (sent as JSON), or anything with a
        `to_csv()` method such as a pandas DataFrame. If `chunk_chars` is set and the
        data is longer, it is split into chunks, each chunk is summarized, and the
        partial summaries are combined into one final answer.
        """
        text = format_data(data)
        if chunk_chars and len(text) > chunk_chars:
            chunks = split_text(text, chunk_chars)
            partials = [
                self.generate(build_prompt(
                    chunk,
                    f"This is part {i} of {len(chunks)} of a larger dataset. Extract everything "
                    f"relevant to the following task, keeping specific figures:\n{instruction}",
                ), **kwargs)
                for i, chunk in enumerate(chunks, 1)
            ]
            combined = "\n\n".join(f"--- Notes on part {i} ---\n{p}" for i, p in enumerate(partials, 1))
            return self.generate(build_prompt(
                combined,
                "These are notes taken from consecutive parts of one dataset. Using them, "
                f"complete the following task as if you had read the whole dataset:\n{instruction}",
            ), **kwargs)
        return self.generate(build_prompt(text, instruction), **kwargs)


def format_data(data):
    if isinstance(data, str):
        return data
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    if hasattr(data, "to_csv"):
        return data.to_csv(index=False)
    return json.dumps(data, indent=2, default=str)


def build_prompt(data_text, instruction):
    return f"<data>\n{data_text}\n</data>\n\n{instruction}"


def split_text(text, size):
    """Split on line boundaries where possible so rows are not cut in half."""
    chunks, current, length = [], [], 0
    for line in text.splitlines(keepends=True):
        while len(line) > size:  # a single line longer than a chunk
            if current:
                chunks.append("".join(current))
                current, length = [], 0
            chunks.append(line[:size])
            line = line[size:]
        if length + len(line) > size and current:
            chunks.append("".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line)
    if current:
        chunks.append("".join(current))
    return chunks
