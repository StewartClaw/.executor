"""Pluggable synchronous AI summarization."""
from .model import AIModel
from .errors import ConfigurationError, ProviderError, SummarizationError
from .providers import setup_anthropic, setup_openai, setup_gemini, setup_ollama, setup_custom

__all__ = ["AIModel", "ConfigurationError", "ProviderError", "SummarizationError",
           "setup_anthropic", "setup_openai", "setup_gemini", "setup_ollama", "setup_custom"]
