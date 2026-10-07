"""Pluggable access to Anthropic, OpenAI, Gemini, Ollama, or a custom endpoint."""

from ._http import ProviderError
from .model import AIModel
from .providers import PROVIDERS, Provider, register_provider

__all__ = ["AIModel", "Provider", "ProviderError", "PROVIDERS", "register_provider"]
