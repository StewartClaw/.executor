class ConfigurationError(ValueError):
    """Invalid model or provider configuration."""


class ProviderError(RuntimeError):
    """A provider request failed or returned unusable text."""


class SummarizationError(RuntimeError):
    """Chunk summaries could not converge within the configured limits."""
