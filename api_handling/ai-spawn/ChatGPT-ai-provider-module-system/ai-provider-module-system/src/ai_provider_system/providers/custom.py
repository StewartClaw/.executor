from ..errors import ConfigurationError, ProviderError


class CustomProvider:
    def __init__(self, callback):
        if not callable(callback):
            raise ConfigurationError("callback must be callable")
        self.callback = callback

    def generate(self, data, instruction):
        try:
            result = self.callback(data, instruction)
        except ProviderError:
            raise
        except Exception:
            raise ProviderError("Custom provider callback failed") from None
        if not isinstance(result, str) or not result.strip():
            raise ProviderError("Custom provider must return nonempty text")
        return result.strip()


def setup_custom(callback):
    """callback(data: str, instruction: str) -> str."""
    return CustomProvider(callback)
