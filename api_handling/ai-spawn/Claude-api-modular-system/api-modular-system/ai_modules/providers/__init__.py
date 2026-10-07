"""One module per provider. Each module exposes `setup(ai_model, model=None, api_key=None, **options)`,
which builds that provider and plugs it into an AIModel."""

from . import anthropic, custom, gemini, ollama, openai
from .base import Provider

# provider name -> that module's setup function
PROVIDERS = {
    "anthropic": anthropic.setup,
    "openai": openai.setup,
    "gemini": gemini.setup,
    "ollama": ollama.setup,
    "custom": custom.setup,
}


def register_provider(name, setup):
    """Plug in a new provider so AIModel(provider=name) can use it.

    `setup` is either a module-style setup function `setup(ai_model, model=None, api_key=None, **options)`
    or a Provider subclass (a setup function is generated for it).
    """
    if isinstance(setup, type):
        if not issubclass(setup, Provider):
            raise TypeError("provider class must subclass ai_modules.Provider")
        cls = setup

        def setup(ai_model, model=None, api_key=None, **options):
            ai_model.attach(cls(model=model, api_key=api_key, **options))
            return ai_model
    elif not callable(setup):
        raise TypeError("setup must be a function or a Provider subclass")
    PROVIDERS[name.lower()] = setup
