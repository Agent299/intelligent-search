import anthropic

from common.config import get_settings


def anthropic_client(**kwargs) -> anthropic.Anthropic:
    """Key comes from .env via Settings; falls back to the SDK's own credential resolution."""
    return anthropic.Anthropic(api_key=get_settings().anthropic_api_key or None, **kwargs)
