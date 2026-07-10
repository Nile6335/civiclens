"""LLM backend selection: local Ollama by default, Anthropic when configured.

LLM_BACKEND=ollama|anthropic (spec feature flag). Anthropic is used only when the
backend is selected AND a key is present; everything else falls back to Ollama.
"""

import logging
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from common.settings import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_chat_model(temperature: float = 0.0) -> BaseChatModel:
    settings = get_settings()
    if settings.llm_backend == "anthropic" and settings.anthropic_api_key:
        from langchain_anthropic import ChatAnthropic

        logger.info("LLM backend: anthropic (%s)", settings.anthropic_model)
        return ChatAnthropic(
            model=settings.anthropic_model,
            temperature=temperature,
            api_key=settings.anthropic_api_key,
            max_tokens=1024,
        )
    from langchain_ollama import ChatOllama

    logger.info("LLM backend: ollama (%s)", settings.ollama_model)
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        num_ctx=4096,
    )


def llm_description() -> str:
    """Human-readable backend descriptor for /health and eval result configs."""
    settings = get_settings()
    if settings.llm_backend == "anthropic" and settings.anthropic_api_key:
        return f"anthropic:{settings.anthropic_model}"
    return f"ollama:{settings.ollama_model}"
