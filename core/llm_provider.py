"""
LLM provider abstraction. factory pattern
tool: an ABC + concrete clients + a factory function, so every downstream
module depends only on `LLMProvider`, never on a specific SDK.

This keeps vision extraction, entity resolution, relationship extraction,
and generation all swappable independently if needed later (e.g. cheaper
model for disambiguation, stronger model for generation).
"""
from __future__ import annotations

import base64
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from pdf_rag_kb.config.settings import LLMConfig

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Raised when a provider call fails after retries."""


class LLMProvider(ABC):
    """Common interface every concrete provider must implement."""

    @abstractmethod
    def complete_text(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        """Plain text completion."""

    @abstractmethod
    def complete_json(self, system: str, prompt: str, max_tokens: int = 1024) -> dict[str, Any]:
        """Completion forced/parsed as JSON. Raises LLMProviderError on
        malformed output after retry."""

    @abstractmethod
    def complete_vision_json(
        self, system: str, prompt: str, image_bytes: bytes, media_type: str = "image/png",
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Vision + text -> structured JSON (used for chart/table extraction)."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding."""

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
        return text.strip().strip("`").strip()

    def _parse_json_or_raise(self, raw_text: str) -> dict[str, Any]:
        cleaned = self._strip_json_fences(raw_text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise LLMProviderError(f"Model did not return valid JSON: {e}\nRaw: {raw_text[:500]}")


class AnthropicProvider(LLMProvider):
    def __init__(self, config: LLMConfig):
        import anthropic  # local import so package doesn't hard-depend on it
        self._config = config
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def complete_text(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        resp = self._client.messages.create(
            model=self._config.anthropic_text_model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    def complete_json(self, system: str, prompt: str, max_tokens: int = 1024) -> dict[str, Any]:
        forced_system = system + "\n\nRespond with ONLY valid JSON. No preamble, no markdown fences."
        raw = self.complete_text(forced_system, prompt, max_tokens)
        return self._parse_json_or_raise(raw)

    def complete_vision_json(
        self, system: str, prompt: str, image_bytes: bytes, media_type: str = "image/png",
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        forced_system = system + "\n\nRespond with ONLY valid JSON. No preamble, no markdown fences."
        resp = self._client.messages.create(
            model=self._config.anthropic_vision_model,
            max_tokens=max_tokens,
            system=forced_system,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text")
        return self._parse_json_or_raise(raw)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Anthropic has no native embeddings endpoint; fall back to a
        # sentence-transformers model for local/dev use.
        from sentence_transformers import SentenceTransformer
        model = _get_cached_st_model()
        return model.encode(texts, convert_to_numpy=False)  # type: ignore[return-value]


class AzureOpenAIProvider(LLMProvider):
    def __init__(self, config: LLMConfig):
        from openai import AzureOpenAI  # local import
        self._config = config
        self._client = AzureOpenAI(
            api_key=config.azure_api_key,
            azure_endpoint=config.azure_endpoint,
            api_version=config.azure_api_version,
        )

    def complete_text(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        resp = self._client.chat.completions.create(
            model=self._config.azure_text_deployment,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    def complete_json(self, system: str, prompt: str, max_tokens: int = 1024) -> dict[str, Any]:
        resp = self._client.chat.completions.create(
            model=self._config.azure_text_deployment,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content or "{}"
        return self._parse_json_or_raise(raw)

    def complete_vision_json(
        self, system: str, prompt: str, image_bytes: bytes, media_type: str = "image/png",
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        resp = self._client.chat.completions.create(
            model=self._config.azure_vision_deployment,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                ]},
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        return self._parse_json_or_raise(raw)

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model="text-embedding-3-large", input=texts)
        return [d.embedding for d in resp.data]


_st_model_cache: Optional[Any] = None


def _get_cached_st_model():
    global _st_model_cache
    if _st_model_cache is None:
        from sentence_transformers import SentenceTransformer
        _st_model_cache = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model_cache


def get_llm_provider(config: LLMConfig) -> LLMProvider:
    """Factory: single place that decides which concrete provider to build."""
    if config.provider == "anthropic":
        return AnthropicProvider(config)
    if config.provider == "azure_openai":
        return AzureOpenAIProvider(config)
    raise ValueError(f"Unknown LLM provider: {config.provider!r}")
