"""LLM provider definitions and model-listing HTTP clients."""

from __future__ import annotations

from enum import StrEnum

import httpx

_HTTP_TIMEOUT = 15


class Provider(StrEnum):
    """Supported LLM providers."""

    OPENAI = "openai"
    GEMINI = "gemini"


def list_gemini_models(api_key: str) -> list[str]:
    """Fetch available Gemini models that support content generation."""
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    headers = {"x-goog-api-key": api_key}
    resp = httpx.get(url, headers=headers, timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    models: list[str] = []
    for m in data.get("models", []):
        name: str = m.get("name", "")
        if name.startswith("models/"):
            name = name[len("models/") :]
        methods = m.get("supportedGenerationMethods", [])
        if "generateContent" in methods:
            models.append(name)
    return sorted(models)


def list_openai_models(api_key: str) -> list[str]:
    """Fetch available OpenAI model IDs."""
    url = "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = httpx.get(url, headers=headers, timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    models: list[str] = []
    for m in data.get("data", []):
        model_id = m.get("id", "")
        if model_id:
            models.append(model_id)
    return sorted(models)
