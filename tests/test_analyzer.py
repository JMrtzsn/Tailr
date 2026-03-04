"""Tests for tailr.analyzer — error detection and API response parsing."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tailr.analyzer import (
    ModelNotFoundError,
)
from tailr.providers import (
    list_gemini_models,
    list_openai_models,
)

from tests.conftest import json_response


# ---------------------------------------------------------------------------
# FitAnalyzer._raise_if_model_not_found — the error-message parsing logic
# ---------------------------------------------------------------------------


class TestRaiseIfModelNotFound:
    """Detects 'not found' variants in upstream errors and converts them."""

    @pytest.mark.parametrize(
        "message",
        [
            "The model `bad` does not exist",
            "Model not found: bad",
            "error: NOT_FOUND for model bad",
        ],
    )
    def test_converts_matching_errors(self, make_analyzer, message: str) -> None:
        analyzer = make_analyzer(model="bad")
        with patch.object(analyzer, "list_models", return_value=["gpt-4o"]):
            with pytest.raises(ModelNotFoundError) as exc_info:
                analyzer._raise_if_model_not_found(Exception(message))
        assert exc_info.value.model == "bad"
        assert exc_info.value.available == ["gpt-4o"]

    def test_ignores_unrelated_errors(self, make_analyzer) -> None:
        analyzer = make_analyzer()
        result = analyzer._raise_if_model_not_found(Exception("connection timeout"))
        assert result is None

    def test_still_works_when_list_models_fails(self, make_analyzer) -> None:
        analyzer = make_analyzer(model="bad")
        with patch.object(analyzer, "list_models", side_effect=Exception("boom")):
            with pytest.raises(ModelNotFoundError) as exc_info:
                analyzer._raise_if_model_not_found(Exception("model not found"))
        assert exc_info.value.available == []


# ---------------------------------------------------------------------------
# list_gemini_models — parses the Gemini REST response
# ---------------------------------------------------------------------------


class TestListGeminiModels:
    def test_strips_models_prefix_and_filters(self) -> None:
        payload = {
            "models": [
                {
                    "name": "models/gemini-pro",
                    "supportedGenerationMethods": ["generateContent"],
                },
                {
                    "name": "models/embedding-001",
                    "supportedGenerationMethods": ["embedContent"],
                },
                {
                    "name": "models/gemini-flash",
                    "supportedGenerationMethods": ["generateContent", "countTokens"],
                },
            ]
        }
        response = json_response(payload)
        with patch("tailr.providers.httpx.get", return_value=response):
            result = list_gemini_models("fake-key")

        # embedding model filtered out, prefix stripped, sorted
        assert result == ["gemini-flash", "gemini-pro"]

    def test_empty_response(self) -> None:
        response = json_response({"models": []})
        with patch("tailr.providers.httpx.get", return_value=response):
            assert list_gemini_models("key") == []


# ---------------------------------------------------------------------------
# list_openai_models — parses the OpenAI REST response
# ---------------------------------------------------------------------------


class TestListOpenAIModels:
    def test_extracts_and_sorts_ids(self) -> None:
        payload = {
            "data": [
                {"id": "gpt-4o"},
                {"id": "gpt-3.5-turbo"},
                {"id": "dall-e-3"},
            ]
        }
        response = json_response(payload)
        with patch("tailr.providers.httpx.get", return_value=response):
            result = list_openai_models("fake-key")

        assert result == ["dall-e-3", "gpt-3.5-turbo", "gpt-4o"]

    def test_skips_empty_ids(self) -> None:
        payload = {"data": [{"id": "gpt-4o"}, {"id": ""}, {}]}
        response = json_response(payload)
        with patch("tailr.providers.httpx.get", return_value=response):
            result = list_openai_models("key")

        assert result == ["gpt-4o"]

