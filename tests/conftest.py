"""Shared test fixtures and factories."""

from __future__ import annotations

import httpx
import pytest

from tailr.analyzer import FitAnalyzer, FitAnalysis
from tailr.providers import Provider


@pytest.fixture()
def make_analysis():
    """Factory fixture for FitAnalysis with sensible defaults."""

    def _factory(**overrides: object) -> FitAnalysis:
        defaults: dict = dict(
            job_title="Backend Engineer",
            company_name="Acme Corp",
            score=72,
            recommendation="STRONG FIT",
            recommendation_reason="Strong match.",
            strengths=["Python", "REST APIs"],
            gaps=["Kubernetes"],
            knowledge_gains=["Helm"],
            interview_focus_areas=["System design"],
            gap_coverage=["Take K8s course"],
        )
        defaults.update(overrides)
        return FitAnalysis(**defaults)

    return _factory


@pytest.fixture()
def make_analyzer():
    """Factory fixture for FitAnalyzer with test defaults."""

    def _factory(**overrides: object) -> FitAnalyzer:
        defaults: dict = dict(
            provider=Provider.OPENAI,
            api_key="test-key",
            model="gpt-4o",
        )
        defaults.update(overrides)
        return FitAnalyzer(**defaults)

    return _factory


def json_response(payload: dict, url: str = "https://example.com") -> httpx.Response:
    """Build an httpx.Response that supports raise_for_status()."""
    request = httpx.Request("GET", url)
    return httpx.Response(200, json=payload, request=request)

