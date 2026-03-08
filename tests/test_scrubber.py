"""Tests for tailr.scrubber — PII removal from CV text."""

from __future__ import annotations

import pytest

from tailr.scrubber import scrub


# ---------------------------------------------------------------------------
# Individual PII categories
# ---------------------------------------------------------------------------


class TestEmailScrubbing:
    @pytest.mark.parametrize(
        "raw",
        [
            "jane.doe@example.com",
            "john+work@company.co.uk",
            "user_name@sub.domain.org",
        ],
    )
    def test_emails_are_replaced(self, raw: str) -> None:
        result = scrub(f"Contact: {raw}")
        assert "[EMAIL]" in result
        assert raw not in result


class TestPhoneScrubbing:
    @pytest.mark.parametrize(
        "raw",
        [
            "+46 70 123 4567",
            "+1 (555) 123-4567",
            "070-123 45 67",
            "555.123.4567",
        ],
    )
    def test_phones_are_replaced(self, raw: str) -> None:
        result = scrub(f"Phone: {raw}")
        assert "[PHONE]" in result
        assert raw not in result


class TestLinkedInScrubbing:
    @pytest.mark.parametrize(
        "raw",
        [
            "https://linkedin.com/in/janedoe",
            "https://www.linkedin.com/in/jane-doe/",
            "http://linkedin.com/in/JaneDoe123",
        ],
    )
    def test_linkedin_urls_are_replaced(self, raw: str) -> None:
        result = scrub(f"LinkedIn: {raw}")
        assert "[LINKEDIN]" in result
        assert raw not in result


class TestGitHubScrubbing:
    @pytest.mark.parametrize(
        "raw",
        [
            "https://github.com/janedoe",
            "https://www.github.com/jane-doe/",
            "http://github.com/JaneDoe123",
        ],
    )
    def test_github_urls_are_replaced(self, raw: str) -> None:
        result = scrub(f"GitHub: {raw}")
        assert "[GITHUB]" in result
        assert raw not in result


class TestGenericUrlScrubbing:
    @pytest.mark.parametrize(
        "raw",
        [
            "https://janedoe.dev",
            "https://my-portfolio.com/projects",
            "http://personal-site.io",
        ],
    )
    def test_generic_urls_are_replaced(self, raw: str) -> None:
        result = scrub(f"Website: {raw}")
        assert "[URL]" in result
        assert raw not in result


class TestAddressScrubbing:
    @pytest.mark.parametrize(
        "raw",
        [
            "123 Main Street",
            "456 Elm Ave, Springfield, IL 62704",
            "789 Oak Blvd",
        ],
    )
    def test_addresses_are_replaced(self, raw: str) -> None:
        result = scrub(raw)
        assert "[ADDRESS]" in result
        assert raw not in result


class TestNameScrubbing:
    def test_first_line_name_is_replaced(self) -> None:
        cv = "Jane Doe\nSenior Backend Engineer\njane@example.com"
        result = scrub(cv)
        assert "[NAME]" in result
        assert "Jane Doe" not in result

    def test_skips_header_words(self) -> None:
        cv = "Curriculum Vitae\nJane Doe\nSenior Engineer"
        result = scrub(cv)
        assert "[NAME]" in result
        assert "Jane Doe" not in result
        # The header word should NOT be replaced
        assert "Curriculum Vitae" in result

    def test_skips_blank_lines(self) -> None:
        cv = "\n\nJohn Smith\nDeveloper"
        result = scrub(cv)
        assert "[NAME]" in result
        assert "John Smith" not in result

    def test_replaces_name_throughout(self) -> None:
        cv = "Jane Doe\nAbout Jane Doe\nJane Doe is a developer."
        result = scrub(cv)
        assert result.count("[NAME]") == 3
        assert "Jane Doe" not in result

    def test_name_with_markdown_heading(self) -> None:
        cv = "# Jane Doe\nSenior Engineer"
        result = scrub(cv)
        assert "[NAME]" in result
        assert "Jane Doe" not in result


# ---------------------------------------------------------------------------
# Integration — realistic CV snippet
# ---------------------------------------------------------------------------


class TestFullCVScrubbing:
    _SAMPLE_CV = """\
# Jane Doe

Senior Backend Engineer

- Email: jane.doe@example.com
- Phone: +46 70 123 4567
- LinkedIn: https://linkedin.com/in/janedoe
- GitHub: https://github.com/janedoe
- Website: https://janedoe.dev
- Address: 123 Main Street, Stockholm

## Experience

Jane Doe has 8+ years of experience building distributed systems.
"""

    def test_all_pii_is_scrubbed(self) -> None:
        result = scrub(self._SAMPLE_CV)

        assert "[NAME]" in result
        assert "[EMAIL]" in result
        assert "[PHONE]" in result
        assert "[LINKEDIN]" in result
        assert "[GITHUB]" in result
        assert "[URL]" in result
        assert "[ADDRESS]" in result

        # No raw PII should remain
        assert "Jane Doe" not in result
        assert "jane.doe@example.com" not in result
        assert "+46 70 123 4567" not in result
        assert "linkedin.com/in/janedoe" not in result
        assert "github.com/janedoe" not in result
        assert "janedoe.dev" not in result
        assert "123 Main Street" not in result

    def test_non_pii_content_preserved(self) -> None:
        result = scrub(self._SAMPLE_CV)

        assert "Senior Backend Engineer" in result
        assert "## Experience" in result
        assert "8+ years of experience" in result
        assert "distributed systems" in result


class TestNoScrubNeeded:
    def test_text_without_pii_is_unchanged(self) -> None:
        text = "Experienced backend engineer with strong Python skills."
        result = scrub(text)
        assert result == text

