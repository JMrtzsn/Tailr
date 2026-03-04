"""Tests for tailr.report — conditional section logic and filename derivation."""

from __future__ import annotations

from pathlib import Path

import pytest

from tailr.report import generate, save


# ---------------------------------------------------------------------------
# Conditional sections — the actual branching logic in generate()
# ---------------------------------------------------------------------------


class TestConditionalSections:
    """STRONG FIT/POSSIBLE FIT get interview sections, WEAK FIT does not."""

    def test_strong_fit_includes_interview_sections(self, make_analysis) -> None:
        md = generate(make_analysis(recommendation="STRONG FIT"))
        assert "Interview Focus Areas" in md
        assert "Gap Strategy" in md

    def test_possible_fit_includes_interview_sections(self, make_analysis) -> None:
        md = generate(make_analysis(recommendation="POSSIBLE FIT"))
        assert "Interview Focus Areas" in md

    def test_weak_fit_excludes_interview_sections(self, make_analysis) -> None:
        md = generate(make_analysis(recommendation="WEAK FIT"))
        assert "Interview Focus Areas" not in md
        assert "Gap Strategy" not in md

    def test_empty_lists_produce_no_section(self, make_analysis) -> None:
        md = generate(make_analysis(strengths=[], gaps=[], knowledge_gains=[]))
        assert "Proof Points" not in md
        assert "Risk Flags" not in md
        assert "Growth Opportunities" not in md

    def test_reason_omitted_when_blank(self, make_analysis) -> None:
        md = generate(make_analysis(recommendation_reason=""))
        assert "### Why?" not in md


# ---------------------------------------------------------------------------
# save() — filename derivation
# ---------------------------------------------------------------------------


class TestSaveFilename:
    """The output filename is derived from the job description path."""

    @pytest.mark.parametrize(
        ("job_stem", "expected_name"),
        [
            ("Backend Engineer", "fit-backend-engineer.md"),
            ("senior-backend-engineer", "fit-senior-backend-engineer.md"),
            ("IKEA", "fit-ikea.md"),
            ("My Cool Job", "fit-my-cool-job.md"),
        ],
    )
    def test_filename_derivation(
        self, tmp_path: Path, make_analysis, job_stem: str, expected_name: str,
    ) -> None:
        result = save(make_analysis(), tmp_path, Path(f"jobs/{job_stem}.md"))
        assert result.name == expected_name

    def test_creates_nested_output_dirs(self, tmp_path: Path, make_analysis) -> None:
        nested = tmp_path / "a" / "b" / "c"
        save(make_analysis(), nested, Path("j.md"))
        assert nested.is_dir()

    def test_written_content_matches_generate(self, tmp_path: Path, make_analysis) -> None:
        analysis = make_analysis()
        path = save(analysis, tmp_path, Path("j.md"))
        assert path.read_text() == generate(analysis)
