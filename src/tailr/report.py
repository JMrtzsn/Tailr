"""Markdown report generation for fit analyses."""

from pathlib import Path

from tailr.analyzer import FitAnalysis


def _section(
    lines: list[str],
    heading: str,
    items: list[str],
    *,
    preamble: str | None = None,
) -> None:
    """Append a markdown section with bullet items (skipped when empty)."""
    if not items:
        return
    lines.append(f"{heading}\n")
    if preamble:
        lines.append(f"{preamble}\n")
    for item in items:
        lines.append(f"- {item}")
    lines.append("")


def generate(analysis: FitAnalysis) -> str:
    """Generate a full markdown report from a fit analysis."""
    lines = [
        "# Fit Report\n",
        f"**Position:** {analysis.job_title}",
        f"**Company:** {analysis.company_name}\n",
        "---\n",
        f"## 📊 Fit Score: {analysis.score}%\n",
        f"## 🎯 Verdict: {analysis.recommendation}\n",
    ]

    if analysis.recommendation_reason:
        lines += ["### Why?\n", f"{analysis.recommendation_reason}\n"]

    _section(lines, "## ✅ Proof Points", analysis.strengths)
    _section(lines, "## ❌ Risk Flags", analysis.gaps)
    _section(
        lines,
        "## 📚 Growth Opportunities",
        analysis.knowledge_gains,
        preamble="Skills and knowledge the candidate would develop in this role:",
    )

    if analysis.recommendation in ("STRONG FIT", "POSSIBLE FIT"):
        _section(
            lines,
            "## 🎓 Interview Focus Areas",
            analysis.interview_focus_areas,
            preamble="Topics worth probing —",
        )
        _section(lines, "## 🔧 Gap Strategy", analysis.gap_coverage)

    return "\n".join(lines)


def save(analysis: FitAnalysis, output_dir: Path, job_path: Path) -> Path:
    """Write the markdown report and return the output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    job_name = job_path.stem.lower().replace(" ", "-")
    path = output_dir / f"fit-{job_name}.md"
    path.write_text(generate(analysis))
    return path
