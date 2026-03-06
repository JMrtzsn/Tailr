"""CV-to-job fit analysis CLI."""

import os
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tailr import report
from tailr.analyzer import FitAnalysis, FitAnalyzer, ModelNotFoundError
from tailr.providers import Provider

LLM_API_KEY_ENV = "LLM_API_KEY"

DEFAULT_MODELS: dict[Provider, str] = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.GEMINI: "gemini-2.5-flash",
}

app = typer.Typer(
    name="tailr-fit",
    help="Evaluate a CV against a job description and generate a fit report.",
    no_args_is_help=True,
)
console = Console()


def _resolve_api_key(api_key: str | None = None) -> str:
    """Resolve API key from argument or environment variable."""
    if api_key:
        return api_key
    if key := os.environ.get(LLM_API_KEY_ENV):
        return key
    raise typer.BadParameter(f"No API key provided. Set {LLM_API_KEY_ENV} or pass --api-key")


def _load_file(path: Path, label: str) -> str:
    """Read a file or exit with an error message."""
    if not path.exists():
        console.print(f"[red]Error: {label} file not found: {path}[/red]")
        raise typer.Exit(1)
    return path.read_text()


def _print_results(analysis: FitAnalysis) -> None:
    """Print the fit results table and verdict to the console."""
    console.print(f"   ✓ Fit Score: {analysis.score}%\n")

    table = Table(title="Fit Report", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Job Title", analysis.job_title)
    table.add_row("Company", analysis.company_name)
    table.add_row("Fit Score", f"{analysis.score}%")
    table.add_row("Verdict", analysis.recommendation)
    table.add_row("Proof Points", str(len(analysis.strengths)))
    table.add_row("Risk Flags", str(len(analysis.gaps)))
    table.add_row("Growth Opportunities", str(len(analysis.knowledge_gains)))
    table.add_row("Interview Focus Areas", str(len(analysis.interview_focus_areas)))
    console.print(table)
    console.print()

    if analysis.recommendation_reason:
        console.print(f"[bold]Why {analysis.recommendation}?[/bold]")
        console.print(f"   {analysis.recommendation_reason}\n")


def _print_summary(analysis: FitAnalysis) -> None:
    """Print a short proof points / risk flags summary."""
    if analysis.strengths:
        console.print("[green]✅ Top Proof Points:[/green]")
        for s in analysis.strengths[:3]:
            console.print(f"   • {s}")
        console.print()

    if analysis.gaps:
        console.print("[red]❌ Key Risk Flags:[/red]")
        for g in analysis.gaps[:3]:
            console.print(f"   • {g}")
        console.print()


def _print_model_not_found(err: ModelNotFoundError) -> None:
    """Print a helpful message when the requested model does not exist."""
    console.print(f"\n[red]Error: Model '{err.model}' not found.[/red]\n")
    if err.available:
        console.print(f"[bold]Available {err.provider.value} models:[/bold]\n")
        for m in err.available:
            console.print(f"  • {m}")
        console.print(
            f"\n[dim]Tip: use --model <name> to pick one, e.g. --model {err.available[0]}[/dim]"
        )
    else:
        console.print("[dim]Could not fetch available models.[/dim]")


@app.callback(invoke_without_command=True)
def fit(
    ctx: typer.Context,
    cv: Annotated[
        Path | None, typer.Option("--cv", "-c", help="Path to CV file (markdown)")
    ] = None,
    job: Annotated[
        Path | None, typer.Option("--job", "-j", help="Path to job description file")
    ] = None,
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path(
        "output"
    ),
    provider: Annotated[
        Provider, typer.Option("--provider", "-p", help="LLM provider to use")
    ] = Provider.GEMINI,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model name")] = None,
    api_key: Annotated[str | None, typer.Option("--api-key", "-k", help="API key")] = None,
    temperature: Annotated[
        float | None,
        typer.Option("--temperature", "-t", help="LLM temperature (0.0-2.0)", min=0.0, max=2.0),
    ] = None,
    max_tokens: Annotated[
        int | None, typer.Option("--max-tokens", help="Maximum output tokens", min=1)
    ] = None,
) -> None:
    """Evaluate a CV against a job description and generate a fit report."""
    if ctx.invoked_subcommand is not None:
        return
    if cv is None or job is None:
        console.print("[red]Error: --cv and --job are required[/red]")
        raise typer.Exit(1)

    resolved_model = model or DEFAULT_MODELS[provider]

    console.print(Panel.fit("🧠 Fit Analysis", style="bold blue"))
    console.print(f"📄 CV:       {cv}")
    console.print(f"💼 Job:      {job}")
    console.print(f"🤖 Provider: {provider.value} ({resolved_model})")
    console.print()

    cv_content = _load_file(cv, "CV")
    job_content = _load_file(job, "Job")

    console.print("🔧 Initialising LLM...")
    console.print("📊 Running fit analysis...")
    try:
        kwargs: dict[str, Any] = {
            "provider": provider,
            "api_key": _resolve_api_key(api_key),
            "model": resolved_model,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        analyzer = FitAnalyzer(**kwargs)
        analysis = analyzer.analyze(cv_content, job_content)
    except ModelNotFoundError as err:
        _print_model_not_found(err)
        raise typer.Exit(1)

    _print_results(analysis)

    console.print("💾 Saving fit report...")
    report_path = report.save(analysis, output, job)

    console.print(Panel.fit("✅ SUCCESS", style="bold green"))
    console.print(f"📝 Report: {report_path}\n")

    _print_summary(analysis)


if __name__ == "__main__":
    app()
