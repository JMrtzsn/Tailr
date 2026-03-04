"""Unified CLI for Tailr - AI-powered fit analysis tool."""

import typer

from tailr.cli import fit

app = typer.Typer(
    name="tailr",
    help="Tailr - AI-powered fit analysis tool",
    no_args_is_help=True,
)

app.add_typer(fit.app, name="fit", help="Fit — Analyse CV readiness against job descriptions")


if __name__ == "__main__":
    app()
