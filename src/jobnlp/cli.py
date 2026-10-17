"""jobnlp command line interface."""

import typer

app = typer.Typer(no_args_is_help=True, help="German job market NLP analyser and CV gap tool.")


@app.command()
def version() -> None:
    """Print the package version."""
    from jobnlp import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
