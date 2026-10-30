"""
kokoro-tts - Local Kokoro Text-to-Speech Service
"""

__version__ = "0.1.0"


def main() -> None:
    """CLI entry point for kokoro-tts."""
    import typer

    from kokoro_tts.cli import app as cli_app

    # Create a new Typer app that wraps the CLI app
    root_app = typer.Typer(
        name="kokoro-tts",
        help="Local Kokoro Text-to-Speech Service",
        add_completion=False,
    )

    root_app.add_typer(cli_app, name=None)  # Main CLI

    root_app()


if __name__ == "__main__":
    main()
