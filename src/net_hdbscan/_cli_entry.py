"""Entry point of the ``net-hdbscan`` command.

The command line needs the optional ``cli`` extra (typer and pyarrow). This
small module gives a clear message instead of an import traceback when the
extra is not installed.
"""

from __future__ import annotations

_MISSING = "The net-hdbscan command needs the 'cli' extra. Install it with: pip install \"net-hdbscan[cli]\""


def main() -> None:
    try:
        from .cli import app
    except ModuleNotFoundError as exc:
        if exc.name in {"typer", "click"}:
            raise SystemExit(_MISSING) from None
        raise
    app()
