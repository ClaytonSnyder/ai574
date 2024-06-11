"""
CLI Entrypoint
"""

import typer

from fy_bot.cli.project import app as project


app = typer.Typer()
app.add_typer(project, name="project")

if __name__ == "__main__":
    app()
