"""
CLI Entrypoint
"""

import typer

from fy_bot.cli.datasource import app as datasource
from fy_bot.cli.project import app as project


app = typer.Typer()
app.add_typer(project, name="project")
app.add_typer(datasource, name="datasource")

if __name__ == "__main__":
    app()
