"""
Data source CLI
"""

from pathlib import Path

import typer

from fy_bot.datasource import add_document


app = typer.Typer()


@app.command()
def download(
    url: str,
    project_name: str,
    projects_path: Path = Path("projects"),
    log_file: str = "fy_bot.log",
    log_level: str = "INFO",
):
    """
    Downloads a datasource into a project

    Args:
        url: Url to the datasource to download
        project_name: Name of the project
        projects_paths: Path where projects are stored. Defaults to Path("projects").
        log_file: Log File. Defaults to "fy_bot.log".
        log_level: Log Level (DEBUG, INFO, WARNING, ERROR). Defaults to "INFO".

    Raises:
        FyBotException: Raised if the project doesn't exist
    """
    add_document(url, project_name, projects_path, log_file, log_level)
