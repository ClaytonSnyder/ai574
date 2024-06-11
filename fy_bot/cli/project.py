"""
Projects CLI
"""

from pathlib import Path

import typer

from fy_bot.project import create_project, delete_project


app = typer.Typer()


@app.command()
def delete(
    name: str,
    projects_path: Path = Path("projects"),
    log_file: str = "fy_bot.log",
    log_level: str = "INFO",
):
    """
    Deletes an existing project

    Args:
        name: Name of the project
        projects_paths: Path where projects are stored. Defaults to Path("projects").
        log_file: Log File. Defaults to "fy_bot.log".
        log_level: Log Level (DEBUG, INFO, WARNING, ERROR). Defaults to "INFO".

    Raises:
        FyBotException: Raised if the project doesn't exist
    """
    delete_project(name, projects_path, log_file, log_level)


@app.command()
def create(
    name: str,
    projects_paths: Path = Path("./projects"),
    log_file: str = "fy_bot.log",
    log_level: str = "INFO",
) -> None:
    """
    Create a new project

    Args:
        name: Name of the project
        projects_paths: Path where projects are stored. Defaults to Path("./projects").
        log_file: Log File. Defaults to "fy_bot.log".
        log_level: Log Level (DEBUG, INFO, WARNING, ERROR). Defaults to "INFO".

    Raises:
        FyBotException: Raised if the project already exists
    """
    create_project(name, projects_paths, log_file, log_level)


if __name__ == "__main__":
    app()
