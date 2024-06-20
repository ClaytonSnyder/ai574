import os
import shutil

from pathlib import Path

from fy_bot.exception import FyBotException
from fy_bot.logger import LoggerFactory


def project_exists(name: str, projects_paths: Path = Path("./projects")) -> bool:
    return os.path.exists(projects_paths / name)


def create_project(
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
    logger = LoggerFactory.get_logger(log_file, log_level)
    logger.info(f"Creating new project: {name}...")
    project_path = projects_paths / name
    downloads_path = projects_paths / name / "downloads"
    raw_path = projects_paths / name / "raw"

    if project_path.exists():
        logger.info(f"Failed to create project: {name}")
        raise FyBotException(f"Project '{name}' already exists")

    os.makedirs(project_path)
    os.makedirs(downloads_path)
    os.makedirs(raw_path)

    logger.info(f"Project Created.")


def delete_project(
    name: str,
    projects_paths: Path = Path("./projects"),
    log_file: str = "fy_bot.log",
    log_level: str = "INFO",
) -> None:
    """
    Deletes an existing project

    Args:
        name: Name of the project
        projects_paths: Path where projects are stored. Defaults to Path("./projects").
        log_file: Log File. Defaults to "fy_bot.log".
        log_level: Log Level (DEBUG, INFO, WARNING, ERROR). Defaults to "INFO".

    Raises:
        FyBotException: Raised if the project doesn't exist
    """
    logger = LoggerFactory.get_logger(log_file, log_level)
    logger.info(f"Deleting project: {name}...")
    project_path = projects_paths / name

    if not project_path.exists():
        logger.info(f"Failed to delete project: {name}")
        raise FyBotException(f"Project '{name}' doesn't exist")

    shutil.rmtree(project_path)
    logger.info(f"Project deleted.")
