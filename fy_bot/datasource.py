import os
import shutil

from logging import Logger
from pathlib import Path

import fitz
import requests

from bs4 import BeautifulSoup
from click import open_file
from ebooklib import ITEM_DOCUMENT, epub

from fy_bot.exception import FyBotException
from fy_bot.logger import LoggerFactory


def add_document(
    project_name: str,
    url: str,
    projects_paths: Path = Path("./projects"),
    log_file: str = "fy_bot.log",
    log_level: str = "INFO",
):
    logger = LoggerFactory.get_logger(log_file, log_level)
    logger.info(f"Adding document {url} to project {project_name}")
    file_path = __download_file(project_name, url, projects_paths, logger)

    if str(file_path).endswith(".epub"):
        content = __extract_text_from_epub(file_path)
    elif str(file_path).lower().endswith(".pdf"):
        content = __extract_text_from_pdf(file_path)
    elif str(file_path).endswith(".txt"):
        content = file_path.read_text()
    else:
        raise FyBotException(
            f"Unknown file type: {file_path}. Cannot extract raw text."
        )

    output_file_path = (
        projects_paths / project_name / "raw" / f"{os.path.basename(file_path)}.raw.txt"
    )

    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write(content)
    logger.info(f"Raw text successfully extracted.")


def __download_file(project_name: str, url: str, projects_paths: Path, logger: Logger):
    file_path = projects_paths / project_name / "downloads" / url.split("/")[-1]

    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as file:
            file.write(response.content)
            logger.info("File downloaded successfully...")
    else:
        logger.error("Failed to download file.")
        response.raise_for_status()

    return file_path


def __extract_text_from_pdf(pdf_path: Path):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    text_content = []

    # Iterate through each page
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text = page.get_text("text")  # type: ignore
        text_content.append(text)

    # Join all text segments into one string
    return "\n".join(text_content)


def __extract_text_from_epub(epub_path: Path):
    # Load the epub file
    book = epub.read_epub(epub_path)
    text_content = []

    # Iterate through the items in the epub file
    for item in book.get_items():
        # We're only interested in document items
        if item.get_type() == ITEM_DOCUMENT:
            # Parse the content with BeautifulSoup
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            # Extract text and append it to the list
            text_content.append(soup.get_text())

    # Join all text segments into one string
    return "\n".join(text_content)
