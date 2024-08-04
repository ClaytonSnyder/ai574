import os
import re

from logging import Logger
from pathlib import Path
from deepmultilingualpunctuation import PunctuationModel
import fitz
import nltk
import requests
import spacy

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub
from nltk.tokenize import sent_tokenize
import torch
from tqdm import tqdm

from fy_bot.exception import FyBotException
from fy_bot.logger import LoggerFactory

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from youtube_transcript_api import YouTubeTranscriptApi

nltk.download("punkt")
spacy_language = spacy.load("en_core_web_sm")
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
model_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
model_gpt2.eval() # type: ignore

def calculate_perplexity(sentence, max_length=1024):
    inputs = tokenizer_gpt2.encode(sentence, return_tensors='pt', max_length=max_length, truncation=True)
    if inputs.shape[1] < 2:
        inputs = torch.cat((inputs, torch.tensor([[tokenizer_gpt2.eos_token_id]])), dim=1)

    with torch.no_grad():
        outputs = model_gpt2(inputs, labels=inputs) # type: ignore
        loss = outputs.loss
    return torch.exp(loss).item()

def __is_syntactically_correct(sentence: str, spacy_language: spacy.language.Language):
    doc = spacy_language(sentence)

    # Basic checks for syntactic correctness
    # 1. There should be one and only one ROOT
    # 2. ROOT should be a verb (in most cases)
    # 3. Ensure basic dependencies like subject and object are present (for transitive verbs)

    root_tokens = [token for token in doc if token.dep_ == "ROOT"]
    if len(root_tokens) != 1:
        return False

    root_token = root_tokens[0]
    if root_token.pos_ != "VERB":
        return False

    # Check for the presence of a subject
    subj_tokens = [token for token in doc if "subj" in token.dep_]
    if len(subj_tokens) == 0:
        return False

    # Additional checks can be added as needed

    perplexity = calculate_perplexity(sentence)

    perplexity_threshold = 50
    if perplexity > perplexity_threshold:
        return False

    return True


def compile_corpus(
    project_name: str,
    projects_paths: Path = Path("./projects"),
    log_file: str = "fy_bot.log",
    log_level: str = "INFO",
) -> None:
    logger = LoggerFactory.get_logger(log_file, log_level)
    logger.info("Compiling corpus...")

    raw_folder = projects_paths / project_name / "raw"
    all_sentences = []
    for file in os.listdir(raw_folder):
        with open(os.path.join(raw_folder, file), "r", encoding="utf-8") as raw_file:
            text = raw_file.read()

        # Remove unwanted characters, extra spaces, and normalize text
        #text = text.lower()  # Convert to lowercase
        text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
        text = re.sub(r"\[.*?\]", "", text)  # Remove text in brackets
        text = re.sub(r"\s*-\s*", "-", text)  # Remove spaces around hyphens
        text = re.sub(
            r"[^\w\s.!?]", "", text
        )  # Remove special characters except punctuation

        # Split text into sentences
        sentences = sent_tokenize(text)

        # Further clean sentences if necessary
        pattern = re.compile(r'\b\d+ \d+ \d+ \d+\b')
        for sentence in tqdm(sentences, f"Cleaning sentences in {file}"):
            if (sentence.strip() != ""
            and not pattern.search(sentence)
            and len(sentence.split(" ")) > 3
            and __is_syntactically_correct(sentence.strip(), spacy_language)):
                all_sentences.append(sentence.strip())

    with open(
        projects_paths / project_name / "corpus.txt", "w", encoding="utf-8"
    ) as corpus_file:
        corpus_file.write(" ".join(all_sentences))

    logger.info("Corpus compilation complete.")

def add_youtube_video(
    project_name: str,
    video_id: str,
    projects_paths: Path = Path("./projects"),
    log_file: str = "fy_bot.log",
    log_level: str = "INFO",
):
    logger = LoggerFactory.get_logger(log_file, log_level)
    logger.info(f"Adding YouTube video {video_id} to project {project_name}")

    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    lines = []
    for item in transcript:
        if "text" in item:
            text = item['text']
            lines.append(text)

    raw = " ".join(lines)

    model = PunctuationModel()
    punctuated_text = model.restore_punctuation(raw)

    output_file_path = (
        projects_paths / project_name / "raw" / f"youtube.{video_id}.raw.txt"
    )

    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write(punctuated_text)
    logger.info(f"Raw text successfully extracted from transcript.")


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
