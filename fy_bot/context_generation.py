import os

from pathlib import Path
from typing import Any, Dict

from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from fy_bot.exception import FyBotException
from fy_bot.logger import LoggerFactory


def generate_context_question(
    project_name: str,
    device: Any,
    projects_paths: Path = Path("./projects"),
    log_file: str = "fy_bot.log",
    log_level: str = "INFO",
) -> Dict[str, str]:
    """
    This method uses the raw unstructured corpus to generate context
    question-answer pairs to be used in training the chatbot. The model
    used to generate this context is a pretrained T5 model

    Args:
        project_name: Name of the project
        projects_paths: Path to where projects are stored. Defaults to Path("./projects").
        log_file: Log file. Defaults to "fy_bot.log".
        log_level: Log level. Defaults to "INFO".

    Raises:
        FyBotException: Raised if the corpus doesnt exist

    Returns:
        Dictionary of Question->Answer pairs
    """
    logger = LoggerFactory.get_logger(log_file, log_level)
    logger.info("Compiling corpus...")
    corpus_file = projects_paths / project_name / "corpus.txt"

    if not os.path.exists(corpus_file):
        raise FyBotException(
            f"Corpus for project {project_name} doesn't exist."
            + " Please compile corpus prior to calling generate_context_question."
        )

    with open(corpus_file, "r", encoding="utf-8") as corpus:
        corpus_content = corpus.read()

    sentences = corpus_content.split("\n")
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    tokenizer = T5Tokenizer.from_pretrained(
        "mrm8488/t5-base-finetuned-question-generation-ap"
    )

    model = T5ForConditionalGeneration.from_pretrained(
        "mrm8488/t5-base-finetuned-question-generation-ap"
    )
    model = model.to(device)  # type: ignore

    # Generate questions for each context
    context_question_pairs = {}
    questions = []
    answers = []
    for sentence in tqdm(sentences, "Generating context questions..."):
        if not sentence.endswith("?"):
            question = __generate_question(sentence, tokenizer, model, device)
            context_question_pairs[question] = sentence
            questions.append(question)
            answers.append(sentence)

    context_content = ""
    for question in tqdm(context_question_pairs, "Writing context..."):
        context_content += f"{question}\n"
        context_content += f"{context_question_pairs[question]}\n"

    context_file = projects_paths / project_name / "context.txt"
    with open(context_file, "w", encoding="utf-8") as context:
        context.write(context_content)

    questions_file = projects_paths / project_name / "questions.txt"
    with open(questions_file, "w", encoding="utf-8") as context:
        context.write("\n".join(questions))

    answers_file = projects_paths / project_name / "answers.txt"
    with open(answers_file, "w", encoding="utf-8") as context:
        context.write("\n".join(answers))

    return context_question_pairs


def __generate_question(context, tokenizer, model, device):
    input_text = f"generate question: {context} </s>"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.to(device)
    outputs = model.generate(
        input_ids=input_ids, max_length=64, num_beams=4, early_stopping=True
    )
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question.replace("question: ", "")
