import os

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import fy_bot
from fy_bot.exception import FyBotException
from fy_bot.logger import LoggerFactory
from nltk.tokenize import sent_tokenize
import yaml

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

SEPARATOR = "__SEP__"

def __get_greetings() -> List[Tuple[str, str]]:
    greetings_file = Path(fy_bot.__file__).parent / "resources" / "greetings.yml"

    with open(greetings_file, 'r', encoding="utf-8") as yaml_file:
        yaml_content = yaml.safe_load(yaml_file)

    greetings = []
    for pairs in yaml_content["conversations"]:
        if len(pairs) == 2:
            question = pairs[0]
            answer = pairs[1]
            greetings.append(((question, answer)))

    return greetings

# Function to generate questions
def generate_question(paragraph, tokenizer, model, device):
    input_text = f"Generate a question given the following context: {paragraph}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True, max_length=512).to(device)
    outputs = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True) # type: ignore
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    torch.cuda.empty_cache()
    return question

def generate_question_answer(paragraph, tokenizer, model, device):
    inputs = tokenizer(paragraph, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=128)
    question_answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
    question_answer = question_answer.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "")
    split_values = question_answer.split(tokenizer.sep_token)
    torch.cuda.empty_cache()
    if len(split_values) == 2:
        return split_values[0], split_values[1]

    return None

# Function to generate answers
def generate_answer(paragraph, question, tokenizer, model, device):
    input_text = f"Answer the following question based on the given context: {question} Context: {paragraph}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True) # type: ignore
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    torch.cuda.empty_cache()
    return answer


def generate_context_question(
    project_name: str,
    device: Any,
    batch_size: int = 3,
    projects_paths: Path = Path("./projects"),
    log_file: str = "fy_bot.log",
    log_level: str = "INFO",
) -> pd.DataFrame:
    """
    This method uses the raw unstructured corpus to generate context
    question-answer pairs to be used in training the chatbot. The model
    used to generate this context is a pretrained T5 model

    Args:
        project_name: Name of the project
        device: Device to run models on (GPU)
        batch_size: Number of sentences to batch together to generate question for
        projects_paths: Path to where projects are stored. Defaults to Path("./projects").
        log_file: Log file. Defaults to "fy_bot.log".
        log_level: Log level. Defaults to "INFO".

    Raises:
        FyBotException: Raised if the corpus doesnt exist

    Returns:
        Dictionary of Question->Answer pairs
    """
    greetings = __get_greetings()

    # Generate questions for each context
    context_question_pairs = {}
    questions = []
    answers = []
    pair_contexts = []
    context_content = ""

    for question, answer in greetings:
        context_question_pairs[question] = answer
        pair_contexts.append(answer)
        questions.append(question)
        answers.append(answer)
        context_content += f"{question}\n"
        context_content += f"{answer}\n"

    logger = LoggerFactory.get_logger(log_file, log_level)
    logger.info("Generating context...")
    corpus_file = projects_paths / project_name / "corpus.txt"

    if not os.path.exists(corpus_file):
        raise FyBotException(
            f"Corpus for project {project_name} doesn't exist."
            + " Please compile corpus prior to calling generate_context_question."
        )

    with open(corpus_file, "r", encoding="utf-8") as corpus:
        corpus_content = corpus.read()

    tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
    model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer").to(device)

    logger.info("Generating Questions...")

    corpus_sentences = sent_tokenize(corpus_content)
    line_chunks = [corpus_sentences[i:i + batch_size] for i in range(0, len(corpus_sentences), batch_size)]

    for line_chunk in tqdm(line_chunks, "Generating Question/Answers..."):
        generated_pair = generate_question_answer("\n".join(line_chunk), tokenizer, model, device)

        if generated_pair is not None:
            question, answer = generated_pair
            pair_context = SEPARATOR.join(line_chunk)

            if answer in pair_context:
                pair_contexts.append(pair_context)
                answers.append(answer.strip())
                questions.append(question.strip())
                context_question_pairs[question.strip()] = answer.strip()
                context_content += f"{question.strip()}\n"
                context_content += f"{answer.strip()}\n"
        torch.cuda.empty_cache()

    context_file = projects_paths / project_name / "context.txt"
    with open(context_file, "w", encoding="utf-8") as context:
        context.write(context_content)

    questions_file = projects_paths / project_name / "questions.txt"
    with open(questions_file, "w", encoding="utf-8") as context:
        context.write("\n".join(questions))

    answers_file = projects_paths / project_name / "answers.txt"
    with open(answers_file, "w", encoding="utf-8") as context:
        context.write("\n".join(answers))

    data = {'question': questions, 'answer': answers, 'context': pair_contexts}
    df = pd.DataFrame(data)
    return df

def __format_df(df: pd.DataFrame, explode_context: bool) -> pd.DataFrame:
    if explode_context:
        new_data = df.set_index(['question', 'answer']).apply(lambda x: x.str.split(SEPARATOR).explode()).reset_index()
    else:
        new_data = df.copy()
        new_data['context'] = new_data["context"].replace(SEPARATOR, "")

    # Clean leading and trailing whitespace
    new_data['context'] = new_data['context'].str.strip()
    new_data['question'] = new_data['question'].str.strip()
    new_data['answer'] = new_data['answer'].str.strip()
    new_data.drop(new_data[new_data['context']==""].index, inplace=True)
    new_data = new_data.reset_index(drop=True)

    return new_data

def __get_start_end_positions(main_string, substring):
    try:
        start_index = main_string.index(substring)
        end_index = start_index + len(substring)-1
        return(pd.Series([start_index,end_index]))
    except ValueError:
        return(pd.Series([None,None]))

def __clean_empty(df: pd.DataFrame) -> pd.DataFrame:
    df['context'] = df['context'].replace('', np.nan)
    df['question'] = df['question'].replace('', np.nan)
    df = df[df['context'].notna()]
    df = df[df['question'].notna()]

    return df

def __prune_data(df: pd.DataFrame) -> pd.DataFrame:
    df['has_answer'] = ((df['start'].notna()))
    df = df.groupby(['question', 'answer', 'has_answer']).apply(lambda x: x.sample(n=1, random_state=11)).reset_index(drop=True)

    return df


def format_data(context: pd.DataFrame, test_split: float = 0.3, explode_context: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This method splits the data into trainand test. Additionally, it can explode the context so that
    there is data for both the questioned answered and not answered.

    Args:
        context: Dataframe containing question, answer, and context
        test_split: Percent of test split
        explode_context: True if the context should be split into separate rows
            and a column should be added highlighting which context contains the answer

    Returns:
        Tuple of train, test, val dataframes
    """
    train_df = context.sample(frac = 1-test_split)
    test_df = context.drop(train_df.index)

    train_df = __format_df(train_df, explode_context)
    test_df = __format_df(test_df, explode_context)

    train_df[["start","end"]] = train_df.apply(lambda row:__get_start_end_positions(row['context'],row["answer"]),axis=1)
    test_df[["start","end"]] = test_df.apply(lambda row:__get_start_end_positions(row['context'],row["answer"]),axis=1)

    train_df = __clean_empty(train_df)
    test_df = __clean_empty(test_df)

    train_df = __prune_data(train_df)
    test_df = __prune_data(test_df)

    return train_df, test_df
