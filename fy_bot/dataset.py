from pathlib import Path
from typing import Any, List
import torch
from torch.utils.data import TensorDataset
from transformers import T5Tokenizer

__tokenizer = T5Tokenizer.from_pretrained('t5-small')

def __encode_data(questions: List[str], answers: List[str]) -> TensorDataset:
    input_ids = []
    attention_masks = []
    target_ids = []
    max_question_length = max(len(s) for s in questions)
    max_answer_length = max(len(s) for s in answers)

    for question, answer in zip(questions, answers):
        encoded_dict_question = __tokenizer.encode_plus(
            question,                      # Sentence to encode
            add_special_tokens = True,     # Add '[CLS]' and '[SEP]'
            max_length = max_question_length,               # Pad & truncate all sentences
            padding = 'max_length',
            return_attention_mask = True,  # Construct attn. masks
            return_tensors = 'pt',         # Return pytorch tensors
        )
        encoded_dict_answer = __tokenizer.encode_plus(
            answer,                        # Sentence to encode
            add_special_tokens = True,     # Add '[CLS]' and '[SEP]'
            max_length = max_answer_length,               # Pad & truncate all sentences
            padding = 'max_length',
            return_attention_mask = True,  # Construct attn. masks
            return_tensors = 'pt',         # Return pytorch tensors
        )

        input_ids.append(encoded_dict_question['input_ids'])
        attention_masks.append(encoded_dict_question['attention_mask'])
        target_ids.append(encoded_dict_answer['input_ids'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    target_ids = torch.cat(target_ids, dim=0)
    return TensorDataset(input_ids, attention_masks, target_ids)

def get_encoded_dataset(
        project_name: str,
        projects_paths: Path = Path("./projects")
    ) -> TensorDataset:
    """
    Gets encoded question/answers object
    """
    answers_file = projects_paths / project_name / "answers.txt"
    questions_file = projects_paths / project_name / "questions.txt"

    with open(answers_file, "r", encoding="utf-8") as answers_file_obj:
        answers = answers_file_obj.readlines()

    with open(questions_file, "r", encoding="utf-8") as questions_file_obj:
        questions = questions_file_obj.readlines()

    return __encode_data(questions, answers)
