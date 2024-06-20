from pathlib import Path
from typing import Any, List, Tuple

import torch
import torch.nn as nn

from transformers import BertForQuestionAnswering, BertTokenizer


def get_tokenized_context(
    project_name: str, projects_paths: Path = Path("projects")
) -> Tuple[Any, Any]:
    answers_file = projects_paths / project_name / "answers.txt"
    questions_file = projects_paths / project_name / "questions.txt"

    with open(answers_file, "r", encoding="utf-8") as answers_file_obj:
        answers = answers_file_obj.readlines()

    with open(questions_file, "r", encoding="utf-8") as questions_file_obj:
        questions = questions_file_obj.readlines()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    question_encodings = tokenizer(
        questions, padding=True, truncation=True, return_tensors="pt"
    )
    answer_encodings = tokenizer(
        answers, padding=True, truncation=True, return_tensors="pt"
    )

    return (question_encodings, answer_encodings)


class BertQA(nn.Module):
    def __init__(self):
        super(BertQA, self).__init__()
        self.bert = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)  # type: ignore
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        return start_logits, end_logits


def build_model() -> Any:
    model = BertQA()
    return model
