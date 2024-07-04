from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn

from transformers import BertTokenizer


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


class Bert(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, n_heads, dropout_prob):
        super(Bert, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(512, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    hidden_dim, n_heads, hidden_dim * 4, dropout_prob
                )
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_dim, 2)  # Assuming binary classification

    def forward(self, input_ids, attention_mask=None):
        seq_len = input_ids.size(1)
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=attention_mask)

        x = x[:, 0]  # Take the representation of [CLS] token
        logits = self.classifier(x)
        return logits


def build_model(
    vocab_size: int = 30522,
    hidden_dim: int = 768,
    n_layers: int = 12,
    n_heads: int = 12,
    dropout_prob: float = 0.1,
) -> Any:
    model = Bert(vocab_size, hidden_dim, n_layers, n_heads, dropout_prob)
    return model
