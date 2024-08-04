from pathlib import Path
from typing import Any, Tuple
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.optim import AdamW
from datasets import Dataset
from torch.utils import data
import nltk
from rouge_score.rouge_scorer import RougeScorer
import os

class GP2QADataset(data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: self.encodings[key][idx].clone().detach() for key in self.encodings}
        return item

def _train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader, desc="Training..."):
        # Train
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    torch.cuda.empty_cache()
    return total_loss / len(data_loader)

def __get_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> GP2QADataset:
    df['text'] = df.apply(lambda row: f"Question: {row['question']} Answer: {row['answer']}", axis=1)
    dataset = Dataset.from_pandas(df[['text']])

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask']) # type: ignore
    tokenized_dataset = tokenized_dataset[:]

    return GP2QADataset(tokenized_dataset)

def __evaluate(model, dataloader, device, tokenizer) -> Tuple[float, float, float, float, float, float]:
    model.eval()
    losses = []
    total_log_likelihood = 0
    total_token_count = 0
    references = []
    hypotheses = []
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    smoothing_function = SmoothingFunction().method1

    for batch in tqdm(dataloader, desc="Evaluating..."):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
        loss = outputs.loss
        logits = outputs.logits

        # Accumulate loss
        losses.append(loss.item())

        # Calculate log-likelihood and token count for perplexity
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch['input_ids'][..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        log_likelihood = -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        non_pad_mask = shift_labels.ne(tokenizer.pad_token_id)
        total_log_likelihood += (log_likelihood * non_pad_mask.view(-1)).sum().item()
        total_token_count += non_pad_mask.sum().item()

        # Collect references and hypotheses for BLEU score
        predictions = torch.argmax(logits, dim=-1)
        references.extend(tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True))
        hypotheses.extend(tokenizer.batch_decode(predictions, skip_special_tokens=True))
        torch.cuda.empty_cache()

    avg_loss = np.mean(losses)
    perplexity = np.exp(total_log_likelihood / total_token_count)

    # Calculate BLEU score
    bleu_scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = nltk.word_tokenize(ref)
        hyp_tokens = nltk.word_tokenize(hyp)
        bleu_scores.append(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing_function))

    avg_bleu_score = np.mean(bleu_scores)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    for ref, hyp in zip(references, hypotheses):
        scores = rouge_scorer.score(ref, hyp)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    avg_rouge1 = np.mean(rouge1_scores)
    avg_rouge2 = np.mean(rouge2_scores)
    avg_rougeL = np.mean(rougeL_scores)

    return avg_loss, perplexity, avg_bleu_score, avg_rouge1, avg_rouge2, avg_rougeL  # type: ignore

def __get_model_path(project_name: str, projects_paths: Path = Path("./projects")) -> str:
    return os.path.join(projects_paths, f"{project_name}/model")

def train(
        project_name: str,
        data: pd.DataFrame,
        val: pd.DataFrame,
        device: str,
        learning_rate: float = 2e-4,
        epochs: int = 8,
        batch_size: int = 2,
        scheduler_warmup_steps: int = 10000,
        max_length: int = 128,
        projects_paths: Path = Path("./projects"),
    ) -> Any:
    torch.cuda.empty_cache()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = __get_dataset(data, tokenizer, max_length)
    test_dataset = __get_dataset(val, tokenizer, max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device) # type: ignore
    model.resize_token_embeddings(len(tokenizer)) # type: ignore

    optimizer = AdamW(model.parameters(), lr=learning_rate) # type: ignore

    num_training_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=scheduler_warmup_steps,
        num_training_steps=num_training_steps
    )

    train_losses = []
    test_losses = []
    bleu_scores = []
    perplexity_scores = []

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = _train_epoch(model, train_dataloader, optimizer, scheduler, device)
        avg_loss, perplexity, avg_bleu_score, avg_rouge1, avg_rouge2, avg_rougeL = __evaluate(model, test_dataloader, device, tokenizer)
        print(f"Epoch - Loss {train_loss}, Val Loss {avg_loss}, Perplexity {perplexity}, BLEU {avg_bleu_score}")
        print(f"Epoch - Rouge 1 {avg_rouge1}, Rouge 2 {avg_rouge2}, Rouge L {avg_rougeL}")

        train_losses.append(train_loss)
        test_losses.append(avg_loss)
        bleu_scores.append(avg_bleu_score)
        rouge1_scores.append(avg_rouge1)
        rouge2_scores.append(avg_rouge2)
        rougeL_scores.append(avg_rougeL)
        perplexity_scores.append(perplexity)

    path = __get_model_path(project_name, projects_paths)

    model.save_pretrained(path) # type: ignore
    tokenizer.save_pretrained(path)

    return model, train_losses, test_losses, bleu_scores, perplexity_scores, rouge1_scores, rouge2_scores, rougeL_scores

def chat(project_name: str, question, device, projects_paths: Path = Path("./projects")):
    path = __get_model_path(project_name, projects_paths)

    model = GPT2LMHeadModel.from_pretrained(path)
    model.to(device) # type: ignore
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    model.eval() # type: ignore

    input_text = f"Question: {question} Answer:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    outputs = model.generate( # type: ignore
        inputs["input_ids"],
        max_length=50,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the answer part
    answer = response.split("Answer:")[1].strip() if "Answer:" in response else response

    return answer
