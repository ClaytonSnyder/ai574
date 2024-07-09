from typing import Any, List, Tuple
from tqdm import tqdm
from transformers import BertConfig, BertModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
from transformers import BertForSequenceClassification
import torch.optim as optim
from transformers import T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score

def get_t5_model(dataset: TensorDataset, device: str) -> Any:
    """
    Pretrained Bert Model
    """
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    model.to(device) # type: ignore
    return model

def train(model: Any, dataset: TensorDataset, device: str, batch_size: int = 8, learning_rate: float = 1e-4, epsilon: float=1e-8, epochs: int=3) -> Tuple[List[float], List[float]]:
    """
    Trains a model
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=RandomSampler(dataset),)
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracies = []

    total_steps = len(dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()

    for epoch in range(0, epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for step, batch in enumerate(tqdm(dataloader)):
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            target_ids = batch[2].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=target_ids)
            loss = outputs.loss
            logits = outputs.logits

            epoch_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            preds = torch.argmax(logits, dim=-1)
            acc = accuracy_score(target_ids.cpu().numpy().flatten(), preds.cpu().numpy().flatten())
            epoch_accuracy += acc
        epoch_loss /= len(dataloader)
        epoch_accuracy /= len(dataloader)

        print(f"{epoch} of {epochs} - Accuracy: {epoch_accuracy} Loss: {epoch_loss}")

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

    return train_accuracies, train_accuracies
