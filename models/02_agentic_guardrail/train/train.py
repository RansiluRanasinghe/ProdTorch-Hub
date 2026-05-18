import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_arch import GuardrailClassifier
from dataset import load_security_data

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):

    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

        loss = loss_fn(outputs, labels)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item()

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')

    return acc, f1, (total_loss / len(data_loader))    