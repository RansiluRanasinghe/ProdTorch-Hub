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

def run_training():

    print("Initializing Agentic Guardrail Training Pipeline...")

    MODEL_NAME = "microsoft/deberta-v3-xsmall"
    EPOCHS = 2
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: Training on {DEVICE}")

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    db_path = os.path.join(root_dir, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("Guardrail_Production")

    with mlflow.start_run():
        mlflow.log_params({
            "model_architecture": "DeBERTa-v3-xsmall",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "device": str(DEVICE)
        })

        train_ds, test_ds = load_security_data(model_name=MODEL_NAME)

        print("\nSubsetting training data to 500 samples for local speed...")
        train_ds.texts = train_ds.texts[:500]
        train_ds.labels = train_ds.labels[:500]

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = GuardrailClassifier(model_name=MODEL_NAME, num_labels=2)
        model.to(DEVICE)

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        total_steps = len(train_loader) * EPOCHS
        schedular = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        loss_fn = nn.CrossEntropyLoss().to(DEVICE)

        for epoch in range(EPOCHS):
            print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")

            train_acc, train_f1, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE, schedular)

            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("train_f1_score", train_f1, step=epoch)
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            print(f"Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} | F1-Score (Unsafe): {train_f1:.4f}")

        print("\nSaving and Registering model to MLflow...")
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="agentic_guardrail_model",
            registered_model_name="AgenticGuardrail",
            pip_requirements=["torch", "transformers", "mlflow", "datasets"]
        )
        
        print("Guardrail model training complete and registered!")

if __name__ == "__main__":
    run_training()            