import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import time

from model_arch import IntentClassifier
from dataset import IntentDataset, create_label_mapping

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, pred = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(pred == labels)
        total_loss += loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    return correct_predictions.double() / len(data_loader.dataset), total_loss / len(data_loader)

def run_training():

    MODEL_NAME = "distilbert-base-uncased"
    EPOCHS = 3
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    INTENTS = ["KNOWLEDGE_BASE", "GENERAL_CONVERSATION", "ACTION_REQUIRED", "OFF_TOPIC", "HELP_REQUEST"]
    label_map = create_label_mapping(INTENTS)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Intent_Router_Production")

    with mlflow.start_run():

        mlflow.log_params({
            "model_architecture": "DistilBERT",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "device": str(DEVICE)
        })

        # In production, load your CSV/Parquet here
        train_texts = ["How do I reset my password?", "Hello there!", "Buy this product now"]
        train_labels = [label_map["KNOWLEDGE_BASE"], label_map["GENERAL_CONVERSATION"], label_map["ACTION_REQUIRED"]]

        train_ds = IntentDataset(train_texts, train_labels, model_name=MODEL_NAME)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = IntentClassifier(model_name=MODEL_NAME, num_labels=len(INTENTS))
        model.to(DEVICE)

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        loss_fn = nn.CrossEntropyLoss().to(DEVICE)

        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS}")
            train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE, scheduler)

            mlflow.log_metric("train_accuracy", train_acc.item(), step=epoch)
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            print(f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="intent_router_model",
            registered_model_name="AgenticIntentRouter",
            pip_requirements=["torch", "transformers", "mlflow"]
        )

        print("Model training complete and logged to MLflow.")

if __name__ == "__main__":
    run_training()                