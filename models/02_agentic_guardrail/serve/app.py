from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer
import mlflow.pytorch
import torch
import time
import os
import sys

from schemas import GuardrailRequest, GuardrailResponse

app = FastAPI(
    title="Agentic Guardrail API",
    description="High-speed security firewall for AI agents.",
    version="1.0.0"
)

model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def load_model():

    global model, tokenizer

    try:
        print("Initializing DeBERTa Tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall", use_fast=True)

        print("Locating MLflow Database...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
        db_path = os.path.join(root_dir, "mlflow.db")

        if not os.path.exists(db_path):
            print(f"WARNING: Database not found at {db_path}")
            return
        
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")

        train_dir = os.path.abspath(os.path.join(current_dir, "../train"))
        sys.path.append(train_dir)

        print("Loading Guardrail model from MLflow Registry (Version 1)...")
        model_uri = "models:/AgenticGuardrail/1"

        model = mlflow.pytorch.load_model(model_uri)
        model.to(device)
        model.eval()
        print("Guardrail Model loaded successfully and ready to block hackers!")

    except Exception as e:
        print(f"Error loading model: {e}")    