from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer
import mlflow.pytorch
import torch
import time
import os

from schemas import IntentRequest, IntentResponse, IntentCategory

app = FastAPI(
    title="Agentic Intent Router API",
    description="Production API for classifying user prompts into system intents.",
    version="1.0.0"
)

model = None
tokenizer = None
decvice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def load_model():

    global model, tokenizer

    try:
        print("Initializing Tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)

        print("Locating MLFlow Database")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
        db_path = os.path.join(root_dir, "mlflow.db")

        if not os.path.exists(db_path):
            print(f"WARNING: Database not found at {db_path}")
            return
        
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")

        print("Loading model from MLflow Registry (Version 1)...")

        model_uri = "models:/AgenticIntentRouter/1"

        model = mlflow.pytorch.load_model(model_uri)
        model.to(decvice)
        model.eval()

        print("Model loaded successfully and ready for inference!")

    except Exception as e:
        print(f"Error loading model: {e}")
    