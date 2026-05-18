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

@app.post("/predict", response_model=GuardrailResponse)
def predict_security(request: GuardrailRequest):

    if model is None:
          raise HTTPException(status_code=503, detail="Model is currently unavailable.")

    start_time = time.time()

    try:

        encoding = tokenizer(
            request.text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted_class = torch.max(probabilities, dim=0)

        is_safe = (predicted_class.item() == 0)
        flag_reason = "SAFE" if is_safe else "PROMPT_INJECTION"

        latency = (time.time() - start_time) * 1000

        return GuardrailResponse(
            is_safe=is_safe,
            confidence=confidence.item(),
            flag_reason=flag_reason,
            latency_ms=round(latency, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

@app.get("/health")
def health_check():
     return {"status": "healthy", "model_loaded": model is not None}             