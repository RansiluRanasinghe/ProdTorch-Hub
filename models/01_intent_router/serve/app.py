from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer
import mlflow.pytorch
import torch
import time
import os
import sys

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

        train_dir = os.path.abspath(os.path.join(current_dir, "../train"))
        sys.path.append(train_dir)

        print("Loading model from MLflow Registry (Version 1)...")

        model_uri = "models:/AgenticIntentRouter/1"

        model = mlflow.pytorch.load_model(model_uri)
        model.to(decvice)
        model.eval()

        print("Model loaded successfully and ready for inference!")

    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/predict", response_model=IntentResponse)
def predict_intent(request: IntentRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Please try again later.")
    
    start_time = time.time()

    try:

        encoding = tokenizer(
           request.text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt', 
        )

        input_ids = encoding["input_ids"].to(decvice)
        attention_mask = encoding["attention_mask"].to(decvice)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_class = torch.max(probabilities, dim=0)

        intents = ["KNOWLEDGE_BASE", "GENERAL_CONVERSATION", "ACTION_REQUIRED", "OFF_TOPIC", "HELP_REQUEST"]
        sorted_intents = sorted(set(intents))
        predicted_intent_str = sorted_intents[predicted_class.item()]

        latency = (time.time() - start_time) * 1000

        return IntentResponse(
            intent=predicted_intent_str,
            confidence=confidence.item(),
            model_version="1.0.0",
            latency_ms=round(latency, 2)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    
@app.get("/health")
def health_check():
       return {"status": "healthy", "model_loaded": model is not None} 