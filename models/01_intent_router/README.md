# Model 01 — Agentic Intent Router

> A high-performance, lightweight intent classification engine designed for AI agents and RAG systems. This model acts as the **routing brain** of a multi-agent pipeline — directing every user prompt to the right tool, pipeline, or handler before any downstream processing begins.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-005571?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-96.4%25-brightgreen?style=flat-square)]()
[![Latency](https://img.shields.io/badge/Latency-~15ms_CPU-blue?style=flat-square)]()

---

## 🧠 Intent Classes

This model classifies raw user text into one of five intents:

| Intent | Description | Example Trigger |
|---|---|---|
| `KNOWLEDGE_BASE` | Triggers a RAG / vector search | *"How do I reset my password?"* |
| `GENERAL_CONVERSATION` | Routes to casual small-talk handler | *"Hey, how's it going?"* |
| `ACTION_REQUIRED` | Triggers a specific function or API call | *"Book a meeting for tomorrow at 3pm"* |
| `OFF_TOPIC` | Identifies queries outside system scope | *"What's the weather in Paris?"* |
| `HELP_REQUEST` | Routes to documentation or human support | *"I'm stuck, I need help"* |

---

## 🚀 Quickstart

Requires **Docker**. No other setup needed.

```bash
# 1. Build the container
docker build -t intent-router:v1 ./models/01_intent_router

# 2. Start the API
docker run -p 8000:8000 intent-router:v1
```

The API is live at **http://localhost:8000**
Interactive Swagger docs at **http://localhost:8000/docs**

---

## 📡 API Reference

### `POST /predict`

Classifies a raw text string into one of the five intent categories.

**Request**

```json
{
  "text": "How do I reset my password using the mobile app?"
}
```

**Response**

```json
{
  "intent": "KNOWLEDGE_BASE",
  "confidence": 0.982,
  "model_version": "1.0.4",
  "latency_ms": 12.5
}
```

**Response Fields**

| Field | Type | Description |
|---|---|---|
| `intent` | `string` | Predicted intent class |
| `confidence` | `float` | Model confidence score `[0.0 – 1.0]` |
| `model_version` | `string` | Deployed model version from MLflow |
| `latency_ms` | `float` | End-to-end inference time in milliseconds |

---

## 🔌 Integration Example

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Cancel my subscription and issue a refund"}
)

result = response.json()

if result["intent"] == "ACTION_REQUIRED" and result["confidence"] > 0.85:
    trigger_cancellation_pipeline(result)
```

---

## 🛠️ Technical Specifications

| Detail | Specification |
|---|---|
| Framework | PyTorch 2.x |
| Architecture | Fine-tuned `DistilBERT-base-uncased` |
| Serving | FastAPI + Uvicorn (async) |
| Experiment Tracking | MLflow |
| Optimization | TorchScript compiled + INT8 quantization |
| Input | Raw UTF-8 text string |
| Output | Intent label + confidence score |

---

## 📊 Performance

Trained on a curated dataset of **15,000 conversational prompts** tailored for enterprise AI environments.

```
Accuracy          96.4%
F1-Score          0.95
Inference Speed   ~15ms  (CPU, no GPU required)
```

To inspect full training logs, loss curves, and artifact history, start the MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then navigate to **http://localhost:5000** and open the `intent-router` experiment.

---

## 📁 Directory Structure

```
01_intent_router/
├── train/
│   ├── dataset.py          # Data loading & preprocessing
│   ├── model.py            # DistilBERT fine-tuning definition
│   ├── train.py            # Training loop with MLflow logging
│   └── evaluate.py         # Metrics & confusion matrix
├── serve/
│   ├── main.py             # FastAPI application
│   ├── schemas.py          # Pydantic request/response models
│   └── predictor.py        # Model loading & inference
├── Dockerfile
├── requirements.txt
└── README.md               # This file
```

---

*Part of the [ML Arsenal](../../README.md) collection — production-grade models, ready to integrate.*