# Model 01 — Agentic Intent Router

> A high-performance, lightweight intent classification engine designed for AI agents and RAG systems. This model acts as the **routing brain** of a multi-agent pipeline — directing every user prompt to the right tool, pipeline, or handler before any downstream processing begins.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-005571?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Model](https://img.shields.io/badge/Model-DistilBERT-blueviolet?style=flat-square)]()
[![Docker](https://img.shields.io/badge/Docker-ready-%230db7ed?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)

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

Requires **Docker**. No other local setup is needed.

> [!IMPORTANT]
> You must run the build command from the **root of the ProdTorch-Hub repository**, not from inside the model folder. Docker requires access to `mlflow.db` at the project root.

```bash
# 1. Build the container (run from repository root)
docker build -f models/01_intent_router/Dockerfile -t intent-router:v1 .

# 2. Start the API
docker run -p 8000:8000 intent-router:v1
```

| Endpoint | URL |
|---|---|
| API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |

---

## 📡 API Reference

### `POST /predict`

Classifies a raw text string into one of the five intent categories.

**Request**

```json
{
  "text": "I need to reset my password using the mobile app."
}
```

**Response**

```json
{
  "intent": "KNOWLEDGE_BASE",
  "confidence": 0.982,
  "model_version": "1.0.0",
  "latency_ms": 268.2,
  "meta_data": null
}
```

**Response Fields**

| Field | Type | Description |
|---|---|---|
| `intent` | `string` | Predicted intent class |
| `confidence` | `float` | Model confidence score `[0.0 – 1.0]` |
| `model_version` | `string` | Deployed model version from MLflow |
| `latency_ms` | `float` | End-to-end inference time in milliseconds |
| `meta_data` | `dict` | Optional metadata dictionary |

---

### `GET /health`

Returns the health status of the API and verifies the MLflow artifact is loaded into memory. Used for **Kubernetes / Docker health checks**.

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

## 🛠️ Local Development & Training

To fine-tune this model on your own dataset, replacing the default training data:

**1. Install dependencies**

```bash
pip install -r models/01_intent_router/requirements.txt
```

**2. Inject your data**

Open `models/01_intent_router/train/train.py`. Locate the `train_texts` and `train_labels` lists and replace them with your own Pandas DataFrame or CSV loading logic.

**3. Run the training loop**

Run from the repository root:

```bash
python models/01_intent_router/train/train.py
```

This trains the model and saves the artifact into a local `mlflow.db` and `mlartifacts/` folder at the project root.

**4. Inspect training metrics**

Launch the MLflow UI pointed at the SQLite tracking database:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Navigate to **http://localhost:5001** to view training graphs, hyperparameters, and the model registry.

---

## 📁 Directory Structure

```
ProdTorch-Hub/
├── mlflow.db                     # SQLite tracking DB  (generated during training)
├── mlartifacts/                  # Model weights       (generated during training)
└── models/
    └── 01_intent_router/
        ├── train/
        │   ├── dataset.py        # PyTorch Dataset & fast tokenization
        │   ├── model_arch.py     # DistilBERT neural network architecture
        │   └── train.py          # Training loop & MLflow SQLite registration
        ├── serve/
        │   ├── app.py            # FastAPI application (loads from MLflow)
        │   └── schemas.py        # Pydantic request/response contracts
        ├── Dockerfile            # Containerization instructions
        └── requirements.txt      # Python dependencies
```

---

*Part of the [ProdTorch-Hub](../../README.md) collection — production-grade PyTorch models, built for the backend.*