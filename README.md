# ML Arsenal 🔥
> *From research to production — in one pull.*

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Models](https://img.shields.io/badge/Models-15-blueviolet?style=for-the-badge)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](http://makeapullrequest.com)

**A curated arsenal of 15 production-grade Machine Learning models** — each one tracked, versioned, containerized, and API-ready. No boilerplate. No dependency hell. Just pull and integrate.

---

## 🎯 The Problem This Solves

Every ML engineer knows the pain: a beautiful Jupyter Notebook that works perfectly — until it hits production and falls apart. Missing dependencies, no API layer, no versioning, no observability. Back to square one.

**ML Arsenal exists to eliminate that gap entirely.**

This is not a research repo. It is not a tutorial collection. It is a battle-tested, enterprise-grade suite of models built for one purpose: **drop straight into your backend and work**.

```bash
# This is the entire integration workflow
docker pull mlarsenalio/<model-name>:latest
docker run -p 8000:8000 mlarsenalio/<model-name>
# Your model is live at http://localhost:8000/docs
```

---

## 👥 Who This Is For

| Audience | Use Case |
|---|---|
| **Software Engineers** | Integrate ML capabilities without becoming an ML expert |
| **AI / ML Engineers** | Skip the infra boilerplate, build on a solid foundation |
| **Backend Developers** | Treat models like any other microservice |
| **DevOps / MLOps** | Standardized deployment patterns across all models |
| **Technical Founders** | Ship AI-powered features in days, not months |

---

## 🏗️ The Production Standard

Every single model in this repository is held to the same uncompromising standard. No exceptions.

```
┌─────────────────────────────────────────────────────┐
│              THE ML ARSENAL CONTRACT                │
├────────────────────┬────────────────────────────────┤
│  📊 TRACKED        │ Full MLflow integration         │
│                    │ Metrics, params, artifacts      │
│                    │ Reproducible experiments        │
├────────────────────┼────────────────────────────────┤
│  ⚡ SERVABLE       │ FastAPI with async support      │
│                    │ Pydantic I/O validation         │
│                    │ Auto-generated /docs endpoint   │
├────────────────────┼────────────────────────────────┤
│  🐳 CONTAINERIZED  │ Optimized multi-stage builds    │
│                    │ docker-compose ready            │
│                    │ One-command deployment          │
├────────────────────┼────────────────────────────────┤
│  🧹 MAINTAINABLE   │ Modular, typed Python code      │
│                    │ Inline documentation            │
│                    │ Benchmarks & performance notes  │
└────────────────────┴────────────────────────────────┘
```

---

## 📦 The Model Collection

> 🚧 **Active Development** — Models are being added progressively. Each one ships only when it meets the full production standard above. Star and watch the repo for updates.

### 🤖 Agentic Utilities
*Tools designed to be used as components in larger AI agent pipelines and RAG systems.*

| # | Model | Description | Status |
|---|---|---|---|
| 01 | `intent-classifier` | Multi-class intent detection for agent routing | 🔜 Coming Soon |
| 02 | `document-embedder` | High-performance text embedding for RAG pipelines | 🔜 Coming Soon |
| 03 | `query-rewriter` | Query reformulation for improved retrieval quality | 🔜 Coming Soon |

### 📝 NLP & Text
*Natural language processing models for real-world text tasks.*

| # | Model | Description | Status |
|---|---|---|---|
| 04 | `sentiment-analyzer` | Fine-grained sentiment analysis (5-class) | 🔜 Coming Soon |
| 05 | `named-entity-recognizer` | Production NER with confidence scores | 🔜 Coming Soon |
| 06 | `text-summarizer` | Extractive + abstractive summarization | 🔜 Coming Soon |
| 07 | `language-detector` | Fast 50+ language detection | 🔜 Coming Soon |

### 🔍 Computer Vision
*Vision models built for backend image processing pipelines.*

| # | Model | Description | Status |
|---|---|---|---|
| 08 | `image-classifier` | Multi-class image classification (ImageNet) | 🔜 Coming Soon |
| 09 | `object-detector` | Real-time object detection with bounding boxes | 🔜 Coming Soon |
| 10 | `image-embedder` | Dense visual feature extraction for similarity search | 🔜 Coming Soon |

### 📊 Data & Analytics
*Structured data models for enterprise analytics workflows.*

| # | Model | Description | Status |
|---|---|---|---|
| 11 | `anomaly-detector` | Unsupervised anomaly detection for time-series | 🔜 Coming Soon |
| 12 | `customer-segmenter` | K-Means / DBSCAN clustering with explainability | 🔜 Coming Soon |
| 13 | `churn-predictor` | Binary classification with calibrated probabilities | 🔜 Coming Soon |

### 🔢 Numeric & Forecasting
*Regression and forecasting models for quantitative tasks.*

| # | Model | Description | Status |
|---|---|---|---|
| 14 | `demand-forecaster` | Multi-step time-series forecasting (Temporal Fusion) | 🔜 Coming Soon |
| 15 | `price-estimator` | Feature-rich regression with uncertainty bounds | 🔜 Coming Soon |

---

## 📂 Repository Architecture

```
ml-arsenal/
│
├── models/
│   └── [model-name]/               # Each model is fully self-contained
│       ├── train/
│       │   ├── dataset.py          # Data loading & preprocessing
│       │   ├── model.py            # PyTorch model definition
│       │   ├── train.py            # Training loop with MLflow logging
│       │   └── evaluate.py         # Evaluation metrics & reporting
│       ├── serve/
│       │   ├── main.py             # FastAPI application
│       │   ├── schemas.py          # Pydantic request/response models
│       │   └── predictor.py        # Model loading & inference logic
│       ├── Dockerfile              # Optimized multi-stage build
│       ├── requirements.txt        # Pinned dependencies
│       └── README.md              # API spec, benchmarks, examples
│
├── infrastructure/
│   └── docker-compose.yml          # Centralized MLflow tracking server
│
├── .github/
│   └── workflows/                  # CI/CD pipelines (coming soon)
│
├── LICENSE
└── README.md
```

---

## ⚙️ Getting Started

### Step 1 — Start the MLflow Tracking Server

Before training any model, spin up the centralized tracking server. All models log to this shared instance.

```bash
cd infrastructure
docker-compose up -d
```

The MLflow UI will be live at **http://localhost:5000**.

### Step 2 — Train a Model

```bash
cd models/<model-name>
pip install -r requirements.txt
python train/train.py
```

All training runs — metrics, hyperparameters, and artifacts — are automatically logged to MLflow.

### Step 3 — Serve the Model

```bash
# Option A: Run locally
uvicorn serve.main:app --host 0.0.0.0 --port 8000 --reload

# Option B: Run in Docker (recommended)
docker build -t ml-arsenal/<model-name> .
docker run -p 8000:8000 ml-arsenal/<model-name>
```

### Step 4 — Explore the API

Navigate to **http://localhost:8000/docs** for the auto-generated, interactive API documentation. Every endpoint includes request schemas, response schemas, and live testing.

---

## 🔌 Integration Example

Every model exposes the same clean REST interface. Here is a universal example:

```python
import requests

# Example: intent-classifier
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Cancel my subscription and issue a refund"}
)

print(response.json())
# {
#   "label": "cancellation_request",
#   "confidence": 0.97,
#   "latency_ms": 12.4
# }
```

Each model's `README.md` contains its full API specification and language-specific code examples.

---

## 🤝 Contributing

We maintain a strict standard for inclusion. Community contributions are welcome, provided they meet the full production standard: a FastAPI serving layer, MLflow tracking, Docker containerization, and a complete `README.md` with benchmarks.

Detailed contribution guidelines — including the evaluation rubric — will be published alongside the first model release.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Developed and maintained by [Ransilu Ranasinghe](https://github.com/ransilu)**

*If this project saves you time, consider giving it a ⭐ — it helps others find it.*

</div>
