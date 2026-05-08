FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY models/01_intent_router/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY mlflow.db /app/mlflow.db
COPY mlruns /app/mlruns

COPY models/01_intent_router /app/models/01_intent_router

WORKDIR /app/models/01_intent_router/serve

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]