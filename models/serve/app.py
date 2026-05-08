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