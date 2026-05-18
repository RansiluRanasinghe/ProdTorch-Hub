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