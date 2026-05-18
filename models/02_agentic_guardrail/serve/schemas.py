from pydantic import BaseModel, Field

class GuardrailRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The raw user prompt to check for malicious intent.")

class GuardrailResponse(BaseModel):

    is_safe: bool = Field(..., description="True if the prompt is safe, False if malicious.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="The model's confidence in its prediction (0.0 to 1.0).")
    flag_reason: str = Field(..., description="The classification category (e.g., SAFE, PROMPT_INJECTION).")
    latency_ms: float = Field(..., description="Inference time in milliseconds.")        