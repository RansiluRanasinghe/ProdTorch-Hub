from pydantic import BaseModel, Field

class GuardrailRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The raw user prompt to check for malicious intent.")    