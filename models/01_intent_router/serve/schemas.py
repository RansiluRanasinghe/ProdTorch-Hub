from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from enum import Enum

class IntentCategory(str, Enum):

    KNOWLEDGE_BASE = "KNOWLEDGE_BASE"
    GENERAL_CONVERSATION = "GENERAL_CONVERSATION"
    ACTION_REQUIRED = "ACTION_REQUIRED"
    OFF_TOPIC = "OFF_TOPIC"
    HELP_REQUEST = "HELP_REQUEST"

class IntentRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The raw text prompt from the user to be routed.",
        examples=["How do I update my billing information?"]
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example" : {
                "text": "I need help with my recent order."
            }
        }
    )

class IntentResponse(BaseModel):

    intent: IntentCategory = Field(
        ...,
        description="The predicted intent label."
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="The softmax confidence score (0.0 to 1.0)."
    )

    model_version: str = Field(..., description="The specific version of the model used.")
    latency_ms: float = Field(..., description="Time taken for inference in milliseconds.")

    meta_data: Optional[dict] = Field(default=None)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "intent": "KNOWLEDGE_BASE",
                "confidence": 0.985,
                "model_version": "1.0.0",
                "latency_ms": 12.4,
                "metadata": {"processor": "cpu"}
            }
        }
    )