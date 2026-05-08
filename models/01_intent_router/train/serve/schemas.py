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