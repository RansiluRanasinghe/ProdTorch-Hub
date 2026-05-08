from pydantic import BaseModel
from typing import Optional
from enum import Enum

class IntentCategory(str, Enum):

    KNOWLEDGE_BASE = "KNOWLEDGE_BASE"
    GENERAL_CONVERSATION = "GENERAL_CONVERSATION"
    ACTION_REQUIRED = "ACTION_REQUIRED"
    OFF_TOPIC = "OFF_TOPIC"
    HELP_REQUEST = "HELP_REQUEST"