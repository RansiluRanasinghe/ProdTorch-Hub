import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class GuardrailClassifier(nn.Module):

    def __init__(self, model_name: str = "microsoft/deberta-v3-xsmall", num_labels: int = 2, dropout_rate: float = 0.3 ):
        super(GuardrailClassifier, self).__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.deberta = AutoModel.from_pretrained(model_name)

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)