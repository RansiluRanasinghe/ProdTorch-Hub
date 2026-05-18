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

    def forward(self, input_ids, attention_mask):

        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)

        hidden_state = outputs.last_hidden_state

        cls_token_state = hidden_state[:, 0, :]

        cls_token_state = self.dropout(cls_token_state)
        logits = self.classifier(cls_token_state)

        return logits    