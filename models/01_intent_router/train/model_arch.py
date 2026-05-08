import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class IntentClassifier(nn.Module):

    def __init__(self, model_name:str, num_labels:int, dropout_rate:float=0.3):
        super(IntentClassifier, self).__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)

        self.pre_classifier = nn.Linear(self.config.dim, self.config.dim)
        self.classifier = nn.Linear(self.config.dim, num_labels)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()