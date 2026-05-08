import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict

class IntentDataset(Dataset):

    def __init__(
            self,
            texts: List[str],
            labels: List[int],
            model_name: str = "distilbert-base-uncased",
            max_length: int = 128
    ):
        
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)