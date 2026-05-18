import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import List, Dict, Tuple

class GuardrailDataset(Dataset):

    def __init__(
            self,
            texts: List[str],
            labels: List[int],
            model_name: str = "microsoft/deberta-v3-xsmall",
            max_length: int = 128
        ):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)