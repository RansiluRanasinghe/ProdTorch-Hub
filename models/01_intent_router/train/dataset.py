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

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

            text = str(self.texts[idx])
            label = self.labels[idx]

            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            return{
                "input_ids" : encoding["input_ids"].flatten(),
                "attention_mask" : encoding["attention_mask"].flatten(),
                "labels" : torch.tensor(label, dtype=torch.long)
            }
    
def create_label_mapping(categories: List[str]) -> Dict[str, int]:

    return {label: i for i, label in enumerate(sorted(set(categories)))}