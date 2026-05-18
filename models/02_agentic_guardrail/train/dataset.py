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

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def load_security_data(model_name: str = "microsoft/deberta-v3-xsmall") -> Tuple[GuardrailDataset, GuardrailDataset]:

    print("Downloading security dataset from Hugging Face...")

    dataset = load_dataset("deepset/prompt-injections")

    train_data = dataset["train"]
    test_data = dataset["test"]

    print(f"Loaded {len(train_data)} training samples and {len(test_data)} testing samples.")

    train_dataset = GuardrailDataset(
        texts=train_data["text"],
        labels=train_data["label"],
        model_name=model_name
    )

    test_dataset = GuardrailDataset(
        texts=test_data["text"],
        labels=test_data["label"],
        model_name=model_name
    )

    return train_dataset, test_dataset

if __name__ == "__main__":

    try:
        train_ds, test_ds = load_security_data()

        sample = train_ds[0]
        original_text = train_ds.texts[0]
        label = train_ds.labels[0]

        print("Verifying dataset loading and tokenization...\n")

        print(f"Original Text: {original_text}")
        print(f"Security Label: {'UNSAFE (1)' if label == 1 else 'SAFE (0)'}")
        print(f"Input IDs     : {sample['input_ids'][:10]}... (Shape: {sample['input_ids'].shape})")
        print(f"Attention Mask: {sample['attention_mask'][:10]}... (Shape: {sample['attention_mask'].shape})")

        print("---------------------------------------------\n")
        print("Pipeline Success: Hugging Face data loaded and tokenized correctly!")

    except Exception as e:
        print(f"Pipeline Failure: An error occurred - {str(e)}")    