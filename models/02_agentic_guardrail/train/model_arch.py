import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class GuardrailClassifier(nn.Module):

    def __init__(self, model_name: str = "microsoft/deberta-v3-xsmall", num_labels: int = 2, dropout_rate: float = 0.3 ):
        super(GuardrailClassifier, self).__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.deberta = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):

        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)

        hidden_state = outputs.last_hidden_state

        cls_token_state = hidden_state[:, 0, :]

        cls_token_state = self.dropout(cls_token_state)
        logits = self.classifier(cls_token_state)

        return logits

if __name__ == "__main__":

    try:
       print("Downloading base model and initializing Guardrail Architecture...")

       model = GuardrailClassifier()

       dummy_input_ids = torch.randint(0, 1000, (1, 128))
       dummy_attention_mask = torch.ones((2, 128), dtype=torch.long)

       print("Running forward pass test...")

       with torch.no_grad():
           logits = model(dummy_input_ids, dummy_attention_mask)

       print(f"Output Logits Shape: {logits.shape}")

       if logits.shape == (2, 2):
           print("Architecture Verification Success: Model built and forward pass functional.")
       else:
           print("Architecture Verification Failed: Unexpected output shape.")

    except Exception as e:
        print(f"Error during model architecture verification: {e}")                      