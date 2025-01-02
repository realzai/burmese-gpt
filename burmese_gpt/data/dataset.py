from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset

from burmese_gpt.config import TrainingConfig


class BurmeseDataset(Dataset):
    def __init__(self, split="train", max_length=128, config: TrainingConfig = None):
        self.dataset = load_dataset(config.dataset_url, split=split)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
        }
