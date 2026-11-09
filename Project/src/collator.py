import torch

from typing import Any
from dataclasses import dataclass

@dataclass
class ChatDataCollator:
    tokenizer: Any
    padding: str | bool = True

    def __call__(self, features: list[dict[str, list[int]]]) -> dict:
        max_len = max(len(feature["input_ids"]) for feature in features)

        labels = []
        input_ids = []
        attention_mask = []

        pad_id = self.tokenizer.pad_token_id
        for feature in features:
            lbl = feature["labels"]
            ids = feature["input_ids"]

            n_pad = max_len - len(ids)

            input_ids.append(ids + [pad_id] * n_pad)
            labels.append(lbl + [-100] * n_pad)          
            attention_mask.append([1] * len(ids) + [0] * n_pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }