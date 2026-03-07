from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class MidiDataCollatorForCausalLM:
    """Collator for causal LM: padding and labels (shifted input_ids, -100 on pad and last position)."""

    tokenizer: Any
    mlm: bool = False
    mlm_probability: float = 0.15
    pad_token_id: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __post_init__(
        self,
    ) -> None:
        if self.mlm:
            raise ValueError(
                "Masked language modeling (MLM) is not supported for causal language modeling. Please set mlm=False.",
            )
        if self.pad_token_id is None:
            if (
                hasattr(
                    self.tokenizer,
                    "pad_token_id",
                )
                and self.tokenizer.pad_token_id is not None
            ):
                self.pad_token_id = self.tokenizer.pad_token_id
            else:
                self.pad_token_id = 0

    def __call__(
        self,
        examples: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        batch_size = len(examples)
        if isinstance(
            examples[0],
            dict,
        ):
            input_ids = [example["input_ids"] for example in examples]
        else:
            input_ids = examples
        max_length = max(len(ids) for ids in input_ids)
        if self.pad_to_multiple_of is not None:
            max_length = (max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of

        padded_input_ids = torch.full(
            (batch_size, max_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,
        )
        for i, ids in enumerate(input_ids):
            if not isinstance(
                ids,
                torch.Tensor,
            ):
                ids = torch.tensor(
                    ids,
                    dtype=torch.long,
                )
            length = len(ids)
            padded_input_ids[i, :length] = ids.detach().clone()
            attention_mask[i, :length] = 1

        labels = padded_input_ids.detach().clone()
        labels = torch.roll(
            labels,
            shifts=-1,
            dims=1,
        )
        labels[:, -1] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
