from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class MidiDataCollatorForCausalLM:
    """
    Data collator for MIDI sequence causal language modeling.
    This works with the ExponentialTimeTokenizer and handles padding and label creation.
    """

    tokenizer: Any
    mlm: bool = False
    mlm_probability: float = 0.15
    pad_token_id: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.mlm:
            raise ValueError(
                "Masked language modeling (MLM) is not supported for causal language modeling. " "Please set mlm=False."
            )

        # Set pad_token_id if not provided
        if self.pad_token_id is None:
            if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
                self.pad_token_id = self.tokenizer.pad_token_id
            else:
                # Default to 0 if no pad token is available
                self.pad_token_id = 0
                print("Warning: No pad_token_id found, using 0 as pad_token_id")

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples for causal language modeling.

        Args:
            examples: List of examples with "input_ids" as a key

        Returns:
            Batch with padded input_ids, attention_mask, and labels
        """
        # Extract input_ids from examples
        if isinstance(examples[0], dict):
            input_ids = [example["input_ids"] for example in examples]
        else:
            input_ids = examples

        # Compute max length for padding
        batch_size = len(input_ids)
        max_length = max(len(ids) for ids in input_ids)

        # If pad_to_multiple_of is set, round up max_length
        if self.pad_to_multiple_of is not None:
            max_length = (max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of

        # Create padded tensors
        padded_input_ids = torch.full((batch_size, max_length), self.pad_token_id, dtype=torch.long)

        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)

        # Fill in the tensors with actual values
        for i, ids in enumerate(input_ids):
            length = len(ids)
            padded_input_ids[i, :length] = ids.detach().clone()
            attention_mask[i, :length] = 1

        # For causal LM, labels are the same as inputs but shifted to the left by one
        labels = padded_input_ids.detach().clone()
        labels = torch.roll(labels, shifts=-1, dims=1)

        # Set the last position (which has no valid prediction target) to -100
        # This is the id that is by default ignored when calculating loss
        labels[:, -1] = -100
        # Mask padding tokens in labels to avoid loss calculation on padding
        mask = attention_mask == 0
        labels[mask] = -100

        # Create the batch
        batch = {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        return batch
