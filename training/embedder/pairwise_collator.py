"""Collator for pairwise embedder: pad notes_first and notes_second sequences separately."""

from typing import Any, Dict, List

import torch


def pairwise_collate(
    examples: List[Dict[str, Any]],
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """Collate batch of {input_ids_first, input_ids_second} into padded tensors."""
    ids_first = [ex["input_ids_first"] for ex in examples]
    ids_second = [ex["input_ids_second"] for ex in examples]
    max_len_first = max(
        len(
            x,
        )
        for x in ids_first
    )
    max_len_second = max(
        len(
            x,
        )
        for x in ids_second
    )
    batch_size = len(
        examples,
    )
    input_ids_first = torch.full(
        (batch_size, max_len_first),
        pad_token_id,
        dtype=torch.long,
    )
    attention_mask_first = torch.zeros(
        (batch_size, max_len_first),
        dtype=torch.long,
    )
    input_ids_second = torch.full(
        (batch_size, max_len_second),
        pad_token_id,
        dtype=torch.long,
    )
    attention_mask_second = torch.zeros(
        (batch_size, max_len_second),
        dtype=torch.long,
    )
    for i in range(
        batch_size,
    ):
        a = ids_first[i]
        b = ids_second[i]
        input_ids_first[i, : len(a)] = torch.tensor(
            a,
            dtype=torch.long,
        )
        attention_mask_first[i, : len(a)] = 1
        input_ids_second[i, : len(b)] = torch.tensor(
            b,
            dtype=torch.long,
        )
        attention_mask_second[i, : len(b)] = 1
    return {
        "input_ids_first": input_ids_first,
        "attention_mask_first": attention_mask_first,
        "input_ids_second": input_ids_second,
        "attention_mask_second": attention_mask_second,
    }
