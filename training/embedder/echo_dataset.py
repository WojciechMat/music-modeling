"""Pairwise embedder: notes_first/notes_second -> tokenize -> ECHO (repeat n times)
-> input_ids_first, input_ids_second."""

import json
from typing import List
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from midi_tokenizers import ExponentialTimeTokenizer


def parse_notes(
    notes_str: str,
) -> pd.DataFrame:
    """Parse notes JSON string to DataFrame. Returns empty DataFrame if invalid."""
    try:
        data = json.loads(
            notes_str,
        )
        if not data:
            return pd.DataFrame()
        if isinstance(
            data,
            dict,
        ) and any(
            isinstance(
                v,
                list,
            )
            for v in data.values()
        ):
            return pd.DataFrame(
                data,
            )
        if isinstance(
            data,
            dict,
        ) and all(k in data for k in ["pitch", "start"]):
            return pd.DataFrame(
                data,
            )
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def notes_to_token_ids(
    notes_str: str,
    tokenizer: ExponentialTimeTokenizer,
) -> List[int]:
    """Tokenize notes_first/notes_second string to list of token ids. Returns [] if invalid."""
    notes_df = parse_notes(
        notes_str,
    )
    if notes_df.empty:
        return []
    required = ["pitch", "start", "end"]
    if not all(c in notes_df.columns for c in required):
        return []
    cols = [c for c in ["pitch", "velocity", "start", "end"] if c in notes_df.columns]
    notes_df = notes_df[cols]
    tokens = tokenizer.encode_notes_df(
        notes_df,
    )
    if not tokens:
        return []
    return list(
        tokens,
    )


def build_echo_input_ids(
    segment_ids: List[int],
    echo_repetitions: int,
    max_seq_length: int,
) -> tuple[List[int], bool]:
    """Repeat segment echo_repetitions times and truncate to max_seq_length.
    Returns (input_ids, was_truncated). was_truncated is True if segment or echoed sequence was cut.
    """
    if not segment_ids or echo_repetitions < 1:
        return [], False
    segment_len = len(segment_ids)
    max_segment = max(1, max_seq_length // echo_repetitions)
    truncated = False
    if segment_len > max_segment:
        segment_ids = segment_ids[:max_segment]
        truncated = True
    echoed = segment_ids * echo_repetitions
    if len(echoed) > max_seq_length:
        echoed = echoed[:max_seq_length]
        truncated = True
    return echoed, truncated


def load_embedding_pairs_dataset(
    dataset_path: str,
    train_split: str,
    eval_split: str,
) -> tuple[Dataset, Dataset]:
    """Load train and eval splits from JSONL directory."""
    path = Path(
        dataset_path,
    )
    if not path.is_dir() or not (path / "train.jsonl").exists():
        raise FileNotFoundError(
            f"No train.jsonl in {dataset_path}. Run scripts/build_embedding_pairs_jsonl first.",
        )
    data_files = {
        "train": str(
            path / "train.jsonl",
        ),
    }
    if (path / "validation.jsonl").exists():
        data_files["validation"] = str(path / "validation.jsonl")
    if (path / "test.jsonl").exists():
        data_files["test"] = str(path / "test.jsonl")
    ds_dict = load_dataset(
        "json",
        data_files=data_files,
    )
    train_ds = ds_dict[train_split]
    eval_ds = ds_dict[eval_split]
    return train_ds, eval_ds


def pairwise_dataset_from_hf(
    hf_dataset: Dataset,
    tokenizer: ExponentialTimeTokenizer,
    max_seq_length: int,
    echo_repetitions: int = 1,
    num_proc: int = 8,
) -> tuple[Dataset, dict]:
    """Map HF embedding-pairs to (input_ids_first, input_ids_second) with ECHO applied to both.
    Each segment is repeated echo_repetitions times and truncated to max_seq_length.
    Returns (dataset, truncation_stats) with keys: num_truncated_first, num_truncated_second, total.
    """

    def map_fn(example: dict) -> dict:
        ids_first = notes_to_token_ids(
            example.get("notes_first", ""),
            tokenizer,
        )
        ids_second = notes_to_token_ids(
            example.get("notes_second", ""),
            tokenizer,
        )
        if not ids_first or not ids_second:
            return {
                "input_ids_first": [],
                "input_ids_second": [],
                "_valid": False,
                "_trunc_first": False,
                "_trunc_second": False,
            }
        input_ids_first, trunc_first = build_echo_input_ids(
            ids_first,
            echo_repetitions,
            max_seq_length,
        )
        input_ids_second, trunc_second = build_echo_input_ids(
            ids_second,
            echo_repetitions,
            max_seq_length,
        )
        if not input_ids_first or not input_ids_second:
            return {
                "input_ids_first": [],
                "input_ids_second": [],
                "_valid": False,
                "_trunc_first": False,
                "_trunc_second": False,
            }
        return {
            "input_ids_first": input_ids_first,
            "input_ids_second": input_ids_second,
            "_valid": True,
            "_trunc_first": trunc_first,
            "_trunc_second": trunc_second,
        }

    out = hf_dataset.map(
        map_fn,
        remove_columns=hf_dataset.column_names,
        num_proc=num_proc,
        desc="pairwise_dataset",
    )
    out = out.filter(lambda x: x["_valid"], num_proc=num_proc)
    num_trunc_first = sum(1 for i in range(len(out)) if out[i].get("_trunc_first"))
    num_trunc_second = sum(1 for i in range(len(out)) if out[i].get("_trunc_second"))
    out = out.remove_columns(["_valid", "_trunc_first", "_trunc_second"])
    truncation_stats = {
        "num_truncated_first": num_trunc_first,
        "num_truncated_second": num_trunc_second,
        "total": len(out),
    }
    return out, truncation_stats
