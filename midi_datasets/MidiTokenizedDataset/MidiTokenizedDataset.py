"""
Tokenized MIDI dataset: load from JSONL produced by scripts/build_midi_tokenized_jsonl.py.

Schema: input_ids (list[int]), source_dataset (str), original_id (str).
"""

from typing import Dict
from pathlib import Path

from datasets import Dataset, load_dataset


def load_from_directory(path: str) -> Dict[str, Dataset]:
    """Load train/validation/test from JSONL files in the given directory.

    Expects train.jsonl, and optionally validation.jsonl, test.jsonl.
    """
    path = Path(path)
    data_files = {}
    if (path / "train.jsonl").exists():
        data_files["train"] = str(path / "train.jsonl")
    if (path / "validation.jsonl").exists():
        data_files["validation"] = str(path / "validation.jsonl")
    if (path / "test.jsonl").exists():
        data_files["test"] = str(path / "test.jsonl")
    if not data_files:
        raise FileNotFoundError(f"No JSONL files found in {path}. Run scripts/build_midi_tokenized_jsonl.py first.")
    return load_dataset("json", data_files=data_files)
