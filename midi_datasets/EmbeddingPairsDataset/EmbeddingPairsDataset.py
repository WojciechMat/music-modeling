"""
Embedding-pairs dataset: load from JSONL produced by scripts/build_embedding_pairs_jsonl.py.

Schema: notes_first (str), notes_second (str), source (str), source_dataset (str), original_id (str).
"""

from pathlib import Path
from typing import Dict, Optional

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
        raise FileNotFoundError(f"No JSONL files found in {path}. Run scripts/build_embedding_pairs_jsonl.py first.")
    return load_dataset("json", data_files=data_files)


def load_embedding_pairs_from_jsonl(
    path: str,
    validation_path: Optional[str] = None,
    test_path: Optional[str] = None,
) -> Dict[str, Dataset]:
    """Load from explicit JSONL file path(s). path = train; optional validation_path, test_path."""
    data_files = {"train": path}
    if validation_path:
        data_files["validation"] = validation_path
    if test_path:
        data_files["test"] = test_path
    return load_dataset("json", data_files=data_files)
