"""
Shared helpers for dataset build scripts: hashing, manifest, progress pool.
"""

import json
import hashlib
import logging
from pathlib import Path
from threading import Thread
from typing import Any, Callable
from datetime import datetime, timezone
from multiprocessing import Pool, Manager

from tqdm import tqdm

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"
PROGRESS_BATCH = 50


def config_hash(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def get_output_dir(
    base_dir: str | Path,
    config_hash_str: str,
) -> Path:
    return Path(base_dir) / config_hash_str


def read_manifest(output_dir: Path) -> dict | None:
    path = output_dir / MANIFEST_FILENAME
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_manifest(
    output_dir: Path,
    config_hash_str: str,
    config: dict,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config_hash": config_hash_str,
        "config": config,
        "built_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    with open(output_dir / MANIFEST_FILENAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def update_latest_symlink(
    base_dir: str | Path,
    config_hash_str: str,
) -> None:
    base = Path(base_dir).resolve()
    latest = base / "latest"
    if latest.exists():
        if latest.is_dir() and not latest.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite directory {latest}. " "Remove or rename it so 'latest' can be a symlink.",
            )
        latest.unlink()
    try:
        latest.symlink_to(config_hash_str, target_is_directory=True)
    except FileExistsError:
        latest.unlink()
        latest.symlink_to(config_hash_str, target_is_directory=True)


def should_skip_build(
    output_dir: Path,
    current_hash: str,
) -> bool:
    if not output_dir.exists():
        return False
    manifest = read_manifest(output_dir)
    if manifest is None:
        return False
    return manifest.get("config_hash") == current_hash


def source_config_id(aggregated_dataset_path: str) -> str:
    path = Path(aggregated_dataset_path).resolve()
    if path.is_dir() and (path / "train.jsonl").exists():
        name = path.name
        if len(name) == 16 and all(c in "0123456789abcdef" for c in name.lower()):
            return name
        return config_hash({"source_path": str(path)})
    return config_hash({"source_path": str(path)})


def load_aggregated_split(
    aggregated_dataset_path: str,
    split: str,
) -> Any:
    from datasets import load_dataset

    path = Path(aggregated_dataset_path).resolve()
    if path.is_dir() and (path / "train.jsonl").exists():
        data_files = {
            "train": str(path / "train.jsonl"),
        }
        if (path / "validation.jsonl").exists():
            data_files["validation"] = str(path / "validation.jsonl")
        if (path / "test.jsonl").exists():
            data_files["test"] = str(path / "test.jsonl")
        ds_dict = load_dataset(
            "json",
            data_files=data_files,
        )
        return ds_dict.get(split)
    return load_dataset(
        aggregated_dataset_path,
        split=split,
    )


def run_pool_with_progress(
    worker_fn: Callable[..., Any],
    worker_args_without_queue: list[tuple],
    n_shards: int,
    total: int | None,
    split_name: str,
    unit: str = "ex",
) -> list[Any]:
    with Manager() as manager:
        progress_queue = manager.Queue()
        worker_args = [
            (
                *args,
                progress_queue,
            )
            for args in worker_args_without_queue
        ]

        def consume() -> None:
            bar = tqdm(total=total, desc=split_name, unit=unit)
            done = 0
            while done < n_shards:
                x = progress_queue.get()
                if x is None:
                    done += 1
                else:
                    bar.update(x)
            bar.close()

        consumer = Thread(target=consume)
        consumer.start()
        pool = Pool(
            processes=n_shards,
        )
        try:
            results = list(pool.imap(worker_fn, worker_args))
        finally:
            pool.close()
            pool.join()
        consumer.join()
    return results


def merge_shard_files(
    out: Path,
    split: str,
    results: list[tuple[str, str, int]],
    sort_key: Callable[[tuple], str],
) -> int:
    total = 0
    path = out / f"{split}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for _split, shard_path, count in sorted(results, key=sort_key):
            if not shard_path:
                continue
            with open(shard_path, "r", encoding="utf-8") as sf:
                f.write(sf.read())
            total += count
            Path(shard_path).unlink()
    return total
