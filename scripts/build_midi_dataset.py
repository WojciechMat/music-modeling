"""
Build aggregated MIDI dataset as JSONL from HuggingFace.
Output: <output_dir>/<config_hash>/. Same config skips rebuild.

Run from project root: python -m scripts.build_midi_dataset ...
"""

import json
import logging
import argparse
from pathlib import Path

from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

from scripts.build_utils import (
    PROGRESS_BATCH,
    config_hash,
    get_output_dir,
    write_manifest,
    merge_shard_files,
    should_skip_build,
    update_latest_symlink,
    run_pool_with_progress,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _record_from_example(
    ex: dict,
    ex_id: int,
    source_dataset: str,
) -> dict | None:
    notes_data = ex.get("notes", ex.get("content", None))
    if notes_data is None:
        for key in ["notes", "content", "midi", "data"]:
            if key in ex:
                notes_data = ex[key]
                break
    if notes_data is None:
        return None
    source_data = ex.get("source", "{}")
    if isinstance(notes_data, dict):
        notes_data = json.dumps(notes_data)
    if isinstance(source_data, dict):
        source_data = json.dumps(source_data)
    return {
        "notes": notes_data,
        "source": source_data,
        "source_dataset": source_dataset,
        "original_id": str(ex.get("original_id", ex_id)),
    }


def _process_midi_shard(args: tuple) -> tuple[str, str, int]:
    split, shard_id, num_shards, output_dir, paths, progress_queue = args
    if split == "train" and isinstance(paths, list):
        ds = concatenate_datasets(
            [
                load_dataset(
                    p,
                    split="train",
                )
                for p in paths
            ],
        )
        source_dataset = paths[0]
    else:
        path = paths if isinstance(paths, str) else paths[0]
        ds = load_dataset(path, split=split)
        source_dataset = path
    shard = ds.shard(num_shards, shard_id)
    out = Path(output_dir)
    shard_path = out / f"{split}_shard_{shard_id:04d}.jsonl"
    count = 0
    batch_count = 0
    with open(shard_path, "w", encoding="utf-8") as f:
        for ex_id, ex in enumerate(shard):
            rec = _record_from_example(ex, ex_id, source_dataset)
            if rec is None:
                continue
            try:
                f.write(
                    json.dumps(rec) + "\n",
                )
                count += 1
                batch_count += 1
                if batch_count >= PROGRESS_BATCH:
                    progress_queue.put(batch_count)
                    batch_count = 0
            except Exception:
                pass
    if batch_count:
        progress_queue.put(batch_count)
    progress_queue.put(None)
    return (split, str(shard_path), count)


def run(
    train_dataset_paths: list[str],
    validation_dataset_path: str,
    output_dir: str,
    splits: tuple[str, ...] = ("train", "validation", "test"),
    num_proc: int = 1,
    force: bool = False,
) -> Path:
    config = {
        "train_dataset_paths": sorted(train_dataset_paths),
        "validation_dataset_path": validation_dataset_path,
        "splits": sorted(splits),
    }
    h = config_hash(config)
    out = get_output_dir(output_dir, h)
    if not force and should_skip_build(out, h):
        logger.info(
            "Config hash %s already built at %s; skipping (use --force to rebuild).",
            h,
            out,
        )
        return out
    out.mkdir(parents=True, exist_ok=True)

    train_datasets = []
    for path in train_dataset_paths:
        try:
            ds = load_dataset(path, split="train")
            logger.info("Loaded train from %s: %d examples", path, len(ds))
            train_datasets.append(ds)
        except Exception as e:
            logger.error("Failed to load %s: %s", path, e)
    if not train_datasets:
        raise ValueError("No valid training datasets found")
    train_dataset = concatenate_datasets(train_datasets)
    logger.info("Train total: %d examples", len(train_dataset))

    val_ds = None
    test_ds = None
    try:
        val_ds = load_dataset(validation_dataset_path, split="validation")
        logger.info(
            "Loaded validation from %s: %d examples",
            validation_dataset_path,
            len(val_ds),
        )
    except Exception as e:
        logger.warning("No validation split: %s", e)
    try:
        test_ds = load_dataset(validation_dataset_path, split="test")
        logger.info(
            "Loaded test from %s: %d examples",
            validation_dataset_path,
            len(test_ds),
        )
    except Exception:
        test_ds = val_ds

    split_list = [
        ("train", train_dataset, train_dataset_paths),
        ("validation", val_ds, validation_dataset_path),
        ("test", test_ds, validation_dataset_path),
    ]
    for split, ds, paths in tqdm(split_list, desc="Splits", unit="split"):
        if split not in splits or ds is None:
            continue
        n = len(ds)
        logger.info("Processing %s (%d examples)", split, n)
        n_shards = min(num_proc, n) if n else 0
        if n_shards <= 1:
            path = out / f"{split}.jsonl"
            source_dataset = paths[0] if isinstance(paths, list) else paths
            count = 0
            with open(path, "w", encoding="utf-8") as f:
                for ex_id, ex in tqdm(
                    enumerate(ds),
                    total=n,
                    desc=split,
                    unit="ex",
                ):
                    try:
                        notes_data = ex.get("notes", ex.get("content", None))
                        if notes_data is None:
                            for key in ["notes", "content", "midi", "data"]:
                                if key in ex:
                                    notes_data = ex[key]
                                    break
                        if notes_data is None:
                            continue
                        source_data = ex.get("source", "{}")
                        if isinstance(notes_data, dict):
                            notes_data = json.dumps(notes_data)
                        if isinstance(source_data, dict):
                            source_data = json.dumps(source_data)
                        f.write(
                            json.dumps(
                                {
                                    "notes": notes_data,
                                    "source": source_data,
                                    "source_dataset": source_dataset,
                                    "original_id": str(ex.get("original_id", ex_id)),
                                },
                            )
                            + "\n",
                        )
                        count += 1
                    except Exception:
                        pass
            logger.info("Wrote %s: %s (%d examples)", split, path, count)
            continue

        worker_args_without_queue = [(split, i, n_shards, str(out), paths) for i in range(n_shards)]
        results = run_pool_with_progress(
            _process_midi_shard,
            worker_args_without_queue,
            n_shards=n_shards,
            total=n,
            split_name=split,
            unit="ex",
        )
        total = merge_shard_files(
            out,
            split,
            results,
            sort_key=lambda r: r[1],
        )
        logger.info(
            "Wrote %s: %s (%d examples, %d workers)",
            split,
            out / f"{split}.jsonl",
            total,
            n_shards,
        )

    write_manifest(out, h, config)
    update_latest_symlink(output_dir, h)
    logger.info("Dataset written to %s (config_hash=%s); latest -> %s", out, h, h)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build aggregated MIDI JSONL from HuggingFace datasets.",
    )
    p.add_argument(
        "--train-paths",
        nargs="+",
        default=["epr-labs/maestro-sustain-v2"],
        help="HuggingFace dataset path(s) for train",
    )
    p.add_argument(
        "--validation-path",
        type=str,
        default="epr-labs/maestro-sustain-v2",
        help="HuggingFace dataset path for validation and test",
    )
    p.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default="./midi_datasets/MidiDataset",
        help="Directory to write split JSONL files",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Splits to export",
    )
    p.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="Number of worker processes per split",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if config hash already exists",
    )
    args = p.parse_args()
    out_path = run(
        train_dataset_paths=args.train_paths,
        validation_dataset_path=args.validation_path,
        output_dir=args.output_dir,
        splits=tuple(args.splits),
        num_proc=args.num_proc,
        force=args.force,
    )
    print("Output path (use as aggregated_dataset_path for derivatives):", out_path)


if __name__ == "__main__":
    main()
