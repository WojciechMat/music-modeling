"""
Build tokenized MIDI dataset as JSONL from aggregated MidiDataset.
Output: <output_dir>/<config_hash>/. Config includes source id and tokenizer params.

Run from project root: python -m scripts.build_midi_tokenized_jsonl ...
"""

import json
import logging
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from midi_tokenizers import ExponentialTimeTokenizer

from scripts.build_utils import (
    PROGRESS_BATCH,
    config_hash,
    get_output_dir,
    write_manifest,
    source_config_id,
    merge_shard_files,
    should_skip_build,
    load_aggregated_split,
    update_latest_symlink,
    run_pool_with_progress,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_notes(notes_str: str) -> pd.DataFrame:
    try:
        data = json.loads(notes_str)
        if isinstance(data, dict) and all(k in data for k in ["pitch", "start"]):
            return pd.DataFrame(data)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def tokenize_and_chunk(
    notes_str: str,
    tokenizer: ExponentialTimeTokenizer,
    context_length: int,
    sliding_window_stride: int,
) -> list[list[int]]:
    notes_df = parse_notes(notes_str)
    if notes_df.empty:
        return []
    required = ["pitch", "start", "end"]
    if not all(c in notes_df.columns for c in required):
        return []
    cols = [c for c in ["pitch", "velocity", "start", "end"] if c in notes_df.columns]
    notes_df = notes_df[cols]
    tokens = tokenizer.encode_notes_df(notes_df)
    if not tokens or len(tokens) < context_length:
        return []
    chunks = []
    for i in range(
        0,
        len(tokens) - context_length + 1,
        sliding_window_stride,
    ):
        chunk = tokens[i : i + context_length]
        if len(chunk) == context_length:
            chunks.append(chunk)
    return chunks


def _process_tokenized_shard(args: tuple) -> tuple[str, str, int]:
    (
        aggregated_dataset_path,
        split,
        shard_id,
        num_shards,
        output_dir,
        context_length,
        stride,
        tokenizer_kwargs,
        progress_queue,
    ) = args
    tokenizer = ExponentialTimeTokenizer.build_tokenizer(tokenizer_kwargs)
    ds = load_aggregated_split(aggregated_dataset_path, split)
    if ds is None:
        progress_queue.put(None)
        return (split, "", 0)
    shard = ds.shard(num_shards, shard_id)
    out = Path(output_dir)
    shard_path = out / f"{split}_shard_{shard_id:04d}.jsonl"
    count = 0
    batch_count = 0
    with open(shard_path, "w", encoding="utf-8") as f:
        for ex_id, ex in enumerate(shard):
            try:
                notes_str = ex["notes"]
                source_dataset = ex.get("source_dataset", "")
                original_id = ex.get("original_id", str(ex_id))
                chunks = tokenize_and_chunk(
                    notes_str,
                    tokenizer,
                    context_length,
                    stride,
                )
                for chunk in chunks:
                    f.write(
                        json.dumps(
                            {
                                "input_ids": chunk,
                                "source_dataset": source_dataset,
                                "original_id": original_id,
                            },
                        )
                        + "\n",
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
    aggregated_dataset_path: str,
    output_dir: str,
    context_length: int = 1024,
    sliding_window_stride: int | None = None,
    min_time_unit: float = 0.01,
    max_time_step: float = 1.0,
    n_velocity_bins: int = 32,
    n_special_ids: int = 1024,
    splits: tuple[str, ...] = ("train", "validation", "test"),
    num_proc: int = 1,
    force: bool = False,
) -> Path:
    stride = sliding_window_stride or context_length
    tokenizer_kwargs = {
        "time_unit": min_time_unit,
        "max_time_step": max_time_step,
        "n_velocity_bins": n_velocity_bins,
        "n_special_ids": n_special_ids,
    }
    source_id = source_config_id(aggregated_dataset_path)
    config = {
        "source_config_id": source_id,
        "context_length": context_length,
        "sliding_window_stride": stride,
        **tokenizer_kwargs,
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

    for split in splits:
        try:
            ds = load_aggregated_split(aggregated_dataset_path, split)
            if ds is None:
                logger.warning("Skipping split %s: no data", split)
                continue
        except Exception as e:
            logger.warning("Skipping split %s: %s", split, e)
            continue

        n_ex = len(ds)
        logger.info("Processing %s (%d examples)", split, n_ex)
        n_shards = min(num_proc, n_ex) if n_ex else 0
        if n_shards <= 1:
            tokenizer = ExponentialTimeTokenizer.build_tokenizer(tokenizer_kwargs)
            path = out / f"{split}.jsonl"
            count = 0
            with open(path, "w", encoding="utf-8") as f:
                for ex_id, ex in tqdm(
                    enumerate(ds),
                    total=n_ex,
                    desc=split,
                    unit="ex",
                ):
                    try:
                        notes_str = ex["notes"]
                        source_dataset = ex.get("source_dataset", "")
                        original_id = ex.get("original_id", str(ex_id))
                        chunks = tokenize_and_chunk(
                            notes_str,
                            tokenizer,
                            context_length,
                            stride,
                        )
                        for chunk in chunks:
                            f.write(
                                json.dumps(
                                    {
                                        "input_ids": chunk,
                                        "source_dataset": source_dataset,
                                        "original_id": original_id,
                                    },
                                )
                                + "\n",
                            )
                            count += 1
                    except Exception:
                        pass
            logger.info("Wrote %s: %s (%d examples)", split, path, count)
            continue

        worker_args_without_queue = [
            (
                aggregated_dataset_path,
                split,
                i,
                n_shards,
                str(out),
                context_length,
                stride,
                tokenizer_kwargs,
            )
            for i in range(n_shards)
        ]
        results = run_pool_with_progress(
            _process_tokenized_shard,
            worker_args_without_queue,
            n_shards=n_shards,
            total=None,
            split_name=split,
            unit="row",
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
        description="Build tokenized MIDI JSONL from MidiDataset.",
    )
    p.add_argument(
        "aggregated_dataset_path",
        type=str,
        help="Path to MidiDataset (dir with train.jsonl or HuggingFace path)",
    )
    p.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default="./midi_datasets/MidiTokenizedDataset",
        help="Directory to write split JSONL files",
    )
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--sliding-window-stride", type=int, default=None)
    p.add_argument("--min-time-unit", type=float, default=0.01)
    p.add_argument("--max-time-step", type=float, default=1.0)
    p.add_argument("--n-velocity-bins", type=int, default=32)
    p.add_argument("--n-special-ids", type=int, default=1024)
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
        aggregated_dataset_path=args.aggregated_dataset_path,
        output_dir=args.output_dir,
        context_length=args.context_length,
        sliding_window_stride=args.sliding_window_stride,
        min_time_unit=args.min_time_unit,
        max_time_step=args.max_time_step,
        n_velocity_bins=args.n_velocity_bins,
        n_special_ids=args.n_special_ids,
        splits=tuple(args.splits),
        num_proc=args.num_proc,
        force=args.force,
    )
    print("Output path (use as dataset_path in training):", out_path)


if __name__ == "__main__":
    main()
