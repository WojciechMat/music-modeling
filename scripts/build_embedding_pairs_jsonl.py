"""
Build embedding-pairs dataset as JSONL from aggregated MidiDataset.
Output: <output_dir>/<config_hash>/. Config includes source id and window/step params.

Run from project root: python -m scripts.build_embedding_pairs_jsonl ...
"""

import json
import logging
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils.note_slicer import RollingWindowNoteSlicer
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

WRITE_BUFFER_SIZE = 1000
LOAD_BATCH_SIZE = 500


def dataframe_to_json_string(df: pd.DataFrame) -> str:
    if df.empty:
        return json.dumps({})
    return json.dumps(
        df.to_dict(orient="list"),
    )


def parse_notes(
    notes_raw: str | dict,
) -> pd.DataFrame:
    """Parse notes to DataFrame. Returns empty DataFrame if invalid."""
    try:
        if isinstance(notes_raw, dict):
            data = notes_raw
        else:
            data = json.loads(notes_raw)
        if not data:
            return pd.DataFrame()
        if isinstance(data, dict) and any(isinstance(v, list) for v in data.values()):
            return pd.DataFrame(data)
        if isinstance(data, dict) and all(k in data for k in ["pitch", "start"]):
            return pd.DataFrame(data)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _write_buffered(
    f: object,
    buffer: list[str],
    line: str,
    flush_size: int = WRITE_BUFFER_SIZE,
) -> None:
    buffer.append(line)
    if len(buffer) >= flush_size:
        f.write("".join(buffer))
        buffer.clear()


def _process_embedding_pairs_shard(args: tuple) -> tuple[str, str, int]:
    (
        aggregated_dataset_path,
        split,
        shard_id,
        num_shards,
        output_dir,
        window_size,
        step_size,
        progress_queue,
    ) = args
    slicer = RollingWindowNoteSlicer(
        window_size=window_size,
        step_size=step_size,
    )
    ds = load_aggregated_split(aggregated_dataset_path, split)
    if ds is None:
        progress_queue.put(None)
        return (split, "", 0)
    shard = ds.shard(num_shards, shard_id)
    n_shard = len(shard)
    out = Path(output_dir)
    shard_path = out / f"{split}_shard_{shard_id:04d}.jsonl"
    count = 0
    batch_count = 0
    write_buf: list[str] = []
    with open(shard_path, "w", encoding="utf-8") as f:
        for batch_start in range(0, n_shard, LOAD_BATCH_SIZE):
            batch_end = min(batch_start + LOAD_BATCH_SIZE, n_shard)
            batch = shard.select(range(batch_start, batch_end))
            for ex_id_off, ex in enumerate(batch):
                ex_id = batch_start + ex_id_off
                try:
                    notes_raw = ex.get("notes", ex.get("content", ""))
                    source = ex.get("source", "{}")
                    source_dataset = ex.get("source_dataset", "unknown")
                    original_id = str(ex.get("original_id", ex_id))
                    if isinstance(source, dict):
                        source = json.dumps(source)
                    notes_df = parse_notes(notes_raw) if isinstance(notes_raw, (str, dict)) else pd.DataFrame(notes_raw)
                    if notes_df.empty:
                        continue
                    pairs = slicer.slice_notes(notes_df)
                    for first, second in pairs:
                        row = json.dumps(
                            {
                                "notes_first": dataframe_to_json_string(first),
                                "notes_second": dataframe_to_json_string(second),
                                "source": source,
                                "source_dataset": source_dataset,
                                "original_id": original_id,
                            },
                        )
                        _write_buffered(f, write_buf, row + "\n")
                        count += 1
                        batch_count += 1
                        if batch_count >= PROGRESS_BATCH:
                            progress_queue.put(batch_count)
                            batch_count = 0
                except Exception:
                    pass
        if write_buf:
            f.write("".join(write_buf))
    if batch_count:
        progress_queue.put(batch_count)
    progress_queue.put(None)
    return (split, str(shard_path), count)


def run(
    aggregated_dataset_path: str,
    output_dir: str,
    window_size: int = 64,
    step_size: int | None = None,
    splits: tuple[str, ...] = ("train", "validation", "test"),
    num_proc: int = 1,
    force: bool = False,
) -> Path:
    step = step_size or window_size
    source_id = source_config_id(aggregated_dataset_path)
    config = {
        "source_config_id": source_id,
        "window_size": window_size,
        "step_size": step,
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
            slicer = RollingWindowNoteSlicer(
                window_size=window_size,
                step_size=step,
            )
            path = out / f"{split}.jsonl"
            count = 0
            write_buf: list[str] = []
            with open(path, "w", encoding="utf-8") as f:
                for batch_start in range(0, n_ex, LOAD_BATCH_SIZE):
                    batch_end = min(batch_start + LOAD_BATCH_SIZE, n_ex)
                    batch = ds.select(range(batch_start, batch_end))
                    for ex_id_off, ex in tqdm(
                        enumerate(batch),
                        total=batch_end - batch_start,
                        desc=split,
                        unit="ex",
                        leave=False,
                    ):
                        ex_id = batch_start + ex_id_off
                        try:
                            notes_raw = ex.get("notes", ex.get("content", ""))
                            source = ex.get("source", "{}")
                            source_dataset = ex.get("source_dataset", "unknown")
                            original_id = str(ex.get("original_id", ex_id))
                            if isinstance(source, dict):
                                source = json.dumps(source)
                            notes_df = (
                                parse_notes(notes_raw)
                                if isinstance(notes_raw, (str, dict))
                                else pd.DataFrame(notes_raw)
                            )
                            if notes_df.empty:
                                continue
                            pairs = slicer.slice_notes(notes_df)
                            for first, second in pairs:
                                row = json.dumps(
                                    {
                                        "notes_first": dataframe_to_json_string(first),
                                        "notes_second": dataframe_to_json_string(second),
                                        "source": source,
                                        "source_dataset": source_dataset,
                                        "original_id": original_id,
                                    },
                                )
                                _write_buffered(f, write_buf, row + "\n")
                                count += 1
                        except Exception:
                            pass
                if write_buf:
                    f.write("".join(write_buf))
            logger.info("Wrote %s: %s (%d pairs)", split, path, count)
            continue

        worker_args_without_queue = [
            (
                aggregated_dataset_path,
                split,
                i,
                n_shards,
                str(out),
                window_size,
                step,
            )
            for i in range(n_shards)
        ]
        results = run_pool_with_progress(
            _process_embedding_pairs_shard,
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
            "Wrote %s: %s (%d pairs, %d workers)",
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
        description="Build embedding-pairs JSONL from MidiDataset (rolling window).",
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
        default="./midi_datasets/EmbeddingPairsDataset",
        help="Directory to write split JSONL files",
    )
    p.add_argument("--window-size", type=int, default=64)
    p.add_argument("--step-size", type=int, default=None)
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
        window_size=args.window_size,
        step_size=args.step_size,
        splits=tuple(args.splits),
        num_proc=args.num_proc,
        force=args.force,
    )
    print("Output path:", out_path)


if __name__ == "__main__":
    main()
