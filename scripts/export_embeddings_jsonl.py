"""
Export embeddings for test split of embedding-pairs dataset.
Uses ECHO (repeat notes_first segment echo_repetitions times, forward, pool last repetition).
Writes tmp/data/<dataset_hash>/<model_name>/split_with_embeddings.jsonl.

Run from project root: python -m scripts.export_embeddings_jsonl <dataset_path> <model_path>
Requires embedding_config.json in the model dir (written by embedder training).
"""

import json
import time
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from midi_tokenizers import ExponentialTimeTokenizer

from training.embedder.echo_dataset import notes_to_token_ids, build_echo_input_ids


def load_tokenizer_from_pretrained(
    pretrained_path: Path,
) -> ExponentialTimeTokenizer:
    tokenizer_path = pretrained_path / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"tokenizer.json missing at {pretrained_path}",
        )
    with open(
        tokenizer_path,
        "r",
    ) as f:
        data = json.load(
            f,
        )
    if "parameters" in data and "tokenizer_config" in data["parameters"]:
        tokenizer_config = data["parameters"]["tokenizer_config"]
    else:
        tokenizer_config = data
    return ExponentialTimeTokenizer.build_tokenizer(
        tokenizer_config,
    )


def build_echo_ids(
    notes_str: str,
    tokenizer: ExponentialTimeTokenizer,
    max_seq_length: int,
    echo_repetitions: int,
) -> list[int]:
    """Tokenize notes_first and build ECHO input (segment repeated echo_repetitions times)."""
    segment_ids = notes_to_token_ids(notes_str, tokenizer)
    if not segment_ids:
        return []
    echoed, _ = build_echo_input_ids(segment_ids, echo_repetitions, max_seq_length)
    return echoed


def mean_pool_hidden(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """(batch, seq, hidden) -> (batch, hidden) using attention_mask."""
    mask = attention_mask.unsqueeze(-1).float()
    sum_hidden = (last_hidden_state * mask).sum(dim=1)
    sum_mask = mask.sum(dim=1).clamp(min=1e-9)
    return sum_hidden / sum_mask


def run(
    dataset_path: str,
    model_path: str,
    split: str = "test",
    output_root: str = "tmp/data",
    max_seq_length: int = 512,
    batch_size: int = 32,
    device: str | None = None,
) -> Path:
    dataset_path = Path(
        dataset_path,
    ).resolve()
    model_path = Path(
        model_path,
    ).resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Dataset path not found: {dataset_path}",
        )
    test_file = dataset_path / f"{split}.jsonl"
    if not test_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {test_file}",
        )
    if not model_path.is_dir() or not (model_path / "config.json").exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}",
        )

    embedding_config_path = model_path / "embedding_config.json"
    if not embedding_config_path.exists():
        raise FileNotFoundError(
            f"embedding_config.json not found inside {model_path}. Train embedder first (it writes this file).",
        )
    with open(embedding_config_path) as f:
        emb_cfg = json.load(f)
    if "echo_repetitions" not in emb_cfg:
        raise ValueError(
            f"embedding_config.json must contain 'echo_repetitions'. Got keys: {list(emb_cfg)}",
        )
    echo_repetitions = int(emb_cfg["echo_repetitions"])

    dataset_hash = dataset_path.name
    model_name = model_path.name
    out_dir = (
        Path(
            output_root,
        )
        / dataset_hash
        / model_name
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    out_path = out_dir / "split_with_embeddings.jsonl"

    tokenizer = load_tokenizer_from_pretrained(
        model_path,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(
        device,
    )
    model.eval()

    ds = load_dataset(
        "json",
        data_files={split: str(test_file)},
        split=split,
    )
    num_examples = len(ds)
    num_batches = (num_examples + batch_size - 1) // batch_size
    print(
        f"Exporting embeddings: split={split}, examples={num_examples}, echo_repetitions={echo_repetitions}, "
        f"batch_size={batch_size}, batches={num_batches}, model={model_path.name}, out={out_path}",
    )
    pad_token_id = (
        getattr(
            tokenizer,
            "pad_token_id",
            None,
        )
        or 0
    )

    def get_input_ids(example: dict) -> list[int]:
        return build_echo_ids(
            example.get("notes_first", ""),
            tokenizer,
            max_seq_length,
            echo_repetitions,
        )

    all_rows = []
    start_time = time.perf_counter()

    for i in tqdm(
        range(0, len(ds), batch_size),
        total=num_batches,
        unit="batch",
        desc="Embedding",
    ):
        batch_examples = [
            ds[j]
            for j in range(
                i,
                min(
                    i + batch_size,
                    len(ds),
                ),
            )
        ]
        input_ids_list = [
            get_input_ids(
                ex,
            )
            for ex in batch_examples
        ]
        valid = [
            idx
            for idx, ids in enumerate(
                input_ids_list,
            )
            if ids
        ]
        if not valid:
            for ex in batch_examples:
                all_rows.append(
                    {
                        "id": len(all_rows),
                        "notes_first": ex.get(
                            "notes_first",
                            "",
                        ),
                        "notes_second": ex.get(
                            "notes_second",
                            "",
                        ),
                        "source": ex.get(
                            "source",
                            "",
                        ),
                        "source_dataset": ex.get(
                            "source_dataset",
                            "",
                        ),
                        "original_id": ex.get(
                            "original_id",
                            "",
                        ),
                        "embedding": [],
                    },
                )
            continue

        max_len = max(
            len(
                input_ids_list[k],
            )
            for k in valid
        )
        batch_input_ids = []
        batch_attention = []
        for idx in range(
            len(
                batch_examples,
            ),
        ):
            ids = input_ids_list[idx]
            if not ids:
                batch_input_ids.append(
                    [pad_token_id],
                )
                batch_attention.append(
                    [0],
                )
            else:
                padding = [pad_token_id] * (
                    max_len
                    - len(
                        ids,
                    )
                )
                batch_input_ids.append(
                    ids + padding,
                )
                batch_attention.append(
                    [1]
                    * len(
                        ids,
                    )
                    + [0]
                    * len(
                        padding,
                    ),
                )

        input_ids_t = torch.tensor(
            batch_input_ids,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.tensor(
            batch_attention,
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            out = model(
                input_ids=input_ids_t,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        last_hidden = out.hidden_states[-1]
        seq_len = input_ids_t.size(1)
        segment_len = seq_len // echo_repetitions
        if segment_len == 0:
            segment_len = seq_len
        last_rep_hidden = last_hidden[:, -segment_len:, :]
        last_rep_mask = attention_mask[:, -segment_len:]
        pooled = mean_pool_hidden(last_rep_hidden, last_rep_mask)
        for idx in range(len(batch_examples)):
            ex = batch_examples[idx]
            row = {
                "id": len(all_rows),
                "notes_first": ex.get("notes_first", ""),
                "notes_second": ex.get("notes_second", ""),
                "source": ex.get("source", ""),
                "source_dataset": ex.get("source_dataset", ""),
                "original_id": ex.get("original_id", ""),
                "embedding": None,
            }
            if input_ids_list[idx]:
                emb_f = pooled[idx].float().cpu().tolist()
                row["embedding"] = emb_f
            else:
                row["embedding"] = []
            all_rows.append(row)

    with open(
        out_path,
        "w",
    ) as f:
        for row in all_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n",
            )

    elapsed = time.perf_counter() - start_time
    print(
        f"Wrote {len(all_rows)} rows to {out_path} in {elapsed:.1f}s ({num_examples / elapsed:.0f} ex/s)",
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export embeddings for test split to split_with_embeddings.jsonl",
    )
    parser.add_argument(
        "dataset_path",
        help="Directory with train.jsonl, test.jsonl (e.g. midi_datasets/EmbeddingPairsDataset/latest)",
    )
    parser.add_argument(
        "model_path",
        help="Embedder checkpoint dir (e.g. checkpoints/note-embedder/embedder-echo-.../final)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split to export (default: test)",
    )
    parser.add_argument(
        "--output-root",
        default="tmp/data",
        help="Root for output: <output_root>/<dataset_hash>/<model_name>/split_with_embeddings.jsonl",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Max sequence length for ECHO input (repeated segment truncated to this)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device (cuda/cpu). Default: auto",
    )
    args = parser.parse_args()
    run(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        split=args.split,
        output_root=args.output_root,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
