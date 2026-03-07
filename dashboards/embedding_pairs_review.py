"""Streamlit dashboard to review embedding-pairs dataset (notes_first / notes_second)."""

import json
import random
from pathlib import Path

import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll

BROWSE_SEED = 4
SAMPLES_PER_PAGE = 6

# Load project dataset module by path to avoid conflict with HuggingFace "datasets" package
project_root = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, project_root / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_embedding_pairs = _load_module(
    "embedding_pairs_dataset", "midi_datasets/EmbeddingPairsDataset/EmbeddingPairsDataset.py"
)
load_embedding_pairs_from_directory = _embedding_pairs.load_from_directory

st.set_page_config(page_title="Embedding Pairs Dataset Review", layout="wide")


def parse_notes(notes_str: str) -> pd.DataFrame:
    """Parse notes from JSON string to DataFrame."""
    try:
        data = json.loads(notes_str)
        if not data:
            return pd.DataFrame()
        if isinstance(data, dict) and any(isinstance(v, list) for v in data.values()):
            return pd.DataFrame(data)
        if isinstance(data, dict) and all(k in data for k in ["pitch", "start"]):
            return pd.DataFrame(data)
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Error parsing notes: {e}")
        return pd.DataFrame()


@st.cache_data
def load_pairs_from_directory(dataset_path: str, split: str, num_examples: int = 1000):
    """Load embedding-pairs dataset from directory (train.jsonl, etc.). Resolves 'latest' symlink."""
    try:
        path = Path(dataset_path).resolve()
        if not path.is_dir():
            st.error(f"Not a directory: {path}")
            return None
        if not (path / "train.jsonl").exists():
            st.error(
                f"No train.jsonl in {path}. Use midi_datasets/EmbeddingPairsDataset/latest or a config-hash subdir."
            )
            return None
        ds_dict = load_embedding_pairs_from_directory(str(path))
        dataset = ds_dict.get(split, ds_dict["train"])
        if len(dataset) > num_examples:
            dataset = dataset.select(range(num_examples))
        return dataset
    except FileNotFoundError as e:
        st.error(str(e))
        return None
    except Exception as e:
        st.error(f"Error loading dataset from {dataset_path}: {e}")
        return None


@st.cache_data
def load_pairs_from_jsonl(jsonl_path: str, num_examples: int = 1000):
    """Load embedding-pairs dataset from a single .jsonl file (as 'train' split)."""
    try:
        from datasets import load_dataset

        path = Path(jsonl_path)
        if not path.exists():
            st.error(f"File not found: {path}")
            return None
        dataset = load_dataset("json", data_files={"train": str(path)}, split="train")
        if len(dataset) > num_examples:
            dataset = dataset.select(range(num_examples))
        return dataset
    except Exception as e:
        st.error(f"Error loading JSONL from {jsonl_path}: {e}")
        return None


def _browse_indices(
    dataset_len: int,
    n: int,
    seed: int,
    shuffle_key: int,
) -> list[int]:
    """Return n random indices from [0, dataset_len) using seed + shuffle_key."""
    rng = random.Random(seed + shuffle_key)
    pool = list(range(dataset_len))
    if len(pool) <= n:
        return pool
    return rng.sample(pool, n)


def render_example(
    example: dict,
    index: int,
) -> None:
    """Render one sample: one-line caption then notes_first | notes_second pianorolls, then concatenated."""
    notes_first_str = example.get("notes_first", "")
    notes_second_str = example.get("notes_second", "")
    df_first = parse_notes(notes_first_str)
    df_second = parse_notes(notes_second_str)

    if df_first.empty and df_second.empty:
        st.warning("Both notes_first and notes_second are empty or invalid.")
        return

    st.caption(
        f"Index {index} · source: {example.get('source_dataset', '—')} "
        f"· original_id: {example.get('original_id', '—')}",
    )
    col_first, col_second = st.columns(2)
    with col_first:
        st.write("**notes_first**")
        if not df_first.empty:
            piece_first = ff.MidiPiece(df_first)
            streamlit_pianoroll.from_fortepyan(piece=piece_first)
    with col_second:
        st.write("**notes_second**")
        if not df_second.empty:
            piece_second = ff.MidiPiece(df_second)
            streamlit_pianoroll.from_fortepyan(piece=piece_second)

    if not df_first.empty and not df_second.empty:
        st.write("**Concatenated (first + second)**")
        t_end_first = df_first["end"].max() if "end" in df_first.columns else df_first["start"].max()
        df_second_shifted = df_second.copy()
        if "start" in df_second_shifted.columns:
            df_second_shifted["start"] = df_second_shifted["start"] + t_end_first
        if "end" in df_second_shifted.columns:
            df_second_shifted["end"] = df_second_shifted["end"] + t_end_first
        combined = pd.concat([df_first, df_second_shifted], ignore_index=True)
        piece_combined = ff.MidiPiece(combined)
        streamlit_pianoroll.from_fortepyan(piece=piece_combined)

    with st.expander("Details"):
        st.write("**Source dataset:**", example.get("source_dataset", "—"))
        st.write("**Original ID:**", example.get("original_id", "—"))
        if example.get("source"):
            try:
                src = json.loads(example["source"]) if isinstance(example["source"], str) else example["source"]
                st.json(src)
            except Exception:
                st.text(example["source"][:500])
        st.write("**notes_first**")
        st.dataframe(df_first, use_container_width=True)
        st.write("**notes_second**")
        st.dataframe(df_second, use_container_width=True)


def main():
    st.title("Embedding Pairs Dataset Review")
    st.caption("Inspect (notes_first, notes_second) records with source metadata.")

    st.sidebar.header("Data source")
    source_type = st.sidebar.radio(
        "Source",
        ["Dataset directory (JSONL)", "Single JSONL file"],
        index=0,
    )

    if source_type == "Dataset directory (JSONL)":
        dataset_path = st.sidebar.text_input(
            "Dataset path",
            value="./midi_datasets/EmbeddingPairsDataset/latest",
            help="Directory with train.jsonl (e.g. midi_datasets/EmbeddingPairsDataset/latest).",
        )
        split = st.sidebar.selectbox("Split", ["train", "validation", "test"], index=0)
        num_examples = st.sidebar.slider(
            "Max examples to load",
            min_value=10,
            max_value=5000,
            value=500,
            step=50,
        )
        with st.spinner("Loading dataset..."):
            dataset = load_pairs_from_directory(dataset_path, split, num_examples)
    else:
        jsonl_path = st.sidebar.text_input(
            "Path to .jsonl file",
            value="",
            help="Each line: JSON with notes_first, notes_second, source, source_dataset, original_id",
        )
        num_examples = st.sidebar.slider(
            "Max examples to load",
            min_value=10,
            max_value=5000,
            value=500,
            step=50,
        )
        with st.spinner("Loading JSONL..."):
            dataset = load_pairs_from_jsonl(jsonl_path, num_examples) if jsonl_path else None
        split = "train"

    if dataset is None:
        st.info("Configure the data source in the sidebar and load a dataset.")
        return

    # Main part: browse samples (multiple per page, random seed 4)
    st.header("Browse samples")
    if "browse_shuffle" not in st.session_state:
        st.session_state.browse_shuffle = 0
    n_show = min(SAMPLES_PER_PAGE, len(dataset))
    indices = _browse_indices(
        len(dataset),
        n_show,
        BROWSE_SEED,
        st.session_state.browse_shuffle,
    )
    if st.button("New random samples", key="new_samples"):
        st.session_state.browse_shuffle += 1
        st.rerun()
    st.caption(f"Random seed {BROWSE_SEED} | showing {len(indices)} samples")

    for idx in indices:
        example = dataset[idx]
        render_example(
            example,
            index=idx,
        )
        st.divider()

    # Bottom: dataset overview and samples from dataset only
    st.header("Dataset overview")
    st.write(f"**Split:** {split} · **Examples loaded:** {len(dataset)}")

    with st.expander("Dataset features"):
        st.json(dataset.features)

    st.subheader("Samples from dataset")
    sample_size = min(5, len(dataset))
    for i in range(sample_size):
        ex = dataset[i]
        preview = {k: (v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v) for k, v in ex.items()}
        with st.expander(f"Example {i}"):
            st.json(preview)

    if "source_dataset" in dataset.features and len(dataset) > 0:
        st.subheader("Source distribution")
        from collections import Counter

        counts = Counter(ex["source_dataset"] for ex in dataset)
        dist = pd.DataFrame(
            list(counts.items()),
            columns=["source_dataset", "count"],
        )
        st.bar_chart(dist.set_index("source_dataset"))


if __name__ == "__main__":
    main()
