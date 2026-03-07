import json
import random
from pathlib import Path
from collections import Counter

import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
import matplotlib.pyplot as plt
from midi_tokenizers import ExponentialTimeTokenizer

BROWSE_SEED = 4
SAMPLES_PER_PAGE = 6

# Load project dataset modules by path to avoid conflict with HuggingFace "datasets" package
project_root = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, project_root / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_midi_dataset = _load_module("midi_dataset", "midi_datasets/MidiDataset/MidiDataset.py")
_tokenized_dataset = _load_module(
    "midi_tokenized_dataset", "midi_datasets/MidiTokenizedDataset/MidiTokenizedDataset.py"
)
load_midi_from_directory = _midi_dataset.load_from_directory
load_tokenized_from_directory = _tokenized_dataset.load_from_directory

st.set_page_config(page_title="MIDI Dataset Explorer", layout="wide")


@st.cache_data
def load_dataset_from_directory(
    dataset_path: str,
    dataset_type: str,
    split: str = "train",
    num_examples: int = 1000,
):
    """Load from directory containing train.jsonl (and optionally validation/test). Resolves 'latest' symlink."""
    try:
        path = Path(dataset_path).resolve()
        if not path.is_dir():
            st.error(f"Not a directory: {path}")
            return None
        if not (path / "train.jsonl").exists():
            st.error(
                f"No train.jsonl in {path}. Use a path like "
                "midi_datasets/MidiDataset/latest or midi_datasets/MidiTokenizedDataset/latest."
            )
            return None
        if dataset_type == "Aggregated Dataset":
            ds_dict = load_midi_from_directory(str(path))
        else:
            ds_dict = load_tokenized_from_directory(str(path))
        dataset = ds_dict.get(split)
        if dataset is None:
            dataset = ds_dict["train"]
        if len(dataset) > num_examples:
            dataset = dataset.select(range(num_examples))
        return dataset
    except FileNotFoundError as e:
        st.error(str(e))
        return None
    except Exception as e:
        st.error(f"Error loading dataset from {dataset_path}: {e}")
        return None


def parse_notes(notes_str):
    """Parse MIDI notes from JSON string to DataFrame."""
    try:
        notes_data = json.loads(notes_str)
        if isinstance(notes_data, dict) and all(key in notes_data for key in ["pitch", "start"]):
            return pd.DataFrame(notes_data)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"Error parsing notes: {e}")
        return pd.DataFrame()


def create_tokenizer(
    min_time_unit=0.01,
    max_time_step=1.0,
    n_velocity_bins=32,
    n_special_ids=1024,
):
    """Create an ExponentialTimeTokenizer with the specified parameters."""
    tokenizer_config = {
        "time_unit": min_time_unit,
        "max_time_step": max_time_step,
        "n_velocity_bins": n_velocity_bins,
        "n_special_ids": n_special_ids,
    }

    tokenizer = ExponentialTimeTokenizer.build_tokenizer(tokenizer_config)
    return tokenizer


def tokenize_and_visualize(notes_df, tokenizer):
    """Tokenize MIDI notes and visualize the tokens."""
    if notes_df.empty:
        st.warning("No notes to tokenize.")
        return

    tokens = tokenizer.tokenize(notes_df)

    # Count token types
    token_types = {}
    for token in tokens:
        if token.startswith("NOTE_ON"):
            token_type = "NOTE_ON"
        elif token.startswith("NOTE_OFF"):
            token_type = "NOTE_OFF"
        elif token.startswith("VELOCITY"):
            token_type = "VELOCITY"
        elif token.endswith("T"):
            token_type = "TIME"
        else:
            token_type = "OTHER"

        token_types[token_type] = token_types.get(token_type, 0) + 1

    # Count individual tokens
    token_counter = Counter(tokens)
    most_common_tokens = token_counter.most_common(20)

    st.write("### Token Statistics")
    st.write(f"Total tokens: {len(tokens)}")
    st.write(f"Unique tokens: {len(token_counter)}")

    # Token type distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(token_types.keys(), token_types.values())
    ax.set_xlabel("Token Type")
    ax.set_ylabel("Count")
    ax.set_title("Token Type Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    # Most common tokens table
    st.write("### Most Common Tokens")
    common_tokens_df = pd.DataFrame(
        most_common_tokens,
        columns=["Token", "Count"],
    )
    common_tokens_df["Percentage"] = common_tokens_df["Count"] / len(tokens) * 100
    st.dataframe(common_tokens_df)

    # Top tokens bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(common_tokens_df["Token"], common_tokens_df["Count"])
    ax.set_xticklabels(
        common_tokens_df["Token"],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("Token")
    ax.set_ylabel("Count")
    ax.set_title("Most Common Tokens")
    plt.tight_layout()
    st.pyplot(fig)

    # Token sequence preview
    st.write("### Token Sequence Preview")
    st.write(tokens[:100])

    if len(tokens) > 100:
        st.write(f"... and {len(tokens) - 100} more tokens")

    # Convert tokens to IDs and show mapping table
    token_to_id = {token: idx for idx, token in enumerate(tokenizer.vocab)}

    # Create a token-to-id preview
    st.write("### Token to ID Mapping Preview")
    token_ids = [token_to_id.get(token, 0) for token in tokens[:20]]
    mapping_data = []

    for i, (token, token_id) in enumerate(zip(tokens[:20], token_ids)):
        mapping_data.append(
            {
                "Position": i,
                "Token": token,
                "ID": token_id,
            },
        )

    st.dataframe(pd.DataFrame(mapping_data))

    # Untokenize and compare
    untokenized_df = tokenizer.untokenize(tokens)
    st.write("### Untokenized Notes")
    st.dataframe(untokenized_df)

    st.write("### Comparison: Original vs Untokenized")
    comp_fig = plt.figure(figsize=(12, 10))

    # Original plot
    ax1 = comp_fig.add_subplot(211)
    for _, note in notes_df.iterrows():
        pitch = note["pitch"]
        start = note["start"]
        duration = note.get("duration", note.get("end", start + 0.5) - start)
        ax1.barh(pitch, duration, left=start, height=0.8, color="blue", alpha=0.7)
    ax1.set_title("Original Notes")
    ax1.set_ylabel("Pitch")

    # Untokenized plot
    ax2 = comp_fig.add_subplot(212)
    for _, note in untokenized_df.iterrows():
        pitch = note["pitch"]
        start = note["start"]
        duration = note.get("duration", note.get("end", start + 0.5) - start)
        ax2.barh(pitch, duration, left=start, height=0.8, color="green", alpha=0.7)
    ax2.set_title("Untokenized Notes")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Pitch")

    plt.tight_layout()
    st.pyplot(comp_fig)


def visualize_tokenized_dataset(dataset, tokenizer=None):
    """Visualize examples from a tokenized dataset."""
    if dataset is None or len(dataset) == 0:
        st.warning("No data to visualize.")
        return

    st.write(f"Total examples: {len(dataset)}")

    # Example selector
    example_idx = st.slider("Select example", 0, len(dataset) - 1, 0)
    example = dataset[example_idx]

    # Example metadata
    st.write("### Example Metadata")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Source dataset: {example.get('source_dataset', 'N/A')}")
    with col2:
        st.write(f"Original ID: {example.get('original_id', 'N/A')}")

    # Tokens
    st.write("### Input Tokens")
    input_ids = example.get("input_ids", [])

    # Token sequence visualization
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(
        range(len(input_ids)),
        input_ids,
        "-o",
        alpha=0.5,
        markersize=3,
    )
    ax.set_xlabel("Position")
    ax.set_ylabel("Token ID")
    ax.set_title("Token Sequence Visualization")
    st.pyplot(fig)

    # Token values
    st.write(f"Number of tokens: {len(input_ids)}")

    # Token distribution
    token_counter = Counter(input_ids)
    most_common_ids = token_counter.most_common(20)

    hist_fig, hist_ax = plt.subplots(figsize=(10, 5))
    hist_ax.hist(input_ids, bins=min(50, len(token_counter)))
    hist_ax.set_xlabel("Token ID")
    hist_ax.set_ylabel("Frequency")
    hist_ax.set_title("Token ID Distribution")
    st.pyplot(hist_fig)

    # If tokenizer is available, show token ID to string mapping
    if tokenizer is not None:
        try:
            # Token ID Preview with string values
            st.write("### Token Sequence with String Values")

            # Create mapping table for preview
            tokens_preview = []

            for i, token_id in enumerate(input_ids[:20]):  # First 20 tokens
                if token_id < len(tokenizer.vocab):
                    token_str = tokenizer.vocab[token_id]
                else:
                    token_str = f"<Unknown Token: {token_id}>"

                tokens_preview.append(
                    {
                        "Position": i,
                        "Token ID": token_id,
                        "Token String": token_str,
                    }
                )

            st.dataframe(pd.DataFrame(tokens_preview))

            # Most common tokens table with string values
            st.write("### Most Common Tokens")
            common_tokens_data = []

            for token_id, count in most_common_ids:
                if token_id < len(tokenizer.vocab):
                    token_str = tokenizer.vocab[token_id]
                else:
                    token_str = f"<Unknown Token: {token_id}>"

                percentage = count / len(input_ids) * 100

                common_tokens_data.append(
                    {
                        "Token ID": token_id,
                        "Token String": token_str,
                        "Count": count,
                        "Percentage": percentage,
                    }
                )

            st.dataframe(pd.DataFrame(common_tokens_data))

            # Show token type distribution
            token_types = {}
            for token_id in input_ids:
                if token_id < len(tokenizer.vocab):
                    token = tokenizer.vocab[token_id]

                    if token.startswith("NOTE_ON"):
                        token_type = "NOTE_ON"
                    elif token.startswith("NOTE_OFF"):
                        token_type = "NOTE_OFF"
                    elif token.startswith("VELOCITY"):
                        token_type = "VELOCITY"
                    elif token.endswith("T"):
                        token_type = "TIME"
                    else:
                        token_type = "OTHER"

                    token_types[token_type] = token_types.get(token_type, 0) + 1

            if token_types:
                st.write("### Token Type Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(token_types.keys(), token_types.values())
                ax.set_xlabel("Token Type")
                ax.set_ylabel("Count")
                ax.set_title("Token Type Distribution")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)

            # Convert all token IDs to tokens
            tokens = []
            for token_id in input_ids:
                if token_id < len(tokenizer.vocab):
                    tokens.append(tokenizer.vocab[token_id])
                else:
                    tokens.append(f"<Unknown Token: {token_id}>")

            # Try to untokenize
            untokenized_df = tokenizer.untokenize(tokens)
            if not untokenized_df.empty:
                st.write("### Untokenized Notes")
                st.dataframe(untokenized_df)
                st.write("### Untokenized Notes Visualization")
                piece = ff.MidiPiece(untokenized_df)
                streamlit_pianoroll.from_fortepyan(piece=piece)
            else:
                st.warning("Untokenization produced empty DataFrame.")
        except Exception as e:
            st.error(f"Error decoding tokens: {e}")


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


def _render_aggregated_sample(
    example: dict,
    index: int,
    tokenizer,
) -> None:
    """Render one aggregated sample: source, notes table, pianoroll, tokenization in expander."""
    with st.expander(
        f"Sample index {index} | source: {example.get('source_dataset', 'N/A')} "
        f"| id: {example.get('original_id', index)}",
        expanded=True,
    ):
        if "source" in example:
            try:
                st.json(json.loads(example["source"]))
            except Exception:
                st.text(example["source"][:200])
        if "notes" not in example:
            st.warning("No notes field.")
            return
        notes_df = parse_notes(example["notes"])
        if notes_df.empty:
            st.warning("No valid notes.")
            return
        st.dataframe(notes_df.head(20))
        piece = ff.MidiPiece(notes_df)
        streamlit_pianoroll.from_fortepyan(piece=piece)
        st.write("Tokenization")
        tokenize_and_visualize(notes_df, tokenizer)


def _render_tokenized_sample(
    example: dict,
    index: int,
    tokenizer,
) -> None:
    """Render one tokenized sample: metadata, token plot, optional untokenized pianoroll."""
    with st.expander(
        f"Sample index {index} | source: {example.get('source_dataset', 'N/A')} "
        f"| id: {example.get('original_id', index)}",
        expanded=True,
    ):
        st.caption(
            f"Source: {example.get('source_dataset', 'N/A')} " f"| Original ID: {example.get('original_id', index)}"
        )
        input_ids = example.get("input_ids", [])
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(
            range(len(input_ids)),
            input_ids,
            "-",
            alpha=0.6,
        )
        ax.set_xlabel("Position")
        ax.set_ylabel("Token ID")
        st.pyplot(fig)
        if tokenizer is not None:
            try:
                tokens = [tokenizer.vocab[tid] if tid < len(tokenizer.vocab) else f"<{tid}>" for tid in input_ids]
                untokenized_df = tokenizer.untokenize(tokens)
                if not untokenized_df.empty:
                    piece = ff.MidiPiece(untokenized_df)
                    streamlit_pianoroll.from_fortepyan(piece=piece)
            except Exception as e:
                st.caption(f"Untokenize failed: {e}")


def main() -> None:
    st.title("MIDI Dataset Explorer")

    st.sidebar.header("Dataset Configuration")
    dataset_type = st.sidebar.radio(
        "Dataset Type",
        ["Aggregated Dataset", "Tokenized Dataset"],
    )
    dataset_path = st.sidebar.text_input(
        "Dataset Path",
        value="./midi_datasets/MidiDataset/latest"
        if dataset_type == "Aggregated Dataset"
        else "./midi_datasets/MidiTokenizedDataset/latest",
        help="Directory with train.jsonl.",
    )
    dataset_split = st.sidebar.selectbox(
        "Dataset Split",
        ["train", "validation", "test"],
        index=0,
    )
    num_examples = st.sidebar.slider(
        "Max Examples to Load",
        min_value=10,
        max_value=10000,
        value=1000,
        step=10,
    )

    st.sidebar.header("Tokenizer Configuration")
    min_time_unit = st.sidebar.number_input(
        "Min Time Unit",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.001,
        format="%.3f",
    )
    max_time_step = st.sidebar.number_input(
        "Max Time Step",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        format="%.1f",
    )
    n_velocity_bins = st.sidebar.slider(
        "Velocity Bins",
        min_value=4,
        max_value=128,
        value=32,
        step=4,
    )
    n_special_ids = st.sidebar.slider(
        "Special IDs",
        min_value=0,
        max_value=2048,
        value=1024,
        step=128,
    )

    tokenizer = create_tokenizer(
        min_time_unit=min_time_unit,
        max_time_step=max_time_step,
        n_velocity_bins=n_velocity_bins,
        n_special_ids=n_special_ids,
    )

    with st.spinner("Loading dataset..."):
        dataset = load_dataset_from_directory(
            dataset_path,
            dataset_type,
            dataset_split,
            num_examples,
        )

    if dataset is None:
        st.error(
            f"Failed to load dataset from {dataset_path}. Please check the path and try again.",
        )
        return

    # Main part: browse samples (multiple per page, random with seed 4)
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
        if dataset_type == "Aggregated Dataset":
            _render_aggregated_sample(
                example,
                idx,
                tokenizer,
            )
        else:
            _render_tokenized_sample(
                example,
                idx,
                tokenizer,
            )

    # Bottom: dataset overview and sample examples only
    st.header("Dataset overview")
    st.write(f"Dataset: {dataset_path}")
    st.write(f"Split: {dataset_split}")
    st.write(f"Number of examples: {len(dataset)}")

    with st.expander("Dataset Features"):
        st.json(dataset.features)

    st.subheader("Samples from dataset")
    sample_size = min(5, len(dataset))
    sample_indices = list(range(sample_size))
    samples = dataset.select(sample_indices)
    for i, sample in enumerate(samples):
        with st.expander(f"Example {i}"):
            st.json(
                {k: (v[:100] + "..." if isinstance(v, str) and len(v) > 100 else v) for k, v in sample.items()},
            )

    if dataset_type == "Tokenized Dataset":
        st.subheader("Dataset statistics")
        seq_lengths = [len(ex["input_ids"]) for ex in dataset]
        avg_length = sum(seq_lengths) / len(seq_lengths)
        st.write(f"Average sequence length: {avg_length:.2f}")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(seq_lengths, bins=30)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Frequency")
        ax.set_title("Sequence Length Distribution")
        st.pyplot(fig)
        if "source_dataset" in dataset.features:
            source_counts = {}
            for ex in dataset:
                src = ex["source_dataset"]
                source_counts[src] = source_counts.get(src, 0) + 1
            if source_counts:
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.bar(source_counts.keys(), source_counts.values())
                ax2.set_xlabel("Source Dataset")
                ax2.set_ylabel("Count")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig2)


if __name__ == "__main__":
    main()
