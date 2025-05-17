import json

import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
import matplotlib.pyplot as plt
from midi_tokenizers import ExponentialTimeTokenizer

from datasets import load_dataset

st.set_page_config(page_title="MIDI Dataset Explorer", layout="wide")


@st.cache_data
def load_hf_dataset(dataset_path, split="train", num_examples=1000):
    """Load a dataset from Hugging Face."""
    try:
        dataset = load_dataset(dataset_path, split=split, trust_remote_code=True)
        if len(dataset) > num_examples:
            dataset = dataset.select(range(num_examples))
        return dataset
    except Exception as e:
        st.error(f"Error loading dataset {dataset_path}: {e}")
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


def create_tokenizer(min_time_unit=0.01, max_time_step=1.0, n_velocity_bins=32, n_special_ids=1024):
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

    st.write("### Token Statistics")
    st.write(f"Total tokens: {len(tokens)}")

    # Token type distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(token_types.keys(), token_types.values())
    ax.set_xlabel("Token Type")
    ax.set_ylabel("Count")
    ax.set_title("Token Type Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    # Tokens review
    st.write("### Tokens")
    st.write(tokens[:100])

    if len(tokens) > 100:
        st.write(f"... and {len(tokens) - 100} more tokens")

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
    ax.plot(range(len(input_ids)), input_ids, "-o", alpha=0.5, markersize=3)
    ax.set_xlabel("Position")
    ax.set_ylabel("Token ID")
    ax.set_title("Token Sequence Visualization")
    st.pyplot(fig)

    # Token values
    st.write(f"Number of tokens: {len(input_ids)}")
    st.write("First 50 tokens:")
    st.write(input_ids[:50])

    # Token distribution
    hist_fig, hist_ax = plt.subplots(figsize=(10, 5))
    hist_ax.hist(input_ids, bins=50)
    hist_ax.set_xlabel("Token ID")
    hist_ax.set_ylabel("Frequency")
    hist_ax.set_title("Token ID Distribution")
    st.pyplot(hist_fig)

    # If tokenizer is available, try to decode and visualize
    if tokenizer is not None:
        try:
            tokens = [tokenizer.vocab[token_id] for token_id in input_ids]

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


def main():
    st.title("MIDI Dataset Explorer")

    # Sidebar for configuration
    st.sidebar.header("Dataset Configuration")

    dataset_type = st.sidebar.radio("Dataset Type", ["Aggregated Dataset", "Tokenized Dataset"])

    dataset_path = st.sidebar.text_input(
        "Dataset Path",
        value="./datasets/MidiDataset" if dataset_type == "Aggregated Dataset" else "./datasets/MidiTokenizedDataset",
    )

    dataset_split = st.sidebar.selectbox("Dataset Split", ["train", "validation", "test"], index=0)

    num_examples = st.sidebar.slider("Max Examples to Load", min_value=10, max_value=10000, value=1000, step=10)

    # Tokenizer configuration in sidebar
    st.sidebar.header("Tokenizer Configuration")

    min_time_unit = st.sidebar.number_input(
        "Min Time Unit", min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.3f"
    )

    max_time_step = st.sidebar.number_input(
        "Max Time Step", min_value=0.1, max_value=10.0, value=1.0, step=0.1, format="%.1f"
    )

    n_velocity_bins = st.sidebar.slider("Velocity Bins", min_value=4, max_value=128, value=32, step=4)

    n_special_ids = st.sidebar.slider("Special IDs", min_value=0, max_value=2048, value=1024, step=128)

    # Create tokenizer
    tokenizer = create_tokenizer(
        min_time_unit=min_time_unit,
        max_time_step=max_time_step,
        n_velocity_bins=n_velocity_bins,
        n_special_ids=n_special_ids,
    )

    # Load dataset
    with st.spinner("Loading dataset..."):
        dataset = load_hf_dataset(dataset_path, dataset_split, num_examples)

    if dataset is None:
        st.error(f"Failed to load dataset from {dataset_path}. Please check the path and try again.")
        return

    # Dataset overview
    st.header("Dataset Overview")
    st.write(f"Dataset: {dataset_path}")
    st.write(f"Split: {dataset_split}")
    st.write(f"Number of examples: {len(dataset)}")

    # Dataset features
    with st.expander("Dataset Features"):
        st.json(dataset.features)

    st.write("Sample Examples")
    sample_size = min(5, len(dataset))
    sample_indices = list(range(sample_size))
    samples = dataset.select(sample_indices)

    for i, sample in enumerate(samples):
        with st.expander(f"Example {i}"):
            st.json({k: (v[:100] + "..." if isinstance(v, str) and len(v) > 100 else v) for k, v in sample.items()})

    if dataset_type == "Aggregated Dataset":
        st.header("Aggregated Dataset Exploration")

        example_idx = st.selectbox("Select example to explore", range(len(dataset)))
        example = dataset[example_idx]

        # Source information
        st.write("### Source Information")
        if "source" in example:
            source_data = json.loads(example["source"])
            st.json(source_data)

        # Notes data
        st.write("### Notes Data")
        if "notes" in example:
            notes_df = parse_notes(example["notes"])
            if not notes_df.empty:
                st.dataframe(notes_df)

                # Visualize notes
                st.write("### Notes Visualization")
                piece = ff.MidiPiece(notes_df)
                streamlit_pianoroll.from_fortepyan(piece=piece)

                # Tokenize and visualize
                st.write("### Tokenization")
                tokenize_and_visualize(notes_df, tokenizer)
            else:
                st.warning("No valid notes data found.")
        else:
            st.warning("No notes field found in example.")

    else:
        st.header("Tokenized Dataset Exploration")
        visualize_tokenized_dataset(dataset, tokenizer)

        st.write("### Dataset Statistics")

        # Calculate average sequence length
        seq_lengths = [len(ex["input_ids"]) for ex in dataset]
        avg_length = sum(seq_lengths) / len(seq_lengths)

        st.write(f"Average sequence length: {avg_length: .2f}")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(seq_lengths, bins=30)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Frequency")
        ax.set_title("Sequence Length Distribution")
        st.pyplot(fig)

        # Source dataset distribution
        if "source_dataset" in dataset.features:
            source_counts = {}
            for ex in dataset:
                source = ex["source_dataset"]
                source_counts[source] = source_counts.get(source, 0) + 1

            if source_counts:
                st.write("### Source Dataset Distribution")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(source_counts.keys(), source_counts.values())
                ax.set_xlabel("Source Dataset")
                ax.set_ylabel("Count")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)


if __name__ == "__main__":
    main()
