import os
import json

import torch
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from transformers import AutoModelForCausalLM
from midi_tokenizers import ExponentialTimeTokenizer


@st.cache_resource
def load_model_and_tokenizer(
    model_path: str,
):
    """Load the trained model and tokenizer."""
    try:
        # Load tokenizer
        tokenizer_path = os.path.join(
            model_path,
            "tokenizer.json",
        )
        with open(
            tokenizer_path,
            "r",
        ) as f:
            tokenizer_config = json.load(
                f,
            )

        tokenizer = ExponentialTimeTokenizer.from_dict(
            tokenizer_config,
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
        )
        model.eval()

        # Move to GPU if available
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        model = model.to(
            device,
        )

        return model, tokenizer, device
    except Exception as e:
        st.error(
            f"Error loading model: {str(e)}",
        )
        return None, None, None


def find_available_checkpoints(
    base_dir: str = "checkpoints",
):
    """Find all available model checkpoints."""
    checkpoints = []

    if not os.path.exists(
        base_dir,
    ):
        return checkpoints

    # Walk through the directory structure
    for root, dirs, files in os.walk(
        base_dir,
    ):
        if "config.json" in files:
            checkpoints.append(
                root,
            )

    return sorted(
        checkpoints,
    )


def midi_to_tokens(
    tokenizer: ExponentialTimeTokenizer,
    midi_file_path: str,
):
    """Convert MIDI file to token sequence using the tokenizer."""
    try:
        # Load MIDI file
        piece = ff.MidiPiece.from_file(
            midi_file_path,
        )
        df = piece.df

        # Encode to tokens
        tokens = tokenizer.encode(
            df,
        )

        return tokens, df
    except Exception as e:
        st.error(
            f"Error converting MIDI to tokens: {str(e)}",
        )
        return None, None


def generate_tokens(
    model,
    tokenizer: ExponentialTimeTokenizer,
    device: torch.device,
    prompt_tokens: list = None,
    max_length: int = 512,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    num_sequences: int = 1,
):
    """Generate token sequences from the model."""
    model.eval()

    with torch.no_grad():
        if (
            prompt_tokens is None
            or len(
                prompt_tokens,
            )
            == 0
        ):
            # Start with BOS token
            input_ids = torch.tensor(
                [[0]],
                dtype=torch.long,
            ).to(
                device,
            )
        else:
            input_ids = torch.tensor(
                [prompt_tokens],
                dtype=torch.long,
            ).to(
                device,
            )

        generated_sequences = []

        for _ in range(
            num_sequences,
        ):
            generated = input_ids.clone()

            for _ in range(
                max_length - generated.shape[1],
            ):
                outputs = model(
                    generated,
                )
                logits = outputs.logits[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = (
                        logits
                        < torch.topk(
                            logits,
                            top_k,
                        )[
                            0
                        ][..., -1, None]
                    )
                    logits[indices_to_remove] = float(
                        "-inf",
                    )

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        logits,
                        descending=True,
                    )
                    cumulative_probs = torch.cumsum(
                        torch.softmax(
                            sorted_logits,
                            dim=-1,
                        ),
                        dim=-1,
                    )

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1,
                        sorted_indices,
                        sorted_indices_to_remove,
                    )
                    logits[indices_to_remove] = float(
                        "-inf",
                    )

                probs = torch.softmax(
                    logits,
                    dim=-1,
                )
                next_token = torch.multinomial(
                    probs,
                    num_samples=1,
                )

                generated = torch.cat(
                    [generated, next_token],
                    dim=1,
                )

                # Check for EOS token (assuming token 1 is EOS)
                if next_token.item() == 1:
                    break

            generated_sequences.append(
                generated[0].cpu().tolist(),
            )

    return generated_sequences


def main():
    st.set_page_config(
        page_title="MIDI Transformer Inference",
        layout="wide",
    )

    st.title(
        "MIDI Transformer Inference Dashboard",
    )
    st.markdown(
        "Generate MIDI token sequences from trained transformer models",
    )

    # Sidebar for model selection and parameters
    with st.sidebar:
        st.header(
            "Model Configuration",
        )

        # Find available checkpoints
        base_checkpoint_dir = st.text_input(
            "Checkpoint Base Directory",
            value="checkpoints",
            help="Base directory where checkpoints are stored",
        )

        available_checkpoints = find_available_checkpoints(
            base_checkpoint_dir,
        )

        if not available_checkpoints:
            st.warning(
                "No checkpoints found. Please specify the correct directory.",
            )
            selected_checkpoint = st.text_input(
                "Manual Model Path",
                help="Enter the full path to your model checkpoint",
            )
        else:
            selected_checkpoint = st.selectbox(
                "Select Checkpoint",
                options=available_checkpoints,
                format_func=lambda x: x.replace(
                    base_checkpoint_dir + "/",
                    "",
                ),
            )

        st.markdown(
            "---",
        )
        st.header(
            "Generation Parameters",
        )

        max_length = st.number_input(
            "Max Length",
            min_value=64,
            max_value=2048,
            value=512,
            step=64,
            help="Maximum number of tokens to generate",
        )

        temperature = st.number_input(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Higher values = more random, lower values = more deterministic",
        )

        top_k = st.number_input(
            "Top-K",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            help="Number of highest probability tokens to keep (0 = disabled)",
        )

        top_p = st.number_input(
            "Top-P (Nucleus)",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.05,
            help="Cumulative probability threshold for nucleus sampling",
        )

        num_sequences = st.number_input(
            "Number of Sequences",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
            help="Number of sequences to generate",
        )

    # Main content area
    if selected_checkpoint and os.path.exists(
        selected_checkpoint,
    ):
        # Load model
        with st.spinner(
            "Loading model and tokenizer...",
        ):
            model, tokenizer, device = load_model_and_tokenizer(
                selected_checkpoint,
            )

        if model is not None and tokenizer is not None:
            st.success(
                f"Model loaded successfully from: `{selected_checkpoint}`",
            )
            st.info(
                f"Running on: **{device}**",
            )

            col1, col2 = st.columns(
                [1, 1],
            )

            with col1:
                st.subheader(
                    "Prompt Configuration",
                )

                # Option 1: Upload MIDI file
                st.markdown(
                    "**Option 1: Upload MIDI File**",
                )
                uploaded_file = st.file_uploader(
                    "Upload MIDI file",
                    type=["mid", "midi"],
                )

                if uploaded_file is not None:
                    # Save uploaded file temporarily
                    temp_path = f"tmp_{uploaded_file.name}"
                    with open(
                        temp_path,
                        "wb",
                    ) as f:
                        f.write(
                            uploaded_file.getbuffer(),
                        )

                    # Convert MIDI to tokens
                    with st.spinner(
                        "Converting MIDI to tokens...",
                    ):
                        prompt_tokens, prompt_df = midi_to_tokens(
                            tokenizer,
                            temp_path,
                        )

                    # Clean up temp file
                    if os.path.exists(
                        temp_path,
                    ):
                        os.remove(
                            temp_path,
                        )

                    if prompt_tokens is not None and prompt_df is not None:
                        st.success(
                            f"Loaded {len(prompt_tokens)} tokens from MIDI",
                        )
                        st.write(
                            "Prompt DataFrame:",
                        )
                        st.dataframe(
                            prompt_df,
                            use_container_width=True,
                        )

                        st.write(
                            "Prompt Piano Roll:",
                        )
                        prompt_piece = ff.MidiPiece(
                            prompt_df,
                        )
                        streamlit_pianoroll.from_fortepyan(
                            piece=prompt_piece,
                            key=str(hash(prompt_piece.df.to_json())),
                        )

                        # Store in session state
                        st.session_state["prompt_tokens"] = prompt_tokens
                        st.session_state["prompt_df"] = prompt_df

                # Option 2: Manual token input
                st.markdown(
                    "**Option 2: Manual Token Input**",
                )
                use_manual_prompt = st.checkbox(
                    "Use manual prompt tokens",
                    value=False,
                )

                manual_prompt_tokens = []
                if use_manual_prompt:
                    prompt_input = st.text_area(
                        "Enter prompt tokens (comma-separated integers)",
                        value="0, 128, 256",
                        help="Enter token IDs separated by commas",
                    )

                    try:
                        manual_prompt_tokens = [
                            int(
                                x.strip(),
                            )
                            for x in prompt_input.split(
                                ",",
                            )
                            if x.strip()
                        ]
                        st.info(
                            f"Manual prompt tokens: {len(manual_prompt_tokens)} tokens",
                        )
                        st.session_state["prompt_tokens"] = manual_prompt_tokens
                        st.session_state["prompt_df"] = None
                    except ValueError:
                        st.error(
                            "Invalid token format. Please enter comma-separated integers.",
                        )
                        manual_prompt_tokens = []

            with col2:
                st.subheader(
                    "Vocabulary Info",
                )
                vocab_size = len(
                    tokenizer.vocab,
                )
                st.metric(
                    "Vocabulary Size",
                    vocab_size,
                )

            # Generation button
            if st.button(
                "Generate",
                type="primary",
                use_container_width=True,
            ):
                # Get prompt tokens from session state if available
                prompt_tokens = st.session_state.get(
                    "prompt_tokens",
                    None,
                )
                prompt_df = st.session_state.get(
                    "prompt_df",
                    None,
                )

                with st.spinner(
                    "Generating token sequences...",
                ):
                    generated_sequences = generate_tokens(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        prompt_tokens=prompt_tokens,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        num_sequences=num_sequences,
                    )

                st.success(
                    f"Generated {len(generated_sequences)} sequence(s)",
                )

                # Display generated sequences as dataframes
                for idx, sequence in enumerate(
                    generated_sequences,
                ):
                    st.header(
                        f"Generated Output {idx + 1}",
                    )

                    with st.spinner(
                        f"Converting sequence {idx + 1} to dataframe...",
                    ):
                        generated_df = tokenizer.decode(sequence)

                    if generated_df is not None:
                        st.write(
                            "Generated DataFrame:",
                        )
                        st.dataframe(
                            generated_df,
                            use_container_width=True,
                        )

                        # Create MidiPiece for visualization
                        generated_piece = ff.MidiPiece(
                            df=generated_df,
                        )

                        # Show piano roll
                        st.subheader(
                            "Piano Roll Visualization",
                        )

                        if prompt_df is not None:
                            # Show side-by-side comparison
                            col_a, col_b = st.columns(
                                2,
                            )
                            with col_a:
                                st.write(
                                    "Prompt:",
                                )
                                prompt_piece = ff.MidiPiece(
                                    prompt_df,
                                )
                                streamlit_pianoroll.from_fortepyan(
                                    piece=prompt_piece, key=f"{hash(prompt_piece.df.to_json())}_2"
                                )
                            with col_b:
                                st.write(
                                    "Generated:",
                                )
                                streamlit_pianoroll.from_fortepyan(
                                    piece=generated_piece, key=str(hash(generated_piece.df.to_json()))
                                )

                            # Show combined view
                            st.write(
                                "Combined View:",
                            )
                            streamlit_pianoroll.from_fortepyan(
                                piece=prompt_piece,
                                secondary_piece=generated_piece,
                                key=f"{hash(prompt_piece.df.to_json())}_{hash(generated_piece.df.to_json)}",
                            )
                        else:
                            # Show only generated
                            st.write(
                                "Generated:",
                            )
                            streamlit_pianoroll.from_fortepyan(
                                piece=generated_piece,
                            )

                        # Show basic statistics
                        col_a, col_b = st.columns(
                            2,
                        )
                        with col_a:
                            st.metric(
                                "Total Events",
                                len(
                                    generated_df,
                                ),
                            )
                        with col_b:
                            st.metric(
                                "Token Count",
                                len(
                                    sequence,
                                ),
                            )
                    else:
                        st.error(
                            f"Failed to convert sequence {idx + 1} to dataframe",
                        )
        else:
            st.error(
                "Failed to load model. Check the checkpoint path and files.",
            )
    else:
        st.info(
            "Please select or specify a valid checkpoint path in the sidebar",
        )
        st.markdown(
            """
        ### Expected Directory Structure:
        ```
        checkpoints/
        ├── project_name/
        │   └── experiment_name/
        │       ├── config.json
        │       ├── pytorch_model.bin
        │       ├── tokenizer.json
        │       ├── epoch_0/
        │       ├── epoch_1/
        │       └── final/
        ```
        """,
        )


if __name__ == "__main__":
    main()
