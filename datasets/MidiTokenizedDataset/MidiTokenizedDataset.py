import json
import logging
from typing import Any, Dict, List, Tuple, Iterator, Optional

import pandas as pd
from datasets.info import DatasetInfo
from midi_tokenizers import ExponentialTimeTokenizer
from datasets.features import Value, Features, Sequence
from datasets.download.download_manager import DownloadManager

import datasets
from datasets import Split, Dataset, BuilderConfig, SplitGenerator, GeneratorBasedBuilder

logger = logging.getLogger(__name__)


class MidiTokenizedDatasetConfig(BuilderConfig):
    """Configuration for the MidiTokenizedDataset."""

    def __init__(
        self,
        name: str = "default",
        version: str = "1.0.0",
        aggregated_dataset_path: str = "./datasets/MidiDataset",
        context_length: int = 1024,
        min_time_unit: float = 0.01,
        max_time_step: float = 1.0,
        n_velocity_bins: int = 32,
        n_special_ids: int = 1024,
        sliding_window_stride: Optional[int] = None,
        num_proc: int = 128,
        description: str = "MIDI data tokenized with ExponentialTimeTokenizer",
    ):
        super().__init__(name=name, version=version, description=description)
        self.aggregated_dataset_path = aggregated_dataset_path
        self.context_length = context_length
        self.min_time_unit = min_time_unit
        self.max_time_step = max_time_step
        self.n_velocity_bins = n_velocity_bins
        self.n_special_ids = n_special_ids
        self.sliding_window_stride = sliding_window_stride or context_length
        self.num_proc = num_proc

    @property
    def tokenizer_config(self):
        """Return the configuration for the tokenizer."""
        return {
            "time_unit": self.min_time_unit,
            "max_time_step": self.max_time_step,
            "n_velocity_bins": self.n_velocity_bins,
            "n_special_ids": self.n_special_ids,
        }


# Define default configurations
BUILDER_CONFIGS = [
    MidiTokenizedDatasetConfig(
        name="default",
        description="Default configuration for tokenized MIDI dataset",
    ),
]


class MidiTokenizedDataset(GeneratorBasedBuilder):
    """Builder that tokenizes an aggregated MIDI dataset using ExponentialTimeTokenizer."""

    DEFAULT_WRITER_BATCH_SIZE = 1000

    BUILDER_CONFIG_CLASS = MidiTokenizedDatasetConfig
    BUILDER_CONFIGS = BUILDER_CONFIGS
    DEFAULT_CONFIG_NAME = "default"

    def __init__(
        self,
        aggregated_dataset_path: str = None,
        context_length: int = 1024,
        min_time_unit: float = 0.01,
        max_time_step: float = 1.0,
        n_velocity_bins: int = 32,
        n_special_ids: int = 1024,
        sliding_window_stride: Optional[int] = None,
        num_proc: int = 128,
        **kwargs,
    ):
        """Initialize the MIDI tokenized dataset builder.

        Args:
            aggregated_dataset_path: Path to the aggregated dataset (local or HF Hub)
            context_length: Maximum context length for tokenization
            min_time_unit: Minimum time unit for the ExponentialTimeTokenizer
            max_time_step: Maximum time step for the ExponentialTimeTokenizer
            n_velocity_bins: Number of velocity bins for the tokenizer
            n_special_ids: Number of special IDs for the tokenizer
            sliding_window_stride: Stride for sliding window tokenization
            num_proc: Number of processes for parallel processing
        """
        if aggregated_dataset_path:
            config = MidiTokenizedDatasetConfig(
                name="custom",
                aggregated_dataset_path=aggregated_dataset_path,
                context_length=context_length,
                min_time_unit=min_time_unit,
                max_time_step=max_time_step,
                n_velocity_bins=n_velocity_bins,
                n_special_ids=n_special_ids,
                sliding_window_stride=sliding_window_stride,
                num_proc=num_proc,
            )
            super().__init__(config=config, **kwargs)
        else:
            super().__init__(**kwargs)

        tokenizer_config = self.config.tokenizer_config
        self.tokenizer = ExponentialTimeTokenizer.build_tokenizer(tokenizer_config)
        logger.info(f"Created tokenizer with config: {tokenizer_config}")

    def _info(self) -> DatasetInfo:
        """Return the dataset info."""
        return DatasetInfo(
            description="MIDI data tokenized with ExponentialTimeTokenizer",
            features=Features(
                {
                    "input_ids": Sequence(feature=Value(dtype="int64", id=None)),
                    "source_dataset": Value(dtype="string", id=None),
                    "original_id": Value(dtype="string", id=None),
                }
            ),
        )

    def _split_generators(
        self,
        dl_manager: DownloadManager,
    ) -> List[SplitGenerator]:
        """Return SplitGenerators with sharding for multiprocessing."""

        # Load the aggregated dataset
        try:
            train_dataset = datasets.load_dataset(
                self.config.aggregated_dataset_path, split="train", trust_remote_code=True
            )
            logger.info(f"Loaded training dataset with {len(train_dataset)} examples")
        except Exception as e:
            logger.error(f"Failed to load training dataset: {e}")
            raise

        try:
            validation_dataset = datasets.load_dataset(
                self.config.aggregated_dataset_path, split="validation", trust_remote_code=True
            )
            logger.info(f"Loaded validation dataset with {len(validation_dataset)} examples")
        except Exception as e:
            logger.error(f"Failed to load validation dataset: {e}")
            validation_dataset = None

        try:
            test_dataset = datasets.load_dataset(
                self.config.aggregated_dataset_path, split="test", trust_remote_code=True
            )
            logger.info(f"Loaded test dataset with {len(test_dataset)} examples")
        except Exception as e:
            logger.warning(f"Failed to load test dataset, using validation dataset instead: {e}")
            test_dataset = validation_dataset

        # Create shards for multiprocessing
        n_train_shards = self.config.num_proc
        train_shards = [train_dataset.shard(n_train_shards, i) for i in range(n_train_shards)]

        # Validation datasets do not need as many shards for processing
        n_val_test_shards = min(self.config.num_proc // 2, 16)

        validation_shards = []
        if validation_dataset:
            validation_shards = [validation_dataset.shard(n_val_test_shards, i) for i in range(n_val_test_shards)]

        test_shards = []
        if test_dataset:
            test_shards = [test_dataset.shard(n_val_test_shards, i) for i in range(n_val_test_shards)]

        return [
            datasets.SplitGenerator(name=Split.TRAIN, gen_kwargs={"dataset_shards": train_shards}),
            datasets.SplitGenerator(name=Split.VALIDATION, gen_kwargs={"dataset_shards": validation_shards}),
            datasets.SplitGenerator(name=Split.TEST, gen_kwargs={"dataset_shards": test_shards}),
        ]

    def _parse_midi_data(
        self,
        notes_str: str,
    ) -> pd.DataFrame:
        """Parse the serialized MIDI notes data into a DataFrame."""
        try:
            notes_data = json.loads(notes_str)

            if isinstance(notes_data, dict) and all(key in notes_data for key in ["pitch", "start"]):
                return pd.DataFrame(notes_data)
            else:
                logger.warning(f"Unexpected notes format: {type(notes_data)}")
                return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Error parsing MIDI notes data: {e}")
            return pd.DataFrame()

    def _tokenize_notes_df(
        self,
        notes_df: pd.DataFrame,
    ) -> List[int]:
        """Tokenize a MIDI DataFrame using the ExponentialTimeTokenizer."""
        try:
            required_columns = ["pitch", "start", "end"]
            if not all(col in notes_df.columns for col in required_columns):
                logger.warning(f"Missing required columns in MIDI data. Available: {notes_df.columns}")
                return []

            # Use only relevant columns
            relevant_columns = ["pitch", "velocity", "start", "end"]
            notes_df = notes_df[relevant_columns]

            # Tokenize the MIDI data
            tokens = self.tokenizer.encode_notes_df(notes_df)

            # Get the token IDs
            vocab = self.tokenizer.vocab

            # Convert tokens to IDs if they are strings
            if tokens and isinstance(tokens[0], str):
                token_ids = []
                for token in tokens:
                    if token in vocab:
                        token_ids.append(vocab.index(token))
                    else:
                        logger.warning(f"Token not in vocabulary: {token}")
                        token_ids.append(0)  # Use padding token as default
                return token_ids

            return tokens

        except Exception as e:
            logger.warning(f"Error tokenizing MIDI data: {e}")
            return []

    def _tokenize_and_chunk(
        self,
        notes_str: str,
    ) -> List[List[int]]:
        """Tokenize MIDI notes data and chunk into overlapping segments."""
        # Parse the MIDI data
        notes_df = self._parse_midi_data(notes_str)

        if notes_df.empty:
            return []

        # Tokenize the MIDI data
        tokens = self._tokenize_notes_df(notes_df)

        if not tokens:
            return []

        # If tokens are fewer than context_length, skip
        if len(tokens) < self.config.context_length:
            return []

        # Create overlapping chunks
        chunks = []
        for i in range(0, len(tokens) - self.config.context_length + 1, self.config.sliding_window_stride):
            chunk = tokens[i : i + self.config.context_length]
            if len(chunk) == self.config.context_length:
                chunks.append(chunk)

        return chunks

    def _generate_examples(
        self,
        dataset_shards: List[Dataset],
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Generate examples from dataset shards.

        Args:
            dataset_shards: List of dataset shards to process

        Yields:
            Tuples of (id, example) where example is a dictionary
        """
        example_counter = 0

        for shard_id, shard in enumerate(dataset_shards):
            for example_id, example in enumerate(shard):
                try:
                    notes_str = example["notes"]

                    source_dataset = example.get("source_dataset", f"shard_{shard_id}")
                    original_id = example.get("original_id", str(example_id))

                    # Tokenize and chunk the MIDI data
                    chunks = self._tokenize_and_chunk(notes_str)

                    for chunk_id, chunk in enumerate(chunks):
                        unique_id = f"{shard_id}_{example_id}_{chunk_id}"

                        yield unique_id, {
                            "input_ids": chunk,
                            "source_dataset": source_dataset,
                            "original_id": original_id,
                        }

                        example_counter += 1

                        # Log progress occasionally
                        if example_counter % 1000 == 0:
                            logger.info(f"Processed {example_counter} examples")

                except Exception as e:
                    logger.error(f"Error processing example {example_id} in shard {shard_id}: {e}")
                    continue
