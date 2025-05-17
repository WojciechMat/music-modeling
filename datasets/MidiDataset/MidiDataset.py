import json
import logging
from typing import Any, Dict, List, Tuple, Iterator

from datasets.info import DatasetInfo
from datasets.features import Value, Features
from datasets.download.download_manager import DownloadManager

import datasets
from datasets import Split, Dataset, BuilderConfig, SplitGenerator, GeneratorBasedBuilder

logger = logging.getLogger(__name__)


class MidiDatasetConfig(BuilderConfig):
    """Configuration for the MidiAggregatedDataset."""

    def __init__(
        self,
        name: str = "default",
        version: str = "1.0.0",
        train_dataset_paths: List[str] = None,
        validation_dataset_path: str = "epr-labs/maestro-sustain-v2",
        num_proc: int = 32,
        description: str = "Aggregated MIDI dataset",
    ):
        self.name = name
        self.version = version
        self.train_dataset_paths = train_dataset_paths or ["epr-labs/maestro-sustain-v2"]
        self.validation_dataset_path = validation_dataset_path
        self.num_proc = num_proc
        self.description = description


BUILDER_CONFIGS = [
    MidiDatasetConfig(
        name="default",
        description="Default configuration for aggregated MIDI dataset",
    ),
]


class MidiDatasetBuilder(GeneratorBasedBuilder):
    """Builder that aggregates multiple MIDI datasets into a single dataset."""

    DEFAULT_WRITER_BATCH_SIZE = 1000

    BUILDER_CONFIG_CLASS = MidiDatasetConfig
    BUILDER_CONFIGS = BUILDER_CONFIGS
    DEFAULT_CONFIG_NAME = "default"

    def __init__(
        self,
        train_dataset_paths: List[str] = None,
        validation_dataset_path: str = None,
        num_proc: int = 128,
        **kwargs,
    ):
        """Initialize the MIDI aggregated dataset builder.

        Args:
            train_dataset_paths: List of paths to training datasets on Hugging Face
            validation_dataset_path: Path to validation dataset on Hugging Face
            num_proc: Number of processes for parallel processing
        """
        if train_dataset_paths:
            config = MidiDatasetConfig(
                name="custom",
                train_dataset_paths=train_dataset_paths,
                validation_dataset_path=validation_dataset_path,
                num_proc=num_proc,
            )
            super().__init__(config=config, **kwargs)
        else:
            super().__init__(**kwargs)

    def _info(self) -> DatasetInfo:
        """Return the dataset info."""
        return DatasetInfo(
            description="Aggregated MIDI datasets",
            features=Features(
                {
                    "notes": Value(dtype="string", id=None),  # JSON string of DataFrame
                    "source": Value(dtype="string", id=None),  # Original source info
                    "source_dataset": Value(dtype="string", id=None),  # Dataset path
                    "original_id": Value(dtype="string", id=None),  # Original example ID
                }
            ),
        )

    def _split_generators(
        self,
        dl_manager: DownloadManager,
    ) -> List[SplitGenerator]:
        """Return SplitGenerators with sharding for multiprocessing."""
        train_datasets = []

        # Load all training datasets
        for path in self.config.train_dataset_paths:
            try:
                ds = datasets.load_dataset(path, split="train")
                logger.info(f"Loaded training dataset from {path} with {len(ds)} examples")
                train_datasets.append(ds)
            except Exception as e:
                logger.error(f"Failed to load dataset {path}: {e}")

        if not train_datasets:
            raise ValueError("No valid training datasets found")

        # Concatenate all training datasets
        train_dataset = datasets.concatenate_datasets(train_datasets)
        logger.info(f"Concatenated {len(train_dataset)} examples from {len(train_datasets)} datasets")

        # Load validation and test datasets
        try:
            validation_dataset = datasets.load_dataset(self.config.validation_dataset_path, split="validation")
            logger.info(f"Loaded validation dataset with {len(validation_dataset)} examples")
        except Exception as e:
            logger.error(f"Failed to load validation dataset: {e}")
            validation_dataset = None

        try:
            test_dataset = datasets.load_dataset(self.config.validation_dataset_path, split="test")
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
        for shard_id, shard in enumerate(dataset_shards):
            for example_id, example in enumerate(shard):
                try:
                    # Check if notes and source fields exist
                    if "notes" not in example and "content" not in example:
                        # Try to find the MIDI data field
                        midi_field = None
                        for field in ["notes", "content", "midi", "data"]:
                            if field in example:
                                midi_field = field
                                break

                        if not midi_field:
                            logger.warning(f"Cannot find MIDI data in example: {example.keys()}")
                            continue

                    notes_data = example.get("notes", example.get("content", None))
                    source_data = example.get("source", "{}")

                    # Ensure notes is a JSON string
                    if isinstance(notes_data, dict):
                        notes_data = json.dumps(notes_data)
                    # Ensure source is a JSON string
                    if isinstance(source_data, dict):
                        source_data = json.dumps(source_data)

                    unique_id = f"{shard_id}_{example_id}"

                    record = {
                        "notes": notes_data,
                        "source": source_data,
                        "source_dataset": shard.info.dataset_name if hasattr(shard, "info") else "unknown",
                        "original_id": str(example_id),
                    }

                    yield unique_id, record

                except Exception as e:
                    logger.error(f"Error processing example {example_id} in shard {shard_id}: {e}")
                    continue
