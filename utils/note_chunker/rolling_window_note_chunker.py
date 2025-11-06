import warnings

import pandas as pd

from utils.note_chunker.base import NoteChunker


class RollingWindowNoteChunker(NoteChunker):
    """Chunk notes using a rolling window approach.

    This chunker splits notes into overlapping or non-overlapping windows
    of a specified size, similar to a rolling window operation.

    Attributes:
        window_size: Number of notes in each chunk.
        step_size: Number of notes to move forward between chunks.
            If equal to window_size, chunks do not overlap.
            If less than window_size, chunks overlap.
    """

    def __init__(
        self,
        window_size: int,
        step_size: int = None,
    ):
        """Initialize the rolling window note chunker.

        Args:
            window_size: Number of notes to include in each chunk.
                Must be positive.
            step_size: Number of notes to advance between chunks.
                Defaults to window_size (no overlap). Must be positive.

        Raises:
            ValueError: If window_size or step_size is not positive.
        """
        if window_size <= 0:
            raise ValueError(
                f"window_size must be positive, got {window_size}",
            )

        if step_size is None:
            step_size = window_size

        if step_size <= 0:
            raise ValueError(
                f"step_size must be positive, got {step_size}",
            )

        self.window_size = window_size
        self.step_size = step_size

    def chunk_notes(
        self,
        notes: pd.DataFrame,
    ) -> list[pd.DataFrame]:
        """Split notes into chunks using a rolling window.

        Args:
            notes: DataFrame containing musical notes.

        Returns:
            List of DataFrames, each containing window_size notes
            (or fewer for the last chunk if there are not enough notes).

        Raises:
            ValueError: If the notes DataFrame is empty.
        """
        if notes.empty:
            warnings.warn(
                "Requested to chunk empty DataFrame - returning empty list",
                UserWarning,
                stacklevel=2,
            )
            return []

        chunks = []
        num_notes = len(notes)

        for start_idx in range(0, num_notes, self.step_size):
            end_idx = min(start_idx + self.window_size, num_notes)
            chunk = notes.iloc[start_idx:end_idx].copy()
            chunks.append(chunk)

            if end_idx >= num_notes:
                break

        return chunks
