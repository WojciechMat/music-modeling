from typing import List
from abc import ABC, abstractmethod

import pandas as pd


class NoteChunker(ABC):
    """Abstract base class for chunking musical notes into fragments.

    This class defines the interface for splitting a DataFrame of musical notes
    into multiple smaller DataFrames based on different chunking strategies.
    """

    @abstractmethod
    def chunk_notes(
        self,
        notes: pd.DataFrame,
    ) -> List[pd.DataFrame]:
        """Split notes DataFrame into a list of note chunks.

        Args:
            notes: DataFrame containing musical notes with their properties.
                Expected to contain columns relevant to musical notation
                (e.g., pitch, start_time, duration, etc.).

        Returns:
            List of DataFrames, where each DataFrame represents a chunk
            of notes from the original input.

        Raises:
            ValueError: If the notes DataFrame is empty or invalid.
        """
        pass
