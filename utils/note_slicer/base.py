"""Base for slicing note data into (notes_first, notes_second) pairs."""

from typing import List, Tuple
from abc import ABC, abstractmethod

import pandas as pd


def shift_note_times(
    notes: pd.DataFrame,
    offset: float,
) -> pd.DataFrame:
    """Shift start/end columns by offset. Returns a new DataFrame."""
    out = notes.copy()
    if "start" in out.columns:
        out["start"] = out["start"] - offset
    if "end" in out.columns:
        out["end"] = out["end"] - offset
    return out


class NoteSlicer(ABC):
    """Splits note data into pairs (notes_first, notes_second) for embedding datasets."""

    @abstractmethod
    def slice_notes(
        self,
        notes: pd.DataFrame,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Split notes into list of (notes_first, notes_second) DataFrame pairs."""
        pass
