"""Rolling-window slicer: first n notes, then next n notes, with configurable step."""

import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

from utils.note_slicer.base import NoteSlicer, shift_note_times


class RollingWindowNoteSlicer(NoteSlicer):
    """
    Slices notes into (notes_first, notes_second) with a rolling window.
    window_size notes per segment; window advances by step_size.
    """

    def __init__(
        self,
        window_size: int,
        step_size: int | None = None,
    ) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if step_size is None:
            step_size = window_size
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        self.window_size = window_size
        self.step_size = step_size

    def slice_notes(
        self,
        notes: pd.DataFrame,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Produce (notes_first, notes_second) pairs. Stops when < 2*window_size notes. Notes in time order (start)."""
        if notes.empty:
            warnings.warn(
                "Requested to slice empty DataFrame - returning empty list",
                UserWarning,
                stacklevel=2,
            )
            return []
        n = len(notes)
        required = 2 * self.window_size
        if n < required:
            return []
        start_arr = np.asarray(notes["start"], dtype=np.float64)
        pairs: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        for start in range(0, n - required + 1, self.step_size):
            end_first = start + self.window_size
            end_second = end_first + self.window_size
            t0 = float(start_arr[start])
            dt = float(start_arr[end_first] - start_arr[end_first - 1])
            first = notes.iloc[start:end_first].copy()
            second = notes.iloc[end_first:end_second].copy()
            first = shift_note_times(first, t0)
            second = shift_note_times(
                second,
                float(start_arr[end_first]) - dt,
            )
            pairs.append((first, second))
        return pairs
