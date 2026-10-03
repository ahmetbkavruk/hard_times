"""Passive event traces with deterministic, local transport on a periodic grid.

These are observer data, never resources or policy inputs. All cells read the
same previous state. New deposits become visible after advance(), and can only
spread on later steps. No random generator is needed or consumed.
"""

from math import isfinite
from numbers import Integral

import numpy as np


FIELD_CHANNELS = ("movement", "trade", "violence")


class FieldSystem:
    def __init__(self, size: int, *, decay: float = 0.02, diffusion: float = 0.2):
        if not isinstance(size, Integral) or size < 1:
            raise ValueError("field size must be a positive integer")
        for name, value in (("decay", decay), ("diffusion", diffusion)):
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"field {name} must be finite and between 0 and 1")
        self.size = size
        self.decay = decay
        self.diffusion = diffusion
        self.t = 0
        self._values = np.zeros((len(FIELD_CHANNELS), size, size), dtype=float)
        self._pending = np.zeros_like(self._values)
        self._cumulative_deposited = np.zeros(len(FIELD_CHANNELS), dtype=float)
        self.history = [self._metrics(np.zeros(len(FIELD_CHANNELS)))]

    def deposit(self, channel: str, cell: tuple[int, int], amount: float = 1.0) -> None:
        """Queue a nonnegative event weight at (x, y), without changing the field."""
        if not isfinite(amount) or amount < 0:
            raise ValueError("field deposit must be finite and nonnegative")
        x, y = cell
        if any(not isinstance(value, Integral) or not 0 <= value < self.size for value in cell):
            raise ValueError("field deposit cell must be inside the grid")
        self._pending[FIELD_CHANNELS.index(channel), y, x] += amount

    def advance(self) -> None:
        """Mix four cardinal directions, decay old traces, then add new events.

        F_new = (1-decay) * ((1-diffusion)*F + diffusion*neighbor_mean) + deposits

        The mixing weights form a convex combination, preserving positivity
        and total weight before decay. Wrapped directions retain their weights
        even when they coincide on grids of side one or two.
        """
        previous = self._values
        neighbors = (np.roll(previous, 1, axis=1) + np.roll(previous, -1, axis=1) + np.roll(previous, 1, axis=2) + np.roll(previous, -1, axis=2)) / 4.0
        self._values = (1.0 - self.decay) * ((1.0 - self.diffusion) * previous + self.diffusion * neighbors) + self._pending
        deposited = self._pending.sum(axis=(1, 2))
        self._cumulative_deposited += deposited
        self._pending.fill(0)
        self.t += 1
        self.history.append(self._metrics(deposited))

    def snapshot(self) -> np.ndarray:
        """Return a detached, read-only array indexed by [channel, y, x]."""
        result = self._values.copy()
        result.setflags(write=False)
        return result

    def layer(self, channel: str) -> np.ndarray:
        """Return a detached, read-only [y, x] layer for plotting."""
        result = self._values[FIELD_CHANNELS.index(channel)].copy()
        result.setflags(write=False)
        return result

    def _metrics(self, deposited) -> dict:
        row = {"t": self.t}
        for index, channel in enumerate(FIELD_CHANNELS):
            values = self._values[index]
            total = float(values.sum())
            # Sum of squared spatial shares: 1/(L*L) for a uniform nonzero
            # field, 1 for one occupied cell, and 0 when the field is empty.
            concentration = float(np.square(values / total).sum()) if total else 0.0
            row.update({
                f"{channel}_deposited": float(deposited[index]),
                f"{channel}_cumulative_deposited": float(self._cumulative_deposited[index]),
                f"{channel}_total": total,
                f"{channel}_peak": float(values.max()),
                f"{channel}_concentration": concentration,
            })
        return row
