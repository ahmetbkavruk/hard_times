"""Action values exchanged between policies and the simulation engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionKind(str, Enum):
    """The small action vocabulary supported by the baseline world."""

    EAT = "eat"
    HARVEST = "harvest"
    MOVE = "move"
    REPRODUCE = "reproduce"


@dataclass(frozen=True)
class Action:
    """A policy decision; the world remains responsible for its consequences."""

    kind: ActionKind
    resource: int | None = None

