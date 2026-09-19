"""Social traits and finite bodies; labels are descriptions, never policy inputs."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class Alignment(str, Enum):
    LAWFUL_GOOD = "lawful_good"
    NEUTRAL_GOOD = "neutral_good"
    CHAOTIC_GOOD = "chaotic_good"
    LAWFUL_NEUTRAL = "lawful_neutral"
    TRUE_NEUTRAL = "true_neutral"
    CHAOTIC_NEUTRAL = "chaotic_neutral"
    LAWFUL_EVIL = "lawful_evil"
    NEUTRAL_EVIL = "neutral_evil"
    CHAOTIC_EVIL = "chaotic_evil"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()


ALIGNMENT_COLORS = {
    Alignment.LAWFUL_GOOD: "#53b7ff",
    Alignment.NEUTRAL_GOOD: "#5ce0b5",
    Alignment.CHAOTIC_GOOD: "#e3e86a",
    Alignment.LAWFUL_NEUTRAL: "#b2a2ff",
    Alignment.TRUE_NEUTRAL: "#d4d9e0",
    Alignment.CHAOTIC_NEUTRAL: "#ffb45c",
    Alignment.LAWFUL_EVIL: "#e075c4",
    Alignment.NEUTRAL_EVIL: "#ad8057",
    Alignment.CHAOTIC_EVIL: "#ff555f",
}


def alignment_for(lawfulness: float, concern: float) -> Alignment:
    """Bin each continuous axis at +/- 1/3, including the boundaries as neutral."""
    for value in (lawfulness, concern):
        if not isfinite(value) or not -1 <= value <= 1:
            raise ValueError("social traits must be finite and between -1 and 1")
    order = "lawful" if lawfulness > 1 / 3 else "chaotic" if lawfulness < -1 / 3 else "neutral"
    care = "good" if concern > 1 / 3 else "evil" if concern < -1 / 3 else "neutral"
    return Alignment("true_neutral" if order == care == "neutral" else f"{order}_{care}")


def reflect_trait(value: float) -> float:
    """Reflect mutations into [-1, 1], without clamping mass at the endpoints."""
    folded = (value + 1) % 4
    return folded - 1 if folded <= 2 else 3 - folded


@dataclass(frozen=True)
class Encounter:
    last_trade: int = -1
    last_harm: int = -1
    trades: int = 0

    @property
    def last_event(self) -> int:
        return max(self.last_trade, self.last_harm)


@dataclass
class Carcass:
    agent_id: int
    x: int
    y: int
    edible_energy: float
