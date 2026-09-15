from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from typing import List, Tuple


@dataclass
class EnvConfig:
    name: str
    map_size: Tuple[float, float]
    start_positions: List[List[float]]
    goal: List[float]
    goal_radius: float
    obstacles: List[dict]

def make_s0_start_pinned() -> EnvConfig:
    """
    S0 — Exactly one agent is trapped right at the start (small U-shaped pocket),
    the others spawn just outside and should head to the goal.
    """
    return EnvConfig(
        name="S0_start_pinned",
        map_size=(30, 30),
        # First agent spawns inside the pocket at (2,2); others are outside it
        start_positions=[[2.0, 2.0], [2.8, 2.2], [2.6, 1.9], [2.9, 2.7]],
        goal=[28, 28],
        goal_radius=0.5,
        obstacles=[
            # A small U-shaped “pen” around (2,2) that’s too tight to escape
            {"type": "wall", "p0": [1.7, 1.7], "p1": [2.3, 1.7], "thickness": 0.6},  # bottom
            {"type": "wall", "p0": [1.7, 1.7], "p1": [1.7, 2.6], "thickness": 0.6},  # left
            {"type": "wall", "p0": [2.3, 1.7], "p1": [2.3, 2.6], "thickness": 0.6},  # right
            # leave the top slightly open, but narrower than robot + safety margin
            {"type": "rock", "center": [2.0, 2.6], "radius": 0.45},  # makes the opening effectively impassable
        ],
    )


def make_s1_split() -> EnvConfig:
    """Scenario S1 — corner trap that forces a split."""

    return EnvConfig(
        name="S1_split_corner_trap",
        map_size=(30.0, 30.0),
        start_positions=[[2, 2], [2.3, 2.1], [2.1, 2.3], [1.9, 2.2]],
        goal=[28, 28],
        goal_radius=0.5,
        obstacles=[
            {"type": "rock", "center": [14.5, 14.5], "radius": 1.0},
            {"type": "rock", "center": [15.8, 14.6], "radius": 0.9},
            {"type": "rock", "center": [14.9, 15.9], "radius": 0.9},
        ],
    )


def make_s2_funnel() -> EnvConfig:
    """Scenario S2 — funnel with a side gap that encourages following free neighbors."""

    return EnvConfig(
        name="S2_funnel_follow",
        map_size=(30.0, 30.0),
        start_positions=[[2, 2], [2.2, 2.1], [2.1, 2.2], [1.9, 2.2], [2.3, 2.0]],
        goal=[28, 28],
        goal_radius=0.5,
        obstacles=[
            {"type": "wall", "p0": [10, 12], "p1": [20, 12], "thickness": 0.6},
            {"type": "wall", "p0": [10, 18], "p1": [20, 18], "thickness": 0.6},
            {"type": "rock", "center": [21.0, 15.0], "radius": 0.8},
        ],
    )


def make_s3_empty() -> EnvConfig:
    """Scenario S3 — empty map for regression testing."""

    return EnvConfig(
        name="S3_empty",
        map_size=(30.0, 30.0),
        start_positions=[[2, 2], [2.2, 2.1], [2.1, 2.2], [1.9, 2.2], [2.3, 2.0]],
        goal=[28, 28],
        goal_radius=0.5,
        obstacles=[],
    )


def save_env_config(cfg: EnvConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


def load_env_config(path: str) -> EnvConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return EnvConfig(**data)
