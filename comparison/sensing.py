from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np


def obstacle_key(obstacle: Any) -> Tuple[float, float, float]:
    center = obstacle_center(obstacle)
    return float(center[0]), float(center[1]), float(obstacle.radius)


def obstacle_center(obstacle: Any) -> np.ndarray:
    if hasattr(obstacle, "x") and hasattr(obstacle, "y"):
        return np.array([obstacle.x, obstacle.y], dtype=float)
    return np.asarray(obstacle.center, dtype=float)


def reveal_obstacles(
    position: np.ndarray,
    obstacles: Iterable[Any],
    sensing_range: float,
    sensing_pad: float = 0.0,
) -> List[Any]:
    """Reveal a full circular obstacle when its surface enters sensing range."""

    visible: List[Any] = []
    position = np.asarray(position, dtype=float)
    threshold = float(sensing_range) + float(sensing_pad)

    for obstacle in obstacles:
        distance_to_surface = (
            float(np.linalg.norm(position - obstacle_center(obstacle)))
            - float(obstacle.radius)
        )
        if distance_to_surface < threshold:
            visible.append(obstacle)

    return visible


def merge_known_obstacles(
    known: Sequence[Any],
    newly_visible: Sequence[Any],
) -> List[Any]:
    merged = list(known)
    keys = {obstacle_key(obstacle) for obstacle in merged}

    for obstacle in newly_visible:
        key = obstacle_key(obstacle)
        if key not in keys:
            merged.append(obstacle)
            keys.add(key)

    return merged
