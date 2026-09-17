from __future__ import annotations

from typing import Any, Iterable, Tuple

import numpy as np

from .collision import segment_is_collision_free


def move_towards(
    position: np.ndarray,
    target: np.ndarray,
    speed_limit: float,
    dt: float,
    known_obstacles: Iterable[Any],
    agent_radius: float,
) -> Tuple[np.ndarray, bool, bool]:
    """Return (new_position, moved, blocked)."""

    position = np.asarray(position, dtype=float)
    target = np.asarray(target, dtype=float)
    displacement = target - position
    distance = float(np.linalg.norm(displacement))

    if distance <= 0.0:
        return position.copy(), False, False

    step_length = min(distance, float(speed_limit) * float(dt))
    candidate = position + displacement / distance * step_length

    if not segment_is_collision_free(
        position,
        candidate,
        known_obstacles,
        agent_radius,
    ):
        return position.copy(), False, True

    return candidate, True, False
