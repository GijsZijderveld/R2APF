from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from .sensing import obstacle_center


def point_to_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    segment = end - start
    denominator = float(np.dot(segment, segment))

    if denominator <= 1e-12:
        return float(np.linalg.norm(point - start))

    fraction = float(np.dot(point - start, segment) / denominator)
    fraction = max(0.0, min(1.0, fraction))
    nearest = start + fraction * segment
    return float(np.linalg.norm(point - nearest))


def segment_is_collision_free(
    start: np.ndarray,
    end: np.ndarray,
    obstacles: Iterable[Any],
    agent_radius: float,
    margin: float = 0.0,
) -> bool:
    clearance = float(agent_radius) + float(margin)

    for obstacle in obstacles:
        distance = point_to_segment_distance(
            obstacle_center(obstacle),
            start,
            end,
        )
        if distance < clearance + float(obstacle.radius):
            return False

    return True


def path_is_collision_free(
    path: Sequence[np.ndarray],
    obstacles: Iterable[Any],
    agent_radius: float,
    start_index: int = 0,
    lookahead_segments: int | None = None,
) -> bool:
    if len(path) < 2:
        return False

    first = max(0, int(start_index))
    last = len(path) - 1
    if lookahead_segments is not None:
        last = min(last, first + int(lookahead_segments))

    for index in range(first, last):
        if not segment_is_collision_free(
            path[index],
            path[index + 1],
            obstacles,
            agent_radius,
        ):
            return False

    return True


def position_is_in_collision(
    position: np.ndarray,
    obstacles: Iterable[Any],
    agent_radius: float,
) -> bool:
    for obstacle in obstacles:
        center_distance = float(
            np.linalg.norm(np.asarray(position, dtype=float) - obstacle_center(obstacle))
        )
        if center_distance < float(agent_radius) + float(obstacle.radius):
            return True
    return False
