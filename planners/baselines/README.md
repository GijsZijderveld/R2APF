# Baseline planners

Each comparison planner should implement the `Planner` protocol from
`comparison/interfaces.py` and return a `PlanningResult`.

Planned baselines:

1. Classical APF — contribution baseline.
2. A* or Theta* — recognizable graph-search baseline.
3. D* Lite — incremental replanning baseline for newly discovered obstacles.
4. RRT* — sampling-based baseline.

The planner modules should contain planning logic only. Shared sensing,
path invalidation, movement, collision checks, stopping conditions, and metrics
belong in `comparison/`.

Register a planner without changing the benchmark runner:

```python
from comparison.registry import register_planner
from planners.baselines.astar import AStarPlanner

register_planner("astar", lambda config: AStarPlanner(**config))
```

Do not compare planner-specific counters directly. For example, an A* node
expansion, an RRT* sample, and an RAPF potential evaluation are not equivalent.
Store those values as diagnostics and use success, path length, collisions,
wall-clock planning time, and replans as the common comparison metrics.
