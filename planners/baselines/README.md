# Baseline planners

All baseline planners implement the `Planner` protocol from
`comparison/interfaces.py` and return a `PlanningResult`.

## Implemented planners

- `APFPlanner`: deterministic classical attractive/repulsive APF. Local-minimum
  and blocked-path failures are reported rather than hidden by RAPF escape logic.
- `AStarPlanner`: deterministic occupancy-grid A*.
- `DStarLitePlanner`: incremental D* Lite that preserves and repairs search
  state when new obstacles become known.
- `RRTStarPlanner`: seeded two-dimensional RRT* with parent selection and
  rewiring.

A* and D* Lite share `grid.py`, so they use identical resolution,
rasterization, obstacle inflation, connectivity, costs, and diagonal-corner
rules. Shared sensing, path invalidation, movement, collision reporting,
stopping conditions, and metrics remain in `comparison/`.

The common policy is `configs/comparison_policy.yaml`. Planner-specific
parameters are explicit in `configs/planners/`.

## Run a smoke comparison

```bash
python scripts/run_comparison.py \
  --planner astar \
  --planner-config configs/planners/astar.json \
  --episodes 1 \
  --output experiments/comparisons/astar_smoke.csv
```

Replace `astar` with `apf`, `dstar_lite`, or `rrt_star` and select the
matching configuration file.

Run contract tests with:

```bash
python -m unittest discover -s tests -v
```

Do not compare planner-specific counters directly. An A* expansion, an RRT*
sample, and an R2APF potential evaluation are not equivalent. Use success,
executed path length, collisions, wall-clock planning time, and replans as the
common comparison metrics.
