# Planner comparison framework

This package provides the controlled layer for comparing R2APF with other
planning families with one shared navigation and measurement layer.

## Comparison policy

`configs/comparison_policy.yaml` is the single source of truth for physical,
sensing, execution, fairness, and reporting choices. It contains two separate
profiles:

- `historical_r2apf` records the values actually used by the preserved RAPF
  implementation, including conflicting values that belong to different
  historical purposes.
- `controlled_comparison` defines the common contract that all planners must
  obey in the new comparison.

Do not silently replace inherited historical values with controlled-comparison
values. Parameters marked `open` or `provisional` must be resolved through
the publication gate before the full experiment campaign.

## Design

- `NavigationAgent` provides the same sensing, path validation, replanning,
  recovery, movement, and goal logic for every planner.
- `RAPFPlannerAdapter` exposes RAPF and R2APF without changing their planning
  algorithms.
- Every other planner implements the small `Planner` protocol and returns a
  `PlanningResult`.
- `runner.py` applies common seeds, stopping conditions, collision checks, and
  metrics.

The shared navigation behavior is regression-tested against the preserved
historical runs. The final campaign is therefore a planner comparison: all
methods use the same agent-level behavior and differ only in planning logic.

## Add a baseline

1. Read and follow `configs/comparison_policy.yaml`.
2. Create a module in `planners/baselines/`.
3. Implement `Planner.plan(...)`.
4. Return `PlanningResult`.
5. Register a factory through `comparison.registry.register_planner`.
6. Add its complete parameters to `configs/planners/`.
7. Run the contract tests and a deterministic smoke campaign.

Keep planner-specific effort counters in `PlanningResult.diagnostics`. Common
claims should use success, executed path length, collisions, planning time, and
replans.
