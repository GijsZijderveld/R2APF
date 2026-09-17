# Planner comparison framework

This package provides the controlled layer for comparing R2APF with other
planning families without modifying the historical RAPF implementation.

## Design

- `RAPFAgentV4Adapter` delegates to `agents/RAPF_Agent_v4.py`.
- `NavigationAgent` provides shared sensing, path validation, replanning, and
  movement for new baseline planners.
- Every baseline implements the small `Planner` protocol and returns a
  `PlanningResult`.
- `runner.py` applies common seeds, stopping conditions, collision checks, and
  metrics.

The adapter uses composition. `RAPF_Agent_v4` does not inherit from a new
class and its source is not changed.

## Comparison levels

The initial campaign is a whole-system comparison: unchanged RAPF v4 versus
baseline planners using the shared baseline agent. A later controlled
planner-only experiment may place an R2APF planner behind `NavigationAgent`,
but that must be reported separately because it is not the historical v4 agent.

## Add a baseline

1. Create a module in `planners/baselines/`.
2. Implement `Planner.plan(...)`.
3. Return `PlanningResult`.
4. Register a factory through `comparison.registry.register_planner`.
5. Add its complete parameters to `configs/planners/`.
6. Run the contract tests and a deterministic smoke campaign.

Keep planner-specific effort counters in `PlanningResult.diagnostics`. Common
claims should use success, executed path length, collisions, planning time, and
replans.
