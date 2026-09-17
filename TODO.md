# TODO

This file records issues found while extracting the R2APF code from `swarm_apf_sim`. They are intentionally **not fixed in the initial extraction**, because the first goal is to preserve the behavior of the thesis implementation.

## Reproducibility and entry points

- [ ] Add a small automated smoke test for the documented demonstration.
- [ ] Run the new comparison contract tests and R2APF smoke campaign in a clean checkout.
- [ ] Record the Python and dependency versions used for the final paper experiments.
- [ ] Add deterministic reference outputs for selected seeds and scenarios.
- [ ] Confirm that all scenarios supported by `make_config` work with the extracted demonstration.
- [ ] Decide whether the paper experiments require additional source files beyond the current single-agent execution path.
- [ ] Establish and document how the Optuna-tuned RAPF v3 parameters map to `RAPF_V4_PARAMS` and `RAPFGlobalPlanner`.
- [ ] Identify the exact command, commit, and parameter set that generated each preserved sensing benchmark table.
- [ ] Replace the broken root `main.py` from the old repository only after defining the intended public command-line interface. It was not copied because it references modules that are no longer present.

## Parameters and behavior to review

- [ ] Review the different collision/safety radii used by the agent, planner, and benchmark code. Do not harmonize them without rerunning and documenting the experiments.
- [ ] Confirm which parameter set represents the final thesis and revised-paper configuration.
- [ ] Document the sensing assumptions: position, time, orientation/heading, sensing range, field of view, and obstacle knowledge.
- [ ] Document the exact meaning and units of all planner and agent parameters.
- [ ] Check hard-coded visualization limits and annotation positions for scenarios other than the current demonstration.

## Naming and structure

- [ ] Decide whether the public method name should consistently be `R2APF`, `RAPF v4`, or another paper-defined name.
- [ ] Rename versioned modules and classes only in a separate, behavior-checked refactor.
- [ ] Clarify the relationship between `rapf_global_planner.py` and `rapf_paper_planner.py`.
- [ ] Add a proper Python package configuration after the stable public API is decided.
- [ ] Remove unused imports and dead/commented code only after tests protect current behavior.

## Benchmarking

- [ ] Correct ambiguous benchmark labels, including code that refers to an improved planner as `oldRAPF`, before publishing new comparison results.
- [x] Separate core runtime dependencies from optional Optuna/tuning dependencies.
- [ ] Replace hard-coded analysis input paths with command-line arguments after the historical baseline is protected by tests.
- [ ] Decide which of `sensing_v4.csv`, `sensing_v4_withvo.csv`, `sensing_v4_manual.csv`, and `sensing_v4_oldRAPF.csv` are publication results versus intermediate runs.
- [ ] Add classical APF and the original RAPF configuration as contribution/ablation baselines.
- [ ] Add D* Lite and RRT* as comparison planners.
- [ ] Add non-invasive planner-only timing for the unchanged RAPF v4 agent; the adapter currently reports no RAPF planning time.
- [ ] Use wall-clock time and collision checks for comparisons across planner families; do not compare unlike iteration definitions directly.
- [ ] Define common sensing, execution, collision checking, stopping criteria, random seeds, and computation budgets.
- [ ] Report success rate, executed path length, planning time, replans, collision/failure rate, and distance travelled before failure.

## Project metadata

- [ ] Select and add an open-source license.
- [ ] Add `CITATION.cff` once the preferred citation and publication details are final.
- [ ] Archive a release and add its DOI after the publication experiment baseline is frozen.
