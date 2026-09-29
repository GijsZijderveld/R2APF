# Planner configurations

The shared experimental choices live in `../comparison_policy.yaml`. That
file is the single source of truth for the historical RAPF and R2APF profiles, the
controlled-comparison contract, parameter rationale, and unresolved decisions.

Keep one configuration file per planner for parameters unique to that planner.
Every value used for a published experiment should be explicit here, including
parameters that otherwise have defaults in code. Do not duplicate common
parameters here with different values.

The `rapf.json` and `r2apf.json` files record the same top-level values from
`NAVIGATION_PARAMS`. The controlled comparison changes only the selected planner;
these files do not replace or modify the agent defaults.

Future grid planners should record only their planner-specific search settings
here; shared grid resolution, connectivity, obstacle rasterization, footprint
inflation, and boundary rules belong in the comparison policy. RRT* should
record its algorithm-specific sampling settings here, while its common budget
and random-seed policy remain in the comparison policy.

For RRT*, `max_samples` is the maximum number of attempted samples in one
planner call. `optimization_samples` is the number of additional attempts after
the first collision-free goal connection, limited by `max_samples`. Set it to
zero to recover first-solution stopping, or to `null` to use the entire sample
budget even after reaching the goal. The configured 600 additional samples
were selected after a preliminary sample-budget sensitivity study. Calls are
still capped at `max_samples`, so fewer than 600 attempts remain when the
first path appears late. Report both executed route length and cumulative
planning time when assessing this setting. Existing comparison CSVs
were produced with the old first-solution implementation and cannot be used
as results for this version of RRT*.
