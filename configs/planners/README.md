# Planner configurations

The shared experimental choices live in `../comparison_policy.yaml`. That
file is the single source of truth for the historical R2APF profile, the
controlled-comparison contract, parameter rationale, and unresolved decisions.

Keep one configuration file per planner for parameters unique to that planner.
Every value used for a published experiment should be explicit here, including
parameters that otherwise have defaults in code. Do not duplicate common
parameters here with different values.

The existing `r2apf.json` records the top-level values from
`RAPF_V4_PARAMS`. It does not replace or modify those defaults.

Future grid planners should record only their planner-specific search settings
here; shared grid resolution, connectivity, obstacle rasterization, footprint
inflation, and boundary rules belong in the comparison policy. RRT* should
record its algorithm-specific sampling settings here, while its common budget
and random-seed policy remain in the comparison policy.
