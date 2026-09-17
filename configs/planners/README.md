# Planner configurations

Keep one configuration file per planner. Every value used for a published
experiment should be explicit here, including parameters that otherwise have
defaults in code.

The existing `r2apf.json` records the top-level values from
`RAPF_V4_PARAMS`. It does not replace or modify those defaults.

Future grid planners should record grid resolution, connectivity, obstacle
rasterization, and footprint inflation. RRT* should record its sampling or
time budget and random seed policy.
