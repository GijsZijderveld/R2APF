# RAPF tuning provenance

This directory contains the final retained Optuna tuning artifact from the original development repository:

- `rapf_tune_v3.py`
- `data/rapf_tuning_official.db`

The database is the original SQLite file and may contain studies from multiple stages of RAPF development.

## Important version relationship

`rapf_tune_v3.py` belongs to the historical tuning workflow and does not tune
the complete shared navigation loop directly.

The later navigation layer introduced sensing, path tracking, and reactive
replanning around `planners/rapf_global_planner.py`. The retained Optuna files
therefore provide tuning history and parameter provenance, not a direct
end-to-end tuning pipeline.

## Inspect the database

Install the experiment dependencies and use Optuna from the repository root. The tuning script defaults to:

```text
sqlite:///experiments/optuna/data/rapf_tuning_official.db
```

Its default study name is:

```text
rapf_local_repair_v5
```

A read-only dashboard can optionally be started after installing `optuna-dashboard`:

```bash
optuna-dashboard sqlite:///experiments/optuna/data/rapf_tuning_official.db
```

Do not continue or modify the historical study database without first making a copy.
