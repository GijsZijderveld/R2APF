# Controlled planner comparisons

Generated comparison results belong in this directory. Do not overwrite the
preserved historical tables in `experiments/benchmarks/`.

Start with a short R2APF smoke campaign:

```bash
python -m scripts.run_comparison \
  --planner r2apf \
  --planner-config configs/planners/r2apf.json \
  --episodes 1 \
  --output experiments/comparisons/r2apf_smoke.csv
```

The default full configuration covers five scenarios and 500 seeds per
scenario. Do not treat smoke-test results as publication results.

To finish a partial campaign, point `--output` at its existing CSV and select
all planners:

```bash
python -m scripts.run_comparison --planner all \
  --output experiments/comparisons/results.csv
```

The runner skips each planner/scenario/seed combination already present in
that CSV and adds missing rows. Use the same campaign and planner settings
as the existing rows. The CSV does not record the settings used for each row.

Summarize the combined raw results with:

```bash
python scripts/summarize_comparison.py \
  experiments/comparisons/results.csv
```

This writes `summary_overall.csv`, `summary_by_scenario.csv`, and
`failure_reasons.csv` under `experiments/comparisons/summary/`. Executed path
length is reported both across all episodes and across successful episodes
only. Unsupported planner-specific diagnostics remain blank.
