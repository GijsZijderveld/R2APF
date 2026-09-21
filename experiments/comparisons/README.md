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

Summarize the combined raw results with:

```bash
python scripts/summarize_comparison.py \
  experiments/comparisons/results.csv
```

This writes `summary_overall.csv`, `summary_by_scenario.csv`, and
`failure_reasons.csv` under `experiments/comparisons/summary/`. Executed path
length is reported both across all episodes and across successful episodes
only. Unsupported planner-specific diagnostics remain blank.
