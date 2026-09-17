# Controlled planner comparisons

Generated comparison results belong in this directory. Do not overwrite the
preserved historical tables in `experiments/benchmarks/`.

Start with a short R2APF smoke campaign:

```bash
python scripts/run_comparison.py \
  --planner r2apf \
  --planner-config configs/planners/r2apf.json \
  --episodes 1 \
  --output experiments/comparisons/r2apf_smoke.csv
```

The default full configuration covers five scenarios and 500 seeds per
scenario. Do not treat smoke-test results as publication results.
