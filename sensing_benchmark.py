import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Mapping from letters to your paper numbering
SCENARIO_MAP = {
    "A": 1,
    "D": 2,
    "B": 3,
    "E": 4,
    "C": 5,
}

DEFAULT_STATS_COLUMNS = [
    "PlanFails",
    "Replans",
    "PathLength",
    "MaxActiveVirt",
    "SmoothnessRad",
    "ComputeTime",
    "BacktrackTries",
]


def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)

    if "Scenario" not in df.columns:
        raise ValueError("CSV must contain a 'Scenario' column")

    df["ScenarioNum"] = df["Scenario"].map(SCENARIO_MAP)
    df = df.dropna(subset=["ScenarioNum"]).copy()
    df["ScenarioNum"] = df["ScenarioNum"].astype(int)

    # Normalize Success column if needed
    if "Success" in df.columns:
        if df["Success"].dtype != bool:
            df["Success"] = df["Success"].astype(str).str.strip().str.lower().map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                    "yes": True,
                    "no": False,
                }
            )

    # Normalize FailReason a bit
    if "FailReason" in df.columns:
        df["FailReason"] = df["FailReason"].fillna("None").astype(str).str.strip()
        df.loc[df["FailReason"] == "", "FailReason"] = "None"

    return df


def filter_success(df, success_only):
    if not success_only:
        return df

    if "Success" not in df.columns:
        raise ValueError("CSV must contain 'Success' column for --success-only")

    return df[df["Success"] == True].copy()


def plot_boxplot(df, column, save_path=None):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in CSV")

    plot_df = df[["ScenarioNum", column]].dropna().copy()

    data = []
    labels = []
    for scenario in [1, 2, 3, 4, 5]:
        vals = plot_df.loc[plot_df["ScenarioNum"] == scenario, column].dropna().values
        if len(vals) > 0:
            data.append(vals)
            labels.append(str(scenario))

    if not data:
        raise ValueError(f"No valid data found for column '{column}'")

    plt.figure(figsize=(7, 4.5))
    plt.boxplot(data, tick_labels=labels)
    plt.title(f"{column} per Scenario")
    plt.xlabel("Scenario")
    plt.yscale("log")
    plt.ylabel(column)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_histogram_by_scenario(df, column, save_path=None, bins=10, scenario=None, xlabel=None, ylabel=None):
    """Plot a histogram / count plot of `column` for each scenario (1..5).

    If the column is numeric, a histogram is drawn. Otherwise categorical counts
    are plotted (bar chart of value counts).
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in CSV")

    scenarios = [1, 2, 3, 4, 5]
    if scenario is not None:
        if scenario not in scenarios:
            raise ValueError(f"Scenario must be one of {scenarios}")
        scenarios = [scenario]

    n = len(scenarios)
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
    if n == 1:
        axs = [axs]

    for ax, sc in zip(axs, scenarios):
        vals = df.loc[df["ScenarioNum"] == sc, column].dropna()
        if vals.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(f"Scenario {sc}")
            ax.set_xlabel(column)
            continue

        # Detect numeric values
        numeric_vals = pd.to_numeric(vals, errors="coerce")
        if numeric_vals.notna().all():
            ax.hist(numeric_vals.values, bins=bins, color="C0", edgecolor="black")
        else:
            counts = vals.astype(str).value_counts().sort_index()
            ax.bar(counts.index.astype(str), counts.values, color="C0", edgecolor="black")
            ax.tick_params(axis="x", rotation=45)

        ax.set_title(f"Scenario {sc}")
        # If a shared xlabel was provided, avoid per-axis x labels
        if xlabel is None:
            ax.set_xlabel(column)

        # Use log scale on y axis and avoid zero lower bound
        try:
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.5)
        except Exception:
            # If matplotlib version doesn't support nonpositive handling, ignore
            pass

    # Shared y-label or default per-axis label
    if ylabel is not None:
        # prefer modern supylabel API, fall back to fig.text
        # position slightly more to the left for readability
        try:
            fig.supylabel(ylabel, x=0.01)
        except Exception:
            fig.text(0.02, 0.5, ylabel, va="center", rotation="vertical")
    else:
        axs[0].set_ylabel("Episodes")

    # Shared x-label if requested
    if xlabel is not None:
        try:
            fig.supxlabel(xlabel)
        except Exception:
            fig.text(0.5, 0.01, xlabel, ha="center")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved histogram to {save_path}")
    else:
        plt.show()


def print_stats(df, columns):
    print("\n=== OVERALL STATISTICS ===")
    for col in columns:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) == 0:
            continue

        mean = series.mean()
        std = series.std()

        print(f"{col}: mean={mean:.4f}, std={std:.4f}")

    print("\n=== PER SCENARIO STATISTICS ===")
    grouped = df.groupby("ScenarioNum")

    for scenario, group in grouped:
        print(f"\nScenario {scenario}")
        for col in columns:
            if col not in group.columns:
                continue

            series = pd.to_numeric(group[col], errors="coerce").dropna()
            if len(series) == 0:
                continue

            mean = series.mean()
            std = series.std()

            print(f"  {col}: mean={mean:.4f}, std={std:.4f}")


def print_success_ratio(df):
    if "Success" not in df.columns:
        raise ValueError("CSV must contain 'Success' column")

    print("\n=== SUCCESS RATIO ===")

    overall = df["Success"].mean()
    total = len(df)
    total_success = int(df["Success"].sum())

    print(f"Overall: {total_success}/{total} = {overall:.4f}")

    print("\nPer Scenario:")
    grouped = df.groupby("ScenarioNum")["Success"]

    for scenario, group in grouped:
        n = len(group)
        s = int(group.sum())
        rate = group.mean()
        print(f"  Scenario {scenario}: {s}/{n} = {rate:.4f}")


def make_failure_table(df, include_percentages=True):
    if "Success" not in df.columns:
        raise ValueError("CSV must contain 'Success' column")
    if "FailReason" not in df.columns:
        raise ValueError("CSV must contain 'FailReason' column")

    scenarios = sorted(df["ScenarioNum"].dropna().unique())

    # Only failed runs for failure-mode counts
    failed_df = df[df["Success"] == False].copy()

    # Count failure modes per scenario
    if len(failed_df) > 0:
        fail_counts = (
            failed_df.groupby(["ScenarioNum", "FailReason"])
            .size()
            .unstack(fill_value=0)
        )
    else:
        fail_counts = pd.DataFrame(index=scenarios)

    # Basic summary
    summary = df.groupby("ScenarioNum").agg(
        TotalRuns=("Success", "size"),
        Successes=("Success", "sum"),
    )
    summary["Failures"] = summary["TotalRuns"] - summary["Successes"]
    summary["SuccessRate"] = summary["Successes"] / summary["TotalRuns"]

    # Merge summary + fail counts
    table = summary.join(fail_counts, how="left").fillna(0)

    # Make integer columns neat
    for col in table.columns:
        if col != "SuccessRate":
            table[col] = table[col].astype(int)

    # Optional percentage columns per failure mode
    if include_percentages and len(fail_counts.columns) > 0:
        for reason in fail_counts.columns:
            pct_col = f"{reason}_PctOfRuns"
            table[pct_col] = table[reason] / table["TotalRuns"]

    return table.reset_index()


def print_failure_table(df):
    table = make_failure_table(df, include_percentages=False)

    print("\n=== FAILURE MODE TABLE ===")
    print(table.to_string(index=False))


def save_failure_table_csv(df, csv_path):
    table = make_failure_table(df, include_percentages=True)
    table.to_csv(csv_path, index=False)
    print(f"Saved failure table to {csv_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument("--column", help="Column to plot")
    parser.add_argument("--histogram", action="store_true", help="Plot histogram/count per scenario for the given column")
    parser.add_argument("--bins", type=int, default=30, help="Number of bins for numeric histograms")
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Which scenario to plot (1-5 or letter A/E/B/C/D). If omitted, plots all scenarios side-by-side.",
    )
    parser.add_argument("--save", help="Save plot to file instead of showing")
    parser.add_argument("--stats", action="store_true", help="Print statistics")
    parser.add_argument("--success-only", action="store_true", help="Use only successful runs")
    parser.add_argument("--all-numeric", action="store_true", help="Use all numeric columns for stats")
    parser.add_argument("--failure-table", action="store_true", help="Print failure mode table")
    parser.add_argument("--failure-csv", help="Save failure mode table to CSV")
    parser.add_argument("--xlabel", help="Shared x-axis label for histogram plots (applies across subplots)")
    parser.add_argument("--ylabel", help="Shared y-axis label for histogram plots (applies across subplots)")

    args = parser.parse_args()

    df = load_and_prepare(args.csv)

    # Plot/stats can optionally use only successful runs
    analysis_df = filter_success(df, args.success_only)

    # Parse scenario argument for histogram mode (accept letter or number)
    parsed_scenario = None
    if args.scenario:
        s = str(args.scenario).strip()
        # try numeric
        try:
            si = int(s)
            parsed_scenario = si
        except Exception:
            # try letter map
            up = s.upper()
            if up in SCENARIO_MAP:
                parsed_scenario = SCENARIO_MAP[up]
            else:
                parser.error("--scenario must be 1-5 or one of: " + ",".join(SCENARIO_MAP.keys()))

    if args.column:
        if args.histogram:
            plot_histogram_by_scenario(
                analysis_df,
                args.column,
                args.save,
                bins=args.bins,
                scenario=parsed_scenario,
                xlabel=args.xlabel,
                ylabel=args.ylabel,
            )
        else:
            plot_boxplot(analysis_df, args.column, args.save)
    elif args.histogram:
        parser.error("--histogram requires --column to be specified")

    if args.stats:
        print_success_ratio(df)

        if args.all_numeric:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != "ScenarioNum"]
            print_stats(analysis_df, numeric_cols)
        else:
            print_stats(analysis_df, DEFAULT_STATS_COLUMNS)

    if args.failure_table:
        print_failure_table(df)

    if args.failure_csv:
        save_failure_table_csv(df, args.failure_csv)


if __name__ == "__main__":
    main()