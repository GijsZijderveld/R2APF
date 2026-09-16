import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Scenario mapping
SCENARIO_MAP = {
    "A": 1,
    "D": 2,
    "B": 3,
    "E": 4,
    "C": 5,
}

DEFAULT_STATS_COLUMNS = [
    "Success",
    "PathLength",
    "UniqueVirtSeen",
    "PlanCalls",
    "PlanningEffortTotal",
]


# ---------------------------
# LOAD / PREP
# ---------------------------

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)

    if "Scenario" not in df.columns:
        raise ValueError("CSV must contain a 'Scenario' column")

    df["ScenarioNum"] = df["Scenario"].map(SCENARIO_MAP)
    df = df.dropna(subset=["ScenarioNum"]).copy()
    df["ScenarioNum"] = df["ScenarioNum"].astype(int)

    if "Success" in df.columns:
        df["Success"] = df["Success"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )

    if "FailReason" in df.columns:
        df["FailReason"] = df["FailReason"].fillna("None").astype(str)

    return df


def filter_success(df, success_only):
    if not success_only:
        return df

    if "Success" not in df.columns:
        raise ValueError("CSV must contain 'Success' column")

    return df[df["Success"] == True].copy()


# ---------------------------
# PLOT COMPARISON
# ---------------------------

def plot_comparison_boxplot(df_old, df_new, column, save_path=None, hide_fliers=False):
    df_old = df_old.copy()
    df_new = df_new.copy()

    df_old["Version"] = "Old"
    df_new["Version"] = "New"

    df = pd.concat([df_old, df_new], ignore_index=True)

    plot_df = df[["ScenarioNum", "Version", column]].dropna()

    data = []
    labels = []

    for scenario in [1, 2, 3, 4, 5]:
        for version in ["Old", "New"]:
            vals = plot_df[
                (plot_df["ScenarioNum"] == scenario)
                & (plot_df["Version"] == version)
            ][column].values

            if len(vals) > 0:
                data.append(vals)
                labels.append(f"{scenario}_{version}")

    plt.figure(figsize=(10, 4.5))
    plt.boxplot(data, tick_labels=labels, showfliers=not hide_fliers)
    plt.title(f"{column} (Old vs New RAPF)")
    plt.ylabel(column)
    plt.yscale("log")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


# ---------------------------
# STATS
# ---------------------------

def print_stats(df, columns):
    print("\n=== OVERALL ===")
    for col in columns:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) == 0:
            continue

        print(f"{col}: mean={series.mean():.4f}, std={series.std():.4f}")

    print("\n=== PER SCENARIO ===")
    grouped = df.groupby("ScenarioNum")

    for scenario, group in grouped:
        print(f"\nScenario {scenario}")
        for col in columns:
            if col not in group.columns:
                continue

            series = pd.to_numeric(group[col], errors="coerce").dropna()
            if len(series) == 0:
                continue

            print(f"  {col}: mean={series.mean():.4f}, std={series.std():.4f}")


def print_single_column_stats(df_old, df_new, column):
    print(f"\n=== {column} COMPARISON ===")

    for name, df in [("Old", df_old), ("New", df_new)]:
        if column not in df.columns:
            continue

        series = pd.to_numeric(df[column], errors="coerce").dropna()

        print(f"\n{name}:")
        print(f"  mean={series.mean():.4f}")
        print(f"  std={series.std():.4f}")
        print(f"  min={series.min():.4f}")
        print(f"  max={series.max():.4f}")


# ---------------------------
# FAILURE TABLE
# ---------------------------

def print_failure_table(df, label):
    if "Success" not in df.columns or "FailReason" not in df.columns:
        print(f"{label}: No failure data")
        return

    print(f"\n=== FAILURE TABLE ({label}) ===")

    failed = df[df["Success"] == False]

    table = (
        failed.groupby(["ScenarioNum", "FailReason"])
        .size()
        .unstack(fill_value=0)
    )

    print(table)


# ---------------------------
# MAIN
# ---------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--column", help="Column to plot")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--single-column", help="Stats for ONE column only")
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--save", help="Save plot")
    parser.add_argument("--hide-fliers", action="store_true")
    parser.add_argument("--failure-table", action="store_true")

    args = parser.parse_args()

    # HARD CODED FILES
    old_path = "experiments/benchmarks/sensing_v4_oldRAPF.csv"
    new_path = "experiments/benchmarks/sensing_v4_manual.csv"

    df_old = load_and_prepare(old_path)
    df_new = load_and_prepare(new_path)

    df_old = filter_success(df_old, args.success_only)
    df_new = filter_success(df_new, args.success_only)

    # ---- PLOT ----
    if args.column:
        plot_comparison_boxplot(
            df_old, df_new, args.column, args.save, args.hide_fliers
        )

    # ---- STATS ----
    if args.stats:
        print("\n=== OLD RAPF ===")
        print_stats(df_old, DEFAULT_STATS_COLUMNS)
        print("\n=== NEW RAPF ===")
        print_stats(df_new, DEFAULT_STATS_COLUMNS)

    # ---- SINGLE COLUMN ----
    if args.single_column:
        print_single_column_stats(df_old, df_new, args.single_column)

    # ---- FAILURE ----
    if args.failure_table:
        print_failure_table(df_old, "OLD")
        print_failure_table(df_new, "NEW")


if __name__ == "__main__":
    main()