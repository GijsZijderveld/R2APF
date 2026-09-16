import pandas as pd

# Hardcoded paths
OLD_PATH = "experiments/benchmarks/sensing_v4_oldRAPF.csv"
NEW_PATH = "experiments/benchmarks/sensing_v4_manual.csv"


def load(csv_path):
    df = pd.read_csv(csv_path)

    if "BlockedSteps" not in df.columns:
        raise ValueError(f"{csv_path} missing 'BlockedSteps' column")

    return df


def analyze(df, name):
    print(f"\n==============================")
    print(f"   {name}")
    print(f"==============================")

    total_runs = len(df)
    total_blocked_steps = df["BlockedSteps"].sum()

    blocked_df = df[df["BlockedSteps"] > 0]
    num_blocked_envs = len(blocked_df)

    print(f"Total runs: {total_runs}")
    print(f"Envs with blocked steps: {num_blocked_envs}")
    print(f"Total blocked steps: {total_blocked_steps}")

    if num_blocked_envs == 0:
        print("No blocked steps found 🎉")
        return

    print("\n--- Affected environments ---")

    cols = []
    if "Scenario" in df.columns:
        cols.append("Scenario")
    if "Seed" in df.columns:
        cols.append("Seed")
    cols.append("BlockedSteps")

    print(blocked_df[cols].to_string(index=False))

    # Per scenario summary
    if "Scenario" in df.columns:
        print("\n--- Per Scenario ---")
        summary = blocked_df.groupby("Scenario").agg(
            env_count=("BlockedSteps", "size"),
            total_blocked=("BlockedSteps", "sum"),
        )
        print(summary)


def main():
    df_old = load(OLD_PATH)
    df_new = load(NEW_PATH)

    analyze(df_old, "OLD RAPF")
    analyze(df_new, "NEW RAPF")


if __name__ == "__main__":
    main()