# python/plot_sweep_gaps_v2.py
#
# What this does:
# - Finds the latest io/sweep_report_*.json
# - Builds a tidy table across scenarios × K
# - Computes THREE metrics per (scenario, K):
#     1) gap_to_best_pct           = 100 * (J_inf - J_mip) / J_mip
#     2) improve_over_baseline_pct = 100 * (J_base - J_inf) / J_base
#     3) relative_gap_pct          = 100 * (J_inf - J_mip) / (J_base - J_mip)
#        (fraction of the baseline→optimal gap that still remains; bounded in [0, 100] if J lies between)
# - Plots each metric vs K with per-scenario scatter points + mean line with std error bars
# - Writes a per-scenario CSV and an aggregated-by-K CSV, plus three PNG figures.
#
# Usage:
#   python python/plot_sweep_gaps_v2.py
#
# Requirements: pandas, matplotlib

import json
import math
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
IO_DIR = ROOT / "io"


def find_latest_sweep(io_dir: Path) -> Path:
    files = list(io_dir.glob("sweep_report_*.json"))
    if not files:
        raise FileNotFoundError(f"No sweep_report_*.json found in {io_dir}")
    return max(files, key=lambda p: p.stat().st_mtime)


def load_records(sweep_path: Path) -> pd.DataFrame:
    with open(sweep_path, "r") as f:
        data = json.load(f)

    rows = []
    scenarios = data.get("scenarios", [])
    for s_idx, scen in enumerate(scenarios):
        gt = scen.get("ground_truth", {})
        J_base = gt.get("baseline", {}).get("J", float("nan"))
        J_mip  = gt.get("mip_opt",  {}).get("J", float("nan"))

        for blok in scen.get("by_N_CANDS", []):
            K = blok.get("N_CANDS")
            best = blok.get("best_result", {}) or {}
            J_inf = best.get("J", float("nan"))

            # skip incomplete rows
            if any(math.isnan(x) for x in [J_inf, J_mip, J_base]) or K is None:
                continue

            # metric 1: gap to best (% of J_mip)
            gap_to_best_pct = float("inf")
            if J_mip != 0:
                gap_to_best_pct = 100.0 * (J_inf - J_mip) / J_mip

            # metric 2: improvement over baseline (% of J_base)
            improve_over_baseline_pct = float("inf")
            if J_base != 0:
                improve_over_baseline_pct = 100.0 * (J_base - J_inf) / J_base

            # metric 3: relative gap (% of baseline->opt gap that remains)
            # (J_inf - J_mip) / (J_base - J_mip)
            denom = (J_base - J_mip)
            relative_gap_pct = float("inf")
            if denom != 0:
                relative_gap_pct = 100.0 * (J_inf - J_mip) / denom

            rows.append({
                "scenario_idx": int(s_idx),
                "K": int(K),
                "J_baseline": J_base,
                "J_mip": J_mip,
                "J_inf": J_inf,
                "gap_to_best_pct": gap_to_best_pct,
                "improve_over_baseline_pct": improve_over_baseline_pct,
                "relative_gap_pct": relative_gap_pct,
            })

    if not rows:
        raise ValueError("No valid (scenario, K) records parsed from sweep report.")

    df = pd.DataFrame(rows)
    # Sort for nice outputs
    df = df.sort_values(["scenario_idx", "K"]).reset_index(drop=True)
    return df


def aggregate_by_K(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("K")
          .agg(
              n=("relative_gap_pct", "count"),
              gap_to_best_pct_mean=("gap_to_best_pct", "mean"),
              gap_to_best_pct_std=("gap_to_best_pct", "std"),
              improve_over_baseline_pct_mean=("improve_over_baseline_pct", "mean"),
              improve_over_baseline_pct_std=("improve_over_baseline_pct", "std"),
              relative_gap_pct_mean=("relative_gap_pct", "mean"),
              relative_gap_pct_std=("relative_gap_pct", "std"),
          )
          .reset_index()
          .sort_values("K")
    )
    # Replace NaN std (n=1) with 0 for plotting
    for c in ["gap_to_best_pct_std", "improve_over_baseline_pct_std", "relative_gap_pct_std"]:
        agg[c] = agg[c].fillna(0.0)
    return agg


def plot_metric_vs_K(df: pd.DataFrame, agg: pd.DataFrame, metric: str, ylabel: str, title: str, out_png: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Per-scenario scatter (light)
    for k, sub in df.groupby("K"):
        ax.scatter([k]*len(sub), sub[metric], alpha=0.4, label=None)

    # Mean with error bars (std)
    mean_col = f"{metric}_mean"
    std_col  = f"{metric}_std"
    ax.errorbar(
        agg["K"], agg[mean_col],
        yerr=agg[std_col],
        fmt="-o", capsize=4
    )

    ax.set_xlabel("Number of candidates K")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    latest = find_latest_sweep(IO_DIR)
    print(f"Using latest sweep report: {latest.name}")

    df = load_records(latest)
    agg = aggregate_by_K(df)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = IO_DIR / f"sweep_metrics_{ts}.csv"
    out_agg = IO_DIR / f"sweep_metrics_agg_{ts}.csv"
    df.to_csv(out_csv, index=False)
    agg.to_csv(out_agg, index=False)
    print(f"Saved per-scenario metrics: {out_csv}")
    print(f"Saved aggregated-by-K metrics: {out_agg}")

    # Mapping from metric name in df to labels and filenames
    specs = [
        (
            "gap_to_best_pct",
            "Gap to best (%)\n$(J_{inf}-J_{mip})/J_{mip}\\times 100$",
            "Gap to Best vs. K",
            IO_DIR / f"plot_gap_to_best_vs_K_{ts}.png",
        ),
        (
            "improve_over_baseline_pct",
            "Improvement over baseline (%)\n$(J_{base}-J_{inf})/J_{base}\\times 100$",
            "Improvement over Baseline vs. K",
            IO_DIR / f"plot_improve_over_baseline_vs_K_{ts}.png",
        ),
        (
            "relative_gap_pct",
            "Relative gap (%)\n$(J_{inf}-J_{mip})/(J_{base}-J_{mip})\\times 100$",
            "Relative Gap (share of remaining gap) vs. K",
            IO_DIR / f"plot_relative_gap_vs_K_{ts}.png",
        ),
    ]

    # Rename agg columns to match the plotting helper expectations
    ren = {
        "gap_to_best_pct_mean": "gap_to_best_pct_mean",
        "gap_to_best_pct_std": "gap_to_best_pct_std",
        "improve_over_baseline_pct_mean": "improve_over_baseline_pct_mean",
        "improve_over_baseline_pct_std": "improve_over_baseline_pct_std",
        "relative_gap_pct_mean": "relative_gap_pct_mean",
        "relative_gap_pct_std": "relative_gap_pct_std",
    }
    # (Already correctly named; kept for clarity)

    for metric, ylabel, title, out_png in specs:
        plot_metric_vs_K(
            df=df,
            agg=agg.rename(columns=ren),
            metric=metric,
            ylabel=ylabel,
            title=title,
            out_png=out_png
        )
        print(f"Saved plot: {out_png}")

    print("Done.")


if __name__ == "__main__":
    main()
