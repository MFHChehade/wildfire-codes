# python/analyze_eval_vselect.py
#
# Analyze io/eval_vselect_jdc_jac_per_xi.csv and produce:
#   - Text + CSV summaries of metrics for each model
#   - Pairwise head-to-head comparisons on J_dc and J_ac
#   - Several publication-ready figures (no titles)
#
# Expected columns in eval_vselect_jdc_jac_per_xi.csv:
#   xi
#   J_dc_<model>    for each model (e.g., zero_shot, sft, dpo)
#   J_ac_<model>
#   J_dc_gt         (optimal DC ground truth from mip_opt)
#   J_dc_base       (baseline DC with no extra switching)
#
# Figures are saved under: io/figs_eval/

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[1]
IO_DIR  = ROOT / "io"
CSV_IN  = IO_DIR / "eval_vselect_jdc_jac_per_xi.csv"
FIG_DIR = IO_DIR / "figs_eval"
FIG_DIR.mkdir(exist_ok=True)

SUMMARY_CSV        = IO_DIR / "eval_stats_per_model.csv"
PAIRWISE_CSV       = IO_DIR / "eval_stats_pairwise.csv"
GLOBAL_SUMMARY_CSV = IO_DIR / "eval_stats_global.csv"


# ----------------- helpers for stats ----------------- #

def detect_models(df: pd.DataFrame) -> List[str]:
    """Infer model names from J_dc_* columns, excluding 'gt' and 'base'."""
    models = []
    for col in df.columns:
        if col.startswith("J_dc_") and col not in ("J_dc_gt", "J_dc_base"):
            name = col[len("J_dc_"):]
            models.append(name)
    models = sorted(set(models))
    return models


def basic_global_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Global numbers independent of model: #scenarios, baseline vs gt, etc."""
    out = {}

    out["n_scenarios"] = int(df["xi"].nunique())

    mask_gt   = df["J_dc_gt"].replace([np.inf, -np.inf], np.nan).notna()
    mask_base = df["J_dc_base"].replace([np.inf, -np.inf], np.nan).notna()
    mask_both = mask_gt & mask_base

    if mask_both.any():
        diff  = df.loc[mask_both, "J_dc_base"] - df.loc[mask_both, "J_dc_gt"]
        ratio = df.loc[mask_both, "J_dc_base"] / df.loc[mask_both, "J_dc_gt"]

        out["baseline_minus_opt_mean"]   = float(diff.mean())
        out["baseline_minus_opt_median"] = float(diff.median())
        out["baseline_minus_opt_std"]    = float(diff.std(ddof=1))
        out["baseline_minus_opt_min"]    = float(diff.min())
        out["baseline_minus_opt_max"]    = float(diff.max())

        out["baseline_over_opt_mean"]   = float(ratio.mean())
        out["baseline_over_opt_median"] = float(ratio.median())
    else:
        for k in [
            "baseline_minus_opt_mean", "baseline_minus_opt_median",
            "baseline_minus_opt_std", "baseline_minus_opt_min",
            "baseline_minus_opt_max", "baseline_over_opt_mean",
            "baseline_over_opt_median",
        ]:
            out[k] = np.nan

    return pd.DataFrame([out])


def per_model_stats(df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    """Compute extensive stats per model on J_dc, J_ac, regret, baseline win rate, etc."""
    rows = []

    for m in models:
        col_jdc = f"J_dc_{m}"
        col_jac = f"J_ac_{m}"

        if col_jdc not in df.columns or col_jac not in df.columns:
            continue

        jdc   = df[col_jdc].replace([np.inf, -np.inf], np.nan)
        jac   = df[col_jac].replace([np.inf, -np.inf], np.nan)
        jgt   = df["J_dc_gt"].replace([np.inf, -np.inf], np.nan)
        jbase = df["J_dc_base"].replace([np.inf, -np.inf], np.nan)

        mask_jdc  = jdc.notna()
        mask_jac  = jac.notna()
        mask_gt   = jgt.notna()
        mask_base = jbase.notna()
        mask_all  = mask_jdc & mask_gt

        regret = (jdc - jgt).where(mask_all)
        ratio_to_gt = (jdc / jgt).where(mask_all)

        mask_valid_base  = mask_jdc & mask_base
        better_than_base = (jdc <= jbase).where(mask_valid_base)
        better_or_equal_opt = (jdc <= jgt + 1e-6).where(mask_all)

        eps_list = [0.01, 0.05, 0.10]
        close_fracs = {}
        for eps in eps_list:
            thresh = jgt * (1.0 + eps)
            is_close = (jdc <= thresh).where(mask_all)
            close_fracs[f"frac_within_{int(eps*100)}pct_opt"] = float(is_close.mean()) if mask_all.any() else np.nan

        row = {
            "model": m,
            "n_valid_J_dc": int(mask_jdc.sum()),
            "n_valid_J_ac": int(mask_jac.sum()),

            "J_dc_mean": float(jdc[mask_jdc].mean()) if mask_jdc.any() else np.nan,
            "J_dc_median": float(jdc[mask_jdc].median()) if mask_jdc.any() else np.nan,
            "J_dc_std": float(jdc[mask_jdc].std(ddof=1)) if mask_jdc.sum() > 1 else 0.0,
            "J_dc_min": float(jdc[mask_jdc].min()) if mask_jdc.any() else np.nan,
            "J_dc_max": float(jdc[mask_jdc].max()) if mask_jdc.any() else np.nan,

            "J_ac_mean": float(jac[mask_jac].mean()) if mask_jac.any() else np.nan,
            "J_ac_median": float(jac[mask_jac].median()) if mask_jac.any() else np.nan,
            "J_ac_std": float(jac[mask_jac].std(ddof=1)) if mask_jac.sum() > 1 else 0.0,
            "J_ac_min": float(jac[mask_jac].min()) if mask_jac.any() else np.nan,
            "J_ac_max": float(jac[mask_jac].max()) if mask_jac.any() else np.nan,

            "regret_mean": float(regret[mask_all].mean()) if mask_all.any() else np.nan,
            "regret_median": float(regret[mask_all].median()) if mask_all.any() else np.nan,
            "regret_std": float(regret[mask_all].std(ddof=1)) if mask_all.sum() > 1 else 0.0,
            "regret_min": float(regret[mask_all].min()) if mask_all.any() else np.nan,
            "regret_max": float(regret[mask_all].max()) if mask_all.any() else np.nan,

            "ratio_to_opt_mean": float(ratio_to_gt[mask_all].mean()) if mask_all.any() else np.nan,
            "ratio_to_opt_median": float(ratio_to_gt[mask_all].median()) if mask_all.any() else np.nan,

            "frac_better_than_baseline": float(better_than_base.mean()) if mask_valid_base.any() else np.nan,
            "frac_leq_opt": float(better_or_equal_opt.mean()) if mask_all.any() else np.nan,
        }

        row.update(close_fracs)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("model")


def pairwise_stats(df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    """Pairwise head-to-head comparison on J_dc and J_ac for all model pairs."""
    rows = []

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            col_jdc_a = f"J_dc_{a}"
            col_jdc_b = f"J_dc_{b}"
            col_jac_a = f"J_ac_{a}"
            col_jac_b = f"J_ac_{b}"

            if not (col_jdc_a in df.columns and col_jdc_b in df.columns):
                continue

            ja = df[col_jdc_a].replace([np.inf, -np.inf], np.nan)
            jb = df[col_jdc_b].replace([np.inf, -np.inf], np.nan)
            mask_dc = ja.notna() & jb.notna()

            if mask_dc.any():
                diff_dc = jb - ja
                frac_a_better_dc = float((diff_dc > 0).mean())
                frac_b_better_dc = float((diff_dc < 0).mean())
                frac_tie_dc       = float((diff_dc == 0).mean())
                delta_mean_dc     = float(diff_dc.mean())
                delta_median_dc   = float(diff_dc.median())
            else:
                frac_a_better_dc = frac_b_better_dc = frac_tie_dc = np.nan
                delta_mean_dc = delta_median_dc = np.nan

            if col_jac_a in df.columns and col_jac_b in df.columns:
                va = df[col_jac_a].replace([np.inf, -np.inf], np.nan)
                vb = df[col_jac_b].replace([np.inf, -np.inf], np.nan)
                mask_ac = va.notna() & vb.notna()
                if mask_ac.any():
                    diff_ac = vb - va
                    frac_a_better_ac = float((diff_ac > 0).mean())
                    frac_b_better_ac = float((diff_ac < 0).mean())
                    frac_tie_ac       = float((diff_ac == 0).mean())
                    delta_mean_ac     = float(diff_ac.mean())
                    delta_median_ac   = float(diff_ac.median())
                else:
                    frac_a_better_ac = frac_b_better_ac = frac_tie_ac = np.nan
                    delta_mean_ac = delta_median_ac = np.nan
            else:
                frac_a_better_ac = frac_b_better_ac = frac_tie_ac = np.nan
                delta_mean_ac = delta_median_ac = np.nan
                mask_ac = pd.Series(False, index=df.index)

            rows.append({
                "pair": f"{a}_vs_{b}",
                "model_a": a,
                "model_b": b,
                "n_overlap_dc": int(mask_dc.sum()),
                "frac_a_better_dc": frac_a_better_dc,
                "frac_b_better_dc": frac_b_better_dc,
                "frac_tie_dc": frac_tie_dc,
                "delta_mean_dc(b-a)": delta_mean_dc,
                "delta_median_dc(b-a)": delta_median_dc,
                "n_overlap_ac": int(mask_ac.sum()),
                "frac_a_better_ac": frac_a_better_ac,
                "frac_b_better_ac": frac_b_better_ac,
                "frac_tie_ac": frac_tie_ac,
                "delta_mean_ac(b-a)": delta_mean_ac,
                "delta_median_ac(b-a)": delta_median_ac,
            })

    return pd.DataFrame(rows)


# ----------------- FIGURES ----------------- #

def _add_boxplot_mean_median_legend_and_labels(bp, stats_means, stats_medians):
    """Add legend entries for mean/median and numeric labels for each box."""
    handles = []
    labels  = []

    if "means" in bp and len(bp["means"]) > 0:
        handles.append(bp["means"][0])
        labels.append("mean")

    if "medians" in bp and len(bp["medians"]) > 0:
        handles.append(bp["medians"][0])
        labels.append("median")

    if handles:
        plt.legend(handles, labels)

    positions = np.arange(1, len(stats_means) + 1)
    for x, mean, med in zip(positions, stats_means, stats_medians):
        if np.isfinite(mean):
            plt.text(x + 0.05, mean, f"{mean:.1f}",
                     va="bottom", ha="left", fontsize=7, rotation=90)
        if np.isfinite(med):
            plt.text(x - 0.05, med, f"{med:.1f}",
                     va="top", ha="right", fontsize=7, rotation=90)


def plot_Jdc_boxplot_5(df: pd.DataFrame, models: List[str]) -> None:
    """
    Boxplot of J_dc with 5 boxes:
      baseline, opt, and each model.
    """
    names = ["baseline", "opt"] + models
    data = []
    labels = []
    means = []
    medians = []

    for name in names:
        if name == "baseline":
            arr = df["J_dc_base"].replace([np.inf, -np.inf], np.nan)
        elif name == "opt":
            arr = df["J_dc_gt"].replace([np.inf, -np.inf], np.nan)
        else:
            col = f"J_dc_{name}"
            if col not in df.columns:
                continue
            arr = df[col].replace([np.inf, -np.inf], np.nan)

        arr = arr.dropna()
        if arr.empty:
            continue

        data.append(arr.values)
        labels.append(name)
        means.append(float(arr.mean()))
        medians.append(float(arr.median()))

    if not data:
        return

    plt.figure(figsize=(7, 4))
    bp = plt.boxplot(
        data,
        tick_labels=labels,
        showmeans=True,
        showfliers=False,
    )
    plt.ylabel("J_dc")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    _add_boxplot_mean_median_legend_and_labels(bp, means, medians)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "box_Jdc_5methods.png", dpi=300)
    plt.close()


def plot_Jac_boxplot_models(df: pd.DataFrame, models: List[str]) -> None:
    """
    Boxplot of J_ac for the LLM models only (we don't have J_ac for baseline/opt).
    """
    data = []
    labels = []
    means = []
    medians = []

    for m in models:
        col = f"J_ac_{m}"
        if col not in df.columns:
            continue
        arr = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if arr.empty:
            continue
        data.append(arr.values)
        labels.append(m)
        means.append(float(arr.mean()))
        medians.append(float(arr.median()))

    if not data:
        return

    plt.figure(figsize=(7, 4))
    bp = plt.boxplot(
        data,
        tick_labels=labels,
        showmeans=True,
        showfliers=False,
    )
    plt.ylabel("J_ac")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    _add_boxplot_mean_median_legend_and_labels(bp, means, medians)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "box_Jac_models.png", dpi=300)
    plt.close()


def plot_regret_histograms_all(df: pd.DataFrame, models: List[str]) -> None:
    """
    Histogram of regret J_dc - J_dc_gt for 4 methods:
      baseline + all models, all in one figure.
    """
    methods = ["baseline"] + models
    plt.figure(figsize=(7, 4))

    all_regrets = []
    for name in methods:
        if name == "baseline":
            j = df["J_dc_base"].replace([np.inf, -np.inf], np.nan)
        else:
            col = f"J_dc_{name}"
            if col not in df.columns:
                continue
            j = df[col].replace([np.inf, -np.inf], np.nan)

        jgt = df["J_dc_gt"].replace([np.inf, -np.inf], np.nan)
        mask = j.notna() & jgt.notna()
        if not mask.any():
            continue
        reg = (j - jgt)[mask].values
        all_regrets.append(reg)

    if not all_regrets:
        plt.close()
        return

    concat = np.concatenate(all_regrets)
    bins = np.linspace(concat.min(), concat.max(), 30)
    linestyles = ["solid", "dashed", "dotted", "dashdot", (0, (5, 1))]

    for i, name in enumerate(methods):
        if name == "baseline":
            j = df["J_dc_base"].replace([np.inf, -np.inf], np.nan)
        else:
            col = f"J_dc_{name}"
            if col not in df.columns:
                continue
            j = df[col].replace([np.inf, -np.inf], np.nan)

        jgt = df["J_dc_gt"].replace([np.inf, -np.inf], np.nan)
        mask = j.notna() & jgt.notna()
        if not mask.any():
            continue
        reg = (j - jgt)[mask].values
        ls = linestyles[i % len(linestyles)]
        plt.hist(
            reg,
            bins=bins,
            density=True,
            histtype="step",
            alpha=0.9,
            linestyle=ls,
            label=name,
        )

    plt.xlabel("J_dc - J_dc_gt")
    plt.ylabel("Density")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hist_regret_Jdc_all_methods.png", dpi=300)
    plt.close()


def plot_regret_cdf_all(df: pd.DataFrame, models: List[str]) -> None:
    """
    CDF of regret J_dc - J_dc_gt for 4 methods:
      baseline + all models, with different marker shapes.
    """
    methods = ["baseline"] + models
    markers = ["o", "s", "^", "x", "D", "v"]

    plt.figure(figsize=(7, 4))

    plotted_any = False
    for i, name in enumerate(methods):
        if name == "baseline":
            j = df["J_dc_base"].replace([np.inf, -np.inf], np.nan)
        else:
            col = f"J_dc_{name}"
            if col not in df.columns:
                continue
            j = df[col].replace([np.inf, -np.inf], np.nan)

        jgt = df["J_dc_gt"].replace([np.inf, -np.inf], np.nan)
        mask = j.notna() & jgt.notna()
        if not mask.any():
            continue

        reg = (j - jgt)[mask].values
        reg = np.sort(reg)
        y = np.linspace(0, 1, len(reg), endpoint=False)

        m = markers[i % len(markers)]
        plt.plot(reg, y, marker=m, linestyle="-", linewidth=1.0, markersize=4, label=name)
        plotted_any = True

    if not plotted_any:
        plt.close()
        return

    plt.xlabel("J_dc - J_dc_gt")
    plt.ylabel("CDF")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cdf_regret_Jdc_all_methods_markers.png", dpi=300)
    plt.close()


def plot_Jac_cdf_models(df: pd.DataFrame, models: List[str]) -> None:
    """
    CDF of J_ac for all models, different marker shapes.
    """
    markers = ["o", "s", "^", "x", "D", "v"]
    plt.figure(figsize=(7, 4))

    plotted_any = False
    for i, m in enumerate(models):
        col = f"J_ac_{m}"
        if col not in df.columns:
            continue
        v = df[col].replace([np.inf, -np.inf], np.nan)
        mask = v.notna()
        if not mask.any():
            continue
        arr = np.sort(v[mask].values)
        y = np.linspace(0, 1, len(arr), endpoint=False)
        marker = markers[i % len(markers)]
        plt.plot(arr, y, marker=marker, linestyle="-", linewidth=1.0, markersize=4, label=m)
        plotted_any = True

    if not plotted_any:
        plt.close()
        return

    plt.xlabel("J_ac")
    plt.ylabel("CDF")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cdf_Jac_all_models_markers.png", dpi=300)
    plt.close()


def plot_Jdc_scatter_all_vs_opt(df: pd.DataFrame, models: List[str]) -> None:
    """
    Single scatter: J_dc_gt on x-axis, J_dc_method on y-axis,
    for baseline, opt, and each model (5 methods).
    """
    jgt = df["J_dc_gt"].replace([np.inf, -np.inf], np.nan)

    methods = ["baseline", "opt"] + models
    markers = ["o", "s", "^", "x", "D", "v"]

    plt.figure(figsize=(7, 5))

    for i, name in enumerate(methods):
        if name == "baseline":
            j = df["J_dc_base"].replace([np.inf, -np.inf], np.nan)
        elif name == "opt":
            j = jgt
        else:
            col = f"J_dc_{name}"
            if col not in df.columns:
                continue
            j = df[col].replace([np.inf, -np.inf], np.nan)

        mask = jgt.notna() & j.notna()
        if not mask.any():
            continue

        x = jgt[mask].values
        y = j[mask].values
        marker = markers[i % len(markers)]
        plt.scatter(x, y, marker=marker, alpha=0.7, edgecolors="none", label=name)

    if not plt.gca().collections:
        plt.close()
        return

    mn = np.nanmin(jgt.values)
    mx = np.nanmax(jgt.values)
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.0)

    plt.xlabel("J_dc_gt")
    plt.ylabel("J_dc")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "scatter_Jdc_all_vs_opt.png", dpi=300)
    plt.close()


def plot_per_scenario_Jdc_lines(df: pd.DataFrame, models: List[str]) -> None:
    """
    Per-scenario line plot of J_dc with different markers per method:
      x-axis: scenario index (sorted by J_dc_gt)
      lines: baseline, opt, and each model.
    """
    base_cols  = ["xi", "J_dc_gt", "J_dc_base"]
    model_cols = [f"J_dc_{m}" for m in models if f"J_dc_{m}" in df.columns]
    cols = [c for c in base_cols + model_cols if c in df.columns]
    d = df[cols].drop_duplicates("xi").copy()

    d = d.sort_values("J_dc_gt")
    x = np.arange(len(d))

    methods = ["baseline", "opt"] + models
    markers = ["o", "s", "^", "x", "D", "v"]

    plt.figure(figsize=(7, 4))

    for i, name in enumerate(methods):
        if name == "baseline":
            if "J_dc_base" not in d.columns:
                continue
            y = d["J_dc_base"].values
        elif name == "opt":
            y = d["J_dc_gt"].values
        else:
            col = f"J_dc_{name}"
            if col not in d.columns:
                continue
            y = d[col].values

        marker = markers[i % len(markers)]
        plt.plot(x, y, marker=marker, linewidth=1.0, markersize=4, label=name)

    plt.xlabel("Scenario index (sorted by J_dc_gt)")
    plt.ylabel("J_dc")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "line_Jdc_per_scenario_all_methods.png", dpi=300)
    plt.close()


# ----------------- main ----------------- #

def main():
    if not CSV_IN.exists():
        raise FileNotFoundError(f"Missing input CSV: {CSV_IN}")

    df = pd.read_csv(CSV_IN)

    # Detect models first
    models = detect_models(df)
    print("Detected models:", models)

    # ---- Filter out rows where any J_ac_* is ~1e6 (AC failure sentinel) ----
    jac_cols = [c for c in df.columns if c.startswith("J_ac_")]
    if jac_cols:
        mask_bad = (df[jac_cols] >= 1e6).any(axis=1)
        n_bad = int(mask_bad.sum())
        if n_bad > 0:
            print(f"Filtering out {n_bad} rows due to J_ac >= 1e6 in at least one model.")
        df = df[~mask_bad].reset_index(drop=True)

    # Recompute models (columns unaffected, but length changed)
    models = detect_models(df)
    print("Models after filtering:", models)

    global_stats_df = basic_global_stats(df)
    per_model_df    = per_model_stats(df, models)
    pairwise_df     = pairwise_stats(df, models)

    global_stats_df.to_csv(GLOBAL_SUMMARY_CSV, index=False)
    per_model_df.to_csv(SUMMARY_CSV, index=False)
    pairwise_df.to_csv(PAIRWISE_CSV, index=False)

    print("\n=== Global stats ===")
    print(global_stats_df.to_string(index=False))

    print("\n=== Per-model stats (excerpt) ===")
    cols_to_show = [
        "model",
        "J_dc_mean",
        "J_ac_mean",
        "regret_mean",
        "ratio_to_opt_mean",
        "frac_better_than_baseline",
        "frac_within_1pct_opt",
        "frac_within_5pct_opt",
        "frac_within_10pct_opt",
    ]
    existing_cols = [c for c in cols_to_show if c in per_model_df.columns]
    print(per_model_df[existing_cols].to_string(index=False))

    print("\n=== Pairwise stats (excerpt) ===")
    cols_pair = [
        "pair",
        "n_overlap_dc",
        "frac_a_better_dc",
        "frac_b_better_dc",
        "frac_tie_dc",
        "delta_mean_dc(b-a)",
    ]
    existing_pair_cols = [c for c in cols_pair if c in pairwise_df.columns]
    print(pairwise_df[existing_pair_cols].to_string(index=False))

    # Figures:
    plot_Jdc_boxplot_5(df, models)
    plot_Jac_boxplot_models(df, models)

    plot_regret_histograms_all(df, models)
    plot_regret_cdf_all(df, models)
    plot_Jac_cdf_models(df, models)

    plot_Jdc_scatter_all_vs_opt(df, models)
    plot_per_scenario_Jdc_lines(df, models)

    print(f"\nSaved stats to:\n  {GLOBAL_SUMMARY_CSV}\n  {SUMMARY_CSV}\n  {PAIRWISE_CSV}")
    print(f"Saved figures to:\n  {FIG_DIR}")

if __name__ == "__main__":
    main()
