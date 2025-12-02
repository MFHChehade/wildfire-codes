# python/run_sweeps.py
# Sweep runner + plotting:
# - Generates/loads multiple PSPS scenarios (contingency masks)
# - For each scenario, evaluates several N_CANDS settings (e.g., 1,4,8,12,16)
# - Plans are "open-only", picking specific (corridor, line)
# - Picks the best feasible plan (min J) per N_CANDS
# - Runs ground-truth (baseline, mip_opt) once per scenario
# - Writes a single JSON report AND immediately produces summary CSVs + 3 plots:
#     1) Gap to best (%) vs K
#     2) Improvement over baseline (%) vs K
#     3) Relative gap (%) vs K

import os, json, time, random, subprocess, math
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt

# ============================
# Paths / constants
# ============================
ROOT       = Path(__file__).resolve().parents[1]
MATLAB_DIR = ROOT / "matlab"
CONFIG     = ROOT / "config"
IO_DIR     = ROOT / "io"
IO_DIR.mkdir(exist_ok=True)

CASE          = os.getenv("CASE_NAME", "case118")
LIMITS_YML    = (CONFIG / "limits.yml").as_posix()
CORRIDORS_F   = CONFIG / "corridor_map.json"

# Sweep controls
SEED              = int(os.getenv("SEED", "7"))
NUM_SCENARIOS     = int(os.getenv("NUM_SCENARIOS", "5"))               # how many PSPS scenarios to evaluate (if not using existing)
USE_EXISTING_PSPS = bool(int(os.getenv("USE_EXISTING_PSPS", "0")))     # 1 => use all existing io/psps_*.json
CANDS_LIST_ENV    = os.getenv("CANDS_LIST", "1,4,8,12,16")             # comma-separated
ONLY_IMPACTED     = bool(int(os.getenv("ONLY_IMPACTED", "1")))         # restrict actions to impacted corridors
K_OPEN_PER_SCEN   = int(os.getenv("K_OPEN_PER_SCEN", "2"))             # forced-open count when sampling PSPS

random.seed(SEED)

# ============================
# MATLAB helpers
# ============================
def _run_matlab_batch(batch: str) -> int:
    cmd = f"matlab -batch \"addpath('{MATLAB_DIR.as_posix()}'); {batch} ;\""
    return subprocess.run(cmd, shell=True).returncode

def get_case_counts() -> Dict[str, int]:
    out = (IO_DIR / f"counts_{int(time.time())}.json").as_posix()
    batch = (
        f"mpc=rundcpf('{CASE}'); "
        f"s=struct('case','{CASE}','buses',size(mpc.bus,1),'lines',size(mpc.branch,1)); "
        f"fid=fopen('{out}','w'); fwrite(fid,jsonencode(s)); fclose(fid); exit"
    )
    rc = _run_matlab_batch(batch)
    if rc != 0:
        raise RuntimeError("MATLAB failed while computing case counts.")
    with open(out, "r") as f:
        return json.load(f)

def call_make_summary(xi_path: Path, out_summary: Path) -> Dict:
    batch = f"make_summary('{CASE}','{xi_path.as_posix()}','{LIMITS_YML}','{out_summary.as_posix()}'); exit"
    rc = _run_matlab_batch(batch)
    if rc != 0:
        raise RuntimeError("MATLAB make_summary failed")
    with open(out_summary, "r") as f:
        return json.load(f)

def call_verify_plan(plan_path: Path, xi_path: Path, out_path: Path) -> Dict:
    batch = (
        f"verify_plan('{CASE}','{plan_path.as_posix()}','{xi_path.as_posix()}',"
        f"'{LIMITS_YML}','{out_path.as_posix()}'); exit"
    )
    rc = _run_matlab_batch(batch)
    if rc != 0:
        return {"feasible": False, "J": 1e18, "shed_MW": None, "notes": "MATLAB error"}
    with open(out_path, "r") as f:
        return json.load(f)

def call_ground_truth(mode: str, xi_path: Path, out_path: Path) -> Dict:
    if mode not in ("baseline", "mip_opt"):
        raise ValueError("mode must be 'baseline' or 'mip_opt'")
    batch = (
        f"run_ground_truth('{CASE}','{xi_path.as_posix()}','{LIMITS_YML}','{out_path.as_posix()}','{mode}'); exit"
    )
    rc = _run_matlab_batch(batch)
    if rc != 0:
        raise RuntimeError(f"MATLAB ground truth failed for mode={mode}")
    with open(out_path, "r") as f:
        return json.load(f)

# ============================
# PSPS & corridors
# ============================
def load_corridor_map() -> Dict[str, List[int]]:
    with open(CORRIDORS_F, "r") as f:
        return json.load(f)

def sample_psps_mask(n_line: int, corrmap: Dict[str, List[int]], k_open: int = 2) -> List[int]:
    """Create PSPS mask: 1=available, 0=forced open; choose k_open from corridor union."""
    xi = [1] * n_line
    union_ids = sorted({int(i) for ids in corrmap.values() for i in ids})
    random.shuffle(union_ids)
    for idx in union_ids[:k_open]:
        if 1 <= idx <= n_line:
            xi[idx - 1] = 0
    return xi

def list_existing_psps(io_dir: Path) -> List[Path]:
    files = list(io_dir.glob("psps_*.json"))
    return sorted(files, key=lambda p: p.stat().st_mtime)

def save_json(path: Path, obj: Any):
    with open(path, "w") as f:
        json.dump(obj, f)

# ============================
# Planning (open-only)
# ============================
def first_eligible_line(summary: Dict, corridor_name: str) -> Optional[int]:
    """Pick the smallest eligible line ID for the corridor."""
    imp = summary.get("impacted_corridors", {})
    if corridor_name not in imp:
        return None
    elig = imp[corridor_name].get("eligible_to_open_lines", [])
    if not elig:
        return None
    try:
        elig_sorted = sorted(int(x) for x in elig)
    except Exception:
        return None
    return elig_sorted[0] if elig_sorted else None

def corridors_pool(summary: Dict, corrmap: Dict[str, List[int]]) -> List[str]:
    impacted_names = list(summary.get("impacted_corridors", {}).keys())
    if ONLY_IMPACTED and impacted_names:
        return impacted_names
    return list(corrmap.keys())

def build_plan_from_corridors(summary: Dict, selected_corridors: List[str]) -> Dict:
    """
    Build a plan by opening the 'first eligible' line in each selected corridor.
    If a corridor has no eligible line, it is skipped.
    """
    actions = []
    seen = set()
    for name in selected_corridors:
        if name in seen:
            continue
        line_id = first_eligible_line(summary, name)
        if line_id is None:
            continue
        actions.append({"name": name, "action": "open", "line": int(line_id)})
        seen.add(name)
    return {"corridor_actions": actions}

def enumerate_candidate_plans(summary: Dict, corrmap: Dict[str, List[int]], n_cands: int) -> List[Dict]:
    """
    Produce up to n_cands open-only plans.
    Each plan selects up to 'toggle_budget' distinct corridors from the pool
    and opens the first eligible line in each.
    """
    budget = int(summary.get("toggle_budget", 3))
    pool = corridors_pool(summary, corrmap)

    plans: List[Dict] = []
    for _ in range(n_cands):
        tmp = pool[:]
        random.shuffle(tmp)
        m = random.randint(0, min(budget, len(tmp)))  # 0..budget actions
        sel = tmp[:m]
        plan = build_plan_from_corridors(summary, sel)
        plans.append(plan)
    return plans

# ============================
# One scenario runner
# ============================
def run_scenario(xi_path: Path, n_cands_values: List[int]) -> Dict:
    """
    For a given PSPS scenario (xi_path):
      - build summary
      - run ground-truth once
      - for each n_cands in list: enumerate plans, verify all, keep best
    Returns a structured dict with all results.
    """
    # Summary
    summary_path = IO_DIR / f"summary_{xi_path.stem}.json"
    summary = call_make_summary(xi_path, summary_path)

    # Corridors
    corrmap = load_corridor_map()

    # Ground truth (once per scenario)
    gt_baseline_out = IO_DIR / f"gt_baseline_{xi_path.stem}_{int(time.time())}.json"
    gt_mip_out      = IO_DIR / f"gt_mip_{xi_path.stem}_{int(time.time())}.json"
    ground_truth = {
        "baseline": call_ground_truth("baseline", xi_path, gt_baseline_out),
        "mip_opt":  call_ground_truth("mip_opt",  xi_path, gt_mip_out),
    }

    # For each N_CANDS value, select best feasible plan
    per_ncands_results = []
    for n_cands in n_cands_values:
        plans = enumerate_candidate_plans(summary, corrmap, n_cands)
        best = {"J": 1e18, "plan": None, "result": None}
        for i, p in enumerate(plans):
            plan_path = IO_DIR / f"plan_{xi_path.stem}_K{n_cands}_{i}.json"
            out_json  = IO_DIR / f"result_{plan_path.stem}.json"
            save_json(plan_path, p)
            res = call_verify_plan(plan_path, xi_path, out_json)
            if res.get("feasible") and res.get("J", 1e18) < best["J"]:
                best = {"J": res["J"], "plan": p, "result": res}
        per_ncands_results.append({
            "N_CANDS": n_cands,
            "candidate_count": len(plans),
            "best_plan": best["plan"],
            "best_result": best["result"],
        })

    return {
        "psps_file": xi_path.as_posix(),
        "summary": {
            "psps_forced_open_count": summary.get("psps_forced_open_count"),
            "impacted_corridors": list(summary.get("impacted_corridors", {}).keys()),
            "toggle_budget": summary.get("toggle_budget"),
        },
        "ground_truth": ground_truth,
        "by_N_CANDS": per_ncands_results,
    }

# ============================
# Plotting helpers (run on the just-created report)
# ============================
def load_records_from_report(report_path: Path) -> pd.DataFrame:
    with open(report_path, "r") as f:
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

            if any(math.isnan(x) for x in [J_inf, J_mip, J_base]) or K is None:
                continue

            gap_to_best_pct = 100.0 * (J_inf - J_mip) / J_mip if J_mip != 0 else float("inf")
            improve_over_baseline_pct = 100.0 * (J_base - J_inf) / J_base if J_base != 0 else float("inf")
            denom = (J_base - J_mip)
            relative_gap_pct = 100.0 * (J_inf - J_mip) / denom if denom != 0 else float("inf")

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

    df = pd.DataFrame(rows).sort_values(["scenario_idx", "K"]).reset_index(drop=True)
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
    for c in ["gap_to_best_pct_std", "improve_over_baseline_pct_std", "relative_gap_pct_std"]:
        agg[c] = agg[c].fillna(0.0)
    return agg

def plot_metric_vs_K(df: pd.DataFrame, agg: pd.DataFrame, metric: str, ylabel: str, title: str, out_png: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    # per-scenario scatter
    for k, sub in df.groupby("K"):
        ax.scatter([k]*len(sub), sub[metric], alpha=0.4, label=None)

    # mean +/- std
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

def generate_plots(report_path: Path):
    """Create CSVs and three plots from a specific sweep report."""
    # Try to reuse the timestamp in the report filename if present
    try:
        stamp = report_path.stem.split("_")[-1]
        int(stamp)
        ts = stamp
    except Exception:
        ts = time.strftime("%Y%m%d_%H%M%S")

    df  = load_records_from_report(report_path)
    agg = aggregate_by_K(df)

    out_csv = IO_DIR / f"sweep_metrics_{ts}.csv"
    out_agg = IO_DIR / f"sweep_metrics_agg_{ts}.csv"
    df.to_csv(out_csv, index=False)
    agg.to_csv(out_agg, index=False)

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
    for metric, ylabel, title, out_png in specs:
        plot_metric_vs_K(df, agg, metric, ylabel, title, out_png)

    print(f"[plots] wrote: {out_csv.name}, {out_agg.name}, and 3 PNGs with suffix {ts}")

# ============================
# Main
# ============================
def main():
    counts = get_case_counts()
    n_line = int(counts["lines"])
    corrmap_all = load_corridor_map()

    # Parse CANDS list
    try:
        n_cands_values = [int(x.strip()) for x in CANDS_LIST_ENV.split(",") if x.strip()]
        if not n_cands_values:
            n_cands_values = [1,4,8,12,16]
    except Exception:
        n_cands_values = [1,4,8,12,16]

    # Build scenarios (PSPS masks)
    scenarios: List[Path] = []
    if USE_EXISTING_PSPS:
        scenarios = list_existing_psps(IO_DIR)
        if not scenarios:
            raise FileNotFoundError("USE_EXISTING_PSPS=1 but no psps_*.json found in io/")
    else:
        # Sample NUM_SCENARIOS masks
        for _ in range(NUM_SCENARIOS):
            xi = sample_psps_mask(n_line, corrmap_all, k_open=K_OPEN_PER_SCEN)
            xi_path = IO_DIR / f"psps_{int(time.time())}_{random.randint(0,9999)}.json"
            save_json(xi_path, xi)
            scenarios.append(xi_path)

    # Run each scenario
    sweep_results = []
    for xi_path in scenarios:
        scenario_res = run_scenario(xi_path, n_cands_values)
        sweep_results.append(scenario_res)

    # Final report
    report = {
        "case": counts["case"],
        "counts": {"buses": counts["buses"], "lines": counts["lines"]},
        "settings": {
            "ONLY_IMPACTED": ONLY_IMPACTED,
            "NUM_SCENARIOS": NUM_SCENARIOS if not USE_EXISTING_PSPS else len(scenarios),
            "USE_EXISTING_PSPS": USE_EXISTING_PSPS,
            "K_OPEN_PER_SCEN": K_OPEN_PER_SCEN,
            "CANDS_LIST": n_cands_values,
            "SEED": SEED,
        },
        "scenarios": sweep_results,
        "notes": "All plans are open-only. Best = min-J feasible per N_CANDS per scenario. Ground truth computed once per scenario.",
    }

    # Save and print report
    out_path = IO_DIR / f"sweep_report_{int(time.time())}.json"
    save_json(out_path, report)
    print(json.dumps(report, indent=2))

    # Generate plots directly from this report
    generate_plots(out_path)


if __name__ == "__main__":
    main()
