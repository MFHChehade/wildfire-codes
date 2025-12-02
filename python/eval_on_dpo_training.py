# Evaluate SFT vs DPO on a few training prompts from io/dpo_openai_prefs.jsonl
# - Queries each model once per prompt
# - Scores with MATLAB verify_plan_hybrid (AC PF) to compare V_pen & report J_dc
# - Saves: io/nb_eval_training_comp.csv (side-by-side results)

import os, json, re, math, random, subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
from openai import OpenAI

# -------------------- Config --------------------
ROOT        = Path(__file__).resolve().parents[0]   # put this file in repo root; adjust if needed
CONFIG      = ROOT / "config"
IO_DIR      = ROOT / "io"
MATLAB_DIR  = ROOT / "matlab"

OPENAI_PREFS_PATH = IO_DIR / "dpo_openai_prefs.jsonl"
LIMITS_YML        = CONFIG / "limits.yml"
CASE_NAME         = os.getenv("CASE_NAME", "case118")

FT_SFT_FILE = CONFIG / "ft_model_sft.txt"
FT_DPO_FILE = CONFIG / "ft_model_dpo.txt"
SFT_MODEL   = FT_SFT_FILE.read_text(encoding="utf-8").strip() if FT_SFT_FILE.exists() else ""
DPO_MODEL   = FT_DPO_FILE.read_text(encoding="utf-8").strip() if FT_DPO_FILE.exists() else ""

API_KEY = os.getenv("OPENAI_API_KEY") or ""
assert API_KEY, "Set OPENAI_API_KEY in your environment."

# sampling
N_SAMPLES = 5
SEED      = 7
MAX_TOKENS = 120
TEMP       = 0.2

# -------------------- LLM prompt grammar --------------------
SYSTEM_MSG = (
    "You are a grid switching assistant.\n"
    "Input: a JSON summary of a PSPS scenario.\n"
    "Output: one line with at most 'toggle_budget' OPEN actions (no other text).\n"
    "Grammar:\n"
    "  - open(Sk:LINE)   e.g., open(S6:135)\n"
    "  - open(LINE)      e.g., open(131)\n"
    "Separate multiple actions with semicolons. No explanations."
)

OPEN_RE = re.compile(r"open\s*\(\s*(?:([sS]\d+)\s*:\s*)?([0-9]+)\s*\)", re.IGNORECASE)

def _clamp(lid: int, n_line: int) -> int:
    return max(1, min(int(lid), n_line))

def parse_actions_to_plan(text: str, toggle_budget: int, n_line: int) -> Dict[str, Any]:
    acts, seen = [], set()
    for (maybe_name, line_id) in OPEN_RE.findall(text or ""):
        lid = _clamp(int(line_id), n_line)
        name = (maybe_name.upper() if maybe_name else "")
        key = (name, lid)
        if key in seen:
            continue
        seen.add(key)
        item = {"action": "open", "line": lid}
        if name:
            item["name"] = name
        acts.append(item)
        if len(acts) >= max(1, int(toggle_budget)):
            break
    return {"corridor_actions": acts}

def plan_to_text(plan: Dict[str, Any]) -> str:
    parts = []
    for a in plan.get("corridor_actions", []):
        lid = int(a["line"])
        name = (a.get("name") or "").strip()
        parts.append(f"open({name+':' if name else ''}{lid})")
    return "; ".join(parts)

# -------------------- MATLAB bridge --------------------
def call_matlab_verify(plan_json: Path, xi_json: Path, out_json: Path) -> Optional[Dict[str, Any]]:
    cmd = (
        f"matlab -batch \"addpath('{MATLAB_DIR.as_posix()}'); "
        f"verify_plan_hybrid('{CASE_NAME}','{plan_json.as_posix()}','{xi_json.as_posix()}','{LIMITS_YML.as_posix()}','{out_json.as_posix()}'); "
        f"exit;\""
    )
    rc = subprocess.run(cmd, shell=True).returncode
    if rc != 0 or (not out_json.exists()):
        return None
    try:
        return json.loads(out_json.read_text(encoding="utf-8"))
    except Exception:
        return None

def v_eff(sc: Dict[str, Any]) -> float:
    v = sc.get("V_pen")
    if isinstance(v, (int, float)) and math.isfinite(v): return float(v)
    j = sc.get("J_ac", float("inf"))
    return float(j) if isinstance(j, (int, float)) and math.isfinite(j) else float("inf")

def infer_toggle_budget(summary: Dict[str, Any]) -> int:
    return int(summary.get("toggle_budget", 3))

def infer_n_line(summary: Dict[str, Any]) -> int:
    return int(summary.get("lines") or summary.get("n_line") or 999999)

def find_xi_path_in_summary(summary: Dict[str, Any]) -> Optional[Path]:
    # If your make_summary included xi path, we’ll find it here.
    for key in ("xi_file", "xi_path", "xi_json"):
        val = summary.get(key)
        if isinstance(val, str) and (val.endswith(".json") or val.endswith(".gz")):
            p = Path(val)
            if p.exists():
                return p
            if (IO_DIR / p.name).exists():
                return IO_DIR / p.name
    return None

# If your summaries *don’t* carry an xi path, map a field to an xi file here:
MANUAL_XI_MAP = {
    # e.g., "scenario_000123": IO_DIR / "xi_scenario_000123.json",
}

def resolve_xi(summary: dict) -> Optional[Path]:
    xi = find_xi_path_in_summary(summary)
    if xi: return xi
    for k in ("scenario_id", "id", "xi_name"):
        if k in summary and summary[k] in MANUAL_XI_MAP:
            return MANUAL_XI_MAP[summary[k]]
    return None

# -------------------- OpenAI --------------------
client = OpenAI(api_key=API_KEY)

def sample_model_output(model: str, prompt_json_str: str, max_tokens: int = 120, temp: float = 0.2) -> str:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_MSG},
                  {"role": "user",   "content": prompt_json_str}],
        temperature=temp,
        max_tokens=max_tokens,
        n=1,
    )
    return (r.choices[0].message.content or "").strip()

# -------------------- Main --------------------
def main():
    assert OPENAI_PREFS_PATH.exists(), f"Missing prefs file: {OPENAI_PREFS_PATH}"
    print("SFT_MODEL:", SFT_MODEL or "(unset)")
    print("DPO_MODEL:", DPO_MODEL or "(unset)")
    assert SFT_MODEL and DPO_MODEL and SFT_MODEL.startswith("ft:") and DPO_MODEL.startswith("ft:"), \
        "Set valid ft:* SFT/DPO model IDs in config files."

    lines = [ln for ln in OPENAI_PREFS_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print("Total training prefs lines:", len(lines))
    random.seed(SEED)
    idxs = sorted(random.sample(range(len(lines)), k=min(N_SAMPLES, len(lines))))
    print("Evaluating indices:", idxs)

    records = []
    for i in idxs:
        obj = json.loads(lines[i])
        # user content should be a JSON summary string
        msgs = obj["input"]["messages"]
        user_msg = next(m for m in msgs if m["role"] == "user")
        prompt_str = user_msg["content"]
        try:
            summary = json.loads(prompt_str)
        except Exception:
            print("Skip: user content is not valid JSON at index", i)
            continue
        records.append({"i": i, "summary": summary, "prompt_str": prompt_str})

    rows = []
    skipped = 0

    for rec in records:
        summary    = dict(rec["summary"])
        prompt_str = rec["prompt_str"]

        xi_path = resolve_xi(summary)
        if xi_path is None:
            print("Skipping (cannot resolve xi_json) at prefs idx", rec["i"])
            skipped += 1
            continue

        toggle_budget = infer_toggle_budget(summary)
        n_line        = infer_n_line(summary)

        # one sample from each model
        outs = {}
        for label, mid in (("sft", SFT_MODEL), ("dpo", DPO_MODEL)):
            text = sample_model_output(mid, prompt_str, MAX_TOKENS, TEMP)
            plan = parse_actions_to_plan(text, toggle_budget, n_line)
            outs[label] = (text, plan)

        # score
        for label in ("sft", "dpo"):
            text, plan = outs[label]
            plan_json = IO_DIR / f"nb_plan_{label}_{rec['i']}.json"
            out_json  = IO_DIR / f"nb_res_{label}_{rec['i']}.json"
            plan_json.write_text(json.dumps(plan), encoding="utf-8")

            sc = call_matlab_verify(plan_json, xi_path, out_json)
            if sc is None:
                print(f"[{label}] verify failed at idx {rec['i']}")
                continue

            rows.append({
                "prefs_idx": rec["i"],
                "model":     label,
                "xi":        xi_path.as_posix(),
                "plan_text": plan_to_text(plan),
                "V_pen":     sc.get("V_pen"),
                "J_dc":      sc.get("J_dc"),
                "ac_ok":     sc.get("ac_ok", sc.get("AC_ok", sc.get("acOK", 0))),
                "feasible_dc": sc.get("feasible_dc", sc.get("feasible", 0)),
            })

    df = pd.DataFrame(rows)
    print(f"\nScored rows: {len(df)} | Skipped (no xi): {skipped}")
    if df.empty:
        print("No data scored; ensure xi paths can be resolved.")
        return

    # Compare on overlap where both SFT & DPO have AC-OK
    sub  = df[df["ac_ok"] == 1].copy()
    pivV = sub.pivot_table(index=["xi","prefs_idx"], columns="model", values="V_pen", aggfunc="first")
    pivJ = sub.pivot_table(index=["xi","prefs_idx"], columns="model", values="J_dc",  aggfunc="first")

    if not {"sft","dpo"}.issubset(pivV.columns):
        print("Not enough overlap for SFT vs DPO comparison.")
        return

    comp = pivV[["sft","dpo"]].rename(columns={"sft":"V_sft","dpo":"V_dpo"}).join(
        pivJ.rename(columns={"sft":"J_sft","dpo":"J_dpo"}), how="left"
    ).reset_index()

    comp["dpo_better_V"] = (comp["V_dpo"] < comp["V_sft"]).astype(int)
    frac_better = comp["dpo_better_V"].mean() if len(comp) else float("nan")
    med_delta   = (comp["V_dpo"] - comp["V_sft"]).median() if len(comp) else float("nan")

    print(f"\nOverlap count (AC-OK both): {len(comp)}")
    print(f"Fraction with DPO lower V_pen than SFT: {frac_better:.3f}")
    print(f"Median ΔV (DPO - SFT): {med_delta:.2f}  (negative is better)")

    out_csv = IO_DIR / "nb_eval_training_comp.csv"
    comp.to_csv(out_csv, index=False)
    print("Saved:", out_csv.as_posix())

if __name__ == "__main__":
    main()
