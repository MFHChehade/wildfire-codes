# python/run_ground_truth.py
import json
import os
import time
import subprocess
from pathlib import Path
from typing import Dict

ROOT       = Path(__file__).resolve().parents[1]
MATLAB_DIR = ROOT / "matlab"
CONFIG     = ROOT / "config"
IO_DIR     = ROOT / "io"
IO_DIR.mkdir(exist_ok=True)

CASE   = os.getenv("CASE_NAME", "case118")
LIMITS = CONFIG / "limits.yml"

def latest_psps(io_dir: Path) -> Path:
    """
    Pick the newest psps_*.json by modification time, unless XI env is set.
    XI env can be an absolute or relative path.
    """
    xi_env = os.getenv("XI")
    if xi_env:
        p = Path(xi_env)
        if not p.exists():
            raise FileNotFoundError(f"XI env points to a missing file: {p}")
        if not p.name.startswith("psps_") or p.suffix != ".json":
            # Not strictly required, but helps catch accidental inputs
            raise ValueError(f"XI must point to a psps_*.json file, got: {p.name}")
        return p

    candidates = list(io_dir.glob("psps_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No psps_*.json files found in {io_dir}. "
                                f"Create one first (e.g., via your pipeline).")
    return max(candidates, key=lambda p: p.stat().st_mtime)

def run(mode: str, xi_path: Path) -> Dict:
    """
    Call MATLAB run_ground_truth for the given mode ('baseline' or 'mip_opt').
    Returns the parsed JSON result written by MATLAB.
    """
    if mode not in ("baseline", "mip_opt"):
        raise ValueError(f"Unsupported mode: {mode}")

    out = IO_DIR / f"gt_{mode}_{int(time.time())}.json"
    cmd = (
        f"matlab -batch \"addpath('{MATLAB_DIR.as_posix()}'); "
        f"run_ground_truth('{CASE}','{xi_path.as_posix()}','{LIMITS.as_posix()}','{out.as_posix()}','{mode}'); exit;\""
    )
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        raise RuntimeError(f"MATLAB ground truth failed for mode={mode} (return code {ret.returncode}).")

    with open(out, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    XI = latest_psps(IO_DIR)
    result = {
        "baseline": run("baseline", XI),
        "mip_opt": run("mip_opt", XI),
    }
    print(json.dumps(result, indent=2))
