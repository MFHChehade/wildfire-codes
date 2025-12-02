# python/make_indices.py
# Build train/test indices from all io/psps_*.json files.

import json, random, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IO_DIR = ROOT / "io"
IO_DIR.mkdir(exist_ok=True)

SEED        = int(os.getenv("SEED", "7"))
TRAIN_FRAC  = float(os.getenv("TRAIN_FRAC", "0.8"))
MIN_COUNT   = int(os.getenv("MIN_COUNT", "1"))   # allow small sets

def main():
    psps = sorted([p.as_posix() for p in IO_DIR.glob("psps_*.json")])
    if len(psps) < MIN_COUNT:
        raise FileNotFoundError(f"Found {len(psps)} psps_*.json; need at least {MIN_COUNT}. Create PSPS first.")

    random.seed(SEED)
    random.shuffle(psps)

    k_train = max(0, min(len(psps), int(round(TRAIN_FRAC * len(psps)))))
    train = psps[:k_train]
    test  = psps[k_train:]

    (IO_DIR / "sft_train_index.json").write_text(json.dumps(train, indent=2), encoding="utf-8")
    (IO_DIR / "sft_test_index.json").write_text(json.dumps(test, indent=2),  encoding="utf-8")
    print(f"Wrote {len(train)} train and {len(test)} test to io/sft_train_index.json / io/sft_test_index.json")

if __name__ == "__main__":
    main()
