# python/sft_pack.py
# Split raw SFT corpus into train/test and emit:
#  - io/sft_train.jsonl, io/sft_test.jsonl (raw pairs with xi_file for audit)
#  - io/sft_openai_chat_train.jsonl (OpenAI chat fine-tune format, TRAIN only)
#  - io/sft_train_index.json, io/sft_test_index.json (lists of xi_file paths)
#
# Deterministic split using a hash of xi_file (or prompt if missing).
# De-duplicates by xi_file (keeps the last seen record).

import json, hashlib, os, re, random
from pathlib import Path
from typing import List, Dict, Any

ROOT    = Path(__file__).resolve().parents[1]
IO_DIR  = ROOT / "io"
RAW_JL  = IO_DIR / "sft_raw.jsonl"

TRAIN_RAW   = IO_DIR / "sft_train.jsonl"
TEST_RAW    = IO_DIR / "sft_test.jsonl"
TRAIN_CHAT  = IO_DIR / "sft_openai_chat_train.jsonl"
TRAIN_INDEX = IO_DIR / "sft_train_index.json"
TEST_INDEX  = IO_DIR / "sft_test_index.json"

# ---- Config (env overridable) ----
TEST_RATIO = float(os.getenv("TEST_RATIO", "0.20"))  # 20% default
SEED       = int(os.getenv("SPLIT_SEED", "13"))
KEEP_EMPTY_TARGETS = bool(int(os.getenv("KEEP_EMPTY_TARGETS", "1")))  # 1=keep "", 0=drop

# System prompt aligned with inference grammar
SYSTEM_MSG = (
    "You are a grid switching assistant. "
    "Given a JSON summary of a PSPS scenario, reply with a single line containing only OPEN actions. "
    "Grammar:\n"
    "- open(Sk:LINE) when the corridor Sk is known (e.g., open(S6:135))\n"
    "- open(LINE) when the corridor is unknown (e.g., open(131))\n"
    "Use at most the toggle_budget actions. No other text."
)

_OPEN_RE = re.compile(r"open\s*\(\s*(?:[sS]\d+\s*:\s*)?\d+\s*\)", re.IGNORECASE)

def _normalize_target(t: str) -> str:
    t = (t or "").strip()
    if t.lower() == "do_nothing":
        return ""
    toks = _OPEN_RE.findall(t)
    return "; ".join(tok.strip() for tok in toks)

def _hash_str(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)

def _key_for_record(r: Dict[str, Any]) -> str:
    # Prefer xi_file to ensure stable split per scenario; fall back to prompt hash
    if r.get("xi_file"):
        return str(r["xi_file"])
    return "PROMPT#" + hashlib.md5(r.get("prompt","").encode("utf-8")).hexdigest()

def main():
    if not RAW_JL.exists():
        raise FileNotFoundError(f"Missing input: {RAW_JL.as_posix()}")

    random.seed(SEED)

    # ---- Load & normalize ----
    # Deduplicate by xi_file (keep the last seen record).
    dedup: Dict[str, Dict[str, Any]] = {}
    total_in = 0
    with RAW_JL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total_in += 1
            rec["target"] = _normalize_target(rec.get("target", ""))
            key = _key_for_record(rec)
            dedup[key] = rec  # last writer wins

    pool = list(dedup.values())
    if not pool:
        raise RuntimeError("No records found after de-duplication.")

    # Optional: drop empty targets from TRAINING (still included in raw/test index)
    train_candidates = pool if KEEP_EMPTY_TARGETS else [r for r in pool if r.get("target","") != ""]

    # ---- Deterministic split ----
    train, test = [], []
    for r in pool:
        key = _key_for_record(r)
        h = _hash_str(key) % 1000
        if h < int(TEST_RATIO * 1000):
            test.append(r)
        else:
            train.append(r)

    # ---- Write RAW splits ----
    with TRAIN_RAW.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")

    with TEST_RAW.open("w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps(r) + "\n")

    # ---- OpenAI chat training file (TRAIN only) ----
    # If KEEP_EMPTY_TARGETS=0, use filtered list to avoid empty assistant messages.
    with TRAIN_CHAT.open("w", encoding="utf-8") as out_f:
        n = 0
        for r in train_candidates:
            # ensure this example belongs to TRAIN split
            key = _key_for_record(r)
            h = _hash_str(key) % 1000
            if h < int(TEST_RATIO * 1000):
                continue  # skip test
            obj = {
                "messages": [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user",   "content": r["prompt"]},
                    {"role": "assistant", "content": r["target"]},
                ]
            }
            out_f.write(json.dumps(obj) + "\n")
            n += 1

    # ---- Indices for downstream scripts ----
    with TRAIN_INDEX.open("w", encoding="utf-8") as f:
        json.dump([r["xi_file"] for r in train if r.get("xi_file")], f, indent=2)

    with TEST_INDEX.open("w", encoding="utf-8") as f:
        json.dump([r["xi_file"] for r in test if r.get("xi_file")], f, indent=2)

    # ---- Stats ----
    print(f"Loaded {total_in} lines; {len(pool)} unique scenarios.")
    print(f"Split done. Train={len(train)}, Test={len(test)}")
    kept_train = sum(1 for r in train_candidates if _hash_str(_key_for_record(r)) % 1000 >= int(TEST_RATIO*1000))
    print(f"- Train (chat FT examples written): {kept_train}")
    print(f"- Wrote {TRAIN_RAW.as_posix()}")
    print(f"- Wrote {TEST_RAW.as_posix()}")
    print(f"- Wrote {TRAIN_CHAT.as_posix()}")
    print(f"- Wrote {TRAIN_INDEX.as_posix()}")
    print(f"- Wrote {TEST_INDEX.as_posix()}")

if __name__ == "__main__":
    main()
