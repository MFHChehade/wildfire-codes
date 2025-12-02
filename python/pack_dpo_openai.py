# python/pack_dpo_openai.py
# Convert io/dpo_pairs.jsonl -> io/dpo_openai_prefs.jsonl in OpenAI DPO format:
# {
#   "input": {"messages":[ ... ]},
#   "preferred_output":     [{"role":"assistant","content":"..."}],
#   "non_preferred_output": [{"role":"assistant","content":"..."}]
# }

import json
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[1]
IO_DIR = ROOT / "io"
IN_JL  = IO_DIR / "dpo_pairs.jsonl"
OUT_JL = IO_DIR / "dpo_openai_prefs.jsonl"

SYSTEM_MSG = (
    "You are a grid switching assistant.\n"
    "Respond with at most 'toggle_budget' OPEN actions in the grammar:\n"
    "open(Sk:LINE) or open(LINE); semicolon-separated; no extra text."
)

def pack_record(rec):
    # Expected input keys in dpo_pairs.jsonl
    #   {"prompt": "<string>", "chosen": "<assistant text>", "rejected": "<assistant text>"}
    prompt   = rec["prompt"]
    chosen   = str(rec["chosen"]).strip()
    rejected = str(rec["rejected"]).strip()

    return {
        "input": {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user",   "content": prompt}
            ]
        },
        "preferred_output":     [{"role": "assistant", "content": chosen}],
        "non_preferred_output": [{"role": "assistant", "content": rejected}],
    }

def main():
    if not IN_JL.exists():
        raise FileNotFoundError(f"Missing input: {IN_JL.as_posix()}")

    n = 0
    with IN_JL.open("r", encoding="utf-8") as fin, OUT_JL.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            # minimal sanity checks
            for k in ("prompt", "chosen", "rejected"):
                if k not in rec:
                    raise KeyError(f"Input record missing '{k}'")

            obj = pack_record(rec)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} DPO pairs to {OUT_JL.as_posix()}")

if __name__ == "__main__":
    main()
