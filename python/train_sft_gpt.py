# python/train_sft_gpt.py
# Fine-tune an OpenAI chat model using io/sft_openai_chat_train.jsonl.
# Saves resulting model ID to:
#   - config/ft_model_sft.txt  (always)
#   - config/ft_model.txt      (if MIRROR_MAIN=1)
#
# Env:
#   OPENAI_API_KEY=...
#   BASE_MODEL=gpt-4.1-mini-2025-04-14   # must be SFT-capable
#   FT_SUFFIX="wildfire-sft"
#   MIRROR_MAIN=0/1                       # also write ft_model.txt
#   USE_VAL=0/1                           # upload optional val file if present

import os, time, sys
from pathlib import Path
from openai import OpenAI

ROOT      = Path(__file__).resolve().parents[1]
CONFIG    = ROOT / "config"
IO_DIR    = ROOT / "io"
TRAIN_JL  = IO_DIR / "sft_openai_chat_train.jsonl"
VAL_JL    = IO_DIR / "sft_openai_chat_val.jsonl"   # optional if you create one
OUT_SFT   = CONFIG / "ft_model_sft.txt"
OUT_MAIN  = CONFIG / "ft_model.txt"                # optional mirror

BASE_MODEL  = os.getenv("BASE_MODEL", "gpt-4.1-mini-2025-04-14")
FT_SUFFIX   = os.getenv("FT_SUFFIX", "").strip() or None
MIRROR_MAIN = bool(int(os.getenv("MIRROR_MAIN", "0")))
USE_VAL     = bool(int(os.getenv("USE_VAL", "1")))  # will only use if file exists

API_KEY = os.getenv("OPENAI_API_KEY") or ""
if not API_KEY:
    print("ERROR: Set OPENAI_API_KEY", file=sys.stderr)

def fail(msg: str):
    print("ERROR:", msg, file=sys.stderr)
    sys.exit(1)

def main():
    if not TRAIN_JL.exists():
        fail(f"Missing training data: {TRAIN_JL.as_posix()}")
    if not API_KEY:
        fail("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=API_KEY)

    # Upload training file
    with TRAIN_JL.open("rb") as f:
        train_file = client.files.create(file=f, purpose="fine-tune")
    print("Uploaded train file:", train_file.id)

    job_kwargs = {
        "model": BASE_MODEL,
        "training_file": train_file.id,
    }
    if FT_SUFFIX:
        job_kwargs["suffix"] = FT_SUFFIX

    # Optional validation file (only if it exists and USE_VAL=1)
    if USE_VAL and VAL_JL.exists():
        with VAL_JL.open("rb") as f:
            val_file = client.files.create(file=f, purpose="fine-tune")
        job_kwargs["validation_file"] = val_file.id
        print("Uploaded val file:", val_file.id)

    print(f"Creating SFT job on base: {BASE_MODEL} (suffix={FT_SUFFIX or 'None'})")
    job = client.fine_tuning.jobs.create(**job_kwargs)
    print("Created job:", job.id)

    # Poll
    spinner = "|/-\\"
    k = 0
    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        status = getattr(job, "status", "unknown")
        print(f"\rStatus: {status} {spinner[k % len(spinner)]}", end="", flush=True)
        k += 1
        if status in ("succeeded", "failed", "cancelled"):
            print()
            break
        time.sleep(8)

    if job.status != "succeeded":
        fail(f"SFT finished with status={job.status}")

    model_id = job.fine_tuned_model
    if not model_id:
        fail("No fine_tuned_model returned on success (unexpected).")

    print("SFT model:", model_id)
    CONFIG.mkdir(exist_ok=True)
    OUT_SFT.write_text(model_id, encoding="utf-8")
    print(f"Wrote SFT model id to {OUT_SFT.as_posix()}")

    if MIRROR_MAIN:
        OUT_MAIN.write_text(model_id, encoding="utf-8")
        print(f"(Mirrored) Wrote model id to {OUT_MAIN.as_posix()}")

if __name__ == "__main__":
    main()
