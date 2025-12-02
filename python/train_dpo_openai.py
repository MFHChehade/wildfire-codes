# python/train_dpo_openai.py  (patched)
import os, sys, time, json
from pathlib import Path
from openai import OpenAI

ROOT    = Path(__file__).resolve().parents[1]
CONFIG  = ROOT / "config"
IO_DIR  = ROOT / "io"
FT_OUT  = CONFIG / "ft_model_dpo.txt"

# ----- Parent SFT model (must start with "ft:")
FT_MODEL_ENV = (os.getenv("FT_MODEL", "") or "").strip()
FT_SFT_FILE  = CONFIG / "ft_model_sft.txt"
if not FT_MODEL_ENV and FT_SFT_FILE.exists():
    FT_MODEL_ENV = FT_SFT_FILE.read_text(encoding="utf-8").strip()

API_KEY = os.getenv("OPENAI_API_KEY") or ""
if not API_KEY:
    print("ERROR: Set OPENAI_API_KEY in your environment.", file=sys.stderr); sys.exit(1)

# ----- Dataset(s)
DATA_PATH = Path(os.getenv("OPENAI_PREFS_PATH", str(IO_DIR / "dpo_openai_prefs.jsonl")))
VAL_PATH  = Path(os.getenv("OPENAI_PREFS_VAL_PATH", "")).resolve() if os.getenv("OPENAI_PREFS_VAL_PATH") else None

# ----- DPO hyperparameters
N_EPOCHS   = int(os.getenv("DPO_EPOCHS", "2"))
BETA       = float(os.getenv("DPO_BETA", "0.1"))
BATCH_SIZE = int(os.getenv("DPO_BATCH_SIZE", "8"))

def fail(msg: str):
    print(f"\nERROR: {msg}", file=sys.stderr); sys.exit(1)

def _normalize_pair_keys(obj: dict) -> dict:
    """
    Accept either:
      preferred_output / non_preferred_output
    or:
      preferred / rejected
    Return a dict with canonical keys: preferred, rejected
    """
    out = dict(obj)
    if "preferred" in obj and "rejected" in obj:
        return out
    if "preferred_output" in obj and "non_preferred_output" in obj:
        out["preferred"] = obj["preferred_output"]
        out["rejected"]  = obj["non_preferred_output"]
        return out
    fail("Dataset line missing preference keys. Expected either "
         "('preferred','rejected') or ('preferred_output','non_preferred_output').")

def _validate_messages(ms, field_name: str):
    if not isinstance(ms, list) or not ms:
        fail(f"`{field_name}` must be a non-empty list of messages.")
    first = ms[0]
    if first.get("role") != "assistant":
        fail(f"First message in `{field_name}` must have role='assistant'.")
    content = first.get("content")
    if content is None or (isinstance(content, str) and not content.strip()):
        fail(f"First message in `{field_name}` must have non-empty `content`.")

def basic_schema_check(raw_obj: dict):
    if not isinstance(raw_obj, dict):
        fail("Dataset line is not a JSON object.")
    if "input" not in raw_obj or not isinstance(raw_obj["input"], dict):
        fail("Dataset line missing `input` object.")
    inp = raw_obj["input"]
    if "messages" not in inp or not isinstance(inp["messages"], list) or not inp["messages"]:
        fail("`input.messages` must be a non-empty list of chat messages.")

    # Normalize to 'preferred'/'rejected' and validate
    obj = _normalize_pair_keys(raw_obj)
    _validate_messages(obj["preferred"], "preferred")
    _validate_messages(obj["rejected"],  "rejected")
    return obj  # normalized

def preview_dataset(path: Path, n: int = 3):
    print(f"Checking dataset: {path.as_posix()}")
    cnt = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except Exception as e:
                fail(f"Line {i} is not valid JSON: {e}")
            _ = basic_schema_check(raw)
            cnt += 1
            if i <= n:
                print(f"  sample[{i}]: OK")
    if cnt == 0:
        fail("Dataset is empty.")
    print(f"Total lines: {cnt}")

def create_client() -> OpenAI:
    return OpenAI(api_key=API_KEY)

def upload_dataset(client: OpenAI, path: Path) -> str:
    with path.open("rb") as f:
        up = client.files.create(file=f, purpose="fine-tune")
    print("Uploaded file id:", up.id)
    return up.id

def start_dpo_job(client: OpenAI, model: str, training_file_id: str, validation_file_id: str | None):
    print(f"Creating DPO job on parent SFT model: {model}")
    method = {
        "type": "dpo",
        "dpo": {
            "hyperparameters": {
                "n_epochs": N_EPOCHS,
                "beta": BETA,
                "batch_size": BATCH_SIZE,
            }
        }
    }
    kwargs = dict(model=model, training_file=training_file_id, method=method)
    if validation_file_id:
        kwargs["validation_file"] = validation_file_id
    return client.fine_tuning.jobs.create(**kwargs)

def poll_job(client: OpenAI, job_id: str):
    spinner = "|/-\\"
    k = 0
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = getattr(job, "status", "unknown")
        print(f"\rStatus: {status} {spinner[k % len(spinner)]}", end="", flush=True)
        k += 1
        if status in ("succeeded", "failed", "cancelled"):
            print()
            return job
        time.sleep(8)

def write_model_id(model_id: str):
    CONFIG.mkdir(exist_ok=True)
    FT_OUT.write_text(model_id, encoding="utf-8")
    print(f"Wrote model id to {FT_OUT.as_posix()}")

def main():
    # Parent checks
    if not FT_MODEL_ENV or not FT_MODEL_ENV.startswith("ft:"):
        fail("No SFT parent found. Set FT_MODEL to an 'ft:' id or write config/ft_model_sft.txt.")
    # Very light guard to catch accidental base model ids
    if ":" not in FT_MODEL_ENV:
        fail("Parent model id looks malformed. Expecting an 'ft:...' id.")

    # Data checks
    if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
        fail(f"Missing or empty dataset: {DATA_PATH.as_posix()}")
    preview_dataset(DATA_PATH)

    client = create_client()
    train_file_id = upload_dataset(client, DATA_PATH)

    val_file_id = None
    if VAL_PATH and VAL_PATH.exists() and VAL_PATH.stat().st_size > 0:
        print(f"Found validation set: {VAL_PATH.as_posix()}")
        val_file_id = upload_dataset(client, VAL_PATH)

    print(f"\nLaunching DPO with hyperparameters: epochs={N_EPOCHS}, beta={BETA}, batch_size={BATCH_SIZE}")
    try:
        job = start_dpo_job(client, FT_MODEL_ENV, train_file_id, val_file_id)
    except Exception as e:
        print("\nFailed to create DPO fine-tune job.", file=sys.stderr)
        print("Common causes:", file=sys.stderr)
        print("  • Parent SFT model/base not eligible for DPO.", file=sys.stderr)
        print("  • Account lacks DPO entitlement or quota.", file=sys.stderr)
        print("  • SDK outdated (pip install -U openai).", file=sys.stderr)
        print("  • Dataset must use 'preferred'/'rejected' (or compatible).", file=sys.stderr)
        print("Server said:\n", e, file=sys.stderr)
        sys.exit(1)

    print("Created job:", job.id, "status:", job.status)
    job = poll_job(client, job.id)

    if job.status != "succeeded":
        print("ERROR: DPO fine-tune finished with status=", job.status, file=sys.stderr)
        sys.exit(1)

    model_id = job.fine_tuned_model
    if not model_id:
        print("ERROR: No fine_tuned_model returned on success.", file=sys.stderr)
        sys.exit(1)

    print("DPO fine-tuned model:", model_id)
    write_model_id(model_id)

if __name__ == "__main__":
    main()
