# python/gpt_client.py
import os
from typing import List

# If you're using OpenAI's official SDK:
# pip install openai
from openai import OpenAI

MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")  # set via env if you want

SYSTEM_PROMPT = """You are a grid assistant.
You will receive a compact JSON summary of a PSPS state and allowed actions.
Output ONLY a single-line action plan in this strict format:
- open(CORRIDOR_NAME); close(CORRIDOR_NAME); nudge(BUS_ID,up|down)
- Use at most the 'toggle_budget' total corridor actions.
- Use at most two nudge() actions.
- Only corridors and buses present in the summary are valid.
- No extra text. No explanations. No JSON.
Examples:
open(S1)
open(S1); nudge(69,up)
open(S1); close(S2); nudge(65,down)
"""

def get_n_plans(summary_text: str, n: int = 3) -> List[str]:
    """
    Returns n raw LLM plan strings, one line each, no JSON.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    out = []
    for _ in range(n):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":summary_text}
            ],
            temperature=0.7,
            n=1
        )
        plan = resp.choices[0].message.content.strip()
        # Defensive cleanup (one line; remove leading/trailing quotes)
        if plan.startswith(("```", "\"", "'")):
            plan = plan.strip("`\"'")
        out.append(plan.replace("\n", " ").strip())
    return out
