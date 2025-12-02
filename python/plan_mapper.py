# python/plan_mapper.py
import json
import re
from typing import Dict, Any, List, Tuple

OPEN_RE  = re.compile(r"\bopen\((?P<corr>[A-Za-z0-9_]+)\)", re.I)
CLOSE_RE = re.compile(r"\bclose\((?P<corr>[A-Za-z0-9_]+)\)", re.I)
NUDGE_RE = re.compile(r"\bnudge\((?P<bus>\d+)\s*,\s*(?P<dir>up|down)\)", re.I)

def load_corridors(path: str) -> Dict[str, List[int]]:
    with open(path, "r") as f:
        return json.load(f)

def parse_plan_text(
    text: str,
    corridor_map: Dict[str, List[int]],
    toggle_budget: int,
    redispatch_cap: int
) -> Dict[str, Any]:
    """
    Parse a single-line plan like: "open(S1); nudge(69,up)"
    Return dict with 'corridor_actions' and 'redispatch'.
    Enforces budget (max corridor actions), clamps to known corridors.
    """
    actions: List[Tuple[str,str]] = []
    for m in OPEN_RE.finditer(text):
        actions.append(("open", m.group("corr")))
    for m in CLOSE_RE.finditer(text):
        actions.append(("close", m.group("corr")))

    # Enforce toggle budget (keep first valid)
    corridor_actions = []
    for act, name in actions:
        if name in corridor_map and len(corridor_actions) < toggle_budget:
            corridor_actions.append({"name": name, "action": act})

    # Parse up to two nudges
    nudges = []
    for m in NUDGE_RE.finditer(text):
        b = int(m.group("bus"))
        d = m.group("dir").lower()
        if d == "up":   sign = "+1"
        elif d == "down": sign = "-1"
        else: continue
        nudges.append({"bus": b, "sign": sign})
        if len(nudges) >= 2:
            break

    return {
        "corridor_actions": corridor_actions,
        "redispatch": nudges
    }
