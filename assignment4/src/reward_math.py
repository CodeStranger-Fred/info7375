import re
import math

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")

def _strip_units(x: str) -> str:
    x = x.strip()
    x = x.replace(",", "")
    x = x.replace(" ", "")
    return x

def _parse_number(s: str):
    s = s.strip()
    try:
        if "/" in s:
            num, den = s.split("/")
            return float(num) / float(den)
        return float(s)
    except Exception:
        return None

def extract_final_answer(text: str):
    # Prefer \boxed{...}
    m = BOXED_RE.findall(text)
    if m:
        return m[-1].strip()
    # Fallback: last line numeric
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        nums = re.findall(r"-?\d+(\.\d+)?(/\d+(\.\d+)?)?", ln)
        if nums:
            return re.findall(r"-?\d+(\.\d+)?(/\d+(\.\d+)?)?", ln)[-1][0]
    return None

def is_equivalent(pred: str, gold: str) -> bool:
    if pred is None or gold is None:
        return False
    a = _strip_units(pred)
    b = _strip_units(gold)

    # Exact match
    if a == b:
        return True

    # Try numeric equivalence
    av = _parse_number(a)
    bv = _parse_number(b)
    if av is not None and bv is not None:
        return math.isclose(av, bv, rel_tol=1e-6, abs_tol=1e-6)

    return False

def is_correct(pred_text: str, gold_solution_text: str) -> bool:
    # Gold often includes a final boxed answer
    gold_ans = extract_final_answer(gold_solution_text)
    pred_ans = extract_final_answer(pred_text)
    return is_equivalent(pred_ans, gold_ans)

def pag_shaped_reward(prev_correct, cur_correct, alpha_pos=1.0, alpha_neg=1.0, alpha_hold=0.2):
    if prev_correct is None:
        return alpha_pos if cur_correct else 0.0
    if (prev_correct is False and cur_correct is True):
        return +alpha_pos
    if (prev_correct is True and cur_correct is False):
        return -alpha_neg
    return +alpha_hold if cur_correct else 0.0
