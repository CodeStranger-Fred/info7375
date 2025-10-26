import json
from typing import List, Dict, Any
from tqdm import tqdm
from reward_math import is_correct, pag_shaped_reward

def estimate_vstar_per_question(offline_samples: List[Dict[str, Any]], beta1: float = 0.5):
    """
    offline_samples: list of dicts with keys:
      question, solution, samples: [ { "text": str, "kind": "p1"|"v"|"p2", "reward": float }, ...]
    We keep a very simple V*(s_question): softmax(beta1 * r) weighted average over policy-turn rewards.
    """
    q2v = {}
    for ex in offline_samples:
        rewards = [s["reward"] for s in ex["samples"] if s["kind"] in ("p1","p2")]
        if not rewards:
            q2v[ex["question"]] = 0.0
        else:
            # softmax-weighted average to favor higher-reward trajectories
            import math
            ws = [math.exp(beta1 * r) for r in rewards]
            s = sum(ws)
            wavg = sum(w * r for w, r in zip(ws, rewards)) / (s if s != 0 else 1.0)
            q2v[ex["question"]] = float(wavg)
    return q2v

def build_offline_samples(pag_rollout, data, k=8):
    """
    For each question, sample k policy answers + verify/revise.
    We only store rewards for policy turns (p1, p2). Verifier gets 0 reward.
    """
    rows = []
    for item in tqdm(data, desc="Stage1 offline sampling"):
        q = item["question"]
        sol = item["solution"]
        samples = []
        for _ in range(k):
            traj = pag_rollout.one_episode(q, sol)
            # policy 1
            p1 = next(t for t in traj if t["role"] == "policy")
            samples.append({"text": p1["text"], "kind":"p1", "reward": p1["reward"]})
            # optional policy 2
            pol2 = [t for t in traj if t["role"] == "policy"][1:]  # any revise turn
            if pol2:
                samples.append({"text": pol2[0]["text"], "kind":"p2", "reward": pol2[0]["reward"]})
        rows.append({"question": q, "solution": sol, "samples": samples})
    return rows

def save_vstar_json(path, q2v: Dict[str,float]):
    with open(path, "w", encoding="utf-8") as f:
        for k, v in q2v.items():
            f.write(json.dumps({"question": k, "vstar": v}) + "\n")

def load_vstar_json(path) -> Dict[str,float]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            out[j["question"]] = float(j["vstar"])
    return out
