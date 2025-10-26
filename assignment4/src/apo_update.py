import math
import torch
import numpy as np

def rolewise_norm(advantages, roles):
    a = np.array(advantages, dtype=np.float32)
    r = np.array(roles)
    out = []
    for role in ["policy", "verifier"]:
        mask = (r == role)
        if mask.sum() == 0:
            continue
        mu = a[mask].mean()
        sd = a[mask].std() + 1e-6
        a[mask] = (a[mask] - mu) / sd
    out = a.tolist()
    return out

def apo_loss(batch_logs, ref_logps, vstar_map, lambda_kl=0.01):
    """
    batch_logs: list of dicts from PAGRollout with keys: role, question, logp, reward
    ref_logps: list of baseline logp values computed under reference (frozen) model for same texts
    vstar_map: dict mapping question -> V*
    """
    advantages = []
    roles = []
    cur_logps = []
    for bl in batch_logs:
        vstar = vstar_map.get(bl["question"], 0.0)
        A = bl["reward"] - vstar
        advantages.append(A)
        roles.append(bl["role"])
        cur_logps.append(bl["logp"])

    advantages = rolewise_norm(advantages, roles)

    # Least-squares style: 0.5 * ( sqrt(ReLU(A*)) * (-logp_cur) )^2 + lambda * KL
    ce = 0.0
    kl_sum = 0.0
    n_kl = 0
    for Astar, cur_lp, ref_lp in zip(advantages, cur_logps, ref_logps):
        w = math.sqrt(max(0.0, Astar))
        ce += 0.5 * (w * (-cur_lp))**2
        if ref_lp is not None:
            kl_sum += (cur_lp - ref_lp)
            n_kl += 1
    ce = ce / max(1, len(cur_logps))
    kl = (kl_sum / max(1, n_kl))
    loss = ce + lambda_kl * kl
    return loss
