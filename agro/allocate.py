"""
AGRO irrigation / relief allocator — the DECISION acceleration proof (agro twin of the
grid shed-plan optimizer).

Given "I have B megalitres of canal water (or N extension-visit slots) this week", it:
  1. generates thousands of candidate allocation plans (greedy-by-benefit/water + random),
  2. evaluates ALL of them as dense matrix ops (CuPy on GPU / NumPy on CPU):
     yield saved, water used, tail-village fairness bonus, budget feasibility,
  3. local-search refines the winners,
  4. returns the top plans with per-village explanation.

Objective: maximize yield-tonnes-saved within the water budget, with a fairness bonus that
favours chronically under-served tail-reach villages (the agro analogue of the grid's
rotational-fairness term). Same GPU throughput story: hundreds of thousands of plans/second
vs an officer hand-comparing a handful.
"""
from __future__ import annotations

import time

import numpy as np


def allocate(village_state: list[dict], budget_ml: float, n_candidates: int = 20000,
             top_k: int = 5, xp=np, seed: int = 0, fairness_w: float = 0.35) -> dict:
    """village_state rows: village_id,name,yield_saveable_t,water_need_ml,past_support_index,
    canal_reach,p_loss. Returns top allocation plans + throughput metrics."""
    t0 = time.time()
    elig = [v for v in village_state
            if v["water_need_ml"] > 1e-3 and v["yield_saveable_t"] > 1e-3]
    if not elig or budget_ml <= 0:
        return {"plans": [], "error": "no eligible villages or zero budget"}

    benefit = xp.asarray([v["yield_saveable_t"] for v in elig], dtype=xp.float32)
    water = xp.asarray([v["water_need_ml"] for v in elig], dtype=xp.float32)
    # fairness bonus: under-served (low past support) villages worth more social value
    fair = xp.asarray([1.0 - v["past_support_index"] for v in elig], dtype=xp.float32)
    F = len(elig)
    total_water = float(water.sum())
    budget = min(budget_ml, total_water)

    rng = xp.random.default_rng(seed) if xp is not np else np.random.default_rng(seed)
    # value density drives both greedy seed and candidate sampling probability
    value = benefit + fairness_w * fair
    density = value / xp.maximum(water, 1e-3)
    p_base = min(0.95, budget / max(total_water, 1e-6))
    per = n_candidates // 3
    masks = xp.concatenate([
        (rng.random((per, F), dtype=xp.float32) < p).astype(xp.float32)
        for p in (p_base * 0.8, p_base, min(0.97, p_base * 1.3))])

    # greedy seed: take highest value-density villages until budget is exhausted
    order = xp.argsort(-density)
    order_h = np.asarray(order.get() if hasattr(order, "get") else order)
    water_h = np.asarray(water.get() if hasattr(water, "get") else water)
    greedy = xp.zeros((1, F), dtype=xp.float32)
    used = 0.0
    for j in order_h:
        if used + float(water_h[int(j)]) <= budget:
            greedy[0, int(j)] = 1.0
            used += float(water_h[int(j)])
    masks = xp.concatenate([greedy, masks])

    # ---- dense vectorized evaluation of ALL plans at once ----
    w_used = masks @ water
    b_saved = masks @ benefit
    fair_tot = masks @ fair
    over = xp.maximum(w_used - budget, 0)
    # maximize benefit+fairness; hard penalty for exceeding the water budget
    score = -(b_saved + fairness_w * fair_tot) + 100.0 * over
    k = min(top_k * 8, masks.shape[0])
    top_idx = xp.argsort(score)[:k]
    eval_s = time.time() - t0

    # ---- local-search refinement on host: try adding any affordable village ----
    b_h = np.asarray(benefit.get() if hasattr(benefit, "get") else benefit)
    f_h = np.asarray(fair.get() if hasattr(fair, "get") else fair)
    mk = np.asarray(masks[top_idx].get() if hasattr(masks, "get") else masks[np.asarray(
        top_idx.get() if hasattr(top_idx, "get") else top_idx)])

    def val(m):
        if float(m @ water_h) > budget + 1e-6:
            return -1e9
        return float(m @ b_h + fairness_w * (m @ f_h))

    order_benefit = np.argsort(-b_h)          # precompute once
    refined, seen = [], set()
    for m in mk[:top_k * 3]:                  # refine only the most promising seeds
        m = m.copy()
        for _pass in range(3):                # bounded passes (fast, near-optimal)
            improved = False
            used_w = float(m @ water_h)
            base = val(m)
            for j in order_benefit:           # greedily add highest-benefit affordable village
                if m[j] < 0.5 and used_w + water_h[j] <= budget:
                    m[j] = 1.0
                    if val(m) > base + 1e-6:
                        improved = True
                        break
                    m[j] = 0.0
            if not improved:
                break
        key = tuple(np.where(m > 0.5)[0].tolist())
        if key and key not in seen:
            seen.add(key)
            refined.append(m)

    plans = []
    for m in refined:
        idxs = np.where(m > 0.5)[0]
        plans.append({
            "villages": [{"village_id": elig[i]["village_id"], "name": elig[i]["name"],
                          "reach": elig[i]["canal_reach"],
                          "yield_saved_t": round(float(b_h[i]), 2),
                          "water_ml": round(float(water_h[i]), 2)} for i in idxs],
            "n_villages": int(len(idxs)),
            "yield_saved_t": round(float(m @ b_h), 2),
            "water_used_ml": round(float(m @ water_h), 2),
            "tail_share": round(float(np.mean([elig[i]["canal_reach"] == "tail"
                                               for i in idxs])) if len(idxs) else 0, 2),
            "fairness_bonus": round(float(m @ f_h), 2),
            "within_budget": bool(float(m @ water_h) <= budget + 1e-6),
            "score": round(-val(m), 3),
        })
    plans.sort(key=lambda p: -p["yield_saved_t"])
    plans = plans[:top_k]

    total_s = time.time() - t0
    n_eval = int(masks.shape[0])
    return {
        "budget_ml": round(budget, 2), "eligible_villages": F,
        "plans_evaluated": n_eval, "eval_seconds": round(total_s, 4),
        "plans_per_second": int(n_eval / max(total_s, 1e-9)),
        "matrix_eval_seconds": round(eval_s, 4),
        "equivalent_manual_hours": round(n_eval * 5.0 / 60.0, 1),
        "backend": "cupy/GPU" if xp is not np else "numpy/CPU",
        "plans": plans,
    }
