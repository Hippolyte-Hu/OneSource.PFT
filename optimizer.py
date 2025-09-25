from __future__ import annotations
import time
import math
import numpy as np
from typing import Dict, Any, List

from scipy.optimize import minimize

from helpers import (
    ML_TOTAL,
    cpk_with_zero_sigma_penalty,
    get_stats,
)

# ---------- Mixture stats (same math, just moved) -------
def mix_mu_sigma_from_map(INGR: dict, ingredients: list[str], vols_g, nutrient: str, is_dry_mode: bool):
    w = np.asarray(vols_g, dtype=float) / ML_TOTAL
    mus = []
    sig = []
    for ing in ingredients:
        mu_i, sd_i = get_stats(INGR, ing, nutrient, is_dry_mode)
        mus.append(mu_i)
        sig.append(sd_i)
    mus = np.asarray(mus, dtype=float)
    sig = np.asarray(sig, dtype=float)
    mu = float(np.dot(w, mus))
    var = float(np.dot(w * w, sig * sig))
    return mu, math.sqrt(max(var, 0.0))


def optimize(
    INGR: dict,
    RECIPE_SPECS: Dict[str, Dict[str, Any]],
    nutrients: List[str],
    ingredients_in_scope: List[str],
    is_dry: bool,
    lam_pen: float,
    strategy: str,
    n_restarts: int,
) -> Dict[str, Any]:
    """
    Multi-start SLSQP optimizer (logic unchanged).
    Returns: dict with best_result, x_opt, opt_score, elapsed_sec
    """
    score_is_avg = ("Average" in strategy)

    def score_from_vols(vols):
        per_cpk = []
        for n in nutrients:
            mu, sigma = mix_mu_sigma_from_map(INGR, ingredients_in_scope, vols, n, is_dry)
            cpk_eff = cpk_with_zero_sigma_penalty(mu, sigma, RECIPE_SPECS[n], lam_pen)
            per_cpk.append(cpk_eff)
        return float(np.mean(per_cpk)) if score_is_avg else float(np.min(per_cpk))

    def objective(x):
        return -score_from_vols(x)  # SLSQP minimizes

    m = len(ingredients_in_scope)
    bounds = [(0.0, ML_TOTAL)] * m
    cons = [{"type": "eq", "fun": lambda x: np.sum(x) - ML_TOTAL}]

    rng = np.random.default_rng(42)
    starts = [np.full(m, ML_TOTAL / m, dtype=float)]
    for _ in range(int(n_restarts) - 1):
        w = rng.dirichlet(np.ones(m))
        starts.append(w * ML_TOTAL)

    best = None
    best_fun = float("inf")

    t0 = time.perf_counter()
    for init in starts:
        res = minimize(
            objective,
            init,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options=dict(maxiter=800, ftol=1e-9, disp=False),
        )
        if res.success and res.fun < best_fun:
            best = res
            best_fun = res.fun
    elapsed_sec = time.perf_counter() - t0

    if best is None:
        raise RuntimeError("Optimization failed to converge. Try adjusting limits.")

    # Renormalize (just in case)
    x_opt = np.clip(best.x, 0.0, None)
    s = float(np.sum(x_opt))
    if s <= 0:
        raise RuntimeError("Optimizer returned a degenerate solution (total weight = 0).")
    x_opt = x_opt * (ML_TOTAL / s)

    opt_score = score_from_vols(x_opt)
    return {
        "best_result": best,
        "x_opt": x_opt,
        "opt_score": opt_score,
        "elapsed_sec": elapsed_sec,
    }
