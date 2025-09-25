"""
optimizer.py
-------------
SLSQP-based optimizer for mixture proportions. This module contains ONLY
optimization + mixture-moment math and is UI-agnostic.

Public API:
  optimize(...): run multi-start SLSQP and return the best solution + timing
"""
from __future__ import annotations

import math
import time
from typing import Dict, List

import numpy as np
from scipy.optimize import minimize

from .settings import ML_TOTAL
from .metrics import cpk_with_zero_sigma_penalty
from .data_io import get_stats


# --- Mixture moment math ---------------------------------------------------
def mix_mu_sigma_from_map(INGR: Dict, ingredients: List[str], vols_g: np.ndarray, nutrient: str, is_dry_mode: bool) -> tuple[float, float]:
    """
    Compute mixture mean/σ for a single nutrient, given ingredient proportions.

    μ_mix  = sum_i (w_i * μ_i)
    σ_mix² = sum_i (w_i² * σ_i²)

    where w_i = g_i / ML_TOTAL.
    """
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


# --- Objective builder -----------------------------------------------------
def _make_objective(INGR: Dict, ingredients_in_scope: List[str], nutrients: List[str], RECIPE_SPECS: Dict[str, Dict],
                    is_dry: bool, lam_pen: float, score_is_avg: bool):
    """
    Build the SLSQP objective closure (we minimize negative of the score).
    The score is either the mean Cpk across nutrients or the min Cpk.
    """
    def score_from_vols(vols):
        per_cpk = []
        for n in nutrients:
            mu, sigma = mix_mu_sigma_from_map(INGR, ingredients_in_scope, vols, n, is_dry)
            cpk_eff = cpk_with_zero_sigma_penalty(mu, sigma, RECIPE_SPECS[n], lam_pen)
            per_cpk.append(cpk_eff)
        return float(np.mean(per_cpk)) if score_is_avg else float(np.min(per_cpk))

    def objective(x):
        return -score_from_vols(x)  # SLSQP minimizes

    return objective, score_from_vols


# --- Optimizer -------------------------------------------------------------
def optimize(*,
             INGR: Dict,
             RECIPE_SPECS: Dict[str, Dict],
             nutrients: List[str],
             ingredients_in_scope: List[str],
             is_dry: bool,
             lam_pen: float,
             strategy: str = "Average Cpk",
             n_restarts: int = 8,
             rng_seed: int = 42) -> Dict:
    """
    Run multi-start SLSQP and return a dict with the best solution.

    Returns:
      {
        "x_opt": np.ndarray of grams per 100 g (len = #ingredients_in_scope),
        "opt_score": float,
        "best_result": scipy OptimizeResult,
        "elapsed_sec": float
      }
    """
    m = len(ingredients_in_scope)
    assert m > 0, "No ingredients to optimize."

    score_is_avg = ("Average" in strategy)
    objective, score_from_vols = _make_objective(
        INGR, ingredients_in_scope, nutrients, RECIPE_SPECS, is_dry, lam_pen, score_is_avg
    )

    bounds = [(0.0, ML_TOTAL)] * m
    cons = [{"type": "eq", "fun": lambda x: np.sum(x) - ML_TOTAL}]  # sum = 100 g

    rng = np.random.default_rng(rng_seed)
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

    # Renormalize (just in case numerical noise adds error to the equality constraint)
    x_opt = np.clip(best.x, 0.0, None)
    s = float(np.sum(x_opt))
    if s <= 0:
        raise RuntimeError("Optimizer returned a degenerate solution (total weight = 0).")
    x_opt = x_opt * (ML_TOTAL / s)

    return {
        "x_opt": x_opt,
        "opt_score": score_from_vols(x_opt),
        "best_result": best,
        "elapsed_sec": elapsed_sec,
    }
