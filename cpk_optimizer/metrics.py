"""
metrics.py
-----------
All Cpk math lives here, including your **sigma≈0 rule**:

  • If σ > EPS:
      - Compute the usual Cpk (two-sided, lower-only, or upper-only)
      - Cap to [-CPK_INF, +CPK_INF]

  • If σ <= EPS:
      - If μ is OUTSIDE spec ⇒ return **-CPK_INF** immediately (hard fail)
      - If μ is INSIDE spec ⇒ return **CPK_INF * exp(-λ * d_norm²)**
            where d_norm measures centering relative to the spec "center".
        (If λ == 0, this simply returns +CPK_INF.)

The "center" and normalization used for d_norm:
  - two-sided: center=(L+U)/2, scale=max((U-L)/2, 1.0)
  - lower   : center=L,        scale=max(0.1*max(|L|,1), 1e-6)
  - upper   : center=U,        scale=max(0.1*max(|U|,1), 1e-6)

This file does NOT depend on Streamlit; it is pure compute.
"""
from __future__ import annotations

import math
from typing import Dict

from .settings import CPK_INF, EPS


# --- Plain Cpk formulas ---------------------------------------------------
def cpk_two_sided(mu: float, sigma: float, L: float, U: float) -> float:
    """Two-sided spec Cpk = min((μ-L)/(3σ), (U-μ)/(3σ))."""
    if sigma <= EPS:
        return CPK_INF if (L <= mu <= U) else -CPK_INF
    return min((mu - L) / (3.0 * sigma), (U - mu) / (3.0 * sigma))


def cpk_lower(mu: float, sigma: float, L: float) -> float:
    """Lower-only spec Cpk = (μ-L)/(3σ)."""
    if sigma <= EPS:
        return CPK_INF if mu >= L else -CPK_INF
    return (mu - L) / (3.0 * sigma)


def cpk_upper(mu: float, sigma: float, U: float) -> float:
    """Upper-only spec Cpk = (U-μ)/(3σ)."""
    if sigma <= EPS:
        return CPK_INF if mu <= U else -CPK_INF
    return (U - mu) / (3.0 * sigma)


def compute_cpk(mu: float, sigma: float, spec: Dict) -> float:
    """Dispatch to the appropriate Cpk formula based on spec['type']."""
    t = spec["type"].lower()
    if t == "two_sided":
        return cpk_two_sided(mu, sigma, spec["L"], spec["U"])
    if t == "lower":
        return cpk_lower(mu, sigma, spec["L"])
    if t == "upper":
        return cpk_upper(mu, sigma, spec["U"])
    raise ValueError("spec['type'] must be one of: 'two_sided' | 'lower' | 'upper'")


# --- Sigma≈0 rule + capping -----------------------------------------------
def cpk_with_zero_sigma_penalty(mu: float, sigma: float, spec: Dict, lam: float) -> float:
    """
    Your business rule for σ≈0 + capping for σ>EPS.
    See module docstring for the full behavior.
    """
    # Normal sigma: compute base Cpk and cap
    if sigma > EPS:
        base = compute_cpk(mu, sigma, spec)
        return max(min(base, CPK_INF), -CPK_INF)

    # Sigma ~ 0: first determine if μ is inside the spec
    t = spec["type"].lower()
    if t == "two_sided":
        L, U = spec["L"], spec["U"]
        inside = (L <= mu <= U)
        center = 0.5 * (L + U)
        scale = max((U - L) / 2.0, 1.0)
    elif t == "lower":
        L = spec["L"]
        inside = (mu >= L)
        center = L
        scale = max(0.1 * max(abs(L), 1.0), 1e-6)
    else:  # 'upper'
        U = spec["U"]
        inside = (mu <= U)
        center = U
        scale = max(0.1 * max(abs(U), 1.0), 1e-6)

    if not inside:
        # Hard fail when σ≈0 AND the mean is out-of-spec
        return -CPK_INF

    # Inside the spec: apply centering penalty around the center
    if lam <= 0.0:
        return CPK_INF

    d_norm = (mu - center) / scale
    val = CPK_INF * math.exp(-lam * (d_norm ** 2))
    return max(min(val, CPK_INF), -CPK_INF)


# --- Utility ---------------------------------------------------------------
def nutrients_present_in_data(ingr_map: Dict) -> set:
    """
    Return the set of nutrient codes present anywhere in the ingredient map.

    The ingestion layer stores each ingredient as:
       ingr_map[ingredient][nutrient] = {mu, sigma, mu_dry, sigma_dry}
    plus optional '_conv' for dry→raw conversion. We skip '_conv' here.
    """
    present = set()
    for _, d in ingr_map.items():
        for k in d.keys():
            if k == "_conv":
                continue
            present.add(k)
    return present
