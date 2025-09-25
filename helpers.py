from __future__ import annotations
import math
from typing import Tuple, Dict, Any

# -------------------------
# Core constants (unchanged)
# -------------------------
ML_TOTAL = 100   # total batch weight (grams)
CPK_INF = 100    # treat any “very large” Cpk as 100
EPS = 1e-12


# -------------------------
# Cpk math (unchanged)
# -------------------------
def cpk_two_sided(mu: float, sigma: float, L: float, U: float) -> float:
    if sigma <= EPS:
        return CPK_INF if (L <= mu <= U) else -CPK_INF
    return min((mu - L) / (3.0 * sigma), (U - mu) / (3.0 * sigma))

def cpk_lower(mu: float, sigma: float, L: float) -> float:
    if sigma <= EPS:
        return CPK_INF if mu >= L else -CPK_INF
    return (mu - L) / (3.0 * sigma)

def cpk_upper(mu: float, sigma: float, U: float) -> float:
    if sigma <= EPS:
        return CPK_INF if mu <= U else -CPK_INF
    return (U - mu) / (3.0 * sigma)

def compute_cpk(mu: float, sigma: float, spec: Dict[str, Any]) -> float:
    t = spec["type"].lower()
    if t == "two_sided": return cpk_two_sided(mu, sigma, spec["L"], spec["U"])
    if t == "lower":     return cpk_lower(mu, sigma, spec["L"])
    if t == "upper":     return cpk_upper(mu, sigma, spec["U"])
    raise ValueError("spec['type'] must be 'two_sided'|'lower'|'upper'")


# ---- Zero-σ handling (updated penalties for all types; logic unchanged from your last version) ----
def cpk_with_zero_sigma_penalty(mu: float, sigma: float, spec: dict, lam: float) -> float:
    """
    - If σ > EPS: compute base Cpk and cap to [-100, +100].
    - If σ <= EPS:
        * If μ is OUTSIDE spec -> return -100 directly (hard fail).
        * If μ is INSIDE spec:
            - two_sided:  +100 * exp(-λ * d_center^2)               (closer to center is better)
            - lower:      +100 * (1 - exp(-λ * d_in^2))              (farther above L is better)
            - upper:      +100 * (1 - exp(-λ * d_in^2))              (farther below U is better)
    """
    if sigma > EPS:
        base = compute_cpk(mu, sigma, spec)
        return max(min(base, CPK_INF), -CPK_INF)

    t = spec["type"].lower()

    if t == "two_sided":
        L, U = spec["L"], spec["U"]
        if not (L <= mu <= U): return -CPK_INF
        if lam <= 0.0: return CPK_INF
        center = 0.5 * (L + U)
        scale = max((U - L) / 2.0, 1.0)
        d_center = (mu - center) / scale
        val = CPK_INF * math.exp(-lam * (d_center ** 2))
        return max(min(val, CPK_INF), -CPK_INF)

    elif t == "lower":
        L = spec["L"]
        if mu < L: return -CPK_INF
        if lam <= 0.0: return CPK_INF
        scale = max(0.1 * max(abs(L), 1.0), 1e-6)
        d_in = max(0.0, mu - L) / scale
        val = CPK_INF * (1.0 - math.exp(-lam * (d_in ** 2)))
        return max(min(val, CPK_INF), -CPK_INF)

    elif t == "upper":
        U = spec["U"]
        if mu > U: return -CPK_INF
        if lam <= 0.0: return CPK_INF
        scale = max(0.1 * max(abs(U), 1.0), 1e-6)
        d_in = max(0.0, U - mu) / scale
        val = CPK_INF * (1.0 - math.exp(-lam * (d_in ** 2)))
        return max(min(val, CPK_INF), -CPK_INF)

    else:
        raise ValueError("spec['type'] must be 'two_sided'|'lower'|'upper'")


# -------------------------
# Data helpers (unchanged)
# -------------------------
def _value_is_tuple_stats(v) -> bool:
    return isinstance(v, (tuple, list)) and len(v) >= 2 and all(isinstance(x, (int, float)) for x in v[:2])

def get_stats(ingr_map: dict, ing: str, nutrient: str, is_dry_mode: bool) -> Tuple[float, float]:
    entry = ingr_map.get(ing, {}).get(nutrient, (0.0, 0.0))
    if _value_is_tuple_stats(entry):
        return (float(entry[0]), float(entry[1]))
    elif isinstance(entry, dict):
        if is_dry_mode:
            mu = float(entry.get("mu_dry", entry.get("mu", 0.0)))
            sigma = float(entry.get("sigma_dry", entry.get("sigma", 0.0)))
        else:
            mu = float(entry.get("mu", 0.0))
            sigma = float(entry.get("sigma", 0.0))
        return (mu, sigma)
    return (0.0, 0.0)

def nutrients_present_in_data(ingr_map: dict) -> set:
    present = set()
    for _, d in ingr_map.items():
        for k in d.keys():
            if k == "_conv":
                continue
            present.add(k)
    return present
