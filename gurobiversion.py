"""
Solve a 'max-min Cpk' recipe design with Gurobi (no Streamlit).

Formulation summary (for each nutrient n):
- Let w be ingredient weights (sum to ML_TOTAL, w >= 0)
- mu_n   = sum_i w_i * mu_in[i, n]                         (affine)
- sigma_n = || (sigma_in[i, n] * w_i)_i ||_2               (SOC via y_n >= ||...||)
- For 'two_sided':   mu_n - L_n >= 3 * t * sigma_n,   U_n - mu_n >= 3 * t * sigma_n
  For 'lower':       mu_n - L_n >= 3 * t * sigma_n
  For 'upper':       U_n - mu_n >= 3 * t * sigma_n
We find the largest t (minimum Cpk across nutrients) that is feasible.

We avoid bilinear (t * sigma_n) by **bisection on t** and using an SOC feasibility
model for each t guess.
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as e:
    raise SystemExit(
        "This script requires Gurobi (gurobipy). Please ensure it's installed and licensed.\n"
        f"Import error: {e}"
    )

# ----------------- constants -----------------
ML_TOTAL = 100.0     # grams total
CPK_INF  = 100.0     # cap for display only (matches your app)
EPS      = 1e-12

# ------------- file paths (edit as needed) -------------
SPEC_PATH = "pov_tool_v8_test.xlsx"  # your spec/limits file
INGR_PATH = "rm_nutrient_distributions_for_optimizer_with_process_losses.xlsx"  # your RM distributions
IS_DRY    = False  # set True to use dried μ/σ if the file contains those columns & to exclude "Water"

# -------- zero-σ penalty for post-hoc reporting (optional) --------
LAMBDA_ZERO_SIGMA = 0.0  # set >0 to see penalized Cpk when sigma ≈ 0 (reporting-only)


# -------------------- IO helpers (mirrors your app) --------------------
def _lower_map(cols):
    return {c.lower().strip(): c for c in cols}

def parse_limits_file(path: str) -> Tuple[Dict[str, dict], pd.DataFrame]:
    """Read CSV/XLSX of limits. Return (spec_map, editable_df)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Limits file not found: {path}")

    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    cols = _lower_map(df.columns)

    def pick(*cands):
        for k in cands:
            if k in cols:
                return cols[k]
        return None

    code_col = pick("nutrient", "bm_code", "nutrientcode", "code", "bm code")
    L_col    = pick("1.2 min_std", "min_std", "min std", "min", "spec_min", "l", "lower", "low")
    U_col    = pick("1.2 max_std", "max_std", "max std", "max", "spec_max", "u", "upper", "up")
    type_col = pick("type", "spec_type")

    if not code_col:
        raise ValueError("Could not find a nutrient code column in the limits file.")

    if L_col: df[L_col] = pd.to_numeric(df[L_col], errors="coerce")
    if U_col: df[U_col] = pd.to_numeric(df[U_col], errors="coerce")

    rows = []
    specs: Dict[str, dict] = {}
    for _, r in df.iterrows():
        code = str(r[code_col]).strip()
        if not code:
            continue
        L = float(r[L_col]) if (L_col and pd.notna(r[L_col])) else None
        U = float(r[U_col]) if (U_col and pd.notna(r[U_col])) else None

        typ = None
        if type_col and pd.notna(r[type_col]):
            t = str(r[type_col]).strip().lower()
            if t in {"two_sided", "lower", "upper"}:
                typ = t
        if typ is None:
            if (L is not None) and (U is not None): typ = "two_sided"
            elif L is not None: typ = "lower"
            elif U is not None: typ = "upper"
            else: continue

        if typ == "two_sided" and (L is not None) and (U is not None) and L > U:
            L, U = U, L

        if typ == "two_sided":
            specs[code] = {"type": "two_sided", "L": L, "U": U}
        elif typ == "lower":
            specs[code] = {"type": "lower", "L": L}
        else:
            specs[code] = {"type": "upper", "U": U}

        rows.append({"nutrient": code, "type": typ, "L": L, "U": U})

    if not specs:
        raise ValueError("No usable limits found (no min/max on any row).")

    return specs, pd.DataFrame(rows, columns=["nutrient", "type", "L", "U"])


def load_ingr_map_from_excel(path: str) -> Dict[str, dict]:
    """
    Expected columns (case-insensitive):
    - 'IS_NAME' or 'Ingredient'
    - 'NUTRIENTCODE'
    - 'chosen_mu' (fallback: 'mu','mean')
    - 'chosen_sigma' (fallback: 'sigma','std')
    Optional (if your sheet includes dried stats/conversion):
    - 'chosen_mu_dried', 'chosen_sigma_dried'
    - 'conversion_from_dry_dosage_to_raw'
    Returns: {ingredient: { nutrient: {mu, sigma, mu_dry, sigma_dry}, '_conv': float? } }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ingredient distribution file not found: {path}")

    df_ing = pd.read_excel(path)
    cols = _lower_map(df_ing.columns)

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    col_ing = pick("is_name", "ingredient")
    col_nut = pick("nutrientcode")
    col_mu  = pick("chosen_mu", "mu", "mean")
    col_sd  = pick("chosen_sigma", "sigma", "std")
    col_mu_d = pick("chosen_mu_dried")
    col_sd_d = pick("chosen_sigma_dried")
    col_conv = pick(
        "conversion_from_dry_dosage_to_raw",
        "from_dry_dosage_to_raw",
        "dry_to_raw",
        "conv_dry_to_raw",
        "conversion_factor_dry_to_raw",
    )

    missing = [n for n, v in {"ingredient": col_ing, "nutrient": col_nut, "mu": col_mu, "sigma": col_sd}.items() if v is None]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")

    use_cols = [col_ing, col_nut, col_mu, col_sd]
    if col_mu_d: use_cols.append(col_mu_d)
    if col_sd_d: use_cols.append(col_sd_d)
    if col_conv: use_cols.append(col_conv)
    df_ing = df_ing[use_cols].copy()

    rename = {col_ing:"Ingredient", col_nut:"Nutrient", col_mu:"mu", col_sd:"sigma"}
    if col_mu_d: rename[col_mu_d] = "mu_dry"
    if col_sd_d: rename[col_sd_d] = "sigma_dry"
    if col_conv: rename[col_conv] = "conv"
    df_ing.columns = [rename.get(c, c) for c in df_ing.columns]

    df_ing["mu"] = pd.to_numeric(df_ing["mu"], errors="coerce").fillna(0.0)
    df_ing["sigma"] = pd.to_numeric(df_ing["sigma"], errors="coerce").fillna(0.0)
    if "mu_dry" in df_ing.columns: df_ing["mu_dry"] = pd.to_numeric(df_ing["mu_dry"], errors="coerce")
    if "sigma_dry" in df_ing.columns: df_ing["sigma_dry"] = pd.to_numeric(df_ing["sigma_dry"], errors="coerce")
    if "conv" in df_ing.columns: df_ing["conv"] = pd.to_numeric(df_ing["conv"], errors="coerce")

    ingr_map: Dict[str, dict] = {}
    for _, r in df_ing.iterrows():
        ing = str(r["Ingredient"])
        nut = str(r["Nutrient"])
        mu = float(r["mu"]); sd = float(r["sigma"])
        mu_dry = float(r["mu_dry"]) if ("mu_dry" in df_ing.columns and not pd.isna(r["mu_dry"])) else mu
        sd_dry = float(r["sigma_dry"]) if ("sigma_dry" in df_ing.columns and not pd.isna(r["sigma_dry"])) else sd
        conv = float(r["conv"]) if ("conv" in df_ing.columns and not pd.isna(r["conv"])) else None

        ingr_map.setdefault(ing, {})
        ingr_map[ing][nut] = {"mu": mu, "sigma": sd, "mu_dry": mu_dry, "sigma_dry": sd_dry}
        if conv is not None:
            ingr_map[ing]["_conv"] = conv

    for ing in ingr_map:
        if "_conv" not in ingr_map[ing]:
            ingr_map[ing]["_conv"] = 1.0

    return ingr_map


# ----------------- small helpers -----------------
def nutrients_present_in_data(ingr_map: Dict[str, dict]) -> set:
    present = set()
    for ing, d in ingr_map.items():
        for k in d.keys():
            if k == "_conv":
                continue
            present.add(k)
    return present


def ingredient_has_any_spec_nutrient(ing: str, INGR: Dict[str, dict], nutrients: List[str]) -> bool:
    keys = [k for k in INGR.get(ing, {}).keys() if k != "_conv"]
    return len(set(keys).intersection(nutrients)) > 0


def get_mu_sigma_for(INGR, ingredient: str, nutrient: str, is_dry: bool) -> Tuple[float, float]:
    rec = INGR.get(ingredient, {}).get(nutrient, None)
    if isinstance(rec, dict):
        if is_dry:
            mu = float(rec.get("mu_dry", rec.get("mu", 0.0)))
            sd = float(rec.get("sigma_dry", rec.get("sigma", 0.0)))
        else:
            mu = float(rec.get("mu", 0.0))
            sd = float(rec.get("sigma", 0.0))
        return mu, sd
    elif isinstance(rec, (list, tuple)) and len(rec) >= 2:
        return float(rec[0]), float(rec[1])
    return 0.0, 0.0


def base_cpk(mu: float, sigma: float, spec: dict) -> float:
    """Plain Cpk (no zero-σ penalty)."""
    if sigma <= EPS:
        # Treat as ±inf depending on inside/outside
        t = spec["type"]
        if t == "two_sided":
            L, U = spec["L"], spec["U"]
            return CPK_INF if (L <= mu <= U) else -CPK_INF
        elif t == "lower":
            L = spec["L"]; return CPK_INF if mu >= L else -CPK_INF
        else:
            U = spec["U"]; return CPK_INF if mu <= U else -CPK_INF
    if spec["type"] == "two_sided":
        L, U = spec["L"], spec["U"]
        return min((mu - L) / (3.0 * sigma), (U - mu) / (3.0 * sigma))
    elif spec["type"] == "lower":
        L = spec["L"]; return (mu - L) / (3.0 * sigma)
    else:
        U = spec["U"]; return (U - mu) / (3.0 * sigma)


def penalized_zero_sigma_cpk(mu: float, sigma: float, spec: dict, lam: float) -> float:
    """Optional reporting-only penalty that matches your rule."""
    if sigma > EPS or lam <= 0.0:
        return base_cpk(mu, sigma, spec)

    # sigma ≈ 0: first check inside/outside
    t = spec["type"]
    if t == "two_sided":
        L, U = spec["L"], spec["U"]
        inside = (L <= mu <= U); center = 0.5 * (L + U); scale = max((U - L) / 2.0, 1.0)
    elif t == "lower":
        L = spec["L"]; inside = (mu >= L); center = L; scale = max(0.1 * max(abs(L), 1.0), 1e-6)
    else:
        U = spec["U"]; inside = (mu <= U); center = U; scale = max(0.1 * max(abs(U), 1.0), 1e-6)

    if not inside:
        return -CPK_INF
    d = (mu - center) / scale
    return CPK_INF * math.exp(-lam * d * d)


# ---------------------- modeling (feasibility for a fixed t) ----------------------
def feasibility_model_for_t(
    t_val: float,
    mu_mat: np.ndarray,          # shape (Nnutr, Ningr)
    sig_mat: np.ndarray,         # shape (Nnutr, Ningr)
    specs_rows: List[dict],      # list aligned with nutrients order
    grams_total: float = ML_TOTAL,
) -> Tuple[bool, np.ndarray | None]:
    """
    Build and solve an SOC feasibility model for a fixed t.
    Returns (is_feasible, w) where w is weights if feasible, else None.
    """
    Nn, Ni = mu_mat.shape
    m = gp.Model("cpk_feasibility")
    m.Params.OutputFlag = 0  # quiet

    # decision: ingredient weights (grams)
    w = m.addVars(Ni, lb=0.0, name="w")

    # helper y_n bounds the sigma_n = || diag(sig_n) * w ||
    y = m.addVars(Nn, lb=0.0, name="y")

    # total = ML_TOTAL
    m.addConstr(gp.quicksum(w[i] for i in range(Ni)) == grams_total, name="total_grams")

    # Per-nutrient SOC + linear side constraints at level t_val
    for n in range(Nn):
        # SOC: || diag(sig_n) * w ||_2 <= y[n]
        # -> sum_i (sig[n,i] * w[i])^2 <= y[n]^2
        m.addQConstr(
            gp.quicksum((sig_mat[n, i] * w[i]) * (sig_mat[n, i] * w[i]) for i in range(Ni))
            <= y[n] * y[n],
            name=f"soc_sig_{n}",
        )

        mu_n = gp.quicksum(mu_mat[n, i] * w[i] for i in range(Ni))
        typ = specs_rows[n]["type"]

        if typ == "two_sided":
            L = float(specs_rows[n]["L"])
            U = float(specs_rows[n]["U"])
            m.addConstr(mu_n - L >= 3.0 * t_val * y[n], name=f"two_low_{n}")
            m.addConstr(U - mu_n >= 3.0 * t_val * y[n], name=f"two_up_{n}")
        elif typ == "lower":
            L = float(specs_rows[n]["L"])
            m.addConstr(mu_n - L >= 3.0 * t_val * y[n], name=f"low_{n}")
        else:  # upper
            U = float(specs_rows[n]["U"])
            m.addConstr(U - mu_n >= 3.0 * t_val * y[n], name=f"up_{n}")

    m.optimize()
    if m.Status == GRB.OPTIMAL:
        w_star = np.array([w[i].X for i in range(Ni)], dtype=float)
        return True, w_star
    else:
        return False, None


def maximize_min_cpk_bisection(
    mu_mat: np.ndarray,
    sig_mat: np.ndarray,
    specs_rows: List[dict],
    grams_total: float = ML_TOTAL,
    tol: float = 1e-4,
    t_hi_cap: float = 50.0,
) -> Tuple[float, np.ndarray]:
    """
    Bisection on t: find largest feasible t.
    """
    # Find an upper bound by expansion
    lo, hi = 0.0, 0.5
    w_best = None
    feas, w = feasibility_model_for_t(lo, mu_mat, sig_mat, specs_rows, grams_total)
    if not feas:
        raise RuntimeError("Feasibility at t=0 failed; check inputs.")
    w_best = w

    # grow hi until infeasible (or cap)
    while hi < t_hi_cap:
        feas, w = feasibility_model_for_t(hi, mu_mat, sig_mat, specs_rows, grams_total)
        if feas:
            lo = hi
            w_best = w
            hi *= 2.0
        else:
            break

    # if even the cap is feasible, keep that
    if hi >= t_hi_cap:
        return lo, w_best

    # binary search between lo (feasible) and hi (infeasible)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        feas, w = feasibility_model_for_t(mid, mu_mat, sig_mat, specs_rows, grams_total)
        if feas:
            lo = mid
            w_best = w
        else:
            hi = mid
        if (hi - lo) <= tol * max(1.0, hi):
            break

    return lo, w_best


# ---------------------- run end-to-end ----------------------
def main(
    spec_path: str = SPEC_PATH,
    ingr_path: str = INGR_PATH,
    is_dry: bool = IS_DRY,
    lambda_zero_sigma: float = LAMBDA_ZERO_SIGMA,
):
    # Load inputs
    RECIPE_SPECS, limits_df = parse_limits_file(spec_path)
    INGR = load_ingr_map_from_excel(ingr_path)

    # (Optional) exclude water in dry mode
    all_ingredients = list(INGR.keys())
    if is_dry:
        all_ingredients = [i for i in all_ingredients if i.lower() != "water"]

    # In-scope nutrients = intersection
    data_nutrients = nutrients_present_in_data(INGR)
    nutrients = [n for n in RECIPE_SPECS.keys() if n in data_nutrients]
    if not nutrients:
        raise RuntimeError("No overlapping nutrients between limits and ingredient dataset.")

    # Eliminate ingredients with none of the in-scope nutrients
    ingredients_in_scope = [i for i in all_ingredients if ingredient_has_any_spec_nutrient(i, INGR, nutrients)]
    if not ingredients_in_scope:
        raise RuntimeError("No ingredients remain after filtering. Check your files/limits.")

    # Build matrices (nutrients x ingredients)
    Nn = len(nutrients)
    Ni = len(ingredients_in_scope)
    mu_mat = np.zeros((Nn, Ni), dtype=float)
    sig_mat = np.zeros((Nn, Ni), dtype=float)

    specs_rows: List[dict] = []
    for n_idx, nut in enumerate(nutrients):
        s = RECIPE_SPECS[nut]
        specs_rows.append({"type": s["type"], "L": s.get("L", None), "U": s.get("U", None)})
        for i_idx, ing in enumerate(ingredients_in_scope):
            mu, sd = get_mu_sigma_for(INGR, ing, nut, is_dry)
            mu_mat[n_idx, i_idx] = mu
            sig_mat[n_idx, i_idx] = sd

    # Solve via bisection
    t0 = time.perf_counter()
    t_star, w_star = maximize_min_cpk_bisection(mu_mat, sig_mat, specs_rows, grams_total=ML_TOTAL)
    elapsed = time.perf_counter() - t0

    # Prepare outputs
    weights_df = pd.DataFrame(
        {"Ingredient": ingredients_in_scope, "Qty (g per 100 g)": np.round(w_star, 6)}
    ).sort_values("Qty (g per 100 g)", ascending=False).reset_index(drop=True)

    # Per nutrient μ, σ, Cpk
    rows = []
    for n_idx, nut in enumerate(nutrients):
        mu = float(np.dot(w_star / ML_TOTAL, mu_mat[n_idx, :])) * ML_TOTAL  # careful! mu is linear in *fraction* weights
        # The above re-multiplies by ML_TOTAL, so rewrite cleaner:
        # w_frac = w_star / ML_TOTAL; mu = w_frac @ mu_i
        w_frac = w_star / ML_TOTAL
        mu = float(np.dot(w_frac, mu_mat[n_idx, :]))
        sigma = math.sqrt(float(np.dot((w_frac ** 2), (sig_mat[n_idx, :] ** 2))))
        spec = RECIPE_SPECS[nut]
        cpk_plain = base_cpk(mu, sigma, spec)
        cpk_pen   = penalized_zero_sigma_cpk(mu, sigma, spec, lam=lambda_zero_sigma)
        rows.append(
            {
                "nutrient": nut,
                "type": spec["type"],
                "L": spec.get("L", None),
                "U": spec.get("U", None),
                "mu": mu,
                "sigma": sigma,
                "Cpk_plain": cpk_plain,
                "Cpk_penalized" if lambda_zero_sigma > 0 else "Cpk_penalized (λ=0)": cpk_pen,
            }
        )
    results_df = pd.DataFrame(rows)
    # Round for display
    for c in ["mu", "sigma", "L", "U", "Cpk_plain", "Cpk_penalized", "Cpk_penalized (λ=0)"]:
        if c in results_df.columns:
            results_df[c] = results_df[c].apply(
                lambda v: None if pd.isna(v) else (round(float(v), 6) if isinstance(v, (int, float)) else v)
            )

    # Print summary
    print("\n=== Max-Min Cpk via Gurobi (bisection SOC) ===")
    print(f"Nutrients used: {len(nutrients)} -> {nutrients}")
    print(f"Ingredients used: {len(ingredients_in_scope)} -> {ingredients_in_scope}")
    print(f"Best minimum Cpk (t*): {t_star:.6f}")
    print(f"Solve time: {elapsed:.3f} s\n")

    print("Per-nutrient μ/σ/Cpk:")
    print(results_df.to_string(index=False))

    print("\nOptimal quantities per 100 g:")
    print(weights_df.to_string(index=False))

    # Also return dataframes (useful in notebooks)
    return {
        "t_star": t_star,
        "elapsed_seconds": elapsed,
        "weights_df": weights_df,
        "results_df": results_df,
        "ingredients": ingredients_in_scope,
        "nutrients": nutrients,
    }


if __name__ == "__main__":
    _ = main(
        spec_path=SPEC_PATH,
        ingr_path=INGR_PATH,
        is_dry=IS_DRY,
        lambda_zero_sigma=LAMBDA_ZERO_SIGMA,
    )
    # In a company Jupyter where gurobipy is available
    res = main(
        spec_path="pov_tool_v8_test.xlsx",
        ingr_path="rm_nutrient_distributions_for_optimizer_with_process_losses.xlsx",
        is_dry=False,                 # or True if you want dried stats & exclude Water
        lambda_zero_sigma=0.0         # set >0 to also show your zero-σ centricity penalty (reporting only)
    )
    res["results_df"], res["weights_df"]
