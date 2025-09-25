import streamlit as st
import pandas as pd
import numpy as np
import itertools as it
import math
import matplotlib.pyplot as plt
import os
import time  # ⬅️ timing
from scipy.optimize import minimize

st.set_page_config(page_title="Cpk Optimizer", page_icon="🧪", layout="wide")

# -------------------------
# Mock DATA LAYER (dicts) — will be overridden by Excel load
# -------------------------
INGR = {
    "Orange": {"sugar": (9.0, 1.0), "vitC": (50.0, 5.0)},
    "Apple": {"sugar": (11.0, 0.8), "vitC": (30.0, 4.0)},
    "Grape": {"sugar": (16.0, 1.2), "vitC": (10.0, 2.0)},
    "Pineapple": {"sugar": (13.5, 0.9), "vitC": (25.0, 3.0)},
    "Water": {"sugar": (0.0, 0.1), "vitC": (0.0, 0.1)},
}

RECIPES = {
    "Citrus Blend": ["Orange", "Apple", "Pineapple", "Water"],
    "Grape Punch": ["Grape", "Apple", "Water"],
    "House Mix": list(INGR.keys()),
}

RECIPE_SPECS = {
    "Citrus Blend": {
        "sugar": {"type": "two_sided", "L": 10.0, "U": 20.0},
        "vitC": {"type": "lower", "L": 20.0},
    },
    "Grape Punch": {
        "sugar": {"type": "two_sided", "L": 10.0, "U": 20.0},
        "vitC": {"type": "lower", "L": 18.0},
    },
    "House Mix": {
        "sugar": {"type": "two_sided", "L": 10.0, "U": 20.0},
        "vitC": {"type": "lower", "L": 22.0},
    },
}

# -------------------------
# Core math helpers
# -------------------------
ML_TOTAL = 100   # total batch weight (grams)
CPK_INF = 100    # treat any “very large” Cpk as 100
EPS = 1e-12

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

def compute_cpk(mu, sigma, spec):
    t = spec["type"].lower()
    if t == "two_sided": return cpk_two_sided(mu, sigma, spec["L"], spec["U"])
    if t == "lower":     return cpk_lower(mu, sigma, spec["L"])
    if t == "upper":     return cpk_upper(mu, sigma, spec["U"])
    raise ValueError("spec['type'] must be 'two_sided'|'lower'|'upper'")

# ---- Zero-σ handling (updated: penalties for all types) + hard cap when σ>EPS ----
def cpk_with_zero_sigma_penalty(mu: float, sigma: float, spec: dict, lam: float) -> float:
    """
    - If σ > EPS: compute base Cpk and cap to [-100, +100].
    - If σ <= EPS:
        * If μ is OUTSIDE spec -> return -100 directly (hard fail).
        * If μ is INSIDE spec:
            - two_sided:  100 * exp(-λ * d_center^2)       (closer to center is better)
            - lower:      100 * (1 - exp(-λ * d_in^2))      (farther above L is better)
            - upper:      100 * (1 - exp(-λ * d_in^2))      (farther below U is better)
      where distances are normalized by a scale to keep λ meaningful.
    """
    # Normal sigma: compute and cap
    if sigma > EPS:
        base = compute_cpk(mu, sigma, spec)
        return max(min(base, CPK_INF), -CPK_INF)

    # σ≈0: special handling by spec type
    t = spec["type"].lower()

    if t == "two_sided":
        L, U = spec["L"], spec["U"]
        if not (L <= mu <= U):
            return -CPK_INF
        if lam <= 0.0:
            return CPK_INF
        center = 0.5 * (L + U)
        scale = max((U - L) / 2.0, 1.0)
        d_norm = (mu - center) / scale
        val = CPK_INF * math.exp(-lam * (d_norm ** 2))
        return max(min(val, CPK_INF), -CPK_INF)

    elif t == "lower":
        L = spec["L"]
        if mu < L:
            return -CPK_INF
        if lam <= 0.0:
            return CPK_INF
        # distance *inside* spec (how far above L)
        scale = max(0.1 * max(abs(L), 1.0), 1e-6)
        d_in = max(0.0, mu - L) / scale
        # reward being deeper inside: 0 at boundary, →100 as distance grows
        val = CPK_INF * (1.0 - math.exp(-lam * (d_in ** 2)))
        return max(min(val, CPK_INF), -CPK_INF)

    elif t == "upper":
        U = spec["U"]
        if mu > U:
            return -CPK_INF
        if lam <= 0.0:
            return CPK_INF
        # distance *inside* spec (how far below U)
        scale = max(0.1 * max(abs(U), 1.0), 1e-6)
        d_in = max(0.0, U - mu) / scale
        # reward being deeper inside: 0 at boundary, →100 as distance grows
        val = CPK_INF * (1.0 - math.exp(-lam * (d_in ** 2)))
        return max(min(val, CPK_INF), -CPK_INF)

    else:
        raise ValueError("spec['type'] must be 'two_sided'|'lower'|'upper'")

def nutrients_present_in_data(ingr_map: dict) -> set:
    present = set()
    for ing, d in ingr_map.items():
        for k in d.keys():
            if k == "_conv":
                continue
            present.add(k)
    return present

# -------------------------
# Ingredient distributions loader (supports Dry + conversion)
# -------------------------
DATA_XLSX = "data/rm_nutrient_distributions_for_optimizer_with_process_losses_v2.xlsx"

def load_ingr_map_from_excel(path: str) -> dict:
    """
    Excel columns (case-insensitive):
      - 'IS_NAME' or 'Ingredient'
      - 'NUTRIENTCODE'
      - 'chosen_mu' (fallback: 'mu','mean')
      - 'chosen_sigma' (fallback: 'sigma','std')
    Optional for Dry mode:
      - 'chosen_mu_dried', 'chosen_sigma_dried'
      - 'conversion_from_dry_dosage_to_raw'  (or similar)
    """
    if not os.path.exists(path):
        st.error(f"Ingredient distribution file not found: {path}")
        return {}

    df_ing = pd.read_excel(path)
    cols = {c.lower(): c for c in df_ing.columns}

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
        st.error(f"Missing required columns in {path}: {', '.join(missing)}")
        return {}

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

    ingr_map: dict[str, dict] = {}
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

def _value_is_tuple_stats(v):
    return isinstance(v, (tuple, list)) and len(v) >= 2 and all(isinstance(x, (int, float)) for x in v[:2])

def get_stats(ingr_map, ing, nutrient, is_dry_mode: bool):
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

# -------------------------
# OPTIMIZER (with Liquid/Dry toggle)
# -------------------------
optimizer_tab, = st.tabs(["Optimizer"])

with optimizer_tab:
    st.title("🧪 Recipe Cpk Optimizer")

    # 1) Upload legislation/spec
    st.subheader("Legislation / spec limits")
    st.caption(
        "Upload CSV/XLSX. Columns can be any of: "
        "nutrient|bm_code|nutrientcode, and min|min_std|1.2 min_std|lower, "
        "max|max_std|1.2 max_std|upper."
    )

    def parse_limits_file(file) -> tuple[dict[str, dict], pd.DataFrame]:
        name = file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        cols = {c.strip().lower(): c for c in df.columns}
        def pick(*cands):
            for k in cands:
                if k in cols:
                    return cols[k]
            return None

        code_col = pick("nutrient", "bm_code", "nutrientcode", "code", "bm code")
        L_col    = pick("1.2 min_std", "min_std", "min std", "min", "spec_min", "l", "lower", "low")
        U_col    = pick("1.2 max_std", "max_std", "max std", "max", "spec_max", "u", "upper", "up")
        type_col = pick("type", "spec_type")  # optional

        if not code_col:
            raise ValueError(f"Couldn’t find a nutrient code column. Found: {list(df.columns)}")

        if L_col: df[L_col] = pd.to_numeric(df[L_col], errors="coerce")
        if U_col: df[U_col] = pd.to_numeric(df[U_col], errors="coerce")

        rows = []
        specs: dict[str, dict] = {}

        for _, r in df.iterrows():
            code = str(r[code_col]).strip()
            if not code:
                continue

            L = float(r[L_col]) if (L_col and pd.notna(r[L_col])) else None
            U = float(r[U_col]) if (U_col and pd.notna(r[U_col])) else None

            typ = None
            if (type_col and pd.notna(r[type_col])):
                t = str(r[type_col]).strip().lower()
                if t in {"two_sided", "lower", "upper"}:
                    typ = t

            if typ is None:
                if (L is not None) and (U is not None):
                    typ = "two_sided"
                elif L is not None:
                    typ = "lower"
                elif U is not None:
                    typ = "upper"
                else:
                    continue

            if typ == "two_sided" and L is not None and U is not None and L > U:
                L, U = U, L

            if typ == "two_sided":
                specs[code] = {"type": "two_sided", "L": L, "U": U}
            elif typ == "lower":
                specs[code] = {"type": "lower", "L": L}
            else:
                specs[code] = {"type": "upper", "U": U}

            rows.append({"nutrient": code, "type": typ, "L": L, "U": U})

        return specs, pd.DataFrame(rows, columns=["nutrient", "type", "L", "U"])

    legis_file = st.file_uploader(
        "Upload limits (CSV or XLSX)", type=["csv", "xlsx", "xls"], key="legis_uploader_new"
    )
    if not legis_file:
        st.warning("Upload limits to continue.")
        st.stop()

    try:
        RECIPE_SPECS, limits_df = parse_limits_file(legis_file)
        if not RECIPE_SPECS:
            st.error("No usable limits found in the file (no min/max on any row).")
            st.stop()
        st.success(f"Loaded {len(RECIPE_SPECS)} nutrient limits.")
    except Exception as e:
        st.error(f"Couldn't read limits: {e}")
        st.stop()

    # 2) Liquid/Dry switch
    is_dry = st.toggle(
        "🧂 Dry recipe mode (uses dried μ/σ and shows raw conversion; Water is excluded)",
        value=False
    )
    st.session_state["is_dry_mode"] = is_dry

    # 3) Load ingredient distributions & set allowed ingredients
    INGR = load_ingr_map_from_excel(DATA_XLSX)
    if not INGR:
        st.stop()

    all_ingredients = list(INGR.keys())
    if is_dry:
        all_ingredients = [i for i in all_ingredients if i.lower() != "water"]

    st.caption(f"Loaded **{len(all_ingredients)}** ingredients from {DATA_XLSX} • Mode: **{'Dry' if is_dry else 'Liquid'}**")

    # ---- Review & adjust limits ----
    st.markdown("#### Review & adjust limits")
    editable = limits_df.copy()
    editable["type"] = editable["type"].str.lower().str.strip()
    editable = editable[editable["type"].isin(["two_sided", "lower", "upper"])].reset_index(drop=True)

    type_options = ["two_sided", "lower", "upper"]
    edited = st.data_editor(
        editable,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "type": st.column_config.SelectboxColumn(options=type_options),
            "L": st.column_config.NumberColumn(format="%.6f"),
            "U": st.column_config.NumberColumn(format="%.6f"),
        },
        key="limits_editor",
    )

    def _valid_row(r):
        if r["type"] == "two_sided": return pd.notna(r["L"]) and pd.notna(r["U"])
        if r["type"] == "lower":     return pd.notna(r["L"])
        if r["type"] == "upper":     return pd.notna(r["U"])
        return False

    edited = edited[edited.apply(_valid_row, axis=1)].copy()

    # Rebuild RECIPE_SPECS from edited table
    RECIPE_SPECS = {}
    for r in edited.itertuples(index=False):
        n, t, L, U = r.nutrient, r.type, r.L, r.U
        L = None if pd.isna(L) else float(L)
        U = None if pd.isna(U) else float(U)
        if t == "two_sided":
            if L > U: L, U = U, L
            RECIPE_SPECS[n] = {"type": "two_sided", "L": L, "U": U}
        elif t == "lower":
            RECIPE_SPECS[n] = {"type": "lower", "L": L}
        else:
            RECIPE_SPECS[n] = {"type": "upper", "U": U}

    # Restrict to nutrients present in ingredient dataset
    data_nutrients = nutrients_present_in_data(INGR)
    nutrients = [n for n in RECIPE_SPECS.keys() if n in data_nutrients]
    missing = sorted(set(RECIPE_SPECS.keys()) - set(nutrients))
    if missing:
        st.warning(
            "Some spec nutrients are not present in the ingredient dataset (or have no μ/σ): "
            + ", ".join(missing)
        )

    if not nutrients:
        st.error("No overlapping nutrients between limits and ingredient dataset. Please check your files.")
        st.stop()

    # Eliminate ingredients BEFORE any implicit zero-fill:
    def ingredient_has_any_spec_nutrient(ing: str) -> bool:
        keys = [k for k in INGR.get(ing, {}).keys() if k != "_conv"]
        return len(set(keys).intersection(nutrients)) > 0

    ingredients_in_scope = [i for i in all_ingredients if ingredient_has_any_spec_nutrient(i)]
    dropped = sorted(set(all_ingredients) - set(ingredients_in_scope))
    if dropped:
        st.info("Excluded ingredients with no in-scope nutrients: " + ", ".join(dropped))

    if not ingredients_in_scope:
        st.error("No ingredients remain after filtering. Adjust your limits or ingredient file.")
        st.stop()

    st.caption("**Nutrients in scope (used in optimization):** " + ", ".join(nutrients))
    st.caption("**Ingredients in scope (used in optimization):** " + ", ".join(ingredients_in_scope))

    # ---------- Mixture stats (uses Liquid/Dry μ/σ) -------
    def mix_mu_sigma_from_map(ingredients, vols_g, nutrient, is_dry_mode: bool):
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

    # ---------------- Sidebar: objective + penalty ----------------
    with st.sidebar:
        st.header("Optimization setup")
        strategy = st.selectbox(
            "Objective",
            ["Average Cpk", "Minimum Cpk (worst nutrient)"],
            help="We maximize this Cpk metric with SLSQP.",
        )
        score_is_avg = ("Average" in strategy)
        n_restarts = st.number_input(
            "Random restarts", 1, 100, 8, help="Multi-start SLSQP to reduce local minima risk."
        )
        lam_pen = st.slider(
            "λ for σ=0 penalty (Option A)",
            min_value=0.0, max_value=5.0, value=1.0, step=0.1,
            help="Penalty only applies when σ≈0. Higher λ penalizes means farther from the spec center more strongly."
        )

    # -------------------- Objective for SLSQP --------------
    def score_from_vols(vols):
        per_cpk = []
        for n in nutrients:
            mu, sigma = mix_mu_sigma_from_map(ingredients_in_scope, vols, n, is_dry)
            cpk_eff = cpk_with_zero_sigma_penalty(mu, sigma, RECIPE_SPECS[n], lam_pen)
            per_cpk.append(cpk_eff)
        return float(np.mean(per_cpk)) if score_is_avg else float(np.min(per_cpk))

    def objective(x):
        # Negative because SLSQP minimizes
        return -score_from_vols(x)

    # ------------------------- Solve with SLSQP ------------------------------

    m = len(ingredients_in_scope)
    bounds = [(0.0, ML_TOTAL)] * m              # x_i >= 0
    cons = [{"type": "eq", "fun": lambda x: np.sum(x) - ML_TOTAL}]  # sum = 100 g

    rng = np.random.default_rng(42)
    starts = [np.full(m, ML_TOTAL / m, dtype=float)]
    for _ in range(int(n_restarts) - 1):
        w = rng.dirichlet(np.ones(m))
        starts.append(w * ML_TOTAL)

    best = None
    best_fun = float("inf")

    run = st.button("🚀 Optimize (SLSQP)", type="primary")
    if not run:
        st.caption("Choose objective, then click **Optimize (SLSQP)**.")
        st.stop()

    t0 = time.perf_counter()  # ⬅️ start timing
    with st.spinner("Solving…"):
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
    elapsed_sec = time.perf_counter() - t0  # ⬅️ end timing

    if best is None:
        st.error("Optimization failed to converge. Try adjusting limits.")
        st.stop()

    # Renormalize (just in case)
    x_opt = np.clip(best.x, 0.0, None)
    s = float(np.sum(x_opt))
    if s <= 0:
        st.error("Optimizer returned a degenerate solution (total weight = 0).")
        st.stop()
    x_opt = x_opt * (ML_TOTAL / s)

    # -------------------------- Present the result --------------------------
    opt_score = score_from_vols(x_opt)
    st.success(f"Optimization finished in {elapsed_sec:.3f} s. Best **{strategy}** = {opt_score:.6f}")

    # Per-nutrient μ, σ, Cpk (with penalty & hard cap)
    rows = []
    for n in nutrients:
        mu, sigma = mix_mu_sigma_from_map(ingredients_in_scope, x_opt, n, is_dry)
        rows.append(
            {
                "nutrient": n,
                "type": RECIPE_SPECS[n]["type"],
                "L": RECIPE_SPECS[n].get("L", None),
                "U": RECIPE_SPECS[n].get("U", None),
                "mu": mu,
                "sigma": sigma,
                "Cpk": cpk_with_zero_sigma_penalty(mu, sigma, RECIPE_SPECS[n], lam_pen),
            }
        )
    df_specs = pd.DataFrame(rows)

    c1, c2 = st.columns([1, 1])

    if not is_dry:
        qty_df = pd.DataFrame(
            {"Ingredient": ingredients_in_scope, "Quantity (g) per 100 g": np.round(x_opt, 6)}
        )
        qty_df = (
            qty_df[qty_df["Quantity (g) per 100 g"] > 0.01]
            .sort_values("Quantity (g) per 100 g", ascending=False)
            .reset_index(drop=True)
        )
        with c1:
            st.subheader("Optimal quantities (per 100 g)")
            st.dataframe(qty_df, use_container_width=True)
    else:
        dried = np.round(x_opt, 6)
        raw = []
        convs = []
        for ing, q in zip(ingredients_in_scope, dried):
            conv = float(INGR.get(ing, {}).get("_conv", 1.0))
            convs.append(conv)
            raw.append(q * conv)

        qty_df = pd.DataFrame(
            {
                "Ingredient": ingredients_in_scope,
                "Dried Qty (g) per 100 g": dried,
                "→ Raw factor (conversion_from_dry_dosage_to_raw)": np.round(convs, 6),
                "Raw Qty (g) per 100 g": np.round(raw, 6),
            }
        )
        qty_df = (
            qty_df[qty_df["Dried Qty (g) per 100 g"] > 0.01]
            .sort_values("Dried Qty (g) per 100 g", ascending=False)
            .reset_index(drop=True)
        )
        with c1:
            st.subheader("Optimal quantities (Dry)")
            st.dataframe(qty_df, use_container_width=True)

    with c2:
        st.subheader("Mixture stats per nutrient")
        show = df_specs.copy()
        for c in ["mu", "sigma", "Cpk", "L", "U"]:
            if c in show.columns:
                show[c] = show[c].apply(
                    lambda v: None
                    if pd.isna(v)
                    else (round(float(v), 6) if isinstance(v, (int, float)) else v)
                )
        st.dataframe(show, use_container_width=True)

    # Single-row CSV export
    out_row = {
        "objective": strategy,
        "score_value": opt_score,
        "mode": "Dry" if is_dry else "Liquid",
        "lambda_penalty": lam_pen,
        "elapsed_seconds": elapsed_sec,                 # ⬅️ timing in CSV
        "best_restart_iters": getattr(best, "nit", None)
    }
    if not is_dry:
        out_row.update({f"Qty {ing} (g per 100 g)": v for ing, v in zip(ingredients_in_scope, x_opt)})
    else:
        for ing, q in zip(ingredients_in_scope, x_opt):
            conv = float(INGR.get(ing, {}).get("_conv", 1.0))
            out_row[f"Dried {ing} (g per 100 g)"] = q
            out_row[f"Raw {ing} (g per 100 g)"] = q * conv
            out_row[f"{ing} factor dry→raw"] = conv

    for r in rows:
        out_row[f"Cpk[{r['nutrient']}]"] = r["Cpk"]
        out_row[f"mu[{r['nutrient']}]"] = r["mu"]
        out_row[f"sigma[{r['nutrient']}]"] = r["sigma"]

    out_df = pd.DataFrame([out_row])
    st.download_button(
        "⬇️ Download optimal recipe (CSV)",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name="cpk_optimal_recipe.csv",
        mime="text/csv",
    )
