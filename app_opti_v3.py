import os
import math
import itertools as it

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# App config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Cpk Optimizer", page_icon="🧪", layout="wide")

# -----------------------------------------------------------------------------
# Demo data (kept as-is)
# -----------------------------------------------------------------------------
INGR = {
    "Orange":    {"sugar": (9.0, 1.0),  "vitC": (50.0, 5.0)},
    "Apple":     {"sugar": (11.0, 0.8), "vitC": (30.0, 4.0)},
    "Grape":     {"sugar": (16.0, 1.2), "vitC": (10.0, 2.0)},
    "Pineapple": {"sugar": (13.5, 0.9), "vitC": (25.0, 3.0)},
    "Water":     {"sugar": (0.0, 0.1),  "vitC": (0.0, 0.1)},
}

RECIPES = {
    "Citrus Blend": ["Orange", "Apple", "Pineapple", "Water"],
    "Grape Punch":  ["Grape", "Apple", "Water"],
    "House Mix":    list(INGR.keys()),
}

RECIPE_SPECS = {
    "Citrus Blend": {
        "sugar": {"type": "two_sided", "L": 10.0, "U": 20.0},
        "vitC":  {"type": "lower",     "L": 20.0},
    },
    "Grape Punch": {
        "sugar": {"type": "two_sided", "L": 10.0, "U": 20.0},
        "vitC":  {"type": "lower",     "L": 18.0},
    },
    "House Mix": {
        "sugar": {"type": "two_sided", "L": 10.0, "U": 20.0},
        "vitC":  {"type": "lower",     "L": 22.0},
    },
}

BASELINE_CPK = {
    "Citrus Blend": {"sugar": 1.10, "vitC": 1.00},
    "Grape Punch":  {"sugar": 1.05, "vitC": 0.95},
    "House Mix":    {"sugar": 1.12, "vitC": 1.00},
}

COST_PER_ML = {
    "Orange": 0.020,
    "Apple":  0.018,
    "Grape":  0.030,
    "Pineapple": 0.025,
    "Water":  0.000,
}

# -----------------------------------------------------------------------------
# Core math helpers (kept as-is)
# -----------------------------------------------------------------------------
ML_TOTAL = 100
CPK_INF = 1e9
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

def mix_mu_sigma(ingredients, vols_ml, nutrient):
    w = [v / ML_TOTAL for v in vols_ml]
    mus, sig = [], []
    for i in ingredients:
        mu_i, sd_i = INGR.get(i, {}).get(nutrient, (0.0, 0.0))  # <-- safe get
        mus.append(mu_i)
        sig.append(sd_i)
    mu = sum(wi * mi for wi, mi in zip(w, mus))
    var = sum((wi ** 2) * (si ** 2) for wi, si in zip(w, sig))
    return mu, math.sqrt(max(var, 0.0))


def average_cpk(cpk_map):
    vals = list(cpk_map.values())
    return float(np.mean(vals)) if vals else float("nan")

def min_cpk(cpk_map):
    vals = list(cpk_map.values())
    return float(np.min(vals)) if vals else float("nan")

def integer_compositions(total_units, k):   
    for cuts in it.combinations(range(1, total_units), k-1):
        prev = 0; parts = []
        for c in cuts + (total_units,):
            parts.append(c - prev); prev = c
        yield tuple(parts)

def generate_mixtures(ingredients, step_ml, min_k, max_k):
    assert ML_TOTAL % step_ml == 0
    units = ML_TOTAL // step_ml
    for k in range(min_k, max_k+1):
        for combo in it.combinations(ingredients, k):
            for parts in integer_compositions(units, k):
                vols = [p*step_ml for p in parts]
                yield combo, vols

def read_legislation_specs(uploaded_file) -> dict[str, dict]:
    if uploaded_file is None:
        return {}
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    lower_map = {c.lower().strip(): c for c in df.columns}
    def pick(*cands):
        for k in cands:
            if k in lower_map:
                return lower_map[k]
        return None
    code_col = pick("bm_code", "nutrientcode", "nutrient", "code")
    L_col    = pick("1.2 min_std", "min_std", "min", "spec_min", "l", "lower")
    U_col    = pick("1.2 max_std", "max_std", "max", "spec_max", "u", "upper")
    if code_col is None:
        raise ValueError("Legislation file must have a nutrient code column (e.g. 'bm_code').")
    if L_col: df[L_col] = pd.to_numeric(df[L_col], errors="coerce")
    if U_col: df[U_col] = pd.to_numeric(df[U_col], errors="coerce")
    specs = {}
    for _, r in df.iterrows():
        code = str(r[code_col]).strip()
        if not code:
            continue
        L = float(r[L_col]) if (L_col and pd.notna(r[L_col])) else None
        U = float(r[U_col]) if (U_col and pd.notna(r[U_col])) else None
        if (L is None) and (U is None):
            continue
        if (L is not None) and (U is not None):
            if L > U:
                L, U = U, L
            specs[code] = {"type": "two_sided", "L": L, "U": U}
        elif L is not None:
            specs[code] = {"type": "lower", "L": L}
        else:
            specs[code] = {"type": "upper", "U": U}
    return specs

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
optimizer_tab, explainer_tab, practice_tab = st.tabs(["Optimizer", "CPk Explainer", "practice tab"])

# -----------------------------------------------------------------------------
# Ingredient distributions from Excel
# -----------------------------------------------------------------------------
DATA_XLSX = "data/ingredient_nutrient_distributions_checked.xlsx"

def load_ingr_map_from_excel(path: str) -> dict:
    if not os.path.exists(path):
        st.error(f"Ingredient distribution file not found: {path}")
        return {}
    df_ing = pd.read_excel(path)
    cols = {c.lower(): c for c in df_ing.columns}
    col_ing = cols.get("is_name", cols.get("ingredient"))
    col_nut = cols.get("nutrientcode", None)
    col_mu  = cols.get("chosen_mu", cols.get("mu", cols.get("mean")))
    col_sd  = cols.get("chosen_sigma", cols.get("sigma", cols.get("std")))
    missing = [n for n, v in {"ingredient": col_ing, "nutrient": col_nut, "mu": col_mu, "sigma": col_sd}.items() if v is None]
    if missing:
        st.error(f"Missing required columns in {path}: {', '.join(missing)}")
        return {}
    df_ing = df_ing[[col_ing, col_nut, col_mu, col_sd]].copy()
    df_ing.columns = ["Ingredient", "Nutrient", "mu", "sigma"]
    df_ing["mu"]    = pd.to_numeric(df_ing["mu"], errors="coerce").fillna(0.0)
    df_ing["sigma"] = pd.to_numeric(df_ing["sigma"], errors="coerce").fillna(0.0)
    ingr_map: dict[str, dict[str, tuple[float, float]]] = {}
    for ing, nut, mu, sd in df_ing.itertuples(index=False, name=None):
        ingr_map.setdefault(str(ing), {})[str(nut)] = (float(mu), float(sd))
    return ingr_map

# -----------------------------------------------------------------------------
# OPTIMIZER tab
# -----------------------------------------------------------------------------
with optimizer_tab:
    st.title("🧪 Recipe Cpk Optimizer")

    # 0) Load ingredient distributions
    INGR_MAP = load_ingr_map_from_excel(DATA_XLSX)
    if not INGR_MAP:
        st.stop()
    all_ingredients = list(INGR_MAP.keys())
    st.caption(f"Loaded **{len(all_ingredients)}** ingredients from `{DATA_XLSX}`.")

    # 1) Limits uploader + robust parser (kept) --------------------------------
    st.subheader("Legislation / spec limits")
    st.caption(
        "Upload CSV/XLSX. Columns can be any of: "
        "`nutrient|bm_code|nutrientcode`, and `min|min_std|1.2 min_std|lower`, "
        "`max|max_std|1.2 max_std|upper`."
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
        type_col = pick("type", "spec_type")   # optional

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
            if type_col and pd.notna(r[type_col]):
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

        limits_df = pd.DataFrame(rows, columns=["nutrient", "type", "L", "U"])
        return specs, limits_df

    legis_file = st.file_uploader("Upload limits (CSV or XLSX)", type=["csv", "xlsx", "xls"], key="legis_uploader_new")
    if not legis_file:
        st.warning("Upload limits to continue.")
        st.stop()

    try:
        SPECS_FLAT, limits_df = parse_limits_file(legis_file)
        if not SPECS_FLAT:
            st.error("No usable limits found in the file (no min/max on any row).")
            st.stop()
        st.success(f"Loaded {len(SPECS_FLAT)} nutrient limits.")
    except Exception as e:
        st.error(f"Couldn't read limits: {e}")
        st.stop()

    nutrients = list(SPECS_FLAT.keys())

    st.write("Nutrients in scope:", ", ".join(nutrients))

    # 2) Review & adjust limits (FIX: editor key depends on file contents) -----
    st.markdown("#### Review & adjust limits")

    editable = limits_df.copy()
    editable["type"] = editable["type"].str.lower().str.strip()
    editable = editable[editable["type"].isin(["two_sided", "lower", "upper"])].reset_index(drop=True)

    type_options = ["two_sided", "lower", "upper"]

    # --- SPEC LOADING FIX ---
    # Re-key the editor with a fingerprint of the uploaded/parsed limits so that
    # re-uploading a different/longer file resets the editor state & rows.
    def _fingerprint_limits(df: pd.DataFrame) -> str:
        if df.empty:
            return "empty"
        canon = df.fillna("").astype(str)[["nutrient", "type", "L", "U"]]
        fp = int(pd.util.hash_pandas_object(canon, index=False).sum())
        return f"{fp}"

    spec_editor_key = f"limits_editor_{_fingerprint_limits(editable)}"

    edited = st.data_editor(
        editable,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "type": st.column_config.SelectboxColumn(options=type_options),
            "L": st.column_config.NumberColumn(format="%.6f"),
            "U": st.column_config.NumberColumn(format="%.6f"),
        },
        key=spec_editor_key,  # <-- key changes when file/content changes
    )

    def _valid_row(r):
        if r["type"] == "two_sided": return pd.notna(r["L"]) and pd.notna(r["U"])
        if r["type"] == "lower":     return pd.notna(r["L"])
        if r["type"] == "upper":     return pd.notna(r["U"])
        return False

    edited = edited[edited.apply(_valid_row, axis=1)].copy()

    # Rebuild a flat nutrient→spec dict (do NOT overwrite RECIPE_SPECS)
    SPECS_FLAT = {}
    for r in edited.itertuples(index=False):
        n, t, L, U = r.nutrient, r.type, r.L, r.U
        L = None if pd.isna(L) else float(L)
        U = None if pd.isna(U) else float(U)
        if t == "two_sided":
            if L > U: L, U = U, L
            SPECS_FLAT[n] = {"type": "two_sided", "L": L, "U": U}
        elif t == "lower":
            SPECS_FLAT[n] = {"type": "lower", "L": L}
        else:
            SPECS_FLAT[n] = {"type": "upper", "U": U}

    st.session_state["SPECS_FLAT"] = SPECS_FLAT
    if not SPECS_FLAT:
        st.warning("No valid limits after edits. Please complete the table.")
        st.stop()

    nutrients = list(SPECS_FLAT.keys())

    st.caption("**Nutrients in scope:** " + ", ".join(nutrients))

    # 3) Optimizer controls ----------------------------------------------------
    with st.sidebar:
        st.header("Optimization setup")
        strategy = st.selectbox(
            "Objective",
            ["Average Cpk", "Minimum Cpk (worst nutrient)"],
            help="We maximize this Cpk metric with SLSQP."
        )
        score_is_avg = ("Average" in strategy)
        n_restarts = st.number_input(
            "Random restarts", 1, 30, 8,
            help="Multi-start SLSQP to reduce local minima risk."
        )

    # 4) Mixture stats helper using the loaded INGR map
    def mix_mu_sigma_from_map(ingredients, vols_ml, nutrient):
        w = np.asarray(vols_ml, dtype=float) / ML_TOTAL
        mus, sig = [], []
        for ing in ingredients:
            mu_i, sd_i = INGR_MAP.get(ing, {}).get(nutrient, (0.0, 0.0))
            mus.append(mu_i)
            sig.append(sd_i)
        mus = np.asarray(mus, dtype=float)
        sig = np.asarray(sig, dtype=float)
        mu = float(np.dot(w, mus))
        var = float(np.dot(w * w, sig * sig))
        return mu, math.sqrt(max(var, 0.0))


    # 5) Objective (maximize selected score; SLSQP minimizes)
    def score_from_vols(vols):
        per_cpk = []
        for n in nutrients:
            mu, sigma = mix_mu_sigma_from_map(all_ingredients, vols, n)
            per_cpk.append(compute_cpk(mu, sigma, SPECS_FLAT[n]))  # use SPECS_FLAT
        return float(np.mean(per_cpk)) if score_is_avg else float(np.min(per_cpk))


    def objective(x):
        return -score_from_vols(x)

    # 6) Solve with SLSQP ------------------------------------------------------
    from scipy.optimize import minimize

    m = len(all_ingredients)
    bounds = [(0.0, ML_TOTAL)] * m
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - ML_TOTAL}]  # sum vols = 100 mL

    rng = np.random.default_rng(42)
    starts = [np.full(m, ML_TOTAL / m, dtype=float)]
    for _ in range(int(n_restarts) - 1):
        w = rng.dirichlet(np.ones(m))
        starts.append(w * ML_TOTAL)

    best = None
    best_fun = float('inf')

    run = st.button("🚀 Optimize (SLSQP)", type="primary")
    if not run:
        st.caption("Choose objective, then click **Optimize (SLSQP)**.")
        st.stop()

    with st.spinner("Solving…"):
        for init in starts:
            res = minimize(
                objective, init, method="SLSQP",
                bounds=bounds, constraints=cons,
                options=dict(maxiter=600, ftol=1e-9, disp=False)
            )
            if res.success and res.fun < best_fun:
                best = res
                best_fun = res.fun

    if best is None:
        st.error("Optimization failed to converge. Try adjusting limits.")
        st.stop()

    # 7) Clean solution & present ---------------------------------------------
    x_opt = np.clip(best.x, 0.0, None)
    s = float(np.sum(x_opt))
    if s <= 0:
        st.error("Optimizer returned a degenerate solution (total volume = 0).")
        st.stop()
    x_opt = x_opt * (ML_TOTAL / s)

    opt_score = score_from_vols(x_opt)
    st.success(f"Optimization finished. Best **{strategy}** = {opt_score:.6f}")

    rows = []
    for n in nutrients:
        mu, sigma = mix_mu_sigma_from_map(all_ingredients, x_opt, n)
        rows.append({
            "nutrient": n,
            "type":  SPECS_FLAT[n]["type"],           # use SPECS_FLAT
            "L":     SPECS_FLAT[n].get("L", None),
            "U":     SPECS_FLAT[n].get("U", None),
            "mu":    mu,
            "sigma": sigma,
            "Cpk":   compute_cpk(mu, sigma, SPECS_FLAT[n]),
        })

    df_specs = pd.DataFrame(rows)

    vol_df = pd.DataFrame({
        "Ingredient": all_ingredients,
        "Volume (mL)": np.round(x_opt, 6)
    })
    vol_df = vol_df[vol_df["Volume (mL)"] > 0.01].sort_values("Volume (mL)", ascending=False).reset_index(drop=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Optimal volumes (per 100 mL)")
        st.dataframe(vol_df, use_container_width=True)
    with c2:
        st.subheader("Mixture stats per nutrient")
        show = df_specs.copy()
        for c in ["mu", "sigma", "Cpk", "L", "U"]:
            if c in show.columns:
                show[c] = show[c].apply(lambda v: None if pd.isna(v) else (round(float(v), 6) if isinstance(v, (int, float)) else v))
        st.dataframe(show, use_container_width=True)

    out_row = {"objective": strategy, "score_value": opt_score}
    out_row.update({f"Vol {ing} (mL)": v for ing, v in zip(all_ingredients, x_opt)})
    for r in rows:
        out_row[f"Cpk[{r['nutrient']}]"] = r["Cpk"]
        out_row[f"mu[{r['nutrient']}]"] = r["mu"]
        out_row[f"sigma[{r['nutrient']}]"] = r["sigma"]

    out_df = pd.DataFrame([out_row])
    st.download_button(
        "⬇️ Download optimal recipe (CSV)",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name="cpk_optimal_recipe.csv",
        mime="text/csv"
    )



# --- CPk Explainer (drop-in for your explainer tab) ---
import matplotlib.pyplot as plt

with explainer_tab:
    st.title("📈 Cpk Explainer(Normal distribution)")

    # Keep the page compact so the chart stays visible after inputs change
    with st.expander("What is Cpk? (formulas & notes)", expanded=False):
        st.markdown(
            "Cpk measures **how well a process fits within specification limits** "
            "considering **centering** (μ) and **spread** (σ)."
        )
        st.markdown("**Two-sided specs (L and U):**")
        st.latex(r"C_{pk}=\min\!\left(\frac{\mu-L}{3\sigma},\,\frac{U-\mu}{3\sigma}\right)")
        st.latex(r"C_p=\frac{U-L}{6\sigma}")
        st.markdown("**Lower-only (L):**")
        st.latex(r"C_{pk}=\frac{\mu-L}{3\sigma}")
        st.markdown("**Upper-only (U):**")
        st.latex(r"C_{pk}=\frac{U-\mu}{3\sigma}")
        st.markdown("**Rule of thumb:** many industries target **Cpk ≥ 1.33**.")

    # Layout: controls on the left, metrics + plot on the right
    left, right = st.columns([1, 2])

    with left:
        spec_type = st.radio(
            "Specification type",
            ["Two-sided (L & U)", "Lower-only (L)", "Upper-only (U)"],
            horizontal=False,
        )

        mu = st.number_input("Process mean (μ)", value=10.0, step=0.1, key="mu")
        sigma = st.number_input("Std dev (σ)", value=1.0, step=0.05,
                                min_value=0.0001, format="%.4f", key="sigma")
        target = st.number_input("(Optional) target/nominal", value=10.0, step=0.1, key="target")

        def Phi(z: float) -> float:
            return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

        cp = None
        if spec_type == "Two-sided (L & U)":
            L = st.number_input("Lower spec (L)", value=7.0, step=0.1, key="L2")
            U = st.number_input("Upper spec (U)", value=13.0, step=0.1, key="U2")
            if L > U:
                st.warning("L > U detected. Swapping to keep L < U.")
                L, U = U, L
            cp = (U - L) / (6.0 * sigma)
            cpk = min((mu - L) / (3.0 * sigma), (U - mu) / (3.0 * sigma))
            p_left  = Phi((L - mu) / sigma)
            p_right = 1.0 - Phi((U - mu) / sigma)
            p_out = p_left + p_right
            L_line, U_line = L, U
        elif spec_type == "Lower-only (L)":
            L = st.number_input("Lower spec (L)", value=7.0, step=0.1, key="L1")
            cpk = (mu - L) / (3.0 * sigma)
            p_out = Phi((L - mu) / sigma)
            L_line, U_line = L, None
        else:  # Upper-only (U)
            U = st.number_input("Upper spec (U)", value=13.0, step=0.1, key="U1")
            cpk = (U - mu) / (3.0 * sigma)
            p_out = 1.0 - Phi((U - mu) / sigma)
            L_line, U_line = None, U

    with right:
        # Metrics at the top so the chart remains in view after reruns
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Cpk", f"{cpk:.3f}")
        with m2:
            if cp is not None:
                st.metric("Cp (spread only)", f"{cp:.3f}")
            else:
                st.empty()
        with m3:
            st.metric("Out-of-spec probability", f"{p_out*100:.4f}%")

        st.caption(f"≈ {p_out*1e6:,.0f} ppm outside specs • In-spec yield = {(1.0-p_out)*100:.4f}%")

        # Plot (compact height, auto-width)
        x_min, x_max = mu - 6 * sigma, mu + 6 * sigma
        xs = np.linspace(x_min, x_max, 800)
        pdf = (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)

        fig, ax = plt.subplots(figsize=(7, 4))  # shorter figure to avoid scrolling
        ax.plot(xs, pdf, linewidth=2)
        ax.axvline(mu, linestyle="--", linewidth=1)
        ax.text(mu, pdf.max()*0.96, "μ", ha="center", va="top")

        if L_line is not None:
            ax.axvline(L_line, linestyle=":", linewidth=1)
            ax.text(L_line, pdf.max()*0.90, "L", ha="center", va="top")
        if U_line is not None:
            ax.axvline(U_line, linestyle=":", linewidth=1)
            ax.text(U_line, pdf.max()*0.90, "U", ha="center", va="top")

        if L_line is not None and U_line is not None:
            ax.fill_between(xs, 0, pdf, where=(xs < L_line) | (xs > U_line), alpha=0.2)
        elif L_line is not None:
            ax.fill_between(xs, 0, pdf, where=(xs < L_line), alpha=0.2)
        elif U_line is not None:
            ax.fill_between(xs, 0, pdf, where=(xs > U_line), alpha=0.2)

        ax.set_xlabel("Quality characteristic")
        ax.set_ylabel("Density")
        ax.set_title("Normal distribution with specification limits")
        st.pyplot(fig, use_container_width=True)


    st.markdown(
        """
**How to read this chart:**
- The bell curve shows process variation.
- Vertical dashed line is **μ**. Dotted lines are **spec limits**.
- Shaded area = probability of being **out of spec**.
- **Cpk increases** when σ shrinks (curve gets narrower) **and/or** μ moves away from the nearest spec limit.
        """
    )


# --- Cpk in Practice (third tab): mix ingredient distributions and see FG robustness ---
import matplotlib.pyplot as plt

with practice_tab:
    st.title("🧪 Cpk in Practice — Recipe Robustness")

    # Compact context pickers
    top1, top2 = st.columns([1, 1])
    with top1:
        recipe = st.selectbox("Recipe", list(RECIPES.keys()), key="practice_recipe")
    allowed = RECIPES[recipe]
    # Prefer uploaded/edited specs (flat across recipes), else use demo per-recipe specs
    SPECS = st.session_state.get("SPECS_FLAT")
    with top2:
        if SPECS:
            nutrient = st.selectbox("Nutrient", list(SPECS.keys()), key="practice_nutrient")
            spec = SPECS[nutrient]
        else:
            nutrient = st.selectbox("Nutrient", list(RECIPE_SPECS[recipe].keys()), key="practice_nutrient")
            spec = RECIPE_SPECS[recipe][nutrient]


    # === Left/Right row: left = sliders, right = PLOT (no metrics here) ===
    left, right = st.columns([1, 2], vertical_alignment="top")

    with left:
        st.markdown("**Set ingredient proportions** *(auto-normalized to 100 mL)*")
        default_share = 100.0 / max(1, len(allowed))
        perc = {ing: st.number_input(f"{ing} (%)", min_value=0.0, value=default_share,
                                     step=1.0, key=f"p_{ing}_practice") for ing in allowed}

    # Compute mixture once
    total = sum(perc.values())
    weights = {ing: (perc[ing] / total if total > 0 else 0.0) for ing in allowed}
    vols_ml = [round(weights[i] * ML_TOTAL, 3) for i in allowed]  # per 100 mL
    mu_mix, sigma_mix = mix_mu_sigma(allowed, vols_ml, nutrient)
    cpk = compute_cpk(mu_mix, sigma_mix, spec)

    def Phi(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    if spec["type"] == "two_sided":
        L, U = spec["L"], spec["U"]
        p_out = Phi((L - mu_mix) / sigma_mix) + (1.0 - Phi((U - mu_mix) / sigma_mix))
        L_line, U_line = L, U
        cp = (U - L) / (6.0 * sigma_mix)
    elif spec["type"] == "lower":
        L = spec["L"]
        p_out = Phi((L - mu_mix) / sigma_mix)
        L_line, U_line = L, None
        cp = None
    else:
        U = spec["U"]
        p_out = 1.0 - Phi((U - mu_mix) / sigma_mix)
        L_line, U_line = None, U
        cp = None

    base_cpk = BASELINE_CPK[recipe][nutrient]
    delta = cpk - base_cpk

    # Ensure Water adds nothing
    INGR["Water"] = {"sugar": (0.0, 0.0), "vitC": (0.0, 0.0)}

    with right:
        # ---- Plot only (aligned with sliders) ----
        ing_params = []
        for ing in allowed:
            mu_i, sig_i = INGR.get(ing, {}).get(nutrient, (0.0, 0.0))  # <-- safe get
            ing_params.append((ing, mu_i, sig_i, weights[ing]))


        cands_min = [mu_mix - 6 * sigma_mix] + [mu_i - 6 * sig_i for _, mu_i, sig_i, _ in ing_params]
        cands_max = [mu_mix + 6 * sigma_mix] + [mu_i + 6 * sig_i for _, mu_i, sig_i, _ in ing_params]
        if L_line is not None: cands_min.append(L_line)
        if U_line is not None: cands_max.append(U_line)
        x_min, x_max = min(cands_min), max(cands_max)

        xs = np.linspace(x_min, x_max, 900)
        pdf_mix = (1.0 / (sigma_mix * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((xs - mu_mix) / sigma_mix) ** 2)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, pdf_mix, linewidth=2, label="Mixture")

        # Fixed per-100 mL ingredient curves (no % in legend)
        for ing, mu_i, sig_i, _ in ing_params:
            if sig_i <= 0:
                continue
            pdf_i = (1.0 / (sig_i * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((xs - mu_i) / sig_i) ** 2)
            ax.plot(xs, pdf_i, linewidth=1, alpha=0.9, label=ing)

        # Mean + specs + shading
        ax.axvline(mu_mix, linestyle="--", linewidth=1)
        ax.text(mu_mix, pdf_mix.max()*0.96, f"μ (mix) {mu_mix:.2f}", ha="center", va="top")
        if L_line is not None:
            ax.axvline(L_line, linestyle=":", linewidth=1)
            ax.text(L_line, pdf_mix.max()*0.90, f"L = {L_line:.2f}", ha="center", va="top")
        if U_line is not None:
            ax.axvline(U_line, linestyle=":", linewidth=1)
            ax.text(U_line, pdf_mix.max()*0.90, f"U = {U_line:.2f}", ha="center", va="top")
        if (L_line is not None) and (U_line is not None):
            ax.axvspan(L_line, U_line, alpha=0.06)

        if (L_line is not None) and (U_line is not None):
            ax.fill_between(xs, 0, pdf_mix, where=(xs < L_line) | (xs > U_line), alpha=0.2)
        elif L_line is not None:
            ax.fill_between(xs, 0, pdf_mix, where=(xs < L_line), alpha=0.2)
        elif U_line is not None:
            ax.fill_between(xs, 0, pdf_mix, where=(xs > U_line), alpha=0.2)

        ax.set_xlabel(f"{nutrient} per 100 mL")
        ax.set_ylabel("Density")
        ax.set_title("Ingredient distributions (fixed) and resulting FG distribution")
        ax.legend(ncol=2, fontsize="small")
        st.pyplot(fig, use_container_width=True)

    # === Full-width metrics BELOW the row (so sliders align with plot top) ===
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("μ (mixture)", f"{mu_mix:.3f}")
    m2.metric("σ (mixture)", f"{sigma_mix:.3f}")
    m3.metric("Cpk (mixture)", f"{cpk:.3f}", f"{delta:+.3f} vs base {base_cpk:.2f}")
    m4.metric("Out-of-spec", f"{p_out*100:.4f}%")
    st.caption(f"≈ {p_out*1e6:,.0f} ppm outside specs • In-spec yield = {(1.0 - p_out)*100:.4f}%")
    if cp is not None:
        st.caption(f"Cp (spread only) = {cp:.3f}")

    # Table below
    st.markdown("**Volumes per 100 mL & ingredient stats**")
    dfv = pd.DataFrame({
        "Ingredient": allowed,
        "Volume (mL)": vols_ml,
        f"{nutrient} μ": [INGR[i][nutrient][0] for i in allowed],
        f"{nutrient} σ": [INGR[i][nutrient][1] for i in allowed],
    })
    st.dataframe(dfv, use_container_width=True)

