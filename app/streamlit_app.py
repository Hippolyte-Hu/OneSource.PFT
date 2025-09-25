from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

from helpers import (
    ML_TOTAL, CPK_INF, EPS,
    cpk_with_zero_sigma_penalty,
    nutrients_present_in_data,
)
from data_reader import (
    DATA_XLSX,
    load_ingr_map_from_excel,
    parse_limits_file,
    ingredient_has_any_spec_nutrient,
)
from optimizer import optimize, mix_mu_sigma_from_map


st.set_page_config(page_title="Cpk Optimizer", page_icon="🧪", layout="wide")

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

    # >>> Eliminate ingredients BEFORE any implicit zero-fill:
    ingredients_in_scope = [i for i in all_ingredients if ingredient_has_any_spec_nutrient(i, INGR, nutrients)]
    dropped = sorted(set(all_ingredients) - set(ingredients_in_scope))
    if dropped:
        st.info("Excluded ingredients with no in-scope nutrients: " + ", ".join(dropped))

    if not ingredients_in_scope:
        st.error("No ingredients remain after filtering. Adjust your limits or ingredient file.")
        st.stop()

    st.caption("**Nutrients in scope (used in optimization):** " + ", ".join(nutrients))
    st.caption("**Ingredients in scope (used in optimization):** " + ", ".join(ingredients_in_scope))

    # ---------------- Sidebar: objective + penalty ----------------
    with st.sidebar:
        st.header("Optimization setup")
        strategy = st.selectbox(
            "Objective",
            ["Average Cpk", "Minimum Cpk (worst nutrient)"],
            help="We maximize this Cpk metric with SLSQP.",
        )
        n_restarts = st.number_input(
            "Random restarts", 1, 30, 8, help="Multi-start SLSQP to reduce local minima risk."
        )
        lam_pen = st.slider(
            "λ for σ=0 penalty (Option A)",
            min_value=0.0, max_value=5.0, value=1.0, step=0.1,
            help="Penalty only applies when σ≈0. Higher λ penalizes means farther from the spec center more strongly."
        )

    # ------------------------- Run optimizer -------------------------------
    run = st.button("🚀 Optimize (SLSQP)", type="primary")
    if not run:
        st.caption("Choose objective, then click **Optimize (SLSQP)**.")
        st.stop()

    try:
        result = optimize(
            INGR=INGR,
            RECIPE_SPECS=RECIPE_SPECS,
            nutrients=nutrients,
            ingredients_in_scope=ingredients_in_scope,
            is_dry=is_dry,
            lam_pen=lam_pen,
            strategy=strategy,
            n_restarts=int(n_restarts),
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

    x_opt = result["x_opt"]
    opt_score = result["opt_score"]
    elapsed_sec = result["elapsed_sec"]

    st.success(f"Optimization finished in {elapsed_sec:.3f} s. Best **{strategy}** = {opt_score:.6f}")

    # Per-nutrient μ, σ, Cpk (with penalty & hard cap)
    rows = []
    for n in nutrients:
        mu, sigma = mix_mu_sigma_from_map(INGR, ingredients_in_scope, x_opt, n, is_dry)
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

    # Quantities table
    if not is_dry:
        qty_df = pd.DataFrame(
            {"Ingredient": ingredients_in_scope, "Quantity (g) per 100 g)": np.round(x_opt, 6)}
        )
        qty_df.columns = ["Ingredient", "Quantity (g) per 100 g"]  # keep original column name
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

    # Export CSV (single row, unchanged)
    out_row = {
        "objective": strategy,
        "score_value": opt_score,
        "mode": "Dry" if is_dry else "Liquid",
        "lambda_penalty": lam_pen,
        "elapsed_seconds": elapsed_sec,
        "best_restart_iters": getattr(result["best_result"], "nit", None),
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
