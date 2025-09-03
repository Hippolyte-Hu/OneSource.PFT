import streamlit as st
import pandas as pd
import numpy as np
import itertools as it
import math
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cpk Optimizer", page_icon="🧪", layout="wide")

# -------------------------
# Mock DATA LAYER (dicts)
# -------------------------
# Per-ingredient nutrient distributions: Normal(mean, std)
INGR = {
    "Orange":    {"sugar": (9.0, 1.0),  "vitC": (50.0, 5.0)},
    "Apple":     {"sugar": (11.0, 0.8), "vitC": (30.0, 4.0)},
    "Grape":     {"sugar": (16.0, 1.2), "vitC": (10.0, 2.0)},
    "Pineapple": {"sugar": (13.5, 0.9), "vitC": (25.0, 3.0)},
    "Water":     {"sugar": (0.0, 0.1),  "vitC": (0.0, 0.1)},
}

# Recipe → allowed ingredients (default “All” for demo)
RECIPES = {
    "Citrus Blend": ["Orange", "Apple", "Pineapple", "Water"],
    "Grape Punch":  ["Grape", "Apple", "Water"],
    "House Mix":    list(INGR.keys()),
}

# (NEW) Legislative specs PER RECIPE (editable in UI)
# type: "two_sided" uses L and U, "lower" uses L, "upper" uses U.
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

# Historical **baseline Cpk per nutrient per recipe**
BASELINE_CPK = {
    "Citrus Blend": {"sugar": 1.10, "vitC": 1.00},
    "Grape Punch":  {"sugar": 1.05, "vitC": 0.95},
    "House Mix":    {"sugar": 1.12, "vitC": 1.00},
}

# Cost per mL (edit to your real numbers)
COST_PER_ML = {
    "Orange": 0.020,
    "Apple": 0.018,
    "Grape": 0.030,
    "Pineapple": 0.025,
    "Water": 0.000,  # free/ignored if you prefer
}

# -------------------------
# Core math helpers
# -------------------------
ML_TOTAL = 100

def cpk_two_sided(mu, sigma, L, U):
    if sigma <= 0: return 1e9
    return min((mu - L)/(3*sigma), (U - mu)/(3*sigma))

def cpk_lower(mu, sigma, L):
    if sigma <= 0: return 1e9
    return (mu - L)/(3*sigma)

def cpk_upper(mu, sigma, U):
    if sigma <= 0: return 1e9
    return (U - mu)/(3*sigma)

def compute_cpk(mu, sigma, spec):
    t = spec["type"].lower()
    if t == "two_sided": return cpk_two_sided(mu, sigma, spec["L"], spec["U"])
    if t == "lower":     return cpk_lower(mu, sigma, spec["L"])
    if t == "upper":     return cpk_upper(mu, sigma, spec["U"])
    raise ValueError("spec['type'] must be 'two_sided'|'lower'|'upper'")

def mix_mu_sigma(ingredients, vols_ml, nutrient):
    w = [v/ML_TOTAL for v in vols_ml]
    mus = [INGR[i][nutrient][0] for i in ingredients]
    sig = [INGR[i][nutrient][1] for i in ingredients]
    mu = sum(wi*mi for wi, mi in zip(w, mus))
    var = sum((wi**2)*(si**2) for wi, si in zip(w, sig))
    return mu, math.sqrt(max(var, 0.0))

def average_cpk(cpk_map):
    vals = list(cpk_map.values())
    return float(np.mean(vals)) if vals else float("nan")

def min_cpk(cpk_map):
    vals = list(cpk_map.values())
    return float(np.min(vals)) if vals else float("nan")

def integer_compositions(total_units, k):
    # strictly positive parts
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

# -------------------------
# TABS
# -------------------------
optimizer_tab, explainer_tab,practice_tab = st.tabs(["Optimizer", "CPk Explainer","practice tab"])

with optimizer_tab:
    # -------------------------
    # UI (original page)
    # -------------------------
    st.title("🧪 Recipe Cpk Optimizer")

    with st.sidebar:
        st.header("Setup")
        recipe = st.selectbox("Recipe", list(RECIPES.keys()))
        allowed = RECIPES[recipe]

        # Strategy drives ranking AND tiering (see evaluate_all)
        strategy = st.selectbox(
            "Optimization strategy (ranking & tiering metric)",
            ["Average Cpk", "Minimum Cpk (worst nutrient)"],
            help="This choice ranks results and also decides tier thresholds."
        )
        score_col = "avg_cpk" if "Average" in strategy else "min_cpk"

        st.caption("Select the ingredients to consider:")
        picked = st.multiselect("Ingredients", allowed, default=allowed)

        st.caption("Search granularity & structure:")
        step_ml = st.slider("Step size (mL)", 5, 25, 10, 5)
        min_k   = st.slider("Min #ingredients", 1, max(1,len(picked)), min(2,len(picked)))
        max_k   = st.slider("Max #ingredients", min_k, len(picked), max(3, min(4, len(picked))))
        top_k   = st.slider("Show top N results", 5, 50, 15, 5)

    # Guardrails
    if not picked:
        st.info("Pick at least one ingredient to continue.")
        st.stop()

    # Nutrients are whatever this recipe's legislation defines
    DEFAULT_SPECS = RECIPE_SPECS[recipe]
    nutrients = list(DEFAULT_SPECS.keys())

    st.subheader(f"Recipe: {recipe}")
    st.write("Baseline Cpks:", BASELINE_CPK[recipe])
    st.write("Chosen ingredients:", ", ".join(picked))

    # ---- Editable legislative bounds for the CURRENT recipe
    st.markdown("### Legislative bounds (editable per recipe)")
    edited_specs = {}
    with st.form("spec_form", clear_on_submit=False):
        cols = st.columns(len(DEFAULT_SPECS))
        for idx, n in enumerate(DEFAULT_SPECS):
            with cols[idx]:
                spec = DEFAULT_SPECS[n].copy()
                st.caption(f"**{n}** · type: `{spec['type']}`")
                if spec["type"] == "two_sided":
                    L = st.number_input(f"{n} · Lower (L)", value=float(spec["L"]), step=0.1, key=f"L_{n}_{recipe}")
                    U = st.number_input(f"{n} · Upper (U)", value=float(spec["U"]), step=0.1, key=f"U_{n}_{recipe}")
                    if L > U:
                        st.warning(f"{n}: L > U detected. Swapping to keep L < U.")
                        L, U = U, L
                    edited_specs[n] = {"type": "two_sided", "L": L, "U": U}
                elif spec["type"] == "lower":
                    L = st.number_input(f"{n} · Lower (L)", value=float(spec["L"]), step=0.1, key=f"L_{n}_{recipe}")
                    edited_specs[n] = {"type": "lower", "L": L}
                else:  # upper
                    U = st.number_input(f"{n} · Upper (U)", value=float(spec["U"]), step=0.1, key=f"U_{n}_{recipe}")
                    edited_specs[n] = {"type": "upper", "U": U}
        specs_applied = st.form_submit_button("Apply limits")

    if specs_applied:
        st.success("Updated limits applied.")

    if not edited_specs:
        edited_specs = {k: v.copy() for k, v in DEFAULT_SPECS.items()}

    # -------------------------
    # Optimization
    # -------------------------
    def evaluate_all(specs):
        rows = []
        base = BASELINE_CPK[recipe]

        # Baseline score for avg/min depending on strategy
        base_avg = float(np.mean([base[n] for n in nutrients]))
        base_min = float(np.min([base[n] for n in nutrients]))

        for combo, vols in generate_mixtures(picked, step_ml, min_k, max_k):
            per = {}
            for n in nutrients:
                mu, sigma = mix_mu_sigma(combo, vols, n)
                cpk = compute_cpk(mu, sigma, specs[n])
                per[n] = {
                    "mu": mu, "sigma": sigma, "cpk": cpk,
                    "baseline": base[n], "delta": cpk - base[n],
                    "pass": cpk >= base[n],
                }

            avg  = average_cpk({n: per[n]["cpk"] for n in nutrients})
            worst = min_cpk({n: per[n]["cpk"] for n in nutrients})
            all_pass = all(per[n]["pass"] for n in nutrients)
            any_pass = any(per[n]["pass"] for n in nutrients)

            # ---- Tiering depends on strategy
            if score_col == "avg_cpk":
                if all_pass:
                    tier, note = "A", "All nutrients ≥ baseline"
                elif avg >= base_avg:
                    bad = [n for n in nutrients if not per[n]["pass"]]
                    tier, note = "B", "Avg ≥ baseline avg; below-baseline: " + ", ".join(bad)
                else:
                    bad = [n for n in nutrients if not per[n]["pass"]]
                    tier, note = "C", "Avg < baseline avg; below-baseline: " + ", ".join(bad)
            else:
                if worst >= base_min:
                    tier, note = "A", "Min Cpk ≥ min baseline"
                else:
                    if any_pass:
                        bad = [n for n in nutrients if not per[n]["pass"]]
                        tier, note = "B", "Min Cpk < min baseline; below-baseline: " + ", ".join(bad)
                    else:
                        bad = [n for n in nutrients if not per[n]["pass"]]
                        tier, note = "C", "Min Cpk < min baseline; all nutrients below baseline"

            # Only include the chosen metric + its baseline
            if score_col == "avg_cpk":
                row = {
                    "tier": tier, "note": note,
                    "avg_cpk": avg,
                    "avg_baseline": base_avg,
                }
            else:
                row = {
                    "tier": tier, "note": note,
                    "min_cpk": worst,
                    "min_baseline": base_min,
                }

            for n in nutrients:
                info = per[n]
                row[f"Cpk[{n}]"]  = info["cpk"]
                row[f"Base[{n}]"] = info["baseline"]
                row[f"Δ[{n}]"]    = info["delta"]
                row[f"Pass[{n}]"] = "✅" if info["pass"] else "❌"

            # volumes
            for i, v in zip(combo, vols):
                row[f"Vol {i} (mL)"] = v
            for i in picked:
                row.setdefault(f"Vol {i} (mL)", 0)
            # >>> NEW: mixture cost per 100 mL
            mix_cost = sum(v * COST_PER_ML.get(i, 0.0) for i, v in zip(combo, vols))
            row["Cost (per 100 mL)"] = mix_cost
            # <<< NEW
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            order = {"A":0, "B":1, "C":2}
            df["tier_order"] = df["tier"].map(order)
            df = df.sort_values(["tier_order", score_col], ascending=[True, False]).drop(columns="tier_order")
        return df


    run = st.button("🚀 Optimize / Re-run", type="primary")

    if run:
        with st.spinner("Optimizing mixtures…"):
            df = evaluate_all(edited_specs)

        if df.empty:
            st.warning("No feasible mixtures generated. Try adjusting step size or min/max ingredients.")
            st.stop()

        # Column order: Tier, note, SCORE, quantities, then other columns
        vol_cols  = [c for c in df.columns if c.startswith("Vol ")]
        diag_cols = list(it.chain.from_iterable([[f"Cpk[{n}]", f"Base[{n}]", f"Δ[{n}]", f"Pass[{n}]"] for n in nutrients]))

        # Only show selected metric + its baseline up front
        if score_col == "avg_cpk":
            base_cols = ["tier", "note", "avg_cpk", "avg_baseline"]
        else:
            base_cols = ["tier", "note", "min_cpk", "min_baseline"]
        # >>> NEW: identify the cost column
        cost_cols = [c for c in df.columns if c == "Cost (per 100 mL)"]
        # <<< NEW
        other_cols = [c for c in df.columns if c not in set(base_cols + vol_cols + diag_cols)]
        cols = [c for c in (base_cols + vol_cols + other_cols + diag_cols) if c in df.columns]
        df = df[cols].copy()

        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]): df[c] = df[c].round(3)

        st.success(f"Found {len(df)} mixtures. Showing top {top_k} per tier.")

        show_A = df.query("tier=='A'").head(top_k)
        show_B = df.query("tier=='B'").head(top_k)
        show_C = df.query("tier=='C'").head(top_k)

        title_metric = "Avg Cpk" if score_col == "avg_cpk" else "Min Cpk"
        st.subheader(f"Tier A — ranked by {title_metric}")
        st.dataframe(show_A if not show_A.empty else pd.DataFrame({"info":["(none)"]}), use_container_width=True)

        st.subheader(f"Tier B — ranked by {title_metric}")
        st.dataframe(show_B if not show_B.empty else pd.DataFrame({"info":["(none)"]}), use_container_width=True)

        with st.expander("Tier C (audit)"):
            st.dataframe(show_C if not show_C.empty else pd.DataFrame({"info":["(none)"]}), use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download all results (CSV)",
            data=csv,
            file_name="cpk_optimized_mixtures.csv",
            mime="text/csv"
        )
    else:
        st.caption("Adjust options and limits above, then click **Optimize / Re-run**.")

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
    with top2:
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
        ing_params = [(ing, INGR[ing][nutrient][0], INGR[ing][nutrient][1], weights[ing]) for ing in allowed]

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




