from __future__ import annotations
import os
import pandas as pd
import streamlit as st

# Path stays the same (logic unchanged)
DATA_XLSX = "data/rm_nutrient_distributions_for_optimizer_with_process_losses.xlsx"


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


def parse_limits_file(file) -> tuple[dict[str, dict], pd.DataFrame]:
    """Identical parsing logic you had in the app."""
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


def ingredient_has_any_spec_nutrient(ing: str, INGR: dict, nutrients: list[str]) -> bool:
    """Used by the app to pre-filter ingredients (unchanged behavior)."""
    keys = [k for k in INGR.get(ing, {}).keys() if k != "_conv"]
    return len(set(keys).intersection(nutrients)) > 0
