"""
data_io.py
-----------
Ingestion and light data utilities. This module isolates all I/O and shape
normalization so the rest of the code can work with a consistent structure.

Key functions:
  - load_ingr_map_from_excel(path): read the RM distributions workbook
  - parse_limits_file(filelike):   parse uploaded limits (CSV/XLSX)
  - get_stats(...):                access (μ, σ) for a given ingredient/nutrient
  - ingredient_has_any_spec_nutrient(...): used to eliminate ingredients
                                           BEFORE any zero-filling
"""
from __future__ import annotations

import os
from typing import Dict, Tuple, List

import pandas as pd


# --- Helpers ---------------------------------------------------------------
def _pick(columns_map: Dict[str, str], *candidates: str) -> str | None:
    """Pick the first candidate column (case-insensitive) that exists."""
    for c in candidates:
        if c in columns_map:
            return columns_map[c]
    return None


# --- Ingredient distributions ---------------------------------------------
def load_ingr_map_from_excel(path: str) -> Dict[str, Dict]:
    """
    Load ingredient → nutrient distribution stats from an Excel workbook.

    Expected columns (case-insensitive):
      - Ingredient name:   'IS_NAME' or 'Ingredient'
      - Nutrient code:     'NUTRIENTCODE'
      - Mean (μ):          'chosen_mu' (or 'mu','mean')
      - Std dev (σ):       'chosen_sigma' (or 'sigma','std')
    Optional:
      - Dried μ/σ:         'chosen_mu_dried', 'chosen_sigma_dried'
      - Dry→raw factor:    'conversion_from_dry_dosage_to_raw' (or similar)

    Returns a nested dict:
       {
         "Ingredient A": {
           "NUT1": {"mu": ..., "sigma": ..., "mu_dry": ..., "sigma_dry": ...},
           "_conv": 1.23   # optional, default 1.0
         },
         ...
       }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Distribution file not found: {path}")

    df = pd.read_excel(path)
    cols = {c.lower(): c for c in df.columns}

    col_ing = _pick(cols, "is_name", "ingredient")
    col_nut = _pick(cols, "nutrientcode")
    col_mu  = _pick(cols, "chosen_mu", "mu", "mean")
    col_sd  = _pick(cols, "chosen_sigma", "sigma", "std")
    col_mu_d = _pick(cols, "chosen_mu_dried")
    col_sd_d = _pick(cols, "chosen_sigma_dried")
    col_conv = _pick(
        cols,
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
    df = df[use_cols].copy()

    rename = {col_ing:"Ingredient", col_nut:"Nutrient", col_mu:"mu", col_sd:"sigma"}
    if col_mu_d: rename[col_mu_d] = "mu_dry"
    if col_sd_d: rename[col_sd_d] = "sigma_dry"
    if col_conv: rename[col_conv] = "conv"
    df.columns = [rename.get(c, c) for c in df.columns]

    df["mu"] = pd.to_numeric(df["mu"], errors="coerce").fillna(0.0)
    df["sigma"] = pd.to_numeric(df["sigma"], errors="coerce").fillna(0.0)
    if "mu_dry" in df.columns: df["mu_dry"] = pd.to_numeric(df["mu_dry"], errors="coerce")
    if "sigma_dry" in df.columns: df["sigma_dry"] = pd.to_numeric(df["sigma_dry"], errors="coerce")
    if "conv" in df.columns: df["conv"] = pd.to_numeric(df["conv"], errors="coerce")

    ingr_map: Dict[str, Dict] = {}
    for _, r in df.iterrows():
        ing = str(r["Ingredient"])
        nut = str(r["Nutrient"])
        mu = float(r["mu"]); sd = float(r["sigma"])
        mu_dry = float(r["mu_dry"]) if ("mu_dry" in df.columns and pd.notna(r["mu_dry"])) else mu
        sd_dry = float(r["sigma_dry"]) if ("sigma_dry" in df.columns and pd.notna(r["sigma_dry"])) else sd
        conv = float(r["conv"]) if ("conv" in df.columns and pd.notna(r["conv"])) else None

        ingr_map.setdefault(ing, {})
        ingr_map[ing][nut] = {"mu": mu, "sigma": sd, "mu_dry": mu_dry, "sigma_dry": sd_dry}
        if conv is not None:
            ingr_map[ing]["_conv"] = conv

    # Default conversion factor is 1.0 when absent
    for ing in ingr_map:
        ingr_map[ing].setdefault("_conv", 1.0)

    return ingr_map


# --- Limits (spec) file ----------------------------------------------------
def parse_limits_file(filelike) -> tuple[Dict[str, Dict], pd.DataFrame]:
    """
    Parse a CSV/XLSX limits/specification file from the Streamlit uploader.

    Flexible column matching (case-insensitive):
      - nutrient code:  'nutrient' | 'bm_code' | 'nutrientcode' | 'code' | 'bm code'
      - lower bound:    '1.2 min_std' | 'min_std' | 'min std' | 'min' | 'spec_min' | 'l' | 'lower' | 'low'
      - upper bound:    '1.2 max_std' | 'max_std' | 'max std' | 'max' | 'spec_max' | 'u' | 'upper' | 'up'
      - optional type:  'type' | 'spec_type' (one of 'two_sided'|'lower'|'upper')

    Returns:
      (specs_dict, table_df)
        specs_dict: { code: {"type":"two_sided"/"lower"/"upper", "L":..., "U":...}, ... }
        table_df:   tidy table with columns ['nutrient','type','L','U'] for UI editing
    """
    if hasattr(filelike, "name") and str(filelike.name).lower().endswith(".csv"):
        df = pd.read_csv(filelike)
    else:
        df = pd.read_excel(filelike)

    cols = {c.strip().lower(): c for c in df.columns}
    code_col = _pick(cols, "nutrient", "bm_code", "nutrientcode", "code", "bm code")
    L_col    = _pick(cols, "1.2 min_std", "min_std", "min std", "min", "spec_min", "l", "lower", "low")
    U_col    = _pick(cols, "1.2 max_std", "max_std", "max std", "max", "spec_max", "u", "upper", "up")
    type_col = _pick(cols, "type", "spec_type")

    if not code_col:
        raise ValueError(f"Couldn’t find a nutrient code column. Found: {list(df.columns)}")

    if L_col: df[L_col] = pd.to_numeric(df[L_col], errors="coerce")
    if U_col: df[U_col] = pd.to_numeric(df[U_col], errors="coerce")

    rows = []
    specs: Dict[str, Dict] = {}

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

    table = pd.DataFrame(rows, columns=["nutrient", "type", "L", "U"])
    return specs, table


# --- Accessors -------------------------------------------------------------
def _value_is_tuple_stats(v) -> bool:
    """True if v looks like a (mu, sigma) tuple/list with numeric first two entries."""
    try:
        return isinstance(v, (tuple, list)) and len(v) >= 2 and all(isinstance(x, (int, float)) for x in v[:2])
    except Exception:
        return False


def get_stats(ingr_map: Dict, ing: str, nutrient: str, is_dry_mode: bool) -> tuple[float, float]:
    """
    Return (μ, σ) for a given ingredient/nutrient pair. If stats are missing,
    returns (0.0, 0.0). The **elimination** of ingredients that do not have any
    in-scope nutrient must be done by the caller BEFORE using this accessor.
    """
    entry = ingr_map.get(ing, {}).get(nutrient, (0.0, 0.0))
    if _value_is_tuple_stats(entry):
        return float(entry[0]), float(entry[1])
    elif isinstance(entry, dict):
        if is_dry_mode:
            mu = float(entry.get("mu_dry", entry.get("mu", 0.0)))
            sigma = float(entry.get("sigma_dry", entry.get("sigma", 0.0)))
        else:
            mu = float(entry.get("mu", 0.0))
            sigma = float(entry.get("sigma", 0.0))
        return mu, sigma
    return 0.0, 0.0


def ingredient_has_any_spec_nutrient(ing: str, ingr_map: Dict, nutrients: List[str]) -> bool:
    """
    Return True iff the ingredient has at least one of the in-scope nutrients
    **as-provided in the raw data** (ignoring any zero-fill logic).
    """
    keys = [k for k in ingr_map.get(ing, {}).keys() if k != "_conv"]
    return len(set(keys).intersection(nutrients)) > 0
