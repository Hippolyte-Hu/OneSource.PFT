"""
settings.py
-------------
Centralized constants used by both the modeling code and the UI.

- ML_TOTAL: total mixture mass the optimizer works with (per 100 g basis).
- CPK_INF : the "very large" Cpk cap you requested (±100). Used both to cap
            finite Cpk values and as the nominal max/min when sigma ~ 0.
- EPS     : small positive number used as the threshold for "sigma ~= 0".
- DATA_XLSX: default path to the ingredient distribution workbook. The UI can
             override this, but keeping a single source of truth here makes
             deployments simpler.
"""
# Total mass for the mixture (grams); the optimizer enforces sum(x_i) = ML_TOTAL.
ML_TOTAL = 100

# Cpk hard cap, also used as +/- value when sigma ~ 0 in your business rule.
CPK_INF = 100

# Numerical threshold below which we treat sigma as "effectively zero".
EPS = 1e-12

# Default path used by the Streamlit app. The file itself is not part of the repo.
DATA_XLSX = "data/rm_nutrient_distributions_for_optimizer_with_process_losses.xlsx"
