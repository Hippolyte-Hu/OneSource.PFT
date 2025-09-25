# Cpk Optimizer (modularized)

A small, modular codebase that separates concerns so **data scientists**, **data engineers**,
and the **Streamlit front-end** can work independently.

## Layout

```text
cpk-optimizer/
├── app/
│   └── streamlit_app.py         # minimal UI that calls into the backend package
└── cpk_optimizer/               # backend package
    ├── __init__.py              # exports common constants
    ├── settings.py              # ML_TOTAL, CPK_INF, EPS, default DATA_XLSX path
    ├── metrics.py               # all Cpk math + sigma≈0 rule
    ├── data_io.py               # ingestion + spec parsing + accessors
    └── optimizer.py             # SLSQP + mixture moments
```

## Quick start

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Place your ingredient distribution workbook at:

```
data/rm_nutrient_distributions_for_optimizer_with_process_losses.xlsx
```

(You can change this default path in `cpk_optimizer/settings.py`.)

## Collaboration notes

- **Data scientists** work in `cpk_optimizer/metrics.py` and `optimizer.py`.
- **Data engineers** wire real sources in `data_io.py` (replace Excel loader as needed).
- **Front-end** iterates in `app/streamlit_app.py` with no backend edits.

## Your sigma≈0 rule (documented)

- If σ > EPS: compute standard Cpk and cap to [-100, +100].  
- If σ <= EPS:
  - If μ is OUTSIDE spec → return **-100** (hard fail).  
  - If μ is INSIDE spec → return **100 · exp(-λ · d_norm²)** to reward centering.
