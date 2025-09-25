"""
cpk_optimizer: backend package for the Cpk Optimizer app.

This package contains the **modeling logic** that your teams can iterate on
independently of the Streamlit UI. Keeping these pieces decoupled lets you
evolve ingestion, metrics, and optimization without touching the front end.

Exports a few stable constants so the UI has a single import surface.
"""
from .settings import ML_TOTAL, CPK_INF, EPS, DATA_XLSX

__all__ = ("ML_TOTAL", "CPK_INF", "EPS", "DATA_XLSX")
