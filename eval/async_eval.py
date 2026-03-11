"""Compatibility shim for older imports.

Primary async eval engine now lives in service.engine.
"""

from service.engine import AsyncRunConfig, RESULTS_DIR, load_dataset, run_async_eval

__all__ = ["AsyncRunConfig", "RESULTS_DIR", "load_dataset", "run_async_eval"]
