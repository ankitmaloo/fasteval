"""Compatibility shim for older imports.

Primary FastAPI app now lives in service.api.
"""

from service.api import (
    app,
    EvalRequest,
    get_run,
    get_status,
    health,
    list_runs,
    resume_eval,
    start_eval,
)

__all__ = [
    "app",
    "EvalRequest",
    "start_eval",
    "resume_eval",
    "get_status",
    "list_runs",
    "get_run",
    "health",
]
