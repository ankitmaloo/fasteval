"""Shared logging for eval. Use `from eval.log import log` everywhere.

Setup via CLI: --log terminal (default) or --log path/to/file.log
Call `setup(dest)` once at startup. All modules use the same logger.
"""

import logging
import sys

log = logging.getLogger("eval")

_configured = False


def setup(dest: str = "terminal"):
    """Configure the eval logger. Call once.
    dest: "terminal" for stderr, or a file path."""
    global _configured
    if _configured:
        return
    _configured = True

    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if dest == "terminal":
        h = logging.StreamHandler(sys.stderr)
    else:
        from pathlib import Path
        p = Path(dest).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        h = logging.FileHandler(str(p), mode="a")
    h.setFormatter(fmt)
    log.addHandler(h)
