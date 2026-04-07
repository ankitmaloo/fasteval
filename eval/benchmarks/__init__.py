"""Benchmark plugin registry.

Plugins are resolved from config.yaml `benchmarks:` section, which maps
a name to a module path and class. Same pattern as providers.

    benchmarks:
      gsm8k:
        module: eval/benchmarks/gsm8k.py
        class: GSM8KPlugin
      kwbench:
        module: eval/benchmarks/kwbench.py
        class: KWBenchPlugin

Optional `scorer` field overrides the plugin's score_case with a standalone
function. Format: `path/to/file.py:function_name`

    benchmarks:
      kwbench_custom:
        module: eval/benchmarks/kwbench.py
        class: KWBenchPlugin
        scorer: eval/scorers/my_reward.py:score
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from eval.benchmarks.base import BaseBenchmarkPlugin, BenchmarkPlugin

_REGISTRY: dict[str, BenchmarkPlugin] = {}

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = Path(__file__).resolve().parent.parent


def register(name: str, plugin_or_cls: BenchmarkPlugin | type) -> None:
    """Register a plugin instance or class. Classes are instantiated on registration."""
    if isinstance(plugin_or_cls, type):
        _REGISTRY[name] = plugin_or_cls()
    else:
        _REGISTRY[name] = plugin_or_cls


def _load_function(spec: str):
    """Load a function from 'path/to/file.py:function_name' spec."""
    if ":" not in spec:
        raise ValueError(f"Scorer spec must be 'path/to/file.py:function_name', got: {spec!r}")
    file_part, func_name = spec.rsplit(":", 1)
    path = Path(file_part)
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.exists():
        raise ValueError(f"Scorer module not found: {path}")

    mod_spec = importlib.util.spec_from_file_location(f"scorer_{path.stem}", str(path))
    if mod_spec is None or mod_spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {path}")
    module = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(module)

    fn = getattr(module, func_name, None)
    if fn is None:
        raise ValueError(f"{path} does not export function '{func_name}'")
    if not callable(fn):
        raise ValueError(f"{path}:{func_name} is not callable")
    return fn


def _wrap_with_scorer(plugin: BenchmarkPlugin, scorer_fn) -> BenchmarkPlugin:
    """Return a wrapper that delegates everything to plugin but overrides score_case."""
    from eval.scorers import ScorerResult

    class _ScorerOverride:
        def __init__(self, inner, fn):
            self._inner = inner
            self._fn = fn
            self.name = inner.name

        def score_case(self, case, answer, artifacts=None):
            result = self._fn(case, answer, artifacts)
            if isinstance(result, ScorerResult):
                return result
            # Allow returning a plain float or (float, dict) tuple
            if isinstance(result, (int, float)):
                return ScorerResult(score=float(result), detail={}, method="custom")
            if isinstance(result, tuple) and len(result) == 2:
                return ScorerResult(score=float(result[0]), detail=result[1], method="custom")
            raise TypeError(
                f"Scorer function must return ScorerResult, float, or (float, dict). Got: {type(result)}"
            )

        def __getattr__(self, name):
            return getattr(self._inner, name)

    return _ScorerOverride(plugin, scorer_fn)


def _load_plugin_from_module(module_path: str, class_name: str) -> BenchmarkPlugin:
    """Dynamic import of a plugin class from a module path."""
    path = Path(module_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.exists():
        raise ValueError(f"Benchmark module not found: {path}")

    spec = importlib.util.spec_from_file_location(f"bench_{path.stem}", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cls = getattr(module, class_name, None)
    if cls is None:
        raise ValueError(f"{path} does not export class '{class_name}'")
    return cls()


def _load_benchmark_config() -> dict[str, dict[str, str]]:
    """Read benchmarks section from config.yaml."""
    config_path = EVAL_DIR / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    benchmarks = payload.get("benchmarks", {})
    if not isinstance(benchmarks, dict):
        return {}
    return benchmarks


def get_plugin(name: str) -> BenchmarkPlugin:
    # Already instantiated
    if name in _REGISTRY:
        return _REGISTRY[name]

    # Resolve from config.yaml
    bench_config = _load_benchmark_config()
    entry = bench_config.get(name)
    if entry is not None and isinstance(entry, dict):
        module_path = entry.get("module")
        class_name = entry.get("class")
        if module_path and class_name:
            plugin = _load_plugin_from_module(module_path, class_name)
            scorer_spec = entry.get("scorer")
            if scorer_spec:
                scorer_fn = _load_function(str(scorer_spec))
                plugin = _wrap_with_scorer(plugin, scorer_fn)
            _REGISTRY[name] = plugin
            return plugin

    known = sorted(set(list(_REGISTRY.keys()) + list(bench_config.keys())))
    raise ValueError(
        f"Unknown benchmark '{name}'. "
        f"Add it to config.yaml under `benchmarks:`. Known: {', '.join(known) or 'none'}"
    )


__all__ = ["BenchmarkPlugin", "BaseBenchmarkPlugin", "register", "get_plugin"]
