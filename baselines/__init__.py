"""Continual-learning baselines: EWC, GPM, A-GEM.

Each method exposes a state-computation helper and (for EWC/A-GEM) an
optimizer wrapper. GPM reuses `NostalgiaOptimizer` for projection and only
provides a subspace-construction helper.

Registry:
    get_baseline(method) -> module or None
"""

from . import ewc, agem, gpm

_BASELINE_MODULES = {
    "ewc": ewc,
    "gpm": gpm,
    "agem": agem,
}


def get_baseline(method: str):
    """Return the baselines submodule for a method, or None if not a baseline."""
    return _BASELINE_MODULES.get(method)


def is_baseline(method: str) -> bool:
    return method in _BASELINE_MODULES