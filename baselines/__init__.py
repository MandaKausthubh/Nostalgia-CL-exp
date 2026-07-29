"""Continual-learning baselines: EWC, GPM, A-GEM, EWC+Nostalgia.

Each method exposes a state-computation helper and (for EWC/A-GEM/EWC+Nostalgia)
an optimizer wrapper. GPM and Nostalgia reuse the projection wrapper.

Registry:
    get_baseline(method) -> module or None
"""

from . import ewc, agem, gpm, ewc_nostalgia, sdft

_BASELINE_MODULES = {
    "ewc": ewc,
    "gpm": gpm,
    "agem": agem,
    "ewc_nostalgia": ewc_nostalgia,
    "sdft": sdft,
}


def get_baseline(method: str):
    """Return the baselines submodule for a method, or None if not a baseline."""
    return _BASELINE_MODULES.get(method)


def is_baseline(method: str) -> bool:
    return method in _BASELINE_MODULES