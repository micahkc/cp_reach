__version__ = "0.1.0"
__author__ = "Micah Condie"
__license__ = "Apache-2.0"


# Lazy-load heavy subpackages to avoid circular imports at package init time.
import importlib
from typing import Any

__all__ = [
    "applications",
    "dynamics",
    "geometry",
    "physics",
    "planning",
    "plotting",
    "reachability",
]

_SUBMODULES = {
    "applications": "cp_reach.applications",
    "dynamics": "cp_reach.dynamics",
    "geometry": "cp_reach.geometry",
    "physics": "cp_reach.physics",
    "planning": "cp_reach.planning",
    "plotting": "cp_reach.plotting",
    "reachability": "cp_reach.reachability",
}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = importlib.import_module(_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__} has no attribute {name!r}")


# Explicitly import key modules
from . import dynamics
from . import planning
from . import reachability
