__version__ = "0.1.0"
__author__ = "Micah Condie"
__license__ = "Apache-2.0"


# Lazy-load heavy subpackages to avoid circular imports at package init time.
import importlib
from typing import Any

__all__ = [
    "config",
    "development",
    "dynamics",
    "geometry",
    "ir",
    "physics",
    "planning",
    "plotting",
    "reachability",
]

_SUBMODULES = {
    "config": "cp_reach.config",
    "development": "cp_reach.development",
    "dynamics": "cp_reach.dynamics",
    "geometry": "cp_reach.geometry",
    "ir": "cp_reach.ir",
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
