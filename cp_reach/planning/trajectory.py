from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class Trajectory:
    """Represents a nominal trajectory and control profile."""

    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def sample(self, t_query: float) -> tuple[np.ndarray, np.ndarray]:
        """Nearest-neighbor sample for convenience."""
        idx = int(np.clip(np.searchsorted(self.t, t_query), 0, len(self.t) - 1))
        return self.x[idx], self.u[idx]
