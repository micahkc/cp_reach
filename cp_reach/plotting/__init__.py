"""
Utility helpers.

Plotting depends on optional extras (e.g., cyecca). Import it lazily to avoid
hard failures when those deps are missing.
"""

try:
    from . import plotting  # noqa: F401
except Exception as exc:  # broad: optional dependency
    # Defer failure until plotting is explicitly used
    plotting = None
    import warnings

    warnings.warn(f"cp_reach.utils.plotting not available: {exc}")
