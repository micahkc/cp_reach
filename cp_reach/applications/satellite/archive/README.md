# Satellite Module Archive

This directory contains experimental versions of satellite reachability algorithms kept for historical reference.

## Archived Files

### `invariant_set2.py` (Oct 29, 2024)
- Second iteration of invariant set computation
- Uses `coupled_dynamics` module instead of separate angular_acceleration
- Different API structure compared to v1
- Used in: `notebooks/satelliteLMI2.ipynb`

### `invariant_set3.py` (Nov 4, 2024)
- Third iteration with simplified assumptions
- Reduced complexity for computational efficiency experiments
- Used in: `notebooks/satellite-outer.ipynb`

### `TH_LTV_v1.py` (Nov 6, 2024)
- Original Tschauner-Hempel Linear Time-Varying parameter bound approach
- Superseded by `TH_LTV.py` (formerly TH_LTV2.py) which includes CasADi integration

## Active Versions

For current implementations, see:
- `../invariant_set.py` - Canonical 2-level Lyapunov API
- `../invariant_set4.py` - Simplified single-level log-linearization (faster)
- `../TH_LTV.py` - Current TH-LTV parameter bounds
