"""
CP_Reach package entry point.

Allows running cp_reach as a module:
    python -m cp_reach analyze --ir model.json
"""

from cp_reach.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
