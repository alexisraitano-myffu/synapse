#!/usr/bin/env python3
"""Convenience launcher — same as `python -m dream_cycle`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dream_cycle import main

if __name__ == "__main__":
    main()
