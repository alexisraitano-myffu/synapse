import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dream_cycle.cycle import run_cycle

if __name__ == "__main__":
    run_cycle()
