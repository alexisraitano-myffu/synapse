import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

from dream_cycle.cycle import run_cycle

if __name__ == "__main__":
    run_cycle()
