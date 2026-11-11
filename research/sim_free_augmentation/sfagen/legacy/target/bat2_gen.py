"""Battery-position candidates (battery/bsfa/bgen.py, unchanged planner and checks) on the calibrated reconstruction."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "battery/bsfa")]
import srecon  # noqa: F401,E402  (calibrated camera + separate cache for every BRecon)
import bgen  # noqa: E402
bgen.ROOT = HERE / "bat2"
bgen.main([int(x) for x in sys.argv[1].split(",")], sys.argv[2])
