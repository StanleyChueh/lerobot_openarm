"""Reconstruct (or load) the given episodes so that their mask caches exist. Usage: recon_masks.py CONFIG [e1,e2] [device]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfagen"))
from sfagen.context import Context
from sfagen.recon import Recon
ctx = Context(sys.argv[1]); dev = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
for e in ([int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 and sys.argv[2] != "all" else ctx.episodes()):
    r = Recon(ctx, e, dev); print(Path(sys.argv[1]).stem, "ep", e, "frames", r.T, "lag", r.lag, "links repaired", getattr(r, "repaired_links", 0), flush=True)
