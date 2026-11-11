import sys, time
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfagen.context import Context
from sfagen.recon import Recon
ctx = Context(sys.argv[1])
for e in [int(x) for x in sys.argv[2].split(",")]:
    t0 = time.time(); r = Recon(ctx, e, "cuda:0"); print(e, "lag", r.lag, round(time.time() - t0), "s", flush=True)
