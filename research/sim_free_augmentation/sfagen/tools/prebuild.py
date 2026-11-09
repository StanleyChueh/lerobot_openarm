import sys, time
sys.path.insert(0, "/home/stanley/Stanley_ws/sfa_hf_export/sfagen")
from sfagen.context import Context
from sfagen.recon import Recon
ctx = Context(sys.argv[1])
for e in [int(x) for x in sys.argv[2].split(",")]:
    t0 = time.time(); r = Recon(ctx, e, "cuda:0"); print(e, "lag", r.lag, round(time.time() - t0), "s", flush=True)
