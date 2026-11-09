import sys, json, itertools
sys.path.insert(0, '.')
import bench, mgw, pandas as pd
eps = [int(x) for x in sys.argv[1].split(",")]; dev = sys.argv[2]
res = {}
for far, ero, tme in [(False, False, 13), (False, False, 21), (False, False, 41)]:
    mgw.MGWForward.FAR, mgw.MGWForward.ERODE, mgw.MGWForward.TM_ERODE = far, ero, tme
    rows = bench.main(eps, ks=(4, 8), step=24, dev=dev, save_examples=False)
    df = pd.DataFrame(rows)
    res[f"far={far} erode={ero} tm={tme}"] = {"d_psnr": float((df.mgw_psnr - df.v3_psnr).mean()), "d_ssim": float((df.mgw_ssim - df.v3_ssim).mean()),
                                              "win_psnr": float((df.mgw_psnr > df.v3_psnr).mean()), "win_ssim": float((df.mgw_ssim > df.v3_ssim).mean())}
    print(res, flush=True)
