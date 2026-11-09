import sys
sys.path.insert(0, '.')
import bench, mgw, pandas as pd
res = {}
for d in (False, True):
    mgw.MGWForward.DESPECKLE = d
    df = pd.DataFrame(bench.main([8], ks=(4, 8), step=24, dev=sys.argv[1], save_examples=False))
    res[f"despeckle={d}"] = {"d_psnr": round(float((df.mgw_psnr - df.v3_psnr).mean()), 3), "d_ssim": round(float((df.mgw_ssim - df.v3_ssim).mean()), 4),
                             "win_psnr": round(float((df.mgw_psnr > df.v3_psnr).mean()), 3), "win_ssim": round(float((df.mgw_ssim > df.v3_ssim).mean()), 3)}
print(res)
