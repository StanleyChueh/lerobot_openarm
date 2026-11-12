"""README dataset table for the v01 candidates, from results/datasets_v01/*/quality_report.json -> results/datasets_v01/TABLE.md."""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results/datasets_v01"
rows = ["| Repo (`ethanCSL/...`) | Demos | Episodes / frames | Coverage /40 | max speed / accel ratio | body decode | wrist valid (L / R) | Gate |", "|---|---|---|---|---|---|---|---|"]
for d in ["fewsrc1", "fewsrc2", "fewsrc4", "fewsrc10", "fewsrc1_wristrender", "fewsrc2_wristrender", "fewsrc4_wristrender", "fewsrc10_wristrender"]:
    p = R / d / "quality_report.json"; n = d[6:].split("_")[0]; v = "_wristrender" if "wrist" in d else ""
    name = f"openarm_pringles_lerobot_real_to_sim_{n}_3cam{v}_v01"
    if not p.exists():
        rows.append(f"| `{name}` | | not built | | | | | |"); continue
    q = json.loads(p.read_text()); t = q["trajectories"]; i = q["images"]; w = list(i["wrist_view_valid_frac"].values())
    link = f"[`{name}`](https://huggingface.co/datasets/ethanCSL/{name})" if q["pass"] else f"`{name}` (not published)"
    gate = "pass" if q["pass"] else "FAIL: " + ", ".join(k for k, ok in {**t["gate"], **i["gate"]}.items() if not ok) + (f" (min {i['wrist_decoded_vs_rendered_psnr_db_min']:.1f} dB < 30)" if not i["gate"]["wrist_decode_psnr"] else "")
    rows.append(f"| {link} | {q['sources']} | {q['episodes']} / {q['frames']:,} | {q['coverage']['within_2cm']} | {t['max_speed_ratio']:.2f} / {t['max_accel_ratio']:.2f} | {i['body_decoded_vs_rendered_psnr_db']} dB | {w[0]:.2f} / {w[1]:.2f} | {gate} |")
tab = "\n".join(rows) + "\n\n`_wristrender` = rendered-wrist **ablation** (the registered wrist gate failed for Pringles); otherwise identical to the main repo. Mask wrist cameras where `wrist_view_valid == 0`."
(R / "TABLE.md").write_text(tab); print(tab)
