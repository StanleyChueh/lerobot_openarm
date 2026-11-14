"""Freeze the source object positions that sfagen's target planner needs (A14; replaces the hand-run v03 script).
For each group config: copy the rig files (calibration.json, table_height.json, lowest_arm.json) from the task's rig
work dir, reconstruct every episode (cached in the group work dir, reused by `sfagen generate`), and record the target
object's table x/y. Prints JSON {task: {episode: [x, y]}} for make_configs.py, plus a per-episode object summary.
Usage: source_positions.py OUT.json CONFIG [CONFIG ...]   (group configs; rig dir = <work_dir>/../rig)"""
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfagen"))
from sfagen.context import Context
from sfagen.recon import Recon

out, cfgs = Path(sys.argv[1]), sys.argv[2:]; res = json.loads(out.read_text()) if out.exists() else {}; summ = {}
for c in cfgs:
    ctx = Context(c); rig = ctx.work.parent / "rig"; task = ctx.work.parent.name
    for f in ("calibration.json", "table_height.json", "lowest_arm.json"):
        if not (ctx.work / f).exists():
            shutil.copy2(rig / f, ctx.work / f)
    ctx = Context(c)                                    # reload with the copied rig files
    target = ctx.cfg["variations"]["object_targets"]["object"]
    for e in ctx.episodes():
        rec = Recon(ctx, e, "cuda:0"); o = next(o for o in rec.objects if o["name"] == target)
        res.setdefault(task, {})[str(e)] = [float(o["table"][0, 3]), float(o["table"][1, 3])]
        summ[f"{task}/{ctx.work.name}/ep{e}"] = {"lag": rec.lag, "objects": rec.info.get("objects"), "table": rec.info.get("table")}
        print(task, ctx.work.name, e, res[task][str(e)], flush=True)
out.write_text(json.dumps(res, indent=1)); out.with_name(out.stem + "_recon_summary.json").write_text(json.dumps(summ, indent=1, default=float))
