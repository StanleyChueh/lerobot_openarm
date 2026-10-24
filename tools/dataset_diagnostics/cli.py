"""LeRobot Dataset Readiness & Coverage Analyzer -- command line.

  python -m tools.dataset_diagnostics analyze --repo-id ethanCSL/openarm_pringles_lerobot_real_10_v00 \\
      --output outputs/readiness/real_10 --adapter openarm \\
      --policy-checkpoint ethanCSL/openarm_pringles_lerobot_real_smolvla_10_v00 \\
      --workspace-config tools/dataset_diagnostics/configs/openarm_pringles_metric.json \\
      --object-pose-source detect:tools/dataset_diagnostics/configs/pringles_can_detect_real.json

  python -m tools.dataset_diagnostics analyze --manifest research/dataset_diagnostics/pringles_manifest.json

  python -m tools.dataset_diagnostics annotate --root /path/to/ds --camera observation.images.body_cam \\
      --output annotations.csv [--propose detect.json] [--interactive]

  python -m tools.dataset_diagnostics fit-transform --correspondences pts.csv --camera observation.images.body_cam \\
      --frame-id openarm_world --output body_cam_to_table.json

--object-pose-source (repeatable; the best source per episode wins: recorded > validated > human > auto):
  none                       declare that no positions exist (coverage = UNKNOWN)
  recorded:<csv>             episode_index,x,y[,z],frame_id  (recorded object poses, frame declared)
  isaac-hdf5:<h5>[#<obj>]    OpenArm/IsaacLab Mimic source HDF5 (matched to episodes by trajectory)
  annotations:<csv>[#<label>] image annotations written by `annotate` (#label: rows of another dataset)
  transform:<json>           validated image->metric transform applied to the image annotations
  detect:<json>              colour-blob proposals in the start frame (UNVERIFIED, provisional only)

The analysis never writes into a dataset directory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .checkpoint import compare_checkpoint, load_checkpoint_meta, training_provenance
from .checks import FAIL, PASS, UNKNOWN, WARN, Check
from .compat import _per_frame_speed, compare_datasets
from .coverage import WorkspaceConfig, analyze_coverage, coverage_fragility
from .loader import dataset_fingerprint, load_dataset
from .positions import (AUTO, HUMAN, RECORDED, VALIDATED, Position, TransformError, detect_color_blob,
                        fit_validated_transform, load_transform, merge_by_priority, read_annotations,
                        read_recorded_poses, write_annotations)
from .report import coverage_checks, overall_line, readiness, render_html, write_json
from .technical import run_technical_checks

TOOL_VERSION = "0.1.0"


def _git(path: Path) -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", "-C", str(path), *a], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return None
    return {"sha": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--porcelain"))}


def _parse_episodes(s):
    if s is None or isinstance(s, list):
        return s
    out = []
    for part in str(s).split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


# ----------------------------------------------------------------------------- positions


def collect_positions(view, specs: list[str], out_dir: Path) -> tuple[dict, list[Check], dict]:
    """Run every declared position source; returns {source_kind: {ep: Position}}, checks, artefacts."""
    sources: dict[str, dict[int, Position]] = {}
    checks: list[Check] = []
    art: dict = {}
    image_ann: dict[int, Position] = {}
    transforms = []
    for spec in specs or ["none"]:
        kind, _, arg = spec.partition(":")
        if kind == "none":
            continue
        if kind == "recorded":
            sources["recorded"] = read_recorded_poses(arg)
        elif kind == "isaac-hdf5":
            from .openarm import isaac_hdf5_poses

            path, _, obj = arg.partition("#")
            pos, chk, match = isaac_hdf5_poses(view, path, object_name=obj or "can")
            sources["isaac-hdf5"] = pos
            checks += chk
            art["isaac_match"] = match
        elif kind == "annotations":  # annotations:<csv>[#<label>] reads another dataset's rows (same episode ids)
            path, _, alias = arg.partition("#")
            image_ann.update(read_annotations(path, dataset=alias or view.label))
        elif kind == "transform":
            transforms.append(arg)
        elif kind == "detect":
            det, chk, frames = detect_positions(view, json.loads(Path(arg).read_text()))
            sources["detect"] = det
            checks += chk
            art["detect_frames"] = frames
        else:
            raise SystemExit(f"unknown --object-pose-source {spec!r}")
    if image_ann:
        sources["annotations"] = image_ann
    for t in transforms:
        try:
            tr = load_transform(t)
        except (TransformError, OSError, ValueError) as exc:
            checks.append(Check("image_to_workspace_transform", FAIL, f"{t}: {exc}"))
            continue
        if not tr.valid:
            checks.append(Check("image_to_workspace_transform", FAIL,
                                f"{t}: held-out error {tr.heldout_max_m:.3f} m > {tr.max_allowed_m} m or "
                                f"n={tr.n_points} < 6 -- metric map refused", tr.to_json()))
            continue
        try:
            sources[f"transform:{Path(t).name}"] = tr.map({**sources.get("detect", {}), **image_ann})
            checks.append(Check("image_to_workspace_transform", PASS,
                                f"{t}: validated (held-out rmse {tr.heldout_rmse_m:.3f} m, n={tr.n_points})",
                                tr.to_json()))
        except TransformError as exc:
            checks.append(Check("image_to_workspace_transform", FAIL, str(exc)))
    # detector validation against recorded poses (how a camera-based estimate earns source B)
    rec = sources.get("isaac-hdf5") or sources.get("recorded")
    det = sources.get("detect")
    if rec and det:
        common = sorted(set(rec) & set(det))
        if len(common) >= 6:
            uv = np.array([[det[e].u, det[e].v] for e in common])
            xy = np.array([[rec[e].x, rec[e].y] for e in common])
            try:
                tr = fit_validated_transform(uv, xy, camera=det[common[0]].camera, frame_id=rec[common[0]].frame_id,
                                             image_size=det[common[0]].image_size, source="detector vs recorded")
                art["detector_transform"] = tr.to_json()
                checks.append(Check("detector_vs_recorded", PASS if tr.valid else WARN,
                                    f"colour detector + homography reproduces recorded positions with leave-one-out "
                                    f"rmse {tr.heldout_rmse_m:.3f} m, max {tr.heldout_max_m:.3f} m (n={tr.n_points}); "
                                    "valid only for THIS camera/scene", tr.to_json()))
            except TransformError as exc:
                checks.append(Check("detector_vs_recorded", WARN, f"could not fit: {exc}"))
        else:
            checks.append(Check("detector_vs_recorded", UNKNOWN, f"{len(common)} common episodes < 6"))
    return sources, checks, art


def detect_positions(view, cfg: dict) -> tuple[dict[int, Position], list[Check], list]:
    cam = cfg["camera"]
    if cam not in view.camera_keys:
        return {}, [Check("detect_positions", FAIL, f"camera {cam} not in dataset {view.camera_keys}")], []
    out, frames, misses = {}, [], []
    for e in sorted(view.data["episode_index"].unique()):
        e = int(e)
        try:
            img = view.read_frame(cam, e, int(cfg.get("frame_index", 0)))
        except Exception as exc:
            misses.append((e, f"decode: {exc}"))
            continue
        d = detect_color_blob(img, cfg["hsv_ranges"], cfg.get("roi"), int(cfg.get("min_area", 200)))
        if d is None:
            misses.append((e, "no blob"))
            frames.append((e, img, None, "no detection"))
            continue
        ref = cfg.get("reference_point", "centroid")
        u, v = (d["u"], d["v"]) if ref == "centroid" else ((d["bbox"][0] + d["bbox"][2]) / 2, d["bbox"][3])
        out[e] = Position(e, AUTO, u=u, v=v, camera=cam, image_size=(img.shape[1], img.shape[0]), verified=False,
                          notes=f"hsv blob {ref}, area {d['area']}", extra={"frame_index": cfg.get("frame_index", 0)})
        frames.append((e, img, (u, v), "auto"))
    st = PASS if not misses else WARN
    return out, [Check("detect_positions", st, f"colour proposals for {len(out)} episodes (UNVERIFIED: image-space "
                                               f"only, a person must confirm them); misses {misses}",
                       {"config": cfg})], frames


# ----------------------------------------------------------------------------- one dataset


def analyze_one(spec: dict, out_root: Path, offline: bool = False) -> dict:
    label = spec["label"]
    out = out_root / label
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {label}")
    view = load_dataset(repo_id=spec.get("repo_id"), root=spec.get("root"), revision=spec.get("revision"),
                        label=label, offline=offline, episodes=_parse_episodes(spec.get("episodes")))
    episodes = sorted(int(e) for e in view.data["episode_index"].unique())
    sections: dict[str, list[Check]] = {s: [] for s in ("technical", "coverage", "quality", "real_sim")}
    sections["technical"] = run_technical_checks(view)
    prov = {"tool": "tools/dataset_diagnostics", "tool_version": TOOL_VERSION,
            "run_at": dt.datetime.now().isoformat(timespec="seconds"), "argv": sys.argv,
            "repo": _git(Path(__file__).resolve().parents[2]), "dataset": dataset_fingerprint(view),
            "role": spec.get("role"), "spec": spec}

    # ------------------------------------------------ checkpoint
    if spec.get("policy_checkpoint"):
        meta = load_checkpoint_meta(spec["policy_checkpoint"], spec.get("checkpoint_revision"), offline)
        sections["technical"] += compare_checkpoint(view, meta)
        prov["checkpoint"] = training_provenance(view, meta)

    # ------------------------------------------------ quality (adapter)
    adapter = spec.get("adapter", "none")
    ep_table = pd.DataFrame({"episode_index": episodes,
                             "length": view.data.groupby("episode_index").size().reindex(episodes).values})
    ep_table = ep_table.set_index("episode_index")
    extras: dict = {"view": view}
    if adapter == "openarm":
        from .openarm import grasp_proxy, run_quality

        opts = spec.get("adapter_options", {})
        ep, summ, qchecks, hold = run_quality(view, gripper_signal=opts.get("gripper_signal", "hybrid"),
                                              overrides=opts.get("quality_overrides"))
        sections["quality"] = qchecks
        extras["hold"] = hold
        if len(ep):
            ep_table = ep_table.join(ep.drop(columns=["length"], errors="ignore"))
        write_json(out / "quality_summary.json", summ)
        proxy, pchecks = grasp_proxy(view)
        sections["quality"] += pchecks
        extras["grasp_proxy"] = proxy
        for e, g in proxy.items():
            ep_table.loc[e, ["grasp_proxy_x", "grasp_proxy_y", "grasp_proxy_z", "grasp_proxy_frame"]] = [
                g["x"], g["y"], g["z"], g["grasp_frame"]]
    else:
        sections["quality"] = [Check("demonstration_quality", UNKNOWN,
                                     "no task adapter given: gripper/handover/tracking diagnostics not run")]

    # ------------------------------------------------ positions & coverage
    sources, pchecks, art = collect_positions(view, spec.get("object_pose_source") or ["none"], out)
    sections["coverage"] += pchecks
    if art.get("detect_frames"):
        plots.plot_annotation_overlay(art["detect_frames"], str(out / "start_frame_proposals.png"))
    if "grasp_proxy" in extras and (sources.get("isaac-hdf5") or sources.get("recorded")):
        from .openarm import validate_grasp_proxy

        sections["quality"].append(validate_grasp_proxy(extras["grasp_proxy"],
                                                        sources.get("isaac-hdf5") or sources.get("recorded")))
    best = merge_by_priority(*sources.values(), episodes=episodes)
    rows = []
    for e in episodes:
        for kind, src in sources.items():
            if e in src:
                rows.append({"dataset": label, "source_kind": kind, **src[e].to_row()})
        if not any(e in s for s in sources.values()):
            rows.append({"dataset": label, "source_kind": "none", **Position(e).to_row()})
    pos_df = pd.DataFrame(rows).drop(columns=[c for c in ("extra_object_traj",) if c in pd.DataFrame(rows).columns])
    pos_df.to_csv(out / "position_annotations.csv", index=False)
    for e, p in best.items():
        ep_table.loc[e, "position_source"] = p.source
        if p.has_metric:
            ep_table.loc[e, ["start_x", "start_y", "start_frame_id"]] = [p.x, p.y, p.frame_id]
        if p.has_image:
            ep_table.loc[e, ["start_u", "start_v", "start_camera"]] = [p.u, p.v, p.camera]

    coverages = {}
    figures = []
    recs_all = []
    for wpath in spec.get("workspace_config") or []:
        cfg = WorkspaceConfig.from_json(wpath)
        tag = cfg.name
        if cfg.frame_type == "metric":
            usable = {e: p for e, p in best.items() if p.has_metric and p.frame_id == cfg.frame_id
                      and p.source in (RECORDED, VALIDATED)}
            provisional = False
        else:
            usable = {e: p for e, p in best.items() if p.has_image and cfg.frame_id in (p.camera, f"image:{p.camera}")}
            if not usable:  # fall back to unverified proposals for a provisional picture
                usable = {e: p for e, p in sources.get("detect", {}).items()}
            provisional = any(p.source == AUTO for p in usable.values())
        if not usable:
            sections["coverage"].append(Check(f"coverage[{tag}]", UNKNOWN,
                                              f"no verified positions in frame {cfg.frame_id}; inputs needed: "
                                              + ("recorded object poses or a validated image->workspace transform "
                                                 "plus image annotations" if cfg.frame_type == "metric" else
                                                 "start-frame annotations (`annotate`)")))
            continue
        boundary_known = (cfg.provenance or {}).get("boundary_measured", True)
        cov = analyze_coverage(usable, cfg, n_recommend=int(spec.get("n_recommend", 5)) if boundary_known else 0)
        if not boundary_known:
            sections["coverage"].append(Check(f"recommendations[{tag}]", UNKNOWN,
                                              "no positions recommended: the workspace boundary is declared "
                                              "unmeasured, so under-covered cells may be unreachable"))
        frag = coverage_fragility(cov.points, cfg)
        for c in coverage_checks(cov, usable, episodes, frag, provisional=provisional):
            c.name = f"{c.name}[{tag}]"
            sections["coverage"].append(c)
        coverages[tag] = cov
        f1, f2, f3 = out / f"position_coverage_{tag}.png", out / f"position_density_{tag}.png", \
            out / f"nearest_demo_distance_{tag}.png"
        ttl = f"{label}: start positions ({'PROVISIONAL, unverified' if provisional else 'verified'})"
        plots.plot_positions([cov], [label], str(f1), ttl)
        plots.plot_density(cov, label, str(f2))
        plots.plot_nearest(cov, label, str(f3))
        figures += [f1, f2, f3]
        for r in cov.recommendations:
            recs_all.append({"dataset": label, "workspace": tag, "provisional": provisional, **r})
        write_json(out / f"coverage_{tag}.json", {"metrics": cov.metrics, "fragility": frag, "regions": cov.regions,
                                                  "duplicates": cov.duplicates, "outside": cov.outside,
                                                  "episodes": cov.episodes, "points": cov.points,
                                                  "config": cfg.__dict__})
    # grasp-position (proxy) coverage: robot-intrinsic, comparable across real and sim
    proxy = extras.get("grasp_proxy") or {}
    for wpath in spec.get("grasp_workspace_config") or []:
        cfg = WorkspaceConfig.from_json(wpath)
        pts = {e: Position(e, "grasp_proxy", x=g["x"], y=g["y"], frame_id=g["frame_id"]) for e, g in proxy.items()}
        if not pts:
            continue
        cov = analyze_coverage(pts, cfg, n_recommend=0)
        coverages[f"grasp:{cfg.name}"] = cov
        f = out / f"grasp_positions_{cfg.name}.png"
        plots.plot_positions([cov], [label], str(f), f"{label}: right-TCP position at grasp (FK proxy, not object "
                                                     "start)", recommendations=False)
        figures.append(f)
        sections["coverage"].append(Check(f"grasp_proxy_spread[{cfg.name}]", PASS,
                                          f"grasp positions span x {cov.metrics['bbox'][0]:.3f}..{cov.metrics['bbox'][1]:.3f}"
                                          f", y {cov.metrics['bbox'][2]:.3f}..{cov.metrics['bbox'][3]:.3f} m; "
                                          f"{cov.metrics['n_independent_groups']} independent at "
                                          f"{cfg.duplicate_radius} m (informational, not a coverage verdict)",
                                          cov.metrics))
    if not spec.get("workspace_config"):
        sections["coverage"].append(Check("coverage", UNKNOWN, "no --workspace-config given"))
    pd.DataFrame(recs_all).to_csv(out / "recommended_positions.csv", index=False)
    ep_table.to_csv(out / "episode_quality.csv")

    ready = readiness(sections)
    write_json(out / "readiness.json", {"dataset": label, **ready})
    write_json(out / "provenance.json", prov)
    print("  " + overall_line(ready))
    return {"label": label, "view": view, "ready": ready, "sections": sections, "coverages": coverages,
            "figures": figures, "extras": extras, "out": out, "prov": prov, "recs": recs_all, "role": spec.get("role"),
            "best": best, "sources": sources}


# ----------------------------------------------------------------------------- comparisons & report


def run(manifest: dict, offline: bool = False) -> dict:
    out_root = Path(manifest["output"])
    out_root.mkdir(parents=True, exist_ok=True)
    results = {d["label"]: analyze_one(d, out_root, offline) for d in manifest["datasets"]}
    extra_html = ""
    comp_rows = []
    for pair in manifest.get("compare", []):
        la, lb = pair
        ra, rb = results[la], results[lb]
        hold = {la: ra["extras"].get("hold", {}), lb: rb["extras"].get("hold", {})}
        checks, tables = compare_datasets(ra["view"], rb["view"], hold_levels=hold)
        tag = f"{la}_vs_{lb}"
        for r in (ra, rb):
            r["sections"]["real_sim"] += [Check(f"{c.name}[{tag}]", c.status, c.detail, c.evidence) for c in checks]
            r["ready"] = readiness(r["sections"])
            write_json(r["out"] / "readiness.json", {"dataset": r["label"], **r["ready"]})
        names = ra["view"].names("observation.state")
        arm = [i for i, n in enumerate(names) if n[:2] in ("RJ", "LJ") and not n.endswith("8.pos")] or None
        speeds = {la: _per_frame_speed(ra["view"], dims=arm), lb: _per_frame_speed(rb["view"], dims=arm)}
        f = out_root / f"real_sim_comparison_{tag}.png"
        plots.plot_real_sim(tables, la, lb, speeds, str(f))
        pd.DataFrame(tables.get("observation.state", [])).to_csv(out_root / f"state_distribution_{tag}.csv", index=False)
        # shared-frame overlays (only frames both datasets actually have)
        for key in set(ra["coverages"]) & set(rb["coverages"]):
            ca, cb = ra["coverages"][key], rb["coverages"][key]
            g = out_root / f"coverage_overlay_{tag}_{key.replace(':', '_')}.png"
            plots.plot_positions([ca, cb], [la, lb], str(g), f"{key}: {la} vs {lb}", recommendations=False)
            extra_html += f"<h3>{key}: {la} vs {lb}</h3>" + _inline(g)
        extra_html += f"<h2>Real/sim comparison: {la} vs {lb}</h2>" + _inline(f)
        comp_rows += [{"pair": tag, "check": c.name, "status": c.status, "detail": c.detail} for c in checks]
    if comp_rows:
        pd.DataFrame(comp_rows).to_csv(out_root / "real_sim_checks.csv", index=False)
    summary_rows = []
    for lab, r in results.items():
        cov_metrics = {k: v.metrics for k, v in r["coverages"].items()}
        summary_rows.append({"dataset": lab, "role": r["role"], **{s: r["ready"][s]["status"] for s in
                                                                   ("technical", "coverage", "quality", "real_sim")},
                             "coverage_metrics": cov_metrics})
    write_json(out_root / "readiness.json", {"datasets": {r["dataset"]: r for r in summary_rows},
                                             "caveat": next(iter(results.values()))["ready"]["caveat"]})
    write_json(out_root / "provenance.json", {lab: r["prov"] for lab, r in results.items()})
    render_html(out_root / "dataset_report.html", manifest.get("title", "Dataset readiness report"),
                {lab: r["ready"] for lab, r in results.items()},
                figures={lab: r["figures"] + ([r["out"] / "start_frame_proposals.png"]
                                              if (r["out"] / "start_frame_proposals.png").exists() else [])
                         for lab, r in results.items()},
                tables={lab: {"recommended positions": r["recs"]} for lab, r in results.items()},
                notes={lab: [f"source: {r['prov']['dataset'].get('repo_id') or r['prov']['dataset']['root']} @ "
                             f"{r['prov']['dataset'].get('resolved_revision')}; role: {r['role']}"]
                       for lab, r in results.items()},
                extra_html=extra_html)
    for lab, r in results.items():  # per-dataset report as well
        render_html(r["out"] / "dataset_report.html", f"{lab}: dataset readiness", {lab: r["ready"]},
                    figures={lab: r["figures"]}, tables={lab: {"recommended positions": r["recs"]}}, notes={})
    print(f"\nwrote {out_root}/dataset_report.html")
    return results


def _inline(p: Path) -> str:
    import base64

    return f"<figure><img alt='{p.stem}' src='data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}'></figure>"


# ----------------------------------------------------------------------------- annotate / fit-transform


def cmd_annotate(args) -> None:
    view = load_dataset(repo_id=args.repo_id, root=args.root, revision=args.revision, label=args.label,
                        offline=args.offline)
    existing = read_annotations(args.output, dataset=view.label) if Path(args.output).exists() else {}
    proposals = {}
    if args.propose:
        proposals, chk, _ = detect_positions(view, {**json.loads(Path(args.propose).read_text()), "camera": args.camera})
        print(chk[0].detail)
    result = dict(existing)
    eps = sorted(int(e) for e in view.data["episode_index"].unique())
    if args.interactive:
        import matplotlib

        matplotlib.use(args.backend)
        import matplotlib.pyplot as plt

        print("click the object centre; [enter]/[space] accept, [s] skip (UNKNOWN), [q] save & quit")
        for e in eps:
            if e in existing and existing[e].source == HUMAN and not args.redo:
                continue
            img = view.read_frame(args.camera, e, args.frame_index)
            prop = proposals.get(e)
            state = {"pt": (prop.u, prop.v) if prop else None, "key": None}
            fig, ax = plt.subplots(figsize=(9, 7))
            ax.imshow(img)
            ax.set_title(f"{view.label} episode {e} ({args.camera}, frame {args.frame_index})")
            mark = ax.scatter(*([[state["pt"][0]], [state["pt"][1]]] if state["pt"] else [[], []]), marker="+", s=200,
                              color="#00ff66", lw=2)

            def on_click(ev):
                if ev.inaxes is ax and ev.xdata is not None:
                    state["pt"] = (ev.xdata, ev.ydata)
                    mark.set_offsets([state["pt"]])
                    fig.canvas.draw_idle()

            def on_key(ev):
                state["key"] = ev.key
                plt.close(fig)

            fig.canvas.mpl_connect("button_press_event", on_click)
            fig.canvas.mpl_connect("key_press_event", on_key)
            plt.show()
            if state["key"] == "q":
                break
            if state["key"] == "s" or state["pt"] is None:
                result.pop(e, None)
                continue
            result[e] = Position(e, HUMAN, u=state["pt"][0], v=state["pt"][1], camera=args.camera,
                                 image_size=(img.shape[1], img.shape[0]), verified=True,
                                 extra={"annotator": args.annotator, "frame_index": args.frame_index,
                                        "timestamp": dt.datetime.now().isoformat(timespec="seconds")})
    else:
        for e, p in proposals.items():
            if e not in result:
                result[e] = p  # unverified proposals; a person must confirm them with --interactive
        frames = [(e, view.read_frame(args.camera, e, args.frame_index),
                   (result[e].u, result[e].v) if e in result else None,
                   result[e].source if e in result else "none") for e in eps]
        sheet = Path(args.output).with_suffix(".png")
        plots.plot_annotation_overlay(frames, str(sheet))
        print(f"review sheet: {sheet}")
    write_annotations(args.output, view.label, result, annotator=args.annotator)
    n_h = sum(p.source == HUMAN for p in result.values())
    print(f"wrote {args.output}: {len(result)} rows ({n_h} human-verified, {len(result) - n_h} unverified)")


def cmd_fit_transform(args) -> None:
    df = pd.read_csv(args.correspondences)
    for c in ("u", "v", "x", "y"):
        if c not in df.columns:
            raise SystemExit(f"{args.correspondences}: needs columns u,v,x,y (pixels -> metres)")
    tr = fit_validated_transform(df[["u", "v"]].to_numpy(), df[["x", "y"]].to_numpy(), camera=args.camera,
                                 frame_id=args.frame_id, max_allowed_m=args.max_error,
                                 image_size=tuple(args.image_size) if args.image_size else None,
                                 source=str(args.correspondences))
    write_json(Path(args.output), tr.to_json())
    print(f"held-out rmse {tr.heldout_rmse_m:.4f} m, max {tr.heldout_max_m:.4f} m, n={tr.n_points}: "
          f"{'VALID' if tr.valid else 'REFUSED'} -> {args.output}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tools.dataset_diagnostics", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="readiness + coverage report")
    a.add_argument("--manifest", help="JSON with {output, title, datasets:[...], compare:[[a,b]]}")
    a.add_argument("--repo-id")
    a.add_argument("--root")
    a.add_argument("--revision")
    a.add_argument("--label")
    a.add_argument("--role", choices=["real", "sim", "other"], default="other")
    a.add_argument("--episodes", help="e.g. 0-9,12 (default: all)")
    a.add_argument("--output", default="outputs/dataset_readiness")
    a.add_argument("--object-pose-source", action="append", default=None)
    a.add_argument("--workspace-config", action="append", default=None)
    a.add_argument("--grasp-workspace-config", action="append", default=None)
    a.add_argument("--policy-checkpoint")
    a.add_argument("--checkpoint-revision")
    a.add_argument("--adapter", choices=["none", "openarm"], default="none")
    a.add_argument("--n-recommend", type=int, default=5)
    a.add_argument("--offline", action="store_true", help="use only the local Hugging Face cache")
    n = sub.add_parser("annotate", help="mark the object centre in each episode's starting frame")
    n.add_argument("--repo-id")
    n.add_argument("--root")
    n.add_argument("--revision")
    n.add_argument("--label")
    n.add_argument("--camera", required=True)
    n.add_argument("--frame-index", type=int, default=0)
    n.add_argument("--output", required=True)
    n.add_argument("--propose", help="detector JSON to pre-fill (unverified) points")
    n.add_argument("--interactive", action="store_true")
    n.add_argument("--redo", action="store_true")
    n.add_argument("--annotator", default="")
    n.add_argument("--backend", default="TkAgg")
    n.add_argument("--offline", action="store_true")
    t = sub.add_parser("fit-transform", help="fit + validate a pixel -> metric-plane homography")
    t.add_argument("--correspondences", required=True)
    t.add_argument("--camera", required=True)
    t.add_argument("--frame-id", required=True)
    t.add_argument("--max-error", type=float, default=0.02)
    t.add_argument("--image-size", type=int, nargs=2)
    t.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "annotate":
        if not (args.repo_id or args.root):
            ap.error("annotate needs --repo-id or --root")
        return cmd_annotate(args)
    if args.cmd == "fit-transform":
        return cmd_fit_transform(args)
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        if args.output != "outputs/dataset_readiness":
            manifest["output"] = args.output
    else:
        if not (args.repo_id or args.root):
            ap.error("analyze needs --repo-id, --root or --manifest")
        manifest = {"output": args.output, "title": f"Dataset readiness: {args.label or args.repo_id or args.root}",
                    "datasets": [{"label": args.label or (args.repo_id or Path(args.root).name).split("/")[-1],
                                  "repo_id": args.repo_id, "root": args.root, "revision": args.revision,
                                  "role": args.role, "episodes": args.episodes,
                                  "object_pose_source": args.object_pose_source,
                                  "workspace_config": args.workspace_config,
                                  "grasp_workspace_config": args.grasp_workspace_config,
                                  "policy_checkpoint": args.policy_checkpoint,
                                  "checkpoint_revision": args.checkpoint_revision, "adapter": args.adapter,
                                  "n_recommend": args.n_recommend}]}
    run(manifest, offline=args.offline)


if __name__ == "__main__":
    main()
