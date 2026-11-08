"""Create new SD10-L1 metadata/schedules/evaluation locks. Never loads a policy or trains.

Refuses to overwrite any generated artifact. Run from repository root with --output-dir pointing
to a NEW folder (default: this folder). Existing historical files are read-only dependencies.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import av
import numpy as np

HERE = Path(__file__).resolve().parent
SD = HERE.parent
REPO = SD.parents[2]
sys.path[:0] = [str(SD), str(HERE)]
from sd10 import common as C
from protocol_tools import *


def dump(p, x):
    with open(p, "x") as f:
        f.write(json.dumps(x, indent=2, allow_nan=False) + "\n")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def files(root):
    return {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": file_sha(p)}
            for p in sorted(root.rglob("*")) if p.is_file()}


def body_frames(ds, ep, want):
    p, t0, _ = ds.video(C.BODY, ep)
    out = {}
    with av.open(str(p)) as cont:
        st = cont.streams.video[0]
        cont.seek(int(max(0, t0 - .5) / st.time_base), stream=st)
        for fr in cont.decode(st):
            if fr.time is None or fr.time < t0 - .5 / 30:
                continue
            k = int(round((fr.time - t0) * 30))
            if k > max(want):
                break
            if k in want and k not in out:
                rgb = fr.to_ndarray(format="rgb24")
                assert rgb.shape == (480, 640, 3)
                out[k] = {"rgb_sha256": array_sha(rgb), "pts": int(fr.pts),
                          "time_base_numerator": int(st.time_base.numerator),
                          "time_base_denominator": int(st.time_base.denominator),
                          "native_pts_s": float(fr.time)}
    assert set(out) == set(want)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output-dir", type=Path, default=HERE); args = ap.parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    expected = ["locked_evaluation_manifest.json", "training_config.json", "training_normalization.json",
                "training_exposure.json", "source_and_runtime_manifest.json", "evaluation_arrays.npz",
                "bootstrap_indices.npz", "learning_rates.npy"] + [f"sampling_seed{s}.npz" for s in SEEDS]
    assert not any((out / f).exists() for f in expected), "Refusing to overwrite a previous freeze"
    subprocess.check_call(["git", "-C", str(REPO), "merge-base", "--is-ancestor", HEAD, "HEAD"])
    assert not git(REPO, "diff", "--name-only")
    assert json.loads((SD / "selected_10_episodes.json").read_text())["selected_episodes"] == ARMS["SD10"]
    assert all(sorted(SLOTS[a]) == ARMS[a] for a in ARMS)
    union = sorted(set().union(*map(set, ARMS.values())))
    assert set(EVAL) == set(range(50)) - set(union) - set(V4A)
    assert not set(EVAL) & set(union)
    for e in union:
        seen = [SLOTS[a].index(e) for a in ARMS if e in ARMS[a]]
        assert len(set(seen)) == 1
    prior = json.loads((SD / "source_provenance_manifest.json").read_text())
    for rel, h in prior["sd10_files_sha256"].items():
        assert file_sha(SD / rel) == h, rel
    inputs = {}
    for name, dsinfo in prior["datasets"].items():
        for f in dsinfo["files"]:
            if f["local"]:
                h = file_sha(f["local"]); assert h == f["sha256"]
                inputs[name + "/" + f["path"]] = {"sha256": h, "path": f["local"], "bytes": f["bytes"]}
    ds = C.pool(); ten = C.ten(); A = {e: ds.vec(e, "action") for e in range(50)}
    S = {e: ds.vec(e, "observation.state") for e in range(50)}
    for e in range(10):
        assert np.array_equal(A[e], ten.vec(e, "action")) and np.array_equal(S[e], ten.vec(e, "observation.state"))
    lengths = {e: len(A[e]) for e in range(50)}
    phase = {e: phases(A[e]) for e in range(50)}
    lids = json.loads((SD / "outputs/lid_centers.json").read_text())
    P = {e: np.array([lids[str(e)]["lid_u"], lids[str(e)]["lid_v"]]) for e in range(50)}
    sources = {str(e): {"capture_id": f"pringles-real-v00-capture-{e:02d}", "episode": e, "length": lengths[e],
               "state_sha256": array_sha(S[e]), "action_sha256": array_sha(A[e]),
               "phase_ids_sha256": array_sha(phase[e]), "command_events_rc_lc_ro": list(events(A[e])),
               "lid_uv_native_px": P[e].tolist(),
               "body_video_window_s": list(ds.video(C.BODY, e)[1:])} for e in range(50)}
    base_rev = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"
    vlm_rev = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
    cache = Path.home() / ".cache/huggingface/hub"
    base = cache / f"models--lerobot--smolvla_base/snapshots/{base_rev}"
    vlm = cache / f"models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/{vlm_rev}"
    lib = REPO.parent / "lerobot"
    assert git(lib, "rev-parse", "HEAD") == LEROBOT_HEAD and not git(lib, "status", "--porcelain")
    runtime = {"git_head": git(REPO, "rev-parse", "HEAD"), "branch": git(REPO, "branch", "--show-current"),
       "pool": {"repo_id": C.POOL_REPO, "revision": C.POOL_REV},
       "original10_lineage_dataset": {"repo_id": C.TEN_REPO, "revision": C.TEN_REV,
                                      "ten_to_pool": {str(e): e for e in range(10)},
                                      "use": "lineage only; ALL arms use pool encoding"},
       "dataset_files": inputs, "captures": sources,
       "pretrained": {"repo_id": "lerobot/smolvla_base", "revision": base_rev, "local_snapshot": str(base), "files": files(base)},
       "backbone_tokenizer": {"repo_id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "revision": vlm_rev,
                              "local_snapshot": str(vlm), "files": files(vlm)},
       "lerobot": {"root": str(lib), "head": LEROBOT_HEAD,
                   "src_py_sha256": {str(p.relative_to(lib)): file_sha(p) for p in sorted((lib / "src/lerobot").rglob("*.py"))}},
       "packages": {k: importlib.metadata.version(k) for k in ["numpy", "scipy", "pandas", "torch", "transformers", "av", "safetensors"]},
       "python": sys.version, "source_selection_manifest_sha256": file_sha(SD / "source_provenance_manifest.json"),
       "v4a_seals": {p.name: file_sha(p) for p in (SD.parent / "v4a").glob("*.sha256")},
       "all_source_capture_ids": union, "evaluation_capture_ids": EVAL, "v4a_excluded": V4A}
    dump(out / "source_and_runtime_manifest.json", runtime)
    cfg = {"study_id": "SD10-L1", "status": "PREREGISTERED_NOT_TRAINED", "source_arms": ARMS, "source_slots": SLOTS,
      "seeds": SEEDS, "optimizer_updates": UPDATES, "batch_size": BATCH, "gradient_accumulation": 1,
      "real_fraction": 1.0, "synthetic_fraction": 0.0, "image_augmentation": False,
      "run_order": [[a, s] for s in SEEDS for a in [list(ARMS)[(j+s) % 3] for j in range(3)]],
      "train_config": {"use_policy_training_preset": False, "use_amp": False, "mixed_precision": "no", "ema": False,
                       "num_workers": 4, "prefetch_factor": 2, "log_freq": 50, "save_freq": 1000,
                       "eval_steps": 0, "env": None, "wandb": False, "push_to_hub": False},
      "optimizer": {"type": "adamw", "lr": .0001, "betas": [.9, .95], "eps": 1e-8,
                    "weight_decay": 1e-10, "grad_clip_norm": 10.0},
      "scheduler": {"type": "cosine_decay_with_warmup", "num_warmup_steps": 33, "num_decay_steps": 1000,
                    "peak_lr": .0001, "decay_lr": .0000025, "lr_by_optimizer_update": "learning_rates.npy",
                    "note": "explicit effective 33/1000 schedule, same as scaling old 1000/30000 preset"},
      "policy_overrides": {"type": "smolvla", "n_obs_steps": 1, "chunk_size": 50, "n_action_steps": 50,
        "input_features": {"observation.state": {"type": "STATE", "shape": [16]},
                           "observation.images.camera1": {"type": "VISUAL", "shape": [3, 480, 640]}},
        "output_features": {"action": {"type": "ACTION", "shape": [16]}},
        "max_state_dim": 32, "max_action_dim": 32, "resize_imgs_with_padding": [512, 512],
        "empty_cameras": 0, "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
        "freeze_vision_encoder": True, "train_expert_only": True, "train_state_proj": True,
        "adapt_to_pi_aloha": False, "use_delta_joint_actions_aloha": False, "compile_model": False,
        "num_steps": 10, "tokenizer_max_length": 48, "pad_language_to": "max_length",
        "vlm_model_name": str(vlm), "load_vlm_weights": True, "use_amp": False, "push_to_hub": False},
      "pretrained_snapshot": str(base), "body_camera_key": C.BODY, "rename_map": {C.BODY: "observation.images.camera1"},
      "native_image_preprocessing": "PyAV RGB uint8 640x480 -> CHW float32 /255 -> policy resize/letterbox512 and [-1,1]",
      "task": TASK, "action_target_offsets": list(range(50)), "padding": "none: starts 0..T-50 inclusive",
      "loss": "standard SmolVLA flow-matching MSE, full 50x16; denominator 12800 per batch",
      "normalization": "per-arm training native rows only, population mean/std, no full-pool stats; frozen training_normalization.json",
      "sampling": "frozen per-seed NPZ, no DataLoader shuffle; 1600 draws/source with paired stratified quantiles",
      "evaluation": {"manifest": "locked_evaluation_manifest.json", "arrays": "evaluation_arrays.npz", "original_body_rgb_only": True,
        "batch_size": 32, "inference_autocast": "bfloat16", "prediction_archive_dtype": "float32",
        "chunk_size": 50, "diffusion_steps": 10, "noise": "stored numpy PCG64 normal(50,32), seed 20261105+1000*episode+frame",
        "state": "recorded state[t]", "targets": "raw recorded action[t:t+50]", "checkpoint": "update1000 only"},
      "statistics": {"bootstrap_draws": 10000, "bootstrap_seed": 20261107,
         "primary": "SD10 vs Original10 far-grasp 14-arm-joint raw MAE; equal recordings, seed averaged",
         "primary_min_relative_improvement": .05, "primary_ci_lower_gt": 0.0, "per_seed_nonnegative": True,
         "max_handover_relative_harm": .02, "max_phase_balanced_relative_harm": .02,
         "random_comparison": "hierarchical secondary after primary screen; SD10 vs fixed Random10",
         "random_min_relative_improvement": .03,
         "handover_gripper_openness_mae_guard": "report both sides; increase >0.02 absolute flags MODIFY even if arm-error gates pass",
         "ci": "paired recording percentile95, conditional on 3 fitted seeds; crossed seed+recording sensitivity also mandatory",
         "causal_position_claims": "PROHIBITED: source selection, timing/session/action distribution and normalizers co-vary"}}
    dump(out / "training_config.json", cfg)
    norm = {"policy": "native training rows only; row weighted population moments in float64, tensors rounded float32",
            "std_floor": 1e-6, "normalizer_eps": 1e-8, "no_clipping": True, "image_normalization": "IDENTITY", "arms": {}}
    for arm, ids in ARMS.items():
        stats = {}
        for key, data in [("observation.state", S), ("action", A)]:
            x = np.concatenate([data[e] for e in ids]).astype(np.float64)
            stats[key] = {"mean": x.mean(0).astype(np.float32).tolist(),
                          "std": np.maximum(x.std(0), 1e-6).astype(np.float32).tolist()}
        norm["arms"][arm] = {"episodes": ids, "native_rows": sum(lengths[e] for e in ids), "stats": stats}
    dump(out / "training_normalization.json", norm)
    exposure = {"updates": UPDATES, "batch_size": BATCH, "draws_per_run": 16000,
                "valid_target_timesteps_per_run": 800000, "scalar_action_loss_terms_per_run": 12800000,
                "source_draws_per_run": 1600, "target_timesteps_per_source_per_run": 80000,
                "phase_exposure_exactly_matched": False, "seeds": {}}
    for seed in SEEDS:
        slot, q = schedule(seed); arrays = {"slot": slot, "quantile": q}; exp = {}
        for arm, ids in SLOTS.items():
            ep = np.array(ids, dtype=np.int16)[slot]
            starts = np.floor(q * np.array([lengths[e] - 49 for e in ids])[slot]).astype(np.int16)
            target_phases = np.stack([phase[int(e)][int(t):int(t)+50] for e,t in zip(ep.flat, starts.flat)])
            assert target_phases.shape == (16000, 50)
            src = {}
            for e in ARMS[arm]:
                mask = ep.flatten() == e; times = starts.flatten()[mask]
                assert mask.sum() == 1600 and times.min() >= 0 and times.max() <= lengths[e] - 50
                row_counts = np.bincount((times[:, None] + np.arange(50)).flatten(), minlength=lengths[e])
                src[str(e)] = {"draws": int(mask.sum()), "unique_input_starts": int(len(np.unique(times))),
                    "full_chunk_eligible_starts": lengths[e]-49, "target_timesteps": int(mask.sum()*50),
                    "target_phase_counts": np.bincount(target_phases[mask].flatten(), minlength=5).tolist(),
                    "input_phase_counts": np.bincount(phase[e][times], minlength=5).tolist(),
                    "native_target_multiplicity": row_counts.tolist()}
            exp[arm] = {"sources": src, "target_phase_counts": np.bincount(target_phases.flatten(), minlength=5).tolist(),
                        "unique_eligible_input_starts": sum(lengths[e]-49 for e in ARMS[arm]),
                        "native_training_rows": sum(lengths[e] for e in ARMS[arm])}
            arrays[arm+"_episode"], arrays[arm+"_frame"] = ep, starts
        np.savez_compressed(out / f"sampling_seed{seed}.npz", **arrays)
        exposure["seeds"][str(seed)] = exp
    dump(out / "training_exposure.json", exposure)
    np.save(out / "learning_rates.npy", learning_rates(), allow_pickle=False)
    entries, items, states, targets, phs, noises = [], [], [], [], [], []
    near, far = [], []
    for e in EVAL:
        ts = eval_anchors(lengths[e]); ims = body_frames(ds, e, set(ts)); rows = ds.rows(e)
        dist = {a: float(min(np.linalg.norm(P[e]-P[s]) for s in ids)) for a,ids in ARMS.items()}
        bin_name = "near" if dist["Original10"] <= 40 else "far"
        (near if bin_name == "near" else far).append(e)
        for t in ts:
            i = len(items); items.append([e,t]); states.append(S[e][t]); targets.append(A[e][t:t+50]); phs.append(phase[e][t:t+50])
            noise = np.random.Generator(np.random.PCG64(20261105+1000*e+t)).standard_normal((50,32)).astype(np.float32)
            noises.append(noise)
            entries.append({"item_index": i, "episode": e, "input_frame": t, "target_frames_inclusive": [t,t+49],
                  "target_phase_ids": phase[e][t:t+50].tolist(), "recording_bin": bin_name,
                  "row_timestamp_s": float(rows.iloc[t].timestamp), "native_video": ims[t],
                  "state_sha256": array_sha(S[e][t]), "target_action_sha256": array_sha(A[e][t:t+50]),
                  "noise_sha256": array_sha(noise)})
        print(f"locked eval episode {e}, {len(ts)} input frames, {bin_name}", flush=True)
    arrays = {"items": np.array(items, dtype=np.int32), "state": np.stack(states), "targets": np.stack(targets),
              "target_phase_ids": np.stack(phs), "noise": np.stack(noises)}
    np.savez_compressed(out / "evaluation_arrays.npz", **arrays)
    bootstrap = {}; rng = np.random.Generator(np.random.PCG64(20261107))
    for title, ids in [("all21", EVAL), ("near",near), ("far",far)]:
        bootstrap[title+"_episodes"] = np.array(ids, dtype=np.int16)
        bootstrap[title+"_indices"] = rng.integers(0, len(ids), (10000,len(ids)), dtype=np.int16)
    bootstrap["seed_indices"] = rng.integers(0,3,(10000,3),dtype=np.int8)
    np.savez_compressed(out / "bootstrap_indices.npz", **bootstrap)
    manifest = {"study_id": "SD10-L1", "status": "FROZEN_EXPLORATORY_EVALUATION_NOT_UNTOUCHED_TEST",
       "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "base_git_head": HEAD,
       "pool_repo_id": C.POOL_REPO, "pool_revision": C.POOL_REV,
       "training_sources": ARMS, "excluded_source_union": union, "v4a_excluded_records": V4A,
       "evaluation_episodes": EVAL, "recording_groups": {"all21": EVAL, "near": near, "far": far},
       "bin_rule": "Initial SAM lid-center distance to nearest Original10 source, near <=40 native body pixels; common to every arm",
       "position_evidence_sha256": file_sha(SD / "outputs/lid_centers.json"),
       "phase_rule": "per native target frame: approach [0,rc-5); grasp [rc-5,rc+15); carry [rc+15,lc-5); handover [lc-5,ro+5); release [ro+5,T)",
       "phase_names": PHASES, "timing_convention": "Recorded native same-row RGB/state/action, no temporal shift. Native PTS saved separately; true camera exposure alignment UNKNOWN; body lag prior is a proxy, not calibration.",
       "item_order": "episode ascending then input frame ascending; stride5 and always include final full-chunk start T-50",
       "num_items": len(items), "num_recordings": len(EVAL), "entries": entries,
       "recordings": {str(e): dict(sources[str(e)], phase_ids=phase[e].tolist(),
                                   distance_to_sources_px={a:float(min(np.linalg.norm(P[e]-P[s]) for s in ids)) for a,ids in ARMS.items()},
                                   allowed_input_frames=eval_anchors(lengths[e])) for e in EVAL},
       "evaluation_arrays_sha256": file_sha(out/"evaluation_arrays.npz"),
       "array_content_sha256": {k:array_sha(v) for k,v in arrays.items()},
       "required_prediction_archives": [f"{a}_s{s}.npz" for a in ARMS for s in SEEDS],
       "prediction_contract": "float32 prediction[N,50,16], exact items[N,2]; checkpoint/config/normalizer/update identities required; missing/duplicate/misordered/nonfinite arrays are INCOMPLETE",
       "historical_exposure": "All evaluation recordings were reused for research/development; selection inspected initial pool positions. Not confirmation, not a publication test.",
       "lineage_exclusion": "No source capture or descendant is evaluation. Alias original_10 episode i == pool episode i. No synthetic data permitted in any arm."}
    dump(out/"locked_evaluation_manifest.json", manifest)
    print(json.dumps({"items":len(items),"near":near,"far":far,"runs":9,"training_started":False}))


if __name__ == "__main__":
    main()
