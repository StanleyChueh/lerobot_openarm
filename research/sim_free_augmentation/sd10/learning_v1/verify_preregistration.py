"""Read-only, model-free SD10-L1 conformance checks. Never imports Torch or loads a policy."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SD = HERE.parent
REPO = SD.parents[2]
sys.path[:0] = [str(HERE), str(SD)]
from protocol_tools import *
from sd10 import common as C


def read(p):
    return json.loads(p.read_text())


def check_seal(path, root):
    for line in path.read_text().splitlines():
        h, rel = line.split(None, 1)
        assert file_sha(root / rel.strip()) == h, rel


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify-rgb", action="store_true")
    ap.add_argument("--preseal",action="store_true",help="Construction checks only, before first seal; no frozen status claimed")
    a = ap.parse_args()
    actual_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    subprocess.check_call(["git", "merge-base", "--is-ancestor", HEAD, "HEAD"], cwd=REPO)
    seal = HERE / "locked_artifacts.sha256"
    if not a.preseal: check_seal(seal, REPO)
    cfg, rt, m = [read(HERE / n) for n in ["training_config.json", "source_and_runtime_manifest.json", "locked_evaluation_manifest.json"]]
    exp, normal = read(HERE / "training_exposure.json"), read(HERE / "training_normalization.json")
    assert cfg["source_arms"] == ARMS and cfg["source_slots"] == SLOTS and cfg["seeds"] == SEEDS
    assert cfg["optimizer_updates"] == 1000 and cfg["batch_size"] == 16 and cfg["gradient_accumulation"] == 1
    assert cfg["real_fraction"] == 1 and cfg["synthetic_fraction"] == 0 and cfg["image_augmentation"] is False
    assert len(cfg["run_order"]) == len(set(tuple(x) for x in cfg["run_order"])) == 9
    assert set(map(tuple,cfg["run_order"])) == {(a,s) for a in ARMS for s in SEEDS}
    assert set(cfg["policy_overrides"]["input_features"]) == {"observation.state","observation.images.camera1"}
    assert cfg["policy_overrides"]["input_features"]["observation.state"]["shape"] == [16]
    assert cfg["policy_overrides"]["output_features"]["action"]["shape"] == [16]
    assert set(m["evaluation_episodes"]) == set(EVAL)
    union = set().union(*[set(s) for s in ARMS.values()])
    assert not union & set(EVAL) and set(range(50)) == union | set(EVAL) | set(V4A)
    assert not set(V4A) & (union | set(EVAL))
    prior = read(SD / "source_provenance_manifest.json")
    for rel,h in prior["sd10_files_sha256"].items(): assert file_sha(SD / rel) == h
    for x in rt["dataset_files"].values(): assert file_sha(x["path"]) == x["sha256"]
    for model in ["pretrained", "backbone_tokenizer"]:
        for rel,x in rt[model]["files"].items(): assert file_sha(Path(rt[model]["local_snapshot"])/rel) == x["sha256"]
    lib = Path(rt["lerobot"]["root"])
    assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=lib,text=True).strip() == LEROBOT_HEAD
    for rel,h in rt["lerobot"]["src_py_sha256"].items(): assert file_sha(lib / rel) == h
    v4a = SD.parent/"v4a"
    for name,h in rt["v4a_seals"].items():
        assert file_sha(v4a/name) == h; check_seal(v4a/name,v4a)
    evidence=read(v4a/"evidence_index.json")["sources"]
    for rel,fact in evidence.items():
        p=Path(rel) if Path(rel).is_absolute() else REPO/rel
        assert file_sha(p)==fact["sha256"]
    ds = C.pool(); A = {e: ds.vec(e,"action") for e in range(50)}; S = {e: ds.vec(e,"observation.state") for e in range(50)}
    lengths = {e: len(A[e]) for e in range(50)}; ph={e:phases(A[e]) for e in range(50)}
    hashes={e:array_sha(np.concatenate([S[e],A[e]],axis=1)) for e in range(50)}
    assert len(set(hashes.values()))==50 and not {hashes[e] for e in union}&{hashes[e] for e in EVAL}
    for e in EVAL:
        rc,lc,ro=events(A[e]); labels=ph[e]
        for t,want in [(rc-6,0),(rc-5,1),(rc+14,1),(rc+15,2),(lc-6,2),(lc-5,3),(ro+4,3),(ro+5,4)]:
            assert labels[t]==want,(e,t,want)
        assert np.array_equal(labels,m["recordings"][str(e)]["phase_ids"])
    for arm,ids in ARMS.items():
        assert normal["arms"][arm]["episodes"]==ids
        for key,data in [("observation.state",S),("action",A)]:
            x=np.concatenate([data[e] for e in ids]).astype(np.float64);stats=normal["arms"][arm]["stats"][key]
            assert np.array_equal(x.mean(0).astype(np.float32),np.array(stats["mean"],np.float32))
            assert np.array_equal(np.maximum(x.std(0),1e-6).astype(np.float32),np.array(stats["std"],np.float32))
    for seed in SEEDS:
        z=np.load(HERE/f"sampling_seed{seed}.npz",allow_pickle=False);slot,q=schedule(seed)
        assert np.array_equal(z["slot"],slot) and np.array_equal(z["quantile"],q)
        for arm,ids in SLOTS.items():
            ep=np.array(ids)[slot];t=np.floor(q*np.array([lengths[e]-49 for e in ids])[slot]).astype(np.int16)
            assert np.array_equal(z[arm+"_episode"],ep) and np.array_equal(z[arm+"_frame"],t)
            counts=np.zeros(5,dtype=np.int64)
            for e in ARMS[arm]:
                starts=t[ep==e];assert len(starts)==1600
                assert starts.min()>=0 and starts.max()<=lengths[e]-50
                labels=np.stack([ph[e][t:t+50] for t in starts])
                assert labels.shape==(1600,50)
                c=np.bincount(labels.flatten(),minlength=5);counts+=c
                ref=exp["seeds"][str(seed)][arm]["sources"][str(e)]
                assert np.array_equal(c,ref["target_phase_counts"])
                assert np.array_equal(np.bincount((starts[:,None]+np.arange(50)).flatten(),minlength=lengths[e]),ref["native_target_multiplicity"])
            assert counts.sum()==800000 and np.array_equal(counts,exp["seeds"][str(seed)][arm]["target_phase_counts"])
        for aa in ARMS:
            for bb in ARMS:
                for e in set(ARMS[aa])&set(ARMS[bb]):
                    keep=z[aa+"_episode"]==e
                    assert np.array_equal(keep,z[bb+"_episode"]==e)
                    assert np.array_equal(z[aa+"_frame"][keep],z[bb+"_frame"][keep])
    assert np.array_equal(np.load(HERE/"learning_rates.npy",allow_pickle=False),learning_rates())
    assert abs(learning_rates()[0]-1e-4/34)<1e-18 and learning_rates()[-1]>2.5e-6
    z=np.load(HERE/"evaluation_arrays.npz",allow_pickle=False)
    assert file_sha(HERE/"evaluation_arrays.npz")==m["evaluation_arrays_sha256"]
    expected=np.array([[e,t] for e in EVAL for t in eval_anchors(lengths[e])],dtype=np.int32)
    assert np.array_equal(z["items"],expected) and len(expected)==834
    for key,h in m["array_content_sha256"].items(): assert array_sha(z[key])==h
    for i,(e,t) in enumerate(expected):
        assert np.array_equal(z["state"][i],S[e][t]) and np.array_equal(z["targets"][i],A[e][t:t+50])
        assert np.array_equal(z["target_phase_ids"][i],ph[e][t:t+50])
        en=m["entries"][i];assert [en["episode"],en["input_frame"]]==[e,t]
        assert en["target_frames_inclusive"]==[int(t),int(t+49)]
        assert array_sha(z["targets"][i])==en["target_action_sha256"]
    assert all(np.isfinite(z[k]).all() for k in z.files)
    oracle=recording_scores(z["targets"].copy(),z["items"],z["targets"],z["target_phase_ids"],lengths)
    assert all(oracle[e][p]==0 for e in EVAL for p in PHASES)
    # Numeric scorer fixtures only; no policy predictions or robot demonstrations are produced.
    fixture=z["targets"].copy();fixture[...,13]+=.14
    known=recording_scores(fixture,z["items"],z["targets"],z["target_phase_ids"],lengths)
    assert all(abs(known[e][p]-.01)<1e-7 for e in EVAL for p in PHASES)
    fixture=z["targets"].copy();fixture[...,14]+=(-1.298*2)
    wrong_grip=recording_scores(fixture,z["items"],z["targets"],z["target_phase_ids"],lengths)
    assert all(wrong_grip[e]["handover_gripper_openness_mae"][0]>1.99 for e in EVAL)
    bi=np.load(HERE/"bootstrap_indices.npz",allow_pickle=False)
    for group,ids in m["recording_groups"].items():
        assert np.array_equal(bi[group+"_episodes"],ids)
        ii=bi[group+"_indices"];assert ii.shape==(10000,len(ids)) and ii.min()>=0 and ii.max()<len(ids)
        ref=np.ones((3,len(ids)));fixture=ref*.9
        for seedidx in [None,bi["seed_indices"]]:
            check=paired_relative(fixture,ref,ii,seedidx)
            assert abs(check["relative_improvement"]-.1)<1e-12 and np.max(abs(np.array(check["ci95"])-.1))<1e-12
    if a.verify_rgb:
        from freeze_inputs import body_frames
        for e in EVAL:
            got=body_frames(ds,e,set(eval_anchors(lengths[e])))
            for en in m["entries"]:
                if en["episode"]==e: assert got[en["input_frame"]]==en["native_video"]
    print(json.dumps({"study":"SD10-L1","decision":"GO for registered design; no training executed",
      "base_head":HEAD,"actual_head":actual_head,"evaluation_recordings":len(EVAL),"evaluation_inputs":len(expected),
      "near":m["recording_groups"]["near"],"far":m["recording_groups"]["far"],
      "registered_runs":9,"training_run_artifacts_checked":0,"training_completed_claimed":False,
      "schedule_draws_per_run":16000,"draws_per_source":1600,"valid_targets_per_run":800000,
      "all_target_phase_boundaries_pass":True,"shared_capture_draws_identical_across_arms":True,
      "oracle_and_known_error_scorer_fixtures_pass":True,"no_clipped_gripper_error_pass":True,
      "paired_bootstrap_numeric_fixtures_pass":True,"rgb_redecoded_and_matched":a.verify_rgb,
      "historical_SD10_hashes_unchanged":True,"v4a_indexed_files_verified":len(evidence),
      "sealed":not a.preseal,"locked_artifacts_manifest_sha256":file_sha(seal) if not a.preseal else None,"PASS":True},indent=2))


if __name__=="__main__":main()
