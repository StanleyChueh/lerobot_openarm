"""Verify a preregistration, target assignments, inputs and statistical fixtures. NO generation."""
from pathlib import Path
import argparse,ast,json,subprocess,sys
from collections import Counter
import numpy as np

HERE=Path(__file__).resolve().parent;SD=HERE.parent;REPO=SD.parents[2]
sys.path[:0]=[str(HERE),str(SD)]
from freeze_registration import sha,SETS,HEAD,quat_R
from paired_stats import paired_effect,meaningful_yield
from sd10 import common as C


def read(p):return json.loads(p.read_text())


def seal_check(p,root):
    for line in p.read_text().splitlines():
        h,rel=line.split(None,1);assert sha(root/rel.strip())==h,rel


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--preseal",action="store_true");a=ap.parse_args()
    subprocess.check_call(["git","merge-base","--is-ancestor",HEAD,"HEAD"],cwd=REPO)
    if not a.preseal:seal_check(HERE/"locked_artifacts.sha256",REPO)
    t,c=read(HERE/"locked_targets.json"),read(HERE/"locked_generation_config.json")
    assert c["conditions"]==SETS and sha(HERE/"locked_targets.json")==c["targets_sha256"]
    assert [x["target_id"] for x in t["targets"]]==[f"T{i:03d}" for i in range(100)]
    uv=np.array([x["requested_lid_uv_native_px"] for x in t["targets"]])
    expected=np.array([[u,v] for v in np.linspace(100,360,10) for u in np.linspace(260,440,10)])
    assert np.array_equal(uv,expected) and len(set(map(tuple,uv)))==100
    lids=read(SD/"outputs/lid_centers.json");P={e:np.array([lids[str(e)]["lid_u"],lids[str(e)]["lid_v"]]) for e in range(50)}
    node=ast.parse((SD.parent/"sfa/robot.py").read_text())
    cam=ast.literal_eval(next(n.value for n in node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="BODY_CAM" for t in n.targets)))
    assert all(cam[k]==v for k,v in t["camera_prior"].items())
    R=quat_R(cam["quat_wxyz"]);p0=np.array(cam["pos"])
    for x in t["targets"]:
        u,v=x["requested_lid_uv_native_px"]
        ray=R@np.array([(u-cam["cx"])/cam["fx"],(v-cam["cy"])/cam["fy"],1.])
        pt=p0+(.32-p0[2])/ray[2]*ray
        assert np.max(abs(pt-x["candidate_lid_xyz_robot_base_m"]))<1e-12
        for arm,ids in SETS.items():
            rank=sorted(ids,key=lambda e:(np.sum((P[e]-[u,v])**2),e));rec=x["assignment"][arm]
            assert rec["ranked_source_ids"]==rank and rec["source_episode"]==rank[0]
            assert abs(rec["distance_pixels"]-np.linalg.norm(P[rank[0]]-[u,v]))<1e-10
        assert (x["near_far"]=="far")==x["underrepresented_by_original10"]==(x["assignment"]["Original10"]["distance_pixels"]>40)
    assert Counter(x["region"] for x in t["targets"])=={"edge":36,"central":16,"intermediate":48}
    assert sum(x["near_far"]=="far" for x in t["targets"])==46
    assert t["fixed_source_sensitivity"]["source_episode"] in set.intersection(*map(set,SETS.values()))
    assert t["fixed_source_sensitivity"]["attempts"]==30 and t["primary_attempts"]==300
    ds,ten=C.pool(),C.ten();fingerprints={}
    for arm,ids in SETS.items():assert len(set(ids))==10 and not set(ids)&set(C.V4A_LOCKED)
    for e,rec in c["source_metadata"].items():
        S,A=ds.vec(int(e),"observation.state"),ds.vec(int(e),"action")
        import hashlib
        assert hashlib.sha256(S.tobytes()).hexdigest()==rec["state_sha256"]
        assert hashlib.sha256(A.tobytes()).hexdigest()==rec["action_sha256"]
        fingerprints[e]=hashlib.sha256(S.tobytes()+A.tobytes()).hexdigest()
        assert len(S)-6-rec["events_rc_lc_ro"][-1]>=15 and rec["events_rc_lc_ro"][0]>3
    assert len(set(fingerprints.values()))==23
    for e in range(10):assert np.array_equal(ten.vec(e,"action"),ds.vec(e,"action")) and np.array_equal(ten.vec(e,"observation.state"),ds.vec(e,"observation.state"))
    for group in ["preserved_source_selection_sha256","inherited_code_and_robot_sha256","historical_seals"]:
        for rel,h in c[group].items():assert sha(REPO/rel)==h,rel
    for d in c["source_dataset_files"].values():assert sha(d["local"])==d["sha256"]
    for f,h in c["models"]["sam"]["files"].items():assert sha(Path(c["models"]["sam"]["snapshot"])/f)==h
    assert sha(c["models"]["raft"]["local"])==c["models"]["raft"]["sha256"]
    assert sha(SD/"learning_v1/locked_artifacts.sha256")==c["learning_seal_sha256"]
    seal_check(SD/"learning_v1/locked_artifacts.sha256",REPO)
    for rel in c["historical_seals"]:seal_check(REPO/rel,(REPO/rel).parent)
    ev=read(SD.parent/"v4a/evidence_index.json")["sources"]
    for rel,f in ev.items():assert sha(Path(rel) if Path(rel).is_absolute() else REPO/rel)==f["sha256"]
    z=np.zeros(100,dtype=int)
    assert not meaningful_yield(z,z)["PASS"] and not meaningful_yield(1-z,1-z)["PASS"]
    win=z.copy();win[np.arange(0,100,11)]=1
    positive=meaningful_yield(win,z);assert positive["PASS"] and abs(positive["difference"]-.1)<1e-12
    assert abs(positive["exact_mcnemar_p_two_sided"]-.001953125)<1e-12
    assert not meaningful_yield(z,win)["PASS"]
    xx=np.r_[np.ones(55),np.zeros(45)];yy=1-xx
    noisy=meaningful_yield(xx,yy);assert abs(noisy["difference"]-.1)<1e-12 and not noisy["PASS"]
    subset=paired_effect(np.ones(30),np.zeros(30),np.arange(30));assert subset["difference"]==1
    assert paired_effect(win,z)==paired_effect(win,z)
    try:paired_effect(np.r_[np.nan,np.zeros(99)],z)
    except AssertionError:pass
    else:raise AssertionError("nonfinite outcome accepted")
    print(json.dumps({"preregistration_checks":"PASS","sealed":not a.preseal,"base_head":HEAD,
        "targets":100,"primary_attempt_budget":300,"fixed_source_control_budget":30,
        "geometry_or_rgb_executed":False,"new_learning_executed":False,"source_sets_unchanged":True,
        "far_targets":46,"source_fingerprints_verified":23,"v4a_files_verified":len(ev),
        "historical_learning_seal_preserved":True,"paired_statistic_fixtures_pass":True,
        "positive_ten_point_fixture":positive,"noisy_ten_point_fixture":noisy,
        "generation_results":"NOT AVAILABLE; final scientific decision not issued"},indent=2))


if __name__=="__main__":main()
