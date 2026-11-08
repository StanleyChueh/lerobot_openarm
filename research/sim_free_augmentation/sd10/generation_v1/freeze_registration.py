"""Create target/config/provenance JSON only. No trajectory synthesis, IK, renderer or policy calls.

Run once; refuses to overwrite. Grid constants do not depend on source-set coverage or L1 outcomes.
"""
from pathlib import Path
import hashlib,json,subprocess,sys
from datetime import datetime,timezone
import numpy as np

HERE=Path(__file__).resolve().parent;SD=HERE.parent;REPO=SD.parents[2]
sys.path.insert(0,str(SD))
from sd10 import common as C

HEAD="8d31437e5779ef80aca9ffd3e6fa94b288d54f5e"
SETS={"Original10":list(range(10)),"SD10":[1,6,7,11,13,14,27,34,42,47],"Random10":[5,6,7,8,18,25,29,32,43,44]}


def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()


def dump(name,x):
    with open(HERE/name,"x") as f:f.write(json.dumps(x,indent=2,allow_nan=False)+"\n")


def quat_R(q):
    w,x,y,z=np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
      [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
      [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def main():
    assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()==HEAD
    assert not subprocess.check_output(["git","diff","--name-only"],cwd=REPO,text=True).strip()
    prior=json.loads((SD/"source_provenance_manifest.json").read_text())
    assert json.loads((SD/"selected_10_episodes.json").read_text())["selected_episodes"]==SETS["SD10"]
    protected={str((SD/f).relative_to(REPO)):h for f,h in prior["sd10_files_sha256"].items()}
    for rel,h in protected.items():assert sha(REPO/rel)==h,rel
    deps={}
    for name,entry in prior["datasets"].items():
        for f in entry["files"]:
            if f["local"]:
                assert sha(f["local"])==f["sha256"]
                deps[name+"/"+f["path"]]={"local":f["local"],"sha256":f["sha256"]}
    lids=json.loads((SD/"outputs/lid_centers.json").read_text())
    P={e:np.array([lids[str(e)]["lid_u"],lids[str(e)]["lid_v"]]) for e in range(50)}
    # Same inherited camera prior as sfa.robot; numerical proxy, NOT accepted independent calibration.
    camera={"pos":[.05621,-.00305,.83062],"quat_wxyz":[.18720,-.68244,.69543,-.12490],
            "fx":607.22,"fy":607.22,"cx":326.4,"cy":238.7,"W":640,"H":480}
    R=quat_R(camera["quat_wxyz"]);p0=np.array(camera["pos"])
    K=np.array([[camera["fx"],0,camera["cx"]],[0,camera["fy"],camera["cy"]],[0,0,1]])
    targets=[]
    for r,v in enumerate(np.linspace(100,360,10)):
        for c,u in enumerate(np.linspace(260,440,10)):
            uv=np.array([u,v]);ray=R@np.linalg.solve(K,[u,v,1]);s=(.32-p0[2])/ray[2]
            assert s>0;xyz=p0+s*ray
            region="edge" if r in (0,9) or c in (0,9) else ("central" if 3<=r<=6 and 3<=c<=6 else "intermediate")
            assignments={}
            for arm,ids in SETS.items():
                ranking=sorted(ids,key=lambda e:(float(np.sum((uv-P[e])**2)),e))
                assignments[arm]={"source_episode":ranking[0],"score_squared_pixels":float(np.sum((uv-P[ranking[0]])**2)),
                    "distance_pixels":float(np.linalg.norm(uv-P[ranking[0]])),"ranked_source_ids":ranking,
                    "rule":"minimum squared native body-lid distance; exact ties lower episode ID; no fallback"}
            d0=assignments["Original10"]["distance_pixels"]
            targets.append({"target_id":f"T{10*r+c:03d}","grid_row":r,"grid_col":c,"region":region,
                "requested_lid_uv_native_px":uv.tolist(),"candidate_lid_xyz_robot_base_m":xyz.tolist(),
                "candidate_can_axis_lid_to_bottom":[0,0,-1],"yaw":"unconstrained cylinder gauge; source yaw retained",
                "pose_evidence":"ESTIMATED_CANDIDATE_NOT_VERIFIED_PHYSICAL_POSE",
                "near_far_reference":"Original10<=40px near, >40px far; fixed across conditions",
                "near_far":"near" if d0<=40 else "far","underrepresented_by_original10":d0>40,
                "spatial_block":5*(r//2)+c//2,"assignment":assignments})
    assert len(targets)==100 and len({tuple(t["requested_lid_uv_native_px"]) for t in targets})==100
    targetdoc={"study":"SD10-G1","status":"FROZEN_BEFORE_GENERATOR_IMPLEMENTATION_AND_EXECUTION",
        "date_utc":datetime.now(timezone.utc).isoformat(),"base_head":HEAD,
        "procedure":"10x10 inclusive Cartesian lattice: u=linspace(260,440,10), v=linspace(100,360,10); row-major",
        "selection_independence":"No generation outcomes or L1 predictions/errors read. Sources never reselected.",
        "camera_prior":camera,"candidate_lid_height_m":.32,"candidate_table_height_m":.09,
        "metric_pose_accuracy":"UNKNOWN: fixed ray/height proxy under inherited unvalidated body calibration",
        "region_meaning":"Edges of registered ROI, not independently verified reachable workspace edges",
        "targets":targets,"primary_attempts":300,
        "fixed_source_sensitivity":{"target_ids":[f"T{i:03d}" for i in range(0,100,11)],"source_episode":6,
            "attempts":30,"not_retries":True,"note":"Same source6 for all three arm labels; independent re-execution, no primary count credit"}}
    dump("locked_targets.json",targetdoc)
    ds=C.pool();ten=C.ten();union=sorted(set().union(*map(set,SETS.values())))
    meta={}
    raw=json.loads((SD/"outputs/features_raw.json").read_text())
    for e in union:
        S,A=ds.vec(e,"observation.state"),ds.vec(e,"action")
        meta[str(e)]={"capture_id":f"pringles-real-v00-capture-{e:02d}","frames":len(S),
            "state_sha256":hashlib.sha256(S.tobytes()).hexdigest(),"action_sha256":hashlib.sha256(A.tobytes()).hexdigest(),
            "events_rc_lc_ro":[raw[str(e)][x] for x in ["rc","lc","ro"]],
            "initial_lid_uv":P[e].tolist(),"duration_s":len(S)/30,
            "grasp_to_receive_s":(raw[str(e)]["lc"]-raw[str(e)]["rc"])/30,
            "observation_quality":{"can_crop_sharpness":raw[str(e)]["can_crop_sharpness"],
                "receiver_from_closure_only":"must be logged by new reconstruction; never treated as measured contact"}}
    for e in range(10):
        assert np.array_equal(ds.vec(e,"action"),ten.vec(e,"action")) and np.array_equal(ds.vec(e,"observation.state"),ten.vec(e,"observation.state"))
    sources={}
    for folder in ["sfa","v2/sfa2","v3/sfa3"]:
        for p in (SD.parent/folder).glob("*.py"):sources[str(p.relative_to(REPO))]=sha(p)
    for p in (REPO/"model").rglob("*"):
        if p.is_file() and (p.suffix in (".urdf",".stl",".STL") or p.name=="calibration.json"):
            sources[str(p.relative_to(REPO))]=sha(p)
    checkpoint=Path.home()/".cache/huggingface/hub/models--facebook--sam2.1-hiera-small/snapshots/ee5bba1d82bb8749febdf90f45e84b687142ba03"
    models={"sam":{"snapshot":str(checkpoint),"revision":checkpoint.name,"files":{p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()}},
        "raft":{"weights":"Raft_Large_Weights.C_T_SKHT_V2","local":str(Path.home()/".cache/torch/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth")}}
    models["raft"]["sha256"]=sha(models["raft"]["local"])
    config={"study":"SD10-G1","base_head":HEAD,"conditions":SETS,"data":{"pool_repo":C.POOL_REPO,"pool_revision":C.POOL_REV,
          "real10_lineage_repo":C.TEN_REPO,"real10_revision":C.TEN_REV,"all_rgb_from_pool":True,"allowed_source_union":union,
          "excluded_v4a":[10,17,24,31,38,45],"camera":"body_cam only; no wrist processing"},
        "targets_sha256":sha(HERE/"locked_targets.json"),"source_metadata":meta,
        "algorithm":{"planner":"v3 G+ waypoint geometry only, common pool/full-initial-pose adapter; no C comparison",
          "renderer":"v3 render3 layered lid/body/robot homographies, unchanged pixels except common input plumbing",
          "initial_pose":"absolute shared candidate position; shortest rotation aligns can axis to down; yaw free",
          "handover":"source native interaction, no pass/regrasp/timing perturbations; D=I Gamma=I tau=t scale=1",
          "reconstruction":"v3/v2 source body-only fitting, SAM small; common fixed body camera/lag; receiver closure-only flagged",
          "body_lag_frames":-1.625,"camera_refit":False,"source_scoring":"locked_targets.json assignments, no quality/IK/outcome reranking",
          "ik":{"method":"pinned Robot.ik DLS","iterations":80,"damping":.001,"joint_margin_rad":.02,
                "warm_start":"half previous solution plus half corresponding source reference","restarts":0},
          "actions":"q_generated[t+source_lead] + (source_action[t]-source_state[t+source_lead]); grippers source ZOH",
          "source_lead":"pinned data.action_lead over 0..6; lower argmin tie; no fit against generated RGB",
          "export_native_interval":"[3,T-6), shared across all sources; no endpoint padding, require all transitions and >=15 post-release frames",
          "provenance_logging":"sidecar only, no condition-specific pixel repair or filtering"},
        "acceptance":{"target_lid_position_error_m":.002,"initial_axis_error_deg":1,"ik_max_position_error_m":.004,
          "ik_max_rotation_error_deg":2,"contact_waypoint_position_m":.002,"contact_waypoint_rotation_deg":1,
          "joint_position_slack_rad":.000001,"joint_speed":"URDF velocity limits, same per joint",
          "joint_accel_max_rad_s2":100,"command_step_max_rad":.25,
          "steady_attachment_error_m":.01,"steady_attachment_rotation_deg":5,
          "receiver_capture_error_m":.01,"receiver_capture_rotation_deg":5,
          "can_step_max_m":.08,"can_step_max_deg":10,"holder_switch_step_m":.005,"holder_switch_step_deg":3,
          "can_table_penetration_max_m":.01,"distal_arm_arm_penetration_max_m":.005,
          "can_nonfinger_link_penetration_max_m":.01,"all_frames_checked":True,
          "no_source_relative_threshold_relaxation":True,"label_construction_abs_tolerance_rad":.000001,
          "source_gripper_chronology_preserved":True,"post_release_min_frames":15},
        "compute":{"seed":20261108,"one_selected_source_attempt_per_primary_target":1,"retry_budget":0,"target_shrink_budget":0,
          "source_preprocessing_cap_s":600,"geometry_cap_s":120,"rgb_encode_decode_cap_s":120,
          "cpu_threads":4,"gpu":"one common GPU model/device, sequential; record actual device/precision before launch",
          "condition_order":"target k ascending, rotate condition order by k mod3; each arm sees T000..T099",
          "preprocessing":"new pool-keyed cold cache for all23; shared capture uses same asset hashes; charge standalone ten-source costs per arm",
          "report":"cold preprocessing, assets/model loads, failures/timeouts, verification, rendering, encoding, I/O and per-accepted-bundle time"},
        "statistics":{"unit":"paired target ID, all100; failures remain zero","bootstrap_draws":10000,"seed":20261108,
          "geometry_min_gain_pp":10,"require_both_controls":True,"paired_ci_lower_gt":0,
          "spatial_block_ci_lower_gt":0,"exact_mcnemar_two_sided_p_max":.05,
          "rgb_clean_bundle_min_gain_pp":10,"h2_common_rendered_min_targets":30,
          "h2_gross_artifact_reduction_min_pp":10,"new_mechanism_claim_permitted":False},
        "models":models,"inherited_code_and_robot_sha256":sources,"source_dataset_files":deps,
        "preserved_source_selection_sha256":protected,
        "historical_seals":{str(p.relative_to(REPO)):sha(p) for p in (SD.parent/"v4a").glob("*.sha256")},
        "learning_seal_sha256":sha(SD/"learning_v1/locked_artifacts.sha256"),
        "status":"REGISTERED_NOT_IMPLEMENTED_NOT_EXECUTED; no final result claimed"}
    dump("locked_generation_config.json",config)
    from collections import Counter
    print(json.dumps({"targets":len(targets),"regions":dict(Counter(t["region"] for t in targets)),
       "far":sum(t["near_far"]=="far" for t in targets),"source_sets":SETS,
       "candidate_xyz_ranges":[np.min([t["candidate_lid_xyz_robot_base_m"] for t in targets],axis=0).tolist(),
        np.max([t["candidate_lid_xyz_robot_base_m"] for t in targets],axis=0).tolist()],"generator_called":False},indent=2))


if __name__=="__main__":main()
