"""Split 10-episode pilot subsets out of lerobot/robotwin_unified with LeRobot's own split_dataset (no re-encoding of
the clean-episode video files, which hold exactly the 50 demo_clean episodes of one task).

Episode -> task mapping (verified 2026-11-14 from the instruction strings and decoded frames, see R01): the 27,500
episodes are 50 blocks of 550 in alphabetical RoboTwin 2.0 task order; in each block the first 50 are demo_clean
(white table, no clutter) and the other 500 demo_randomized. Pre-registered pick: the first K demo_clean episodes.
Usage: robotwin_subset.py OUT_DIR task [task ...]   (HF_HUB_OFFLINE=1; needed files must already be downloaded)
"""
import json, sys
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import split_dataset

REPO, REV = "lerobot/robotwin_unified", "1287871839fae2296bc27b88a5457c3e1eba8e1f"
ROOT = Path.home() / f".cache/huggingface/hub/datasets--lerobot--robotwin_unified/snapshots/{REV}"
TASKS = """adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock click_bell dump_bin_bigbin
grab_roller handover_block handover_mic hanging_mug lift_pot move_can_pot move_pillbottle_pad move_playingcard_away
move_stapler_pad open_laptop open_microwave pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right
place_bread_basket place_bread_skillet place_burger_fries place_can_basket place_cans_plasticbox place_container_plate
place_dual_shoes place_empty_cup place_fan place_mouse_pad place_object_basket place_object_scale place_object_stand
place_phone_stand place_shoe press_stapler put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object shake_bottle
shake_bottle_horizontally stack_blocks_three stack_blocks_two stack_bowls_three stack_bowls_two stamp_seal turn_switch""".split()
assert len(TASKS) == 50
BLOCK, CLEAN, K = 550, 50, 10


def episodes(task, k=K):
    b = TASKS.index(task) * BLOCK
    return list(range(b, b + k))


if __name__ == "__main__":
    out, tasks = Path(sys.argv[1]), sys.argv[2:]
    for t in tasks:
        eps = episodes(t)
        if (out / t).exists():
            sys.exit(f"refusing to overwrite {out / t}")
        src = LeRobotDataset(REPO, root=ROOT, revision=REV, episodes=eps)
        ds = split_dataset(src, {t: eps}, output_dir=out)[t]
        rec = {"task": t, "source_repo": REPO, "source_revision": REV, "source_episodes": eps,
               "rule": "first K demo_clean episodes of the task block", "episodes": ds.meta.total_episodes,
               "frames": ds.meta.total_frames, "task_strings": len(ds.meta.tasks)}
        (out / t / "subset.json").write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec))
