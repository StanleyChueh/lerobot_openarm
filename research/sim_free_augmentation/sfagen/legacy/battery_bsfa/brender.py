"""MGW forward renderer (render_v4/mgw.py, unchanged algorithm and configuration) on the battery reconstruction.
The 'can' slot of MGW is the battery: an 18650 cylinder (r 9.25 mm, L 65 mm; origin = end-cap centre, z = axis)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), "/home/stanley/Stanley_ws/sfa_hf_export/render_v4"]
import mgw  # noqa: E402
import brecon  # noqa: E402
mgw.rc.CAN_RADIUS, mgw.rc.CAN_HEIGHT = brecon.BAT_R, brecon.BAT_L
assert mgw.MGWForward.TM_ERODE == 21 and mgw.MGWForward.DESPECKLE and mgw.MGWForward.CLEAR_CAD == 0


def renderer(rec, device):
    return mgw.MGWForward(rec, device=device)


def candidate_video(rec, z):
    return mgw.candidate_video(rec, z)
