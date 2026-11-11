"""MGW forward renderer on the calibrated battery reconstruction; object slot = station (box proxy geometry) or battery
(18650 cylinder). Configuration = battery-v00 configuration (Pringles config + part-aware fill)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "battery/bsfa"), str(HERE.parent / "render_v4")]
import srecon, brecon, mgw  # noqa: E402
mgw.MGWForward.PART_FILL = True
assert mgw.MGWForward.TM_ERODE == 21 and mgw.MGWForward.DESPECKLE and mgw.MGWForward.CLEAR_CAD == 0


def renderer(rec, device, obj):
    if obj == "station":
        mgw.can_points = lambda: srecon.box_points(rec.st_ext, 0.0015)
    else:
        mgw.rc.CAN_RADIUS, mgw.rc.CAN_HEIGHT = brecon.BAT_R, brecon.BAT_L
        mgw.can_points = _cyl
    return mgw.MGWForward(rec, device=device, calib=srecon.mgw_calib_file())


_cyl = mgw.can_points
