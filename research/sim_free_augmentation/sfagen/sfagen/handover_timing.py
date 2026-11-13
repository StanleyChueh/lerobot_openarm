"""Fixed-duration hand-over timing port: v3 planner3.py make_spec's non-dual clock."""
import numpy as np
from .timing import replay_reference


def handover_clock(T, grasp, receiver_close, giver_released, offset):
    source = np.array([0, grasp + 12, receiver_close, giver_released, T - 1], float)
    output = source + np.array([0, 0, offset, offset, 0])
    if not (np.diff(source) > 0).all() or not (np.diff(output) > 0).all():
        return None
    return np.interp(np.arange(T), output, source), output, source


def plan(rec, offset):
    if len(rec.objects) != 1 or len(rec.arms) != 2 or not offset:
        return None
    donor = rec.objects[0]["arm"]; receiver = next(s for s in rec.arms if s != donor)
    grasp = int(rec.objects[0]["grasp_frame"])
    close = next((t for t in rec.ev[receiver]["closes"] if t > grasp), None)
    opened = next((t for t in rec.ev[donor]["opens"] if close is not None and t > close), None)
    if opened is None:
        return None
    released = next((t for t in range(opened, rec.T) if rec.sp["g"+donor][t] >= .7), opened)
    clock = handover_clock(rec.T, grasp, close, released, offset)
    if clock is None:
        return None
    tau, output, source = clock
    from .plan import action_lead
    lead = action_lead(rec.state, rec.action)
    state, action = replay_reference(rec, tau, lead)
    split = rec.robot.split(state); edited = np.abs(tau - np.arange(rec.T)) > 1e-9
    marks = dict(t_grasp=grasp, t_rejoin=rec.T-1, dt_frames=int(offset), t_window_start=close+int(offset),
                 t_window_end=released+int(offset), knots_output=output.tolist(), knots_source=source.tolist(), time_scale=1.)
    return dict(family="interaction_timing", object=None, shift_m=[0., 0., 0.], marks=marks, w=edited.astype(float),
                state=state, action=action, ref_state=state.copy(), ref_action=action.copy(), tau=tau,
                q={s: split[s] for s in rec.arms}, ik_err=np.zeros((rec.T, 2)), lead=lead, moved_arms=list(rec.arms),
                objects=[np.stack([rec.obj_pose(k, t) for t in tau]) for k in range(len(rec.objects))],
                edited=edited, t_rejoin=rec.T-1)
