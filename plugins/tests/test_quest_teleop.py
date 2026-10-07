"""Synthetic-Quest test of OpenArmQuest: hold, anchor, tracking, triggers, reset. No hardware.

    env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python plugins/tests/test_quest_teleop.py
"""
import json, socket, threading, time
import numpy as np
from lerobot_teleoperator_openarm_quest import OpenArmQuest, OpenArmQuestConfig
from lerobot_robot_openarm_umeow.common import MOTOR_KEYS

PORT = 5996
state = {"rc": [0.25, 1.1, 0.3], "lc": [-0.25, 1.1, 0.3], "rt": 0.0, "lt": 0.0, "x": False, "y": False, "run": True}
def sender():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while state["run"]:
        pose = lambda p: {"x": p[0], "y": p[1], "z": p[2], "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
        msg = {"t": time.time(), "rc": pose(state["rc"]), "lc": pose(state["lc"]), "rf": pose([0, 1.5, 0]),
               "rt": state["rt"], "lt": state["lt"], "x": state["x"], "y": state["y"], "a": False, "b": False, "v": 0, "vl": 0, "vr": 0}
        s.sendto(json.dumps(msg).encode(), ("127.0.0.1", PORT)); time.sleep(1 / 72)
threading.Thread(target=sender, daemon=True).start()

t = OpenArmQuest(OpenArmQuestConfig(port=PORT, episode_buttons=False)); t.connect()
d = t.driver
def press(k):
    state[k] = True; time.sleep(0.1); state[k] = False; time.sleep(0.1)
home_cmd = d.home_driver.copy()
time.sleep(1.0)
c = d.command(); print("1 HELD before X: equals home driver:", np.allclose(c[[*range(7), *range(8, 15)]], home_cmd[[*range(7), *range(8, 15)]]), "| status", d.status())

press("x"); time.sleep(0.5)
c = d.command(); print("2 after X, no motion: max |cmd-home| arm joints = %.4f rad" % np.abs(c[[*range(7), *range(8, 15)]] - home_cmd[[*range(7), *range(8, 15)]]).max())

# Move the right controller +5 cm along Unity x over 1 s; expected EE delta in arm_origin = R_FRAME @ (0.05,0,0) = (0,-0.05,0)
for i in range(50):
    state["rc"][0] = 0.25 + 0.05 * (i + 1) / 50; time.sleep(0.02)
time.sleep(1.5)
c = d.command()
# FK on a separate model, so as not to race the IK thread over its MjData.
from lerobot_teleoperator_openarm_quest.ik_driver import build_kinematics
fk = build_kinematics(t.config.ik_xml, "home", t.config.ik_args)
pr0, pl0 = fk.fk_bimanual(home_cmd[:8], home_cmd[8:]); pr, pl = fk.fk_bimanual(c[:8], c[8:])
print("3 right EE moved", np.round(pr[:3] - pr0[:3], 4), "(expect ~[0, -0.05, 0]); left EE moved", np.round(pl[:3] - pl0[:3], 4), "(expect ~0)")

state["rt"] = 1.0; time.sleep(0.3); a = t.get_action(); state["rt"] = 0.0; time.sleep(0.3); b = t.get_action()
g = t.calib["right"]["gripper"]
print("4 right trigger 1.0 -> RJ8 %.3f (closed_raw %.3f); trigger 0.0 -> RJ8 %.3f (open_raw %.3f)" % (a["RJ8.pos"], g["closed_raw"], b["RJ8.pos"], g["open_raw"]))

press("y"); time.sleep(0.3)
c = d.command(); print("5 after Y: back at home:", np.allclose(c[[*range(7), *range(8, 15)]], home_cmd[[*range(7), *range(8, 15)]]), "live:", d.status()["live"])

s0 = d.status(); time.sleep(2.0); s1 = d.status()
print("6 IK solve rate while packets flow: %.0f Hz (target %g), Quest packets %.0f Hz, failed solves %d" % ((s1["solves"] - s0["solves"]) / 2, t.config.ik_hz, (s1["packets"] - s0["packets"]) / 2, s1["failed_solves"]))
t0 = time.perf_counter(); n = 300
for _ in range(n): t.get_action()
print("7 get_action() cost: %.2f ms" % ((time.perf_counter() - t0) / n * 1e3), "| keys match robot order:", list(t.get_action()) == MOTOR_KEYS)
state["run"] = False; t.disconnect()
