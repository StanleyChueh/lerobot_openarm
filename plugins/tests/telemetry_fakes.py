"""A fake dual-arm DM bus for the telemetry tests: what openarm_can's Motor objects hold AND what a passive tap
on the interface sees, generated from one true state with the library's own wire format. No CAN traffic.

Frames are encoded the way openarm_can 1.2.8 encodes them (dm_motor_control.cpp: clamp, then truncate to
12/16 bits), and the Motor state is set to the DECODE of the reply it last processed -- so the "library"
here and a real one disagree with the true state in exactly the same way.
"""
import time

import openarm_can as oa

import lerobot_robot_openarm_umeow  # noqa: F401  (adds the repo root to sys.path)
from lerobot_robot_openarm_umeow import telemetry as tel

TYPES = [oa.MotorType.DM8009] * 2 + [oa.MotorType.DM4340] * 2 + [oa.MotorType.DM4310] * 4
LIMITS = {f"{s}J{i}": tel.motor_limits(t) for s in "LR" for i, t in enumerate(TYPES, start=1)}
SEND = list(range(0x01, 0x09))
RECV = list(range(0x11, 0x19))
FOLLOWER_ORDER = [f"{s}J{i}" for s in "RL" for i in range(1, 9)]  # follower._motor_channels(): right first


def d2u(x, lo, hi, bits):
    """CanPacketEncoder::double_to_uint."""
    x = min(max(x, lo), hi)
    return int((x - lo) / (hi - lo) * ((1 << bits) - 1))


def enc_state(status, motor_id, q, dq, tau, t_mos, t_rotor, lim):
    p, v, t = lim
    qu, dqu, tu = d2u(q, -p, p, 16), d2u(dq, -v, v, 12), d2u(tau, -t, t, 12)
    return bytes([(status << 4) | motor_id, qu >> 8, qu & 0xFF, dqu >> 4, ((dqu & 0xF) << 4) | (tu >> 8), tu & 0xFF,
                  t_mos, t_rotor])


def enc_mit(q, dq, kp, kd, tau, lim):
    """CanPacketEncoder::pack_mit_control_data."""
    p, v, t = lim
    qu, dqu, tu = d2u(q, -p, p, 16), d2u(dq, -v, v, 12), d2u(tau, -t, t, 12)
    kpu, kdu = d2u(kp, 0, 500, 12), d2u(kd, 0, 5, 12)
    return bytes([qu >> 8, qu & 0xFF, dqu >> 4, ((dqu & 0xF) << 4) | (kpu >> 8), kpu & 0xFF, kdu >> 4,
                  ((kdu & 0xF) << 4) | (tu >> 8), tu & 0xFF])


def enc_ctrl(code):
    return b"\xff" * 7 + bytes([code])


def enc_refresh(send_id):
    return bytes([send_id & 0xFF, (send_id >> 8) & 0xFF, 0xCC, 0, 0, 0, 0, 0])


def lsb(lim, which):
    p, v, t = lim
    return {"q": 2 * p / 65535, "dq": 2 * v / 4095, "tau": 2 * t / 4095, "kp": 500 / 4095, "kd": 5 / 4095}[which]


class FakeMotor:
    def __init__(self):
        self.q = self.dq = self.tau = 0.0
        self.t_mos = self.t_rotor = 0
        self.fail = False

    def get_position(self):
        if self.fail:
            raise RuntimeError("injected motor read failure")
        return self.q

    def get_velocity(self):
        return self.dq

    def get_torque(self):
        return self.tau

    def get_state_tmos(self):
        return self.t_mos

    def get_state_trotor(self):
        return self.t_rotor


class FakeTap:
    """Stands in for telemetry.WireTap: a queue of (can_id, payload, kernel ns)."""

    def __init__(self):
        self.available, self.dropped, self.errors, self.queue = True, 0, 0, []

    def read(self):
        out, self.queue = self.queue, []
        return out

    def close(self):
        self.available = False


class FakeWire:
    """Both arms: true joint state -> frames on the tap -> the Motor state openarm_can would end up with.

      drop    joints whose replies never arrive (a motor off the bus): no frame, Motor state frozen
      unread  joints whose refresh reply reaches the wire only AFTER the library's drain finished, so the Motor
              holds the previous frame -- the reply to the last MIT command (processed on the next read)
    """

    def __init__(self):
        self.taps = {"R": FakeTap(), "L": FakeTap()}
        self.motors = {j: FakeMotor() for j in tel.JOINTS}
        self.true = {j: [0.0, 0.0, 0.0] for j in tel.JOINTS}
        self.status, self.temps = 0x1, (31, 29)
        self.inbox = {j: [] for j in tel.JOINTS}
        self.carry = {j: [] for j in tel.JOINTS}
        self.drop, self.unread = set(), set()
        self.commands = []

    def channels(self):
        return [(j, self.motors[j]) for j in FOLLOWER_ORDER]

    def _reply(self, j, t_ns):
        i = int(j[2:])
        data = enc_state(self.status, i, *self.true[j], *self.temps, LIMITS[j])
        self.taps[j[0]].queue.append((RECV[i - 1], data, t_ns))
        return tel.decode_state(data, LIMITS[j])

    def command(self, sent: dict, vel: dict, tau_ff: dict, kp: float, kd: float):
        """follower.send_action: one MIT frame per motor; each motor answers with a state frame."""
        self.commands.append(dict(sent))
        for j in tel.JOINTS:
            i = int(j[2:])
            t = time.time_ns() - 3000
            self.taps[j[0]].queue.append(
                (SEND[i - 1], enc_mit(sent[f"{j}.pos"], vel.get(f"{j}.vel", 0.0), kp, kd, tau_ff.get(j, 0.0), LIMITS[j]), t))
            self.true[j][0] = sent[f"{j}.pos"]  # perfect tracking, like the other mocks
            if j not in self.drop:
                self.inbox[j].append(self._reply(j, t + 1000))

    def ctrl(self, side: str, code: int):
        for i in range(1, 9):
            self.taps[side].queue.append((SEND[i - 1], enc_ctrl(code), time.time_ns() - 1000))

    def read(self) -> dict:
        """follower._read_motor_positions_once: refresh every motor, drain, report positions."""
        for j in tel.JOINTS:
            i = int(j[2:])
            t = time.time_ns() - 2000
            self.taps[j[0]].queue.append((tel.REFRESH_ID, enc_refresh(SEND[i - 1]), t))
            self.inbox[j] = self.carry[j] + self.inbox[j]
            self.carry[j] = []
            if j in self.drop:
                continue
            st = self._reply(j, t + 1000)
            (self.carry if j in self.unread else self.inbox)[j].append(st)
        for j in tel.JOINTS:
            if self.inbox[j]:
                _, q, dq, tau, t_mos, t_rotor = self.inbox[j][-1]
                m = self.motors[j]
                m.q, m.dq, m.tau, m.t_mos, m.t_rotor = q, dq, tau, t_mos, t_rotor
            self.inbox[j] = []
        return {f"{j}.pos": self.motors[j].q for j in tel.JOINTS}
