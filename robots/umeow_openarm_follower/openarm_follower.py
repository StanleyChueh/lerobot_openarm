#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import time

import numpy as np
import pinocchio as pin
import openarm_can as oa

from multiprocessing import Process, Array

from functools import cached_property

from lerobot.cameras.utils import make_cameras_from_configs

from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.robots.robot import Robot
from .can_monitor import CanFeedbackMonitor, is_fault, status_name
from .config_openarm_follower import OpenArmFollowerConfig

logger = logging.getLogger(__name__)


class OpenArmFollower(Robot):
    config_class = OpenArmFollowerConfig
    name = "openarm_follower"

    def __init__(self, config: OpenArmFollowerConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        
        self.right_arm = oa.OpenArm(self.config.right_port, self.config.enable_fd)
        self.left_arm  = oa.OpenArm(self.config.left_port,  self.config.enable_fd)
        
        self.right_refresh_thread = None
        self.left_refresh_thread = None
        
        # joint3 (index 2) raised from kp=20/kd=2 -> kp=150/kd=8 on 2026-07-03: bench-tested
        # via safe_probe.py at kp=40/60/100/150 (all clean, no jitter/overshoot observed),
        # steady-state hold error under this joint's own gravity/friction load improved from
        # ~70% of commanded delta reached at kp=40 to ~92% at kp=150, matching the same
        # diminishing-returns curve seen on joint1's earlier gain sweep. Applies to both
        # arms (LJ3 and RJ3 share this index).
        # 50 1.0, 45 1.0
        # self.KPs = [ 200.0, 200.0, 200.0, 40.0, 40.0, 40.0, 40.0,  3.0 ]
        # self.KDs = [   3.0,   3.0,   3.0,  1.5,  1.5,  1.5,  1.5,   0.3]
        self.KPs = [ 60.0,  50.0,  50.0,  60.0,  20.0, 40.0, 20.0,  3.0 ] #RJ6=30.0
        self.KDs = [   2.0,   2.0,  2.0,  2.5,  1.0,  1.2,  1.0,  0.3 ]       
        self.model = pin.buildModelFromUrdf(self.config.model_path)
        self.data = self.model.createData()
        
        self.goal_pos = None
        
        self._is_connected = False
        
        self._shared_array = Array('d', 16)  # Shared array for 16 doubles

        # Passive per-channel state-frame counters, opened in configure(). See can_monitor.py:
        # without them, "the feedback stopped refreshing" and "the joint did not move" are the
        # same observation, and every caller here has to guess which one it is looking at.
        self._monitors: list[CanFeedbackMonitor] = []

    @property
    def _motors_ft(self) -> dict[str, type]:
        obs_dict = {}
        
        for i in range(8):
            obs_dict[f'RJ{i+1}.pos'] = float
            obs_dict[f'LJ{i+1}.pos'] = float

        return obs_dict

    @property        # self.KPs = [ 200.0, 100.0, 150.0, 120.0, 20.0, 45.0, 20.0,  20.0 ]
        # self.KDs = [   5.0,   5.0,   8.0,  6.0,  1.0,  2.0,  1.0,   1.0 ]
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = False) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        if calibrate and not self.is_calibrated:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        self._is_connected = True
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        raise NotImplementedError('is_calibrated property not implemented in OpenArmFollower')

    def calibrate(self) -> None:
        raise NotImplementedError('calibrate() method not implemented in OpenArmFollower')

    # enable_all()/disable_all() are fire-and-forget: no ack, no return value, no readback. A
    # dropped frame, or a motor that is not on the bus at all, is indistinguishable from success
    # at the call site -- which is how a session starts with part of the robot unarmed and only
    # finds out when a joint fails to track. Both paths below therefore SEND MORE THAN ONCE, and
    # the enable path VERIFIES (see _silent_channels) before declaring the robot connected.
    _ENABLE_ATTEMPTS = 5
    _ENABLE_SETTLE_S = 0.15    # let the enable land and its state frame come back before checking
    _DISABLE_ATTEMPTS = 3
    _DISABLE_GAP_S = 0.05
    _PROBE_TIMEOUT_US = 50_000  # once per connect/teardown, so a generous timeout costs nothing
    _PROBE_PASSES = 3           # refresh/drain cycles per enable attempt; an intermittent channel
                                # (see _silent_channels) needs more than one chance to answer, and
                                # each pass is a few milliseconds

    def _motor_channels(self) -> list[tuple[str, "oa.Motor"]]:
        """[("RJ1", motor), ...] for all sixteen channels, keyed like get_observation()."""
        channels = []
        for prefix, arm in (("R", self.right_arm), ("L", self.left_arm)):
            for i, motor in enumerate(arm.get_arm().get_motors(), start=1):
                channels.append((f"{prefix}J{i}", motor))
            channels.append((f"{prefix}J8", arm.get_gripper().get_motor()))
        return channels

    def _silent_channels(self) -> list[str]:
        """Channels that have never answered a CAN frame since this process started.

        The test is the reported temperature, not the position and not is_enabled():

          - A Motor object starts at t_mos = t_rotor = 0 and only leaves 0 by decoding a real
            state frame, whose last two bytes are the MOSFET and rotor temperatures in whole
            degrees C (verified against candump on this rig: 0x1B 0x19 = 27C, 25C at room
            temperature). No real motor reads 0 C in both, so both-zero means nothing has ever
            come back from that channel.
          - Position cannot answer this question. A motor that drops off the bus keeps reporting
            its last position forever, which is plausible, unchanging, and identical to a joint
            that is simply holding still -- the exact ambiguity that let the right arm run for
            days with J5-J8 unreachable (2026-08-26).
          - is_enabled() cannot answer it either: it returned False for all sixteen channels on a
            run where five joints tracked their commands to within 0.03 rad. See
            reset_to_rest_pose.py's module docstring.

        This says the channel is REACHABLE, which is strictly weaker than "this motor will now
        execute commands" -- a motor can answer and still refuse to act. It is, however, the
        strongest claim obtainable without moving the robot, and it catches the failure that
        actually happens at startup.

        CUMULATIVE, not per-probe. A per-probe test asks "did this channel answer THIS request",
        which is a question about one round trip and not about reachability, and it condemns a
        channel for a single dropped frame. It did exactly that on 2026-08-26: a run refused to
        connect with "RJ1 never answered a CAN frame after 5 enable attempts" while RJ1 was
        answering a raw probe on the same bus 79 times out of 80 -- the drops were coming from the
        request pattern (see _REQUEST_STAGGER_S), not from the motor, and the gate was about to
        send someone looking for a broken shoulder harness. A channel therefore clears the moment
        it has answered ANY probe since the taps opened, which is the top of configure(), so the
        window is this connect and nothing earlier. What stays out is the channel that never
        answers at all -- the case that actually strands a session.
        """
        for _ in range(self._PROBE_PASSES):
            self._read_motor_positions_once()   # refresh + a drain that waits for the answers
        totals: dict[str, int] = {}
        for mon in self._monitors:
            if mon.available:
                totals.update(mon.total)
        if totals:
            return [key for key, n in sorted(totals.items()) if n == 0]
        return [
            key for key, motor in self._motor_channels()
            if motor.get_state_tmos() == 0 and motor.get_state_trotor() == 0
        ]

    def _reply_rates(self) -> dict[str, int]:
        """Frames each channel has sent since the taps opened. Empty when no tap is open."""
        totals: dict[str, int] = {}
        for mon in self._monitors:
            if mon.available:
                totals.update(mon.total)
        return totals

    def _disable_arms(self) -> list[str]:
        """Best-effort de-energise of both arms. Returns the sides that never accepted a disable.

        Each arm is independent: one arm's exception must not strand the other. These two calls
        used to run bare and in sequence, so anything the right arm raised propagated out of
        disconnect() and left the LEFT arm fully enabled, holding its last MIT command, behind a
        traceback that pointed at teardown rather than at an armed robot.

        The frame is sent _DISABLE_ATTEMPTS times per arm even when nothing raises, because a
        silently dropped frame and a delivered one look identical from here.
        """
        failed = []
        for side, arm in (("right", self.right_arm), ("left", self.left_arm)):
            sent, last_error = 0, None
            for _ in range(self._DISABLE_ATTEMPTS):
                try:
                    arm.disable_all()
                    sent += 1
                except Exception as e:  # noqa: BLE001 -- one arm's failure must not strand the other
                    last_error = e
                time.sleep(self._DISABLE_GAP_S)
            if sent == 0:
                failed.append(side)
                logger.error(
                    f"{self}: disable_all() FAILED on the {side} arm after"
                    f" {self._DISABLE_ATTEMPTS} attempts"
                    f" ({type(last_error).__name__}: {last_error}) -- it may still be energised."
                )
        return failed

    def _channels_still_armed(self) -> dict[str, str]:
        """After a disable, which channels still say they are ENABLED (or cannot be asked).

        disable_all() is fire-and-forget and the arm's own answer is the only way to know it
        landed. That answer exists -- it is the DM status nibble the passive tap decodes -- and
        until this checked it, nothing did. Observed 2026-08-26: a run ended with RJ1-RJ4
        reporting "disabled" and RJ5-RJ8 reporting "enabled", because the wrist group had dropped
        off the bus mid-run and never received the disable frame. It sat energised, holding its
        last MIT command, while the script printed "Robot disconnected."

        Returns {channel: reason}, empty when everything is verifiably down.
        """
        if not any(mon.available for mon in self._monitors):
            return {}
        try:
            self._mark_read_cycle()
            self._request_positions()
            self._drain_until_answered(self.config.recv_first_timeout_us,
                                       self.config.recv_mop_timeout_us
                                       or self.config.recv_first_timeout_us)
            status = self.get_feedback_status()
        except Exception as e:  # noqa: BLE001 -- teardown must not be blocked by a diagnostic
            logger.warning(f"{self}: could not verify motors de-energised ({type(e).__name__}: {e})")
            return {}
        live = {}
        for key, code in sorted(status.items()):
            if code == 0x1:
                live[key] = "still reports ENABLED"
            elif code is None:
                live[key] = "never answered -- cannot be confirmed disabled"
        return live

    def _report_still_armed(self, live: dict[str, str]) -> None:
        """Tell a human, in the terminal, which motors did not confirm they are down."""
        if not live:
            return
        detail = "\n".join(f"           {key}: {why}" for key, why in live.items())
        print(
            f"\n[DANGER] {self}: {len(live)} channel(s) did NOT confirm they de-energised:\n"
            f"{detail}\n"
            "         A channel that missed the disable is STILL ENERGISED and holding its last\n"
            "         commanded position -- it will resist being moved and can snap back if it\n"
            "         has drifted. This happens when a motor drops off the bus mid-run: it keeps\n"
            "         its last command and never hears the disable.\n"
            "         Run:  python emergency_disable.py --side <side>\n"
            "         If those channels are still off the bus, software cannot reach them at all"
            " -- CUT POWER to that arm before touching it.",
            flush=True,
        )

    def _report_disable_failures(self, failed: list[str]) -> None:
        """Tell a human, in the terminal, that an arm may still be live.

        Not a log line: this is the one message that must survive whatever else teardown is doing,
        so it is printed and flushed before anything that could raise on the way out.
        """
        for side in failed:
            print(
                f"\n[DANGER] {self}: the {side} arm did NOT accept disable_all() -- assume it is"
                f" STILL ENERGISED and holding its last command.\n"
                f"         Run:  python emergency_disable.py --side {side}\n"
                f"         Then LOOK at that arm's status LEDs. If they do not go red/off,"
                f" CUT POWER -- do not retry in software.",
                flush=True,
            )

    # SocketCAN bring-up for this adapter, kept next to the check that tells you to run it. The
    # DM-Tech DM-USB2FDCAN is a gs_usb device with TWO channels on one USB interface, so can0 and
    # can1 come and go together, and it does NOT support restart-ms ("Device doesn't support
    # restart from Bus Off") -- an interface that goes down stays down until it is re-created by
    # hand. Nothing on this machine brings them up at boot.
    _CAN_BRINGUP_HINT = (
        "sudo ip link set {port} down; sleep 0.5; "
        "sudo ip link set {port} type can bitrate 1000000 sample-point 0.75 "
        "dbitrate 5000000 dsample-point 0.75 fd on; "
        "sudo ip link set {port} up; sleep 1"
    )

    @staticmethod
    def _can_link_problem(port: str) -> str | None:
        """Why `port` cannot carry traffic right now, or None if it looks usable.

        This exists because writing to a down CAN interface RAISES NOTHING. enable_all(),
        disable_all() and every MIT command return normally, the motors never hear a word, and the
        only visible symptom is that the whole bus went quiet -- which reads exactly like unplugged
        motors. Confirmed 2026-08-26: hours went into hunting a missing adapter and a broken wrist
        harness for what was an interface that had silently gone down. The link state is one
        sysfs read away, needs no privileges, and separates the two cases outright.

        Deliberately advisory: any surprise while reading sysfs returns None (usable) rather than
        blocking a connect. A diagnostic must not become a new way to fail.
        """
        base = f"/sys/class/net/{port}"
        try:
            if not os.path.isdir(base):
                return (f"no such network interface -- the USB-CAN adapter is not enumerated."
                        f" Check `lsusb | grep 1d50` and the adapter's USB cable")
            with open(f"{base}/flags") as f:
                flags = int(f.read().strip(), 16)
            if not flags & 0x1:  # IFF_UP
                return "interface is administratively DOWN"
            with open(f"{base}/operstate") as f:
                operstate = f.read().strip()
            try:
                with open(f"{base}/carrier") as f:
                    carrier = int(f.read().strip())
            except OSError:
                carrier = None   # sysfs returns EINVAL for carrier while the link is down
            # carrier gets the benefit of the doubt over operstate: gs_usb reports operstate "up"
            # here, but a driver that leaves it "unknown" while carrying traffic must not be
            # reported as broken.
            if operstate != "up" and carrier != 1:
                return (f"interface is UP but operstate is '{operstate}' (no carrier) -- the CAN"
                        f" controller did not start, or went bus-off. Note gs_usb can take about a"
                        f" second to come up, so a check immediately after `ip link set up` can"
                        f" also land here")
        except OSError as e:
            logger.debug(f"could not read link state for {port}: {e} -- not blocking the connect")
            return None
        return None

    @staticmethod
    def _tx_packets(port: str) -> int | None:
        """Frames this interface has actually transmitted, or None if unreadable."""
        try:
            with open(f"/sys/class/net/{port}/statistics/tx_packets") as f:
                return int(f.read().strip())
        except OSError:
            return None

    def _stalled_tx_ports(self, before: dict[str, int | None]) -> list[str]:
        """Ports whose transmit counter has not moved since `before`, i.e. the frames we wrote
        never left the host.

        A gs_usb channel can wedge with the link still UP and the carrier still on: the qdisc
        backlog fills (txqueuelen is 10 on these), stops draining, and every further write fails
        with ENOBUFS while `ip link` keeps reporting ERROR-ACTIVE. Observed 2026-08-26 on can0 --
        11 packets stuck in the backlog, 32409 dropped, tx_packets frozen for tens of seconds with
        no process running, while can1 on the same adapter was clean.

        Without this check that state is indistinguishable from a dead robot: nothing is
        transmitted, so nothing answers, and configure() blames sixteen silent motors and sends
        someone to look at the arm. Only an interface re-create fixes it (see _CAN_BRINGUP_HINT);
        no amount of retrying in software will.
        """
        stalled = []
        for port, was in before.items():
            now = self._tx_packets(port)
            if was is not None and now is not None and now == was:
                stalled.append(port)
        return stalled

    def _check_can_links(self) -> None:
        """Refuse to connect over an interface that cannot carry traffic. See _can_link_problem."""
        problems = [(port, why) for port in (self.config.right_port, self.config.left_port)
                    if (why := self._can_link_problem(port)) is not None]
        if not problems:
            return
        detail = "\n".join(
            f"  {port}: {why}\n    fix: {self._CAN_BRINGUP_HINT.format(port=port)}"
            for port, why in problems
        )
        raise DeviceNotConnectedError(
            f"{self}: {len(problems)} CAN interface(s) cannot carry traffic. Nothing was sent to"
            f" the motors -- this is a host-side problem, not a robot one.\n{detail}"
        )

    def configure(self) -> None:
        """Register the motors, then enable until every channel is verifiably answering.

        The link check runs first so that the two failure modes stay separable: an interface that
        cannot carry traffic and a motor that is not on the bus produce the identical silence, and
        only the former is fixable from the host (see _can_link_problem).

        Raises rather than returning a half-armed robot. Enabling used to be one unchecked
        enable_all() per arm, so a channel that missed its frame -- or was not on the bus at all
        -- produced a robot that looked connected and then quietly failed to move that joint. The
        retry costs at most _ENABLE_ATTEMPTS x _ENABLE_SETTLE_S once per session.
        """
        self._check_can_links()
        tx_before = {port: self._tx_packets(port)
                     for port in (self.config.right_port, self.config.left_port)}

        for arm in (self.right_arm, self.left_arm):
            arm.init_arm_motors(self.config.motor_types, self.config.send_ids,
                                self.config.recv_ids, self.config.motor_modes)
            arm.init_gripper_motor(self.config.gripper_motor_type, self.config.gripper_motor_send_id,
                                   self.config.gripper_motor_recv_id, self.config.gripper_motor_mode)
            arm.set_callback_mode_all(oa.CallbackMode.STATE)

        self._open_monitors()

        silent = []
        for attempt in range(1, self._ENABLE_ATTEMPTS + 1):
            self.right_arm.enable_all()
            self.left_arm.enable_all()
            time.sleep(self._ENABLE_SETTLE_S)

            silent = self._silent_channels()
            if not silent:
                logger.info(f"{self}: all 16 channels answering after enable attempt {attempt}.")
                self._warn_intermittent_channels(attempt)
                return
            logger.warning(
                f"{self}: no CAN response from {', '.join(silent)} after enable attempt"
                f" {attempt}/{self._ENABLE_ATTEMPTS} -- re-sending enable."
            )

        # Never leave a partially-enabled robot behind an exception: whatever DID come up is
        # armed right now, and the caller is about to stop running.
        self._report_disable_failures(self._disable_arms())

        # Before blaming the robot, check the frames actually left the host. A wedged transmit
        # queue produces exactly this symptom and is fixed on this side, not on the arm.
        stalled = self._stalled_tx_ports(tx_before)
        if stalled:
            detail = "\n".join(f"  {port}\n    fix: {self._CAN_BRINGUP_HINT.format(port=port)}"
                               for port in stalled)
            raise DeviceNotConnectedError(
                f"{self}: {', '.join(stalled)} accepted no transmissions during bring-up -- the"
                " interface's tx_packets counter did not move while enable frames were being"
                " written, so nothing reached the bus and the silent channels below say nothing"
                " about the motors. This is a wedged interface (the queue fills, stops draining,"
                " and writes fail with ENOBUFS while the link still reports UP with carrier)."
                " Re-create it; retrying in software cannot clear it."
                f"\n{detail}\n  Then re-run. Check `tc -s qdisc show dev <port>`: a backlog that"
                " does not drain with nothing running is this fault."
            )

        raise DeviceNotConnectedError(
            f"{self}: {len(silent)} channel(s) never answered a CAN frame after"
            f" {self._ENABLE_ATTEMPTS} enable attempts: {', '.join(silent)}."
            " Those motors are not reachable on the bus -- refusing to run a partially armed"
            " robot. Both CAN interfaces were UP with carrier when this started, so this is the"
            " robot side: check that arm's power, and the CAN wiring for that section of the chain"
            " (a contiguous tail of the chain going silent points at the link upstream of the"
            " first one listed). Note a motor that never got a disable is still energised and"
            " holding its last command."
        )

    _DRAIN_SPIN_S = 50e-6      # keep the wait loop off a hot spin without coarsening it

    # Both CAN channels hang off ONE USB adapter (`parentdev` is the same for can0 and can1 in
    # `ip -d link`), and it cannot transmit one channel's burst while receiving the other's
    # answers without losing some. Issuing right.refresh_all() and left.refresh_all() back to back
    # -- 16 frames handed over with nothing read in between -- reliably destroyed the replies that
    # come back FIRST, i.e. the lowest CAN ids. Measured 2026-08-26 over 40 reads:
    #
    #     back to back        RJ1 fresh on   2% of reads, all 16 fresh on  2%
    #     staggered by 2 ms   RJ1 fresh on  92% of reads, all 16 fresh on 82%
    #
    # and the same burst sent to one channel alone answered 8/8 every single time. This is not a
    # marginal motor on the right shoulder, which is what it looks like from the outside and what
    # the connect gate was about to condemn RJ1 as.
    _REQUEST_STAGGER_S = 0.002

    def _request_positions(self, arms=None) -> None:
        """Ask the given arms (default: both) for a state frame, one channel at a time.

        The gap is the point -- see _REQUEST_STAGGER_S. A single arm is sent immediately.
        """
        arms = list(arms if arms is not None else (self.right_arm, self.left_arm))
        for i, arm in enumerate(arms):
            if i:
                time.sleep(self._REQUEST_STAGGER_S)
            arm.refresh_all()

    def _arms_for(self, channels) -> list:
        """The arm objects owning `channels` (["RJ5", "LJ1", ...]), right before left."""
        sides = {key[0] for key in channels}
        return [arm for prefix, arm in (("R", self.right_arm), ("L", self.left_arm))
                if prefix in sides]

    def _drain_until_answered(self, first_us: int, mop_us: int) -> list[str]:
        """Drain both sockets until every channel has answered the outstanding refresh, or the
        configured deadline passes. Returns the channels still missing.

        This replaces a fixed number of recv_all() calls, which was a race the low-priority half
        of the bus kept losing. recv_all()'s timeout argument does not make it wait on this build
        (see OpenArmFollowerConfig.recv_deadline_us for the measurement), so `recv_rounds` rounds
        of it amounted to 2-3 ms of polling rather than 2-3 ms of patience, and the motors answer
        in ascending CAN-id order over roughly a millisecond on an idle bus -- longer once a
        control loop's own command replies are sharing the wire. Whether the window closed before
        J5-J8 got their turn came down to USB scheduling, which is why the same joints read fresh
        on one run and sat frozen for seconds on the next.

        The wait is evidence-driven: the passive taps say which channels have actually answered
        this request, so the loop stops the instant the set is complete and a healthy read costs
        no more than the old spin did. With no tap open there is nothing to be driven by, so it
        falls back to the old round count plus a floor on elapsed time -- still a fix for the
        race, just without the early exit.
        """
        deadline = time.perf_counter() + self.config.recv_deadline_us / 1e6
        retry_after = self.config.recv_retry_us / 1e6
        next_retry = time.perf_counter() + retry_after if retry_after > 0 else float("inf")
        rounds = 0
        while True:
            timeout = first_us if rounds == 0 else mop_us
            self.right_arm.recv_all(timeout)
            self.left_arm.recv_all(timeout)
            rounds += 1
            self._poll_monitors()
            missing = self._pending_channels()
            if missing is not None and not missing:
                return []
            now = time.perf_counter()
            if now >= deadline:
                break
            if missing and now >= next_retry:
                # A request or its answer was dropped, so waiting longer for it is pointless --
                # ask again, and only the arm that owes us something. Cheap, because it only
                # happens when a channel is actually missing.
                self._request_positions(self._arms_for(missing))
                next_retry = time.perf_counter() + retry_after
                continue
            if missing is None and rounds >= max(1, self.config.recv_rounds):
                # No tap to tell us when we are done: spend the rest of the deadline as an actual
                # wait, then take one more pass at whatever landed during it.
                time.sleep(max(0.0, deadline - time.perf_counter()))
                self.right_arm.recv_all(mop_us)
                self.left_arm.recv_all(mop_us)
                return []
            time.sleep(self._DRAIN_SPIN_S)
        missing = self._pending_channels() or []
        if missing:
            logger.debug(f"{self}: no state frame from {', '.join(sorted(missing))} within"
                         f" {self.config.recv_deadline_us}us -- those positions are retained state.")
        return missing

    def _mark_read_cycle(self) -> None:
        for mon in self._monitors:
            mon.mark_cycle()

    def _pending_channels(self) -> list[str] | None:
        """Channels yet to answer since _mark_read_cycle(), or None when no tap is open."""
        if not any(mon.available for mon in self._monitors):
            return None
        missing = []
        for mon in self._monitors:
            missing.extend(mon.pending())
        return missing

    def _open_monitors(self) -> None:
        """Open one passive state-frame tap per CAN interface (see can_monitor.py).

        Fail-soft by construction: a tap that cannot be opened reports itself unavailable and
        every consumer degrades to "no freshness data", never to a failed connect. The taps
        observe the same frames openarm_can consumes rather than competing for them, so this
        cannot make the control path read less.
        """
        self._close_monitors()
        right = {0x10 + i: f"RJ{i}" for i in range(1, 8)}
        right[self.config.gripper_motor_recv_id] = "RJ8"
        left = {0x10 + i: f"LJ{i}" for i in range(1, 8)}
        left[self.config.gripper_motor_recv_id] = "LJ8"
        self._monitors = [
            CanFeedbackMonitor(self.config.right_port, right, fd=self.config.enable_fd),
            CanFeedbackMonitor(self.config.left_port, left, fd=self.config.enable_fd),
        ]

    def _close_monitors(self) -> None:
        for mon in self._monitors:
            mon.close()
        self._monitors = []

    def _poll_monitors(self) -> None:
        for mon in self._monitors:
            mon.poll()

    def take_feedback_counts(self) -> dict[str, int]:
        """State frames received per channel since the previous call, keyed "RJ1".."LJ8".

        This is the answer to "was that reading fresh?", and it is a count of frames on the wire
        rather than a guess made from the decoded value. Zero means the motor sent nothing in
        that window, so get_position() returned retained state; non-zero means the reading is
        current and an unchanged position is the joint genuinely not having moved.

        Empty dict when no tap could be opened -- callers must treat that as "unknown", not as
        "no frames".
        """
        self._poll_monitors()
        counts: dict[str, int] = {}
        for mon in self._monitors:
            counts.update(mon.take_counts())
        return counts

    def get_bus_evidence(self) -> dict:
        """What the WIRE says, as opposed to what the decoded values suggest.

        Two independent readings, both of which openarm_can discards:

          - "esi": per channel, how many of its frames were sent with the CAN-FD Error State
            Indicator set, i.e. by a transmitter that had already gone error-passive. A node on
            its way to bus-off sets ESI before it goes silent, so this separates "that motor was
            fighting the wire and lost" from "that motor was fine and then stopped".
          - "errors": error frames raised by the CAN controller itself, by class. Note this is
            strictly better evidence than `ip -s link`'s bus-errors column, which only counts
            once berr-reporting is enabled on the link and therefore reads zero on a bus that is
            visibly failing.

        Empty when no tap could be opened.
        """
        esi: dict[str, int] = {}
        errors: dict[int, int] = {}
        for mon in self._monitors:
            if not mon.available:
                continue
            esi.update(mon.esi)
            for cls, n in mon.errors.items():
                errors[cls] = errors.get(cls, 0) + n
        return {"esi": esi, "errors": errors}

    def get_feedback_status(self) -> dict[str, int | None]:
        """Each channel's most recent DM status nibble (see can_monitor.DM_STATUS), or None.

        This is the real "is this motor armed / has it faulted" signal. Motor.is_enabled() is not:
        openarm_can defines Motor::set_enabled() and never calls it from anywhere, so it reports
        False for all sixteen channels forever regardless of what the motors are doing.
        """
        status: dict[str, int | None] = {}
        for mon in self._monitors:
            status.update(mon.status)
        return status

    # A channel below this share of the best channel's frame count is answering intermittently
    # rather than reliably. Not fatal -- the read re-asks and waits (_drain_until_answered), so
    # positions still come back fresh -- but it costs time on every read it misses. Set well below
    # 1 on purpose: probe counts differ by a frame or two between channels for entirely benign
    # reasons, and a gate that cries wolf about those is worse than no gate.
    _INTERMITTENT_FRACTION = 0.5

    def _warn_intermittent_channels(self, attempt: int) -> None:
        """Print the channels that answered, but not every time, during the connect probes.

        Silence at connect is fatal and already handled. This is the case in between, which used
        to be invisible: a channel that answers most of the time passes the gate, then quietly
        makes every read pay the recv deadline waiting for the frames it drops.
        """
        totals = self._reply_rates()
        if not totals:
            return
        best = max(totals.values())
        if best == 0:
            return
        weak = {k: n for k, n in sorted(totals.items())
                if n < best * self._INTERMITTENT_FRACTION}
        if not weak:
            return
        detail = ", ".join(f"{k} {n}/{best}" for k, n in weak.items())
        print(
            f"\n[WARN] {self}: intermittent CAN feedback from {detail} frames over"
            f" {attempt * self._PROBE_PASSES} probe(s). Every other channel answered {best}."
            "\n       Not fatal: the read re-asks and waits (recv_deadline_us) so positions stay"
            " fresh, but each miss costs time. If this is ONE channel while the rest are clean,"
            " suspect that joint's CAN connector; if it is the first motor of an arm, suspect the"
            " request pattern instead (see _REQUEST_STAGGER_S) before suspecting the hardware.",
            flush=True,
        )

    def setup_motors(self) -> None:
        raise NotImplementedError('setup_motors() method not implemented in OpenArmFollower')

    # A human-scale OpenArm joint should never legitimately approach the motor's full
    # +/-12.5 rad encoder range. An intermittent read glitch (consistent with a stale/
    # never-updated Motor object for one CAN response) has been observed to land near
    # this extreme, on a different joint almost every call, even with a generous
    # recv_all() timeout -- diagnosed 2026-07-01. Reject such readings outright rather
    # than trust them, and retry until two consecutive PLAUSIBLE reads agree.
    _PLAUSIBLE_ARM_JOINT_RANGE = 3.2  # rad

    def _read_motor_positions_once(self) -> dict:
        # recv_all()'s timeout is MICROSECONDS (see OpenArm::recv_all in openarm_can), not
        # milliseconds -- the 500us default was too short for a reliable USB-CAN round trip and
        # was observed returning stale/never-updated positions, which is why this was hard-coded
        # to 8 rounds of 50_000us. Measurement (profile_can_read.py, 2026-08-21, 30 samples per
        # setting) showed what that actually costs and what it buys:
        #
        #   * That profiling read recv_all() as blocking its FULL timeout on an empty socket, and
        #     it does not: re-measured 2026-08-26 against openarm_can 1.2.8, recv_all() returns in
        #     0.04-0.16 ms on an empty socket whether it is passed 500 us or 200 000 us. The 801ms
        #     figure was the read being slow for some other reason, not evidence about the
        #     timeout, and the conclusion drawn from it -- that rounds x timeout is a WAIT you can
        #     tune -- was wrong. That is why the drain below no longer counts rounds: see
        #     _drain_until_answered() and OpenArmFollowerConfig.recv_deadline_us.
        #   * refresh_all() is called ONCE but recv_all() N times, so rounds 2..N have no refresh
        #     response of their own outstanding. In a control loop they feed instead on the
        #     feedback frames send_action() leaves unread (each MIT command makes every motor
        #     reply; 10 interpolation substeps is ~80 frames per arm per cycle).
        #   * That backlog is why the same read costs 8-59ms inside deploy_smolvla_pickup_
        #     jointspace.py rather than 801ms -- and why it DEGRADES: as the backlog depletes,
        #     one more round per cycle hits the timeout, and the observed cycle time climbed in
        #     exact 100ms steps (110ms -> 210ms -> 310ms, i.e. 9.1 Hz -> 3.2 Hz) heading for the
        #     no-backlog 801ms.
        #   * A long timeout does NOT buy freshness, but for a blunter reason than that note gave:
        #     the timeout buys nothing at all. What freshness needs is a wait, and until
        #     _drain_until_answered() there was none -- the whole drain covered 2-3 ms of spinning
        #     while the motors answer 0.26 ms (J1) to 1.02 ms (J8) after the request, in ascending
        #     CAN-id order. J5-J8 lost that race whenever USB scheduling ran a little late, which
        #     is how their feedback froze for seconds while J1-J4 stayed fresh on the same bus.
        #
        # Hence the split: round 1 keeps a real timeout because it is the only one with a
        # refresh_all() response outstanding, while the mop-up rounds -- whose job is to drain
        # send_action's backlog so it can neither deplete nor accumulate -- are bounded so that a
        # dry buffer costs microseconds. Draining to empty every cycle is what stops the runaway.
        # Defaults in OpenArmFollowerConfig reproduce the old behaviour exactly; callers opt in.
        cfg = self.config
        mop_us = cfg.recv_mop_timeout_us if cfg.recv_mop_timeout_us is not None else cfg.recv_first_timeout_us

        self._mark_read_cycle()
        self._request_positions()
        self._drain_until_answered(cfg.recv_first_timeout_us, mop_us)

        obs_dict = {}
        for i, motor in enumerate(self.right_arm.get_arm().get_motors()):
            obs_dict[f'RJ{i+1}.pos'] = motor.get_position()
        obs_dict['RJ8.pos'] = self.right_arm.get_gripper().get_motor().get_position()

        for i, motor in enumerate(self.left_arm.get_arm().get_motors()):
            obs_dict[f'LJ{i+1}.pos'] = motor.get_position()
        obs_dict['LJ8.pos'] = self.left_arm.get_gripper().get_motor().get_position()
        return obs_dict

    def get_motor_health(self) -> dict:
        """Per-channel motor state BEYOND position: enabled flag, MOSFET/rotor temperatures, torque
        and velocity. Keyed like the observation ("RJ1".."RJ8", "LJ1".."LJ8").

        get_observation() reads get_position() and nothing else, which makes a motor that has
        dropped off the bus or tripped its own protection completely invisible: the Motor object
        retains its last position, so the read stays plausible and unchanging, the control loop
        keeps commanding a channel that is no longer listening, and the joint physically goes
        wherever gravity leaves it while the logs insist it is holding station. is_enabled() is the
        direct question, and the temperatures say whether a thermal cutout is why.

        Reads retained Motor state only -- no refresh_all()/recv_all() of its own -- so it is
        cheap, and it reports whatever the most recent successful read populated. Call it right
        after a real read (or after a run) rather than in isolation.
        """
        status = self.get_feedback_status()
        frames = {}
        for mon in self._monitors:
            frames.update(mon.total)
        health = {}
        for prefix, arm in (("R", self.right_arm), ("L", self.left_arm)):
            motors = list(arm.get_arm().get_motors()) + [arm.get_gripper().get_motor()]
            for i, motor in enumerate(motors, start=1):
                key = f"{prefix}J{i}"
                code = status.get(key)
                health[key] = {
                    "enabled": motor.is_enabled(),   # always False; kept only so nothing that
                                                     # reads this key breaks. Use "status".
                    "status": code,
                    "status_name": status_name(code),
                    "faulted": is_fault(code),
                    "frames": frames.get(key),
                    "t_mos": motor.get_state_tmos(),
                    "t_rotor": motor.get_state_trotor(),
                    "torque": motor.get_torque(),
                    "velocity": motor.get_velocity(),
                    "position": motor.get_position(),
                }
        return health

    @classmethod
    def _find_implausible_key(cls, pos: dict) -> str | None:
        for k, v in pos.items():
            if k.endswith('8.pos'):
                continue  # gripper can legitimately sit near an encoder extreme
            if abs(v) > cls._PLAUSIBLE_ARM_JOINT_RANGE:
                return k
        return None

    def _read_motor_positions_stable(self, max_attempts: int = 8) -> dict:
        """Retry until a single PLAUSIBLE read is obtained.

        This does NOT require two consecutive reads to numerically agree -- unlike a
        stationary bring-up check (see safe_probe.py's read_positions_stable, which is
        only ever used at rest), get_observation() is routinely called while the arm is
        actively moving (teleop, replay), where real motion between two reads a few
        milliseconds apart is expected and NOT a sign of a bad read. The one robust,
        motion-independent signal we have is the plausibility bound: a human-scale arm
        joint should never legitimately read near the motor's full +/-12.5 rad encoder
        range, which is exactly the signature of the intermittent glitch diagnosed
        2026-07-01/02 (consistent with a stale/never-updated Motor object for one CAN
        response).
        """
        for attempt in range(1, max_attempts + 1):
            cur = self._read_motor_positions_once()
            bad_key = self._find_implausible_key(cur)
            if bad_key is None:
                return cur
            logger.warning(f"{self} read {bad_key}={cur[bad_key]:+.4f} rad, implausible for an arm joint"
                            f" (> {self._PLAUSIBLE_ARM_JOINT_RANGE} rad), retry {attempt}/{max_attempts}")
        raise RuntimeError(
            f"{self}: no plausible position read after {max_attempts} attempts. Refusing to report"
            " untrustworthy positions -- this points to a real communication reliability issue."
        )

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()

        obs_dict = self._read_motor_positions_stable()

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: RobotAction, target_vel: dict[str, float]) -> RobotAction:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action (RobotAction): The goal positions for the motors.
            target_vel (dict[str, float]): The target velocities for each joint.

        Returns:
            RobotAction: The action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        vel = target_vel or {}
        # q / tau index layout, from pinocchio's depth-first walk of the URDF tree (root ->
        # left arm -> left hand -> both left finger prismatics -> right arm -> right fingers):
        #    0..6  LJ1..LJ7      7,8  left finger_joint1/2   (LJ8 drives one, the mimic is 0.0)
        #    9..15 RJ1..RJ7    16,17  right finger_joint1/2  (likewise for RJ8)
        # The right-arm block was indexed one slot high (tau[10..17]) from 32e5cd8 until
        # 2026-08-13: RJ1 was fed RJ2's gravity torque, and RJ7/RJ8 got a finger's ~0 N
        # prismatic force, i.e. no feedforward at all. It went unnoticed because every session
        # since was left-arm only; it showed up as a constant per-joint offset (err ~= dtau/kp,
        # 0.05-0.07 rad at these gains) on the right arm alone in the sim-vs-real replay plots.
        q = np.array([
            action['LJ1.pos'], action['LJ2.pos'], action['LJ3.pos'], action['LJ4.pos'],
            action['LJ5.pos'], action['LJ6.pos'], action['LJ7.pos'], action['LJ8.pos'], 0.0,
            action['RJ1.pos'], action['RJ2.pos'], action['RJ3.pos'], action['RJ4.pos'],
            action['RJ5.pos'], action['RJ6.pos'], action['RJ7.pos'], action['RJ8.pos'], 0.0,
        ], np.float32)
        # Slots 7/8 and 16/17 are the finger joints, and they are PRISMATIC with a 0.000-0.044 m
        # travel -- metres, not radians. What lands in them here is a raw gripper MOTOR angle,
        # whose calibrated range on this robot spans about +-1.3 rad, so pinocchio was being told
        # the finger sits up to 1.3 m out of a 44 mm slot and returned the gravity of that
        # imaginary lever: measured 2026-08-25, feeding -1.2980 instead of a legal 0.044 moves the
        # joint5 feedforward from -0.049 to -0.524 N-m, i.e. ~0.024 rad of steady-state error at
        # kp=20 on LJ5/RJ5. It went unnoticed while calibration.json happened to put the gripper's
        # open position near raw 0 (inside the legal range by luck); re-zeroing both grippers at
        # the closed stop moved open to +-1.2 and made every commanded-open pose wrong.
        #
        # Clamping to the model's own limits rather than converting through calibration keeps this
        # module free of calibration knowledge, and a gripper's finger extension is worth so little
        # arm gravity torque (0.0648 -> 0.0492 N-m on joint5 across its whole legal travel) that
        # the residual from clamping instead of mapping is far below what the gains resolve.
        q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)
        tau: np.ndarray = pin.computeGeneralizedGravity(self.model, self.data, q)
        
        # self.right_arm.get_arm().posvel_control_all([
        #     oa.PosVelParam(q=action['RJ1.pos'], dq=20.0),
        #     oa.PosVelParam(q=action['RJ2.pos'], dq=20.0),
        #     oa.PosVelParam(q=action['RJ3.pos'], dq=20.0),
        #     oa.PosVelParam(q=action['RJ4.pos'], dq=20.0),
        #     oa.PosVelParam(q=action['RJ5.pos'], dq=20.0),
        #     oa.PosVelParam(q=action['RJ6.pos'], dq=20.0),
        #     oa.PosVelParam(q=action['RJ7.pos'], dq=20.0)
        # ])
        # self.right_arm.get_gripper().posvel_control_all([
        #     oa.PosVelParam(q=action['RJ8.pos'] + 0.08, dq=20.0)
        # ])
        self.right_arm.get_arm().mit_control_all([
            oa.MITParam(q=action['RJ1.pos'], dq=vel.get('RJ1.vel', 0.0), tau=tau[9], kp=self.KPs[0], kd=self.KDs[0]),
            oa.MITParam(q=action['RJ2.pos'], dq=vel.get('RJ2.vel', 0.0), tau=tau[10], kp=self.KPs[1], kd=self.KDs[1]),
            oa.MITParam(q=action['RJ3.pos'], dq=vel.get('RJ3.vel', 0.0), tau=tau[11], kp=self.KPs[2], kd=self.KDs[2]),
            oa.MITParam(q=action['RJ4.pos'], dq=vel.get('RJ4.vel', 0.0), tau=tau[12], kp=self.KPs[3], kd=self.KDs[3]),
            oa.MITParam(q=action['RJ5.pos'], dq=vel.get('RJ5.vel', 0.0), tau=tau[13], kp=self.KPs[4], kd=self.KDs[4]),
            oa.MITParam(q=action['RJ6.pos'], dq=vel.get('RJ6.vel', 0.0), tau=tau[14], kp=self.KPs[5], kd=self.KDs[5]),
            oa.MITParam(q=action['RJ7.pos'], dq=vel.get('RJ7.vel', 0.0), tau=tau[15], kp=self.KPs[6], kd=self.KDs[6]),
        ])
        self.right_arm.get_gripper().mit_control_all([
            oa.MITParam(q=action['RJ8.pos'], dq=0.0, tau=tau[16], kp=self.KPs[7], kd=self.KDs[7])
        ])

        # Same shared-adapter problem the read path has (see _REQUEST_STAGGER_S): handing the
        # adapter the left arm's eight command frames while the right arm's eight answers are
        # still coming back costs those answers, and the ones lost are whichever motors reply
        # last. Nothing here reads them, but the next get_observation() would have, and a command
        # burst that is still draining out of the adapter is also what the following state request
        # has to queue behind. Cheap insurance: one gap per tick, no gap if only one arm is in
        # use.
        time.sleep(self._REQUEST_STAGGER_S)

        
        # self.left_arm.get_arm().posvel_control_all([
        #     oa.PosVelParam(q=action['LJ1.pos'], dq=20.0),
        #     oa.PosVelParam(q=action['LJ2.pos'], dq=20.0),
        #     oa.PosVelParam(q=action['LJ3.pos'], dq=20.0),
        #     oa.PosVelParam(q=action['LJ4.pos'], dq=20.0),
        #     oa.PosVelParam(q=action['LJ5.pos'], dq=20.0),
        #     oa.PosVelParam(q=action['LJ6.pos'], dq=20.0),
        #     oa.PosVelParam(q=action['LJ7.pos'], dq=20.0)
        # ])
        # self.left_arm.get_gripper().posvel_control_all([
        #     oa.PosVelParam(q=action['LJ8.pos'] + 0.08, dq=20.0)
        # ])
        self.left_arm.get_arm().mit_control_all([
            oa.MITParam(q=action['LJ1.pos'], dq=vel.get('LJ1.vel', 0.0), tau=tau[0], kp=self.KPs[0], kd=self.KDs[0]),
            oa.MITParam(q=action['LJ2.pos'], dq=vel.get('LJ2.vel', 0.0), tau=tau[1], kp=self.KPs[1], kd=self.KDs[1]),
            oa.MITParam(q=action['LJ3.pos'], dq=vel.get('LJ3.vel', 0.0), tau=tau[2], kp=self.KPs[2], kd=self.KDs[2]),
            oa.MITParam(q=action['LJ4.pos'], dq=vel.get('LJ4.vel', 0.0), tau=tau[3], kp=self.KPs[3], kd=self.KDs[3]),
            oa.MITParam(q=action['LJ5.pos'], dq=vel.get('LJ5.vel', 0.0), tau=tau[4], kp=self.KPs[4], kd=self.KDs[4]),
            oa.MITParam(q=action['LJ6.pos'], dq=vel.get('LJ6.vel', 0.0), tau=tau[5], kp=self.KPs[5], kd=self.KDs[5]),
            oa.MITParam(q=action['LJ7.pos'], dq=vel.get('LJ7.vel', 0.0), tau=tau[6], kp=self.KPs[6], kd=self.KDs[6]),
        ])
        self.left_arm.get_gripper().mit_control_all([
            oa.MITParam(q=action['LJ8.pos'], dq=0.0, tau=tau[7], kp=self.KPs[7], kd=self.KDs[7])
        ])
        
        return action

    
    def disconnect(self):
        """De-energise both arms (see _disable_arms), then release the cameras.

        A disable failure is reported and carried on from, never raised: what follows it is
        exactly what still has to happen -- the other arm, and the cameras. is_connected is
        cleared either way, so a retry is not blocked by the check below.

        Returning without an exception is NOT evidence that the motors de-energised. disable_all()
        is fire-and-forget, and there is no "is disabled" signal to read back: is_enabled()
        returned False for all sixteen channels on a run where five joints tracked their commands
        to within 0.03 rad (see reset_to_rest_pose.py's module docstring), and a motor that has
        dropped off the bus answers nothing at all -- confirmed 2026-08-26, when this arm's
        J5-J8 stayed powered and holding torque while disable frames went out on 0x05..0x08 and
        nothing ever came back on 0x15..0x18. The status LEDs are the only trustworthy
        confirmation, which is why the failure path instructs a human instead of asserting
        anything.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        failed = self._disable_arms()
        still_live = self._channels_still_armed()

        self._is_connected = False
        self._close_monitors()

        # Reported before the cameras are touched: a camera that throws on release must not be
        # able to swallow the one message that says a robot arm is still live.
        self._report_disable_failures(failed)
        self._report_still_armed(still_live)

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected."
                    + (f" ({', '.join(failed)} arm(s) failed to disable)" if failed else ""))