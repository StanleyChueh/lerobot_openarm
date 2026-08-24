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
        self.KPs = [ 50.0,  50.0,  50.0,  60.0,  20.0, 40.0, 20.0,  3.0 ] #RJ6=30.0
        self.KDs = [   2.0,   2.0,  2.0,  2.5,  1.0,  1.2,  1.0,  0.3 ]       
        self.model = pin.buildModelFromUrdf(self.config.model_path)
        self.data = self.model.createData()
        
        self.goal_pos = None
        
        self._is_connected = False
        
        self._shared_array = Array('d', 16)  # Shared array for 16 doubles

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

    def configure(self) -> None:
        self.right_arm.init_arm_motors(self.config.motor_types, self.config.send_ids, self.config.recv_ids, self.config.motor_modes)
        self.right_arm.init_gripper_motor(self.config.gripper_motor_type, self.config.gripper_motor_send_id, self.config.gripper_motor_recv_id, self.config.gripper_motor_mode)
        self.right_arm.set_callback_mode_all(oa.CallbackMode.STATE)
        self.right_arm.enable_all()
        
        self.left_arm.init_arm_motors(self.config.motor_types, self.config.send_ids, self.config.recv_ids, self.config.motor_modes)
        self.left_arm.init_gripper_motor(self.config.gripper_motor_type, self.config.gripper_motor_send_id, self.config.gripper_motor_recv_id, self.config.gripper_motor_mode)
        self.left_arm.set_callback_mode_all(oa.CallbackMode.STATE)
        self.left_arm.enable_all()

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
        #   * recv_all() blocks its FULL timeout whenever the socket has nothing buffered. Read
        #     cost came out as exactly rounds x 2 arms x timeout at every setting measured, so
        #     8 x 50_000us is 801ms per read -- on a 20 Hz control loop whose entire cycle budget
        #     is 50ms.
        #   * refresh_all() is called ONCE but recv_all() N times, so rounds 2..N have no refresh
        #     response of their own outstanding. In a control loop they feed instead on the
        #     feedback frames send_action() leaves unread (each MIT command makes every motor
        #     reply; 10 interpolation substeps is ~80 frames per arm per cycle).
        #   * That backlog is why the same read costs 8-59ms inside deploy_smolvla_pickup_
        #     jointspace.py rather than 801ms -- and why it DEGRADES: as the backlog depletes,
        #     one more round per cycle hits the timeout, and the observed cycle time climbed in
        #     exact 100ms steps (110ms -> 210ms -> 310ms, i.e. 9.1 Hz -> 3.2 Hz) heading for the
        #     no-backlog 801ms.
        #   * A long timeout does NOT buy freshness. rounds=1 at 50_000us blocked its full 100ms
        #     with nothing arriving, and still returned plausible positions -- i.e. retained Motor
        #     state. Whether a frame arrives is decided by whether one was generated, not by how
        #     long this waits. So a shorter timeout does not make a stale read more likely; it
        #     makes the same stale read 100x cheaper, and the plausibility retry in
        #     _read_motor_positions_stable() is unchanged either way.
        #
        # Hence the split: round 1 keeps a real timeout because it is the only one with a
        # refresh_all() response outstanding, while the mop-up rounds -- whose job is to drain
        # send_action's backlog so it can neither deplete nor accumulate -- are bounded so that a
        # dry buffer costs microseconds. Draining to empty every cycle is what stops the runaway.
        # Defaults in OpenArmFollowerConfig reproduce the old behaviour exactly; callers opt in.
        cfg = self.config
        mop_us = cfg.recv_mop_timeout_us if cfg.recv_mop_timeout_us is not None else cfg.recv_first_timeout_us

        self.right_arm.refresh_all()
        self.left_arm.refresh_all()
        self.right_arm.recv_all(cfg.recv_first_timeout_us)
        self.left_arm.recv_all(cfg.recv_first_timeout_us)
        for _ in range(max(0, cfg.recv_rounds - 1)):
            self.right_arm.recv_all(mop_us)
            self.left_arm.recv_all(mop_us)

        obs_dict = {}
        for i, motor in enumerate(self.right_arm.get_arm().get_motors()):
            obs_dict[f'RJ{i+1}.pos'] = motor.get_position()
        obs_dict['RJ8.pos'] = self.right_arm.get_gripper().get_motor().get_position()

        for i, motor in enumerate(self.left_arm.get_arm().get_motors()):
            obs_dict[f'LJ{i+1}.pos'] = motor.get_position()
        obs_dict['LJ8.pos'] = self.left_arm.get_gripper().get_motor().get_position()
        return obs_dict

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
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.right_arm.disable_all()
        self.left_arm.disable_all()
        self._is_connected = False
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")