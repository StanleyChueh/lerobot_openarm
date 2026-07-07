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
        self.KPs = [ 200.0, 100.0, 150.0, 120.0, 20.0, 45.0, 20.0,  20.0 ]
        self.KDs = [   5.0,   5.0,   8.0,  6.0,  1.0,  2.0,  1.0,   1.0 ]
        
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

    @property
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
        self.right_arm.refresh_all()
        self.left_arm.refresh_all()
        for _ in range(8):
            # recv_all()'s timeout is MICROSECONDS (see OpenArm::recv_all in openarm_can),
            # not milliseconds -- the 500us default is too short for a reliable USB-CAN
            # round trip and was observed returning stale/never-updated positions.
            self.right_arm.recv_all(50_000)
            self.left_arm.recv_all(50_000)

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

    def send_action(self, action: RobotAction) -> RobotAction:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action (RobotAction): The goal positions for the motors.

        Returns:
            RobotAction: The action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

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
            oa.MITParam(q=action['RJ1.pos'], dq=0.0, tau=tau[10], kp=self.KPs[0], kd=self.KDs[0]),
            oa.MITParam(q=action['RJ2.pos'], dq=0.0, tau=tau[11], kp=self.KPs[1], kd=self.KDs[1]),
            oa.MITParam(q=action['RJ3.pos'], dq=0.0, tau=tau[12], kp=self.KPs[2], kd=self.KDs[2]),
            oa.MITParam(q=action['RJ4.pos'], dq=0.0, tau=tau[13], kp=self.KPs[3], kd=self.KDs[3]),
            oa.MITParam(q=action['RJ5.pos'], dq=0.0, tau=tau[14], kp=self.KPs[4], kd=self.KDs[4]),
            oa.MITParam(q=action['RJ6.pos'], dq=0.0, tau=tau[15], kp=self.KPs[5], kd=self.KDs[5]),
            oa.MITParam(q=action['RJ7.pos'], dq=0.0, tau=tau[16], kp=self.KPs[5], kd=self.KDs[5]),
        ])
        self.right_arm.get_gripper().mit_control_all([
            oa.MITParam(q=action['RJ8.pos'], dq=0.0, tau=tau[17], kp=self.KPs[7], kd=self.KDs[7])
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
            oa.MITParam(q=action['LJ1.pos'], dq=0.0, tau=tau[0], kp=self.KPs[0], kd=self.KDs[0]),
            oa.MITParam(q=action['LJ2.pos'], dq=0.0, tau=tau[1], kp=self.KPs[1], kd=self.KDs[1]),
            oa.MITParam(q=action['LJ3.pos'], dq=0.0, tau=tau[2], kp=self.KPs[2], kd=self.KDs[2]),
            oa.MITParam(q=action['LJ4.pos'], dq=0.0, tau=tau[3], kp=self.KPs[3], kd=self.KDs[3]),
            oa.MITParam(q=action['LJ5.pos'], dq=0.0, tau=tau[4], kp=self.KPs[4], kd=self.KDs[4]),
            oa.MITParam(q=action['LJ6.pos'], dq=0.0, tau=tau[5], kp=self.KPs[5], kd=self.KDs[5]),
            oa.MITParam(q=action['LJ7.pos'], dq=0.0, tau=tau[6], kp=self.KPs[5], kd=self.KDs[5]),
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