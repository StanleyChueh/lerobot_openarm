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

import numpy as np
import pinocchio as pin
import logging
import time
import openarm_can as oa

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.teleoperators import Teleoperator
from .config_openarm_leader import OpenArmConfig


logger = logging.getLogger(__name__)


class OpenArmLeader(Teleoperator):
    config_class = OpenArmConfig
    name = "openarm_leader"

    def __init__(self, config: OpenArmConfig):
        super().__init__(config)
        self.config = config
        self.right_arm = oa.OpenArm(self.config.right_port, self.config.enable_fd)
        self.left_arm  = oa.OpenArm(self.config.left_port,  self.config.enable_fd)

        self._is_connected = False
        
        self.model = pin.buildModelFromUrdf(self.config.model_path)
        self.data = self.model.createData()
        # print("show model.nq: ", self.model.nq)

    @property
    def action_features(self) -> dict[str, type]:
        action = {}
        
        for i in range(8):
            action[f'RJ{i+1}.pos'] = float
            action[f'LJ{i+1}.pos'] = float
        
        return action

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = False) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        if calibrate and not self.is_calibrated:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()
        self._is_connected = True
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        raise NotImplementedError('is_calibrated property not implemented in OpenArmLeader')

    def calibrate(self) -> None:
        raise NotImplementedError('calibrate() method not implemented in OpenArmLeader')

    def configure(self) -> None:
        self.right_arm.init_arm_motors(self.config.motor_types, self.config.send_ids, self.config.recv_ids)
        self.right_arm.init_gripper_motor(self.config.gripper_motor_type, self.config.gripper_motor_send_id, self.config.gripper_motor_recv_id)
        self.right_arm.get_arm().set_control_mode_all(oa.ControlMode.MIT) # type: ignore
        self.right_arm.get_gripper().set_control_mode_all(oa.ControlMode.MIT) # type: ignore
        self.right_arm.set_callback_mode_all(oa.CallbackMode.STATE)
        self.right_arm.enable_all()
        
        self.left_arm.init_arm_motors(self.config.motor_types, self.config.send_ids, self.config.recv_ids)
        self.left_arm.init_gripper_motor(self.config.gripper_motor_type, self.config.gripper_motor_send_id, self.config.gripper_motor_recv_id)
        self.left_arm.get_arm().set_control_mode_all(oa.ControlMode.MIT) # type: ignore
        self.left_arm.get_gripper().set_control_mode_all(oa.ControlMode.MIT) # type: ignore
        self.left_arm.set_callback_mode_all(oa.CallbackMode.STATE)
        self.left_arm.enable_all()

    def setup_motors(self) -> None:
        raise NotImplementedError('setup_motors() method not implemented in OpenArmLeader')

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.time()
        
        self.right_arm.refresh_all()
        self.left_arm.refresh_all()
        for _ in range(8):
            self.right_arm.recv_all(250)
            self.left_arm.recv_all(250)

        action = {}
        for i, motor in enumerate(self.right_arm.get_arm().get_motors()):
            action[f'RJ{i+1}.pos'] = motor.get_position()
        action['RJ8.pos'] = self.right_arm.get_gripper().get_motor().get_position()
        
        for i, motor in enumerate(self.left_arm.get_arm().get_motors()):
            action[f'LJ{i+1}.pos'] = motor.get_position()
        action['LJ8.pos'] = self.left_arm.get_gripper().get_motor().get_position()
        
        
        q = np.array([
            action['LJ1.pos'], action['LJ2.pos'], action['LJ3.pos'], action['LJ4.pos'],
            action['LJ5.pos'], action['LJ6.pos'], action['LJ7.pos'], -action['LJ8.pos'] / 25.0, -action['LJ8.pos'] / 25.0,
            action['RJ1.pos'], action['RJ2.pos'], action['RJ3.pos'], action['RJ4.pos'],
            action['RJ5.pos'], action['RJ6.pos'], action['RJ7.pos'], -action['RJ8.pos'] / 25.0, -action['RJ8.pos'] / 25.0,
        ], np.float32)
        tau: np.ndarray = pin.computeGeneralizedGravity(self.model, self.data, q) * 0.87
        self.right_arm.get_arm().mit_control_all([
            oa.MITParam(kp=0.0, kd=0.0, q=0.0, dq=0.0, tau=tau[9]),
            oa.MITParam(kp=0.0, kd=0.0, q=0.0, dq=0.0, tau=tau[10]),
            oa.MITParam(kp=0.0, kd=0.0, q=0.0, dq=0.0, tau=tau[11]),
            oa.MITParam(kp=0.0, kd=0.0, q=0.0, dq=0.0, tau=tau[12]),
            oa.MITParam(kp=0.0, kd=0.0, q=0.0, dq=0.0, tau=tau[13]),
            oa.MITParam(kp=0.0, kd=0.0, q=0.0, dq=0.0, tau=tau[14]),
            oa.MITParam(kp=0.0, kd=0.0, q=0.0, dq=0.0, tau=tau[15] * 0.5)
        ])
        self.right_arm.get_gripper().mit_control_all([
            oa.MITParam(kp=0.0, kd=0.0, q=0.0, dq=0.0, tau=0.0),
        ])
        
        self.left_arm.get_arm().mit_control_all([
            oa.MITParam(kp=0.0, kd=0.0, q=0.0, dq=0.0, tau=tau[0]),
            oa.MITParam(kp=0.0, kd=0.0, q=0.0, dq=0.0, tau=tau[1]),
            oa.MITParam(kp=0.0, kd=0.0, q=0.0, dq=0.0, tau=tau[2]),
            oa.MITParam(kp=0.0, kd=0.0, q=0.0, dq=0.0, tau=tau[3]),
            oa.MITParam(kp=0.0, kd=0.0, q=0.0, dq=0.0, tau=tau[4]),
            oa.MITParam(kp=0.0, kd=0.0, q=0.0, dq=0.0, tau=tau[5]),
            oa.MITParam(kp=0.0, kd=0.0, q=0.0, dq=0.0, tau=tau[6] * 0.5)
        ])
        self.left_arm.get_gripper().mit_control_all([
            oa.MITParam(kp=0.0, kd=0.0, q=0.0, dq=0.0, tau=0.0),
        ])
        for _ in range(8):
            self.right_arm.recv_all(250)
            self.left_arm.recv_all(250)
        
        dt_ms = (time.time() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._is_connected = False
        time.sleep(0.1)
        self.right_arm.disable_all()
        self.left_arm.disable_all()
        time.sleep(2 * 1e-3)
        self.right_arm.recv_all()
        self.left_arm.recv_all()

        logger.info(f"{self} disconnected.")