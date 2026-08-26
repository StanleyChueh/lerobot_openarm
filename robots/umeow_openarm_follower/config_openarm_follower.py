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

from dataclasses import dataclass, field

from openarm_can import MotorType, ControlMode
from lerobot.cameras.camera import CameraConfig

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("umeow_openarm_follower")
@dataclass
class OpenArmFollowerConfig(RobotConfig):
    # Port to connect to the arm
    right_port: str
    left_port:  str
    
    enable_fd: bool
    
    model_path: str
    
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    
    motor_types: list[MotorType] = field(default_factory=lambda: [
        MotorType.DM8009, MotorType.DM8009,
        MotorType.DM4340, MotorType.DM4340,
        MotorType.DM4310, MotorType.DM4310, MotorType.DM4310
    ])
    
    send_ids = [ 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07 ]
    recv_ids = [ 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17 ]
    
    # motor_modes: list[ControlMode] = field(default_factory=lambda: [
    #     ControlMode.POS_VEL, ControlMode.POS_VEL,
    #     ControlMode.POS_VEL, ControlMode.POS_VEL,
    #     ControlMode.POS_VEL, ControlMode.POS_VEL, ControlMode.POS_VEL
    # ])
    
    motor_modes: list[ControlMode] = field(default_factory=lambda: [
        ControlMode.MIT, ControlMode.MIT,
        ControlMode.MIT, ControlMode.MIT,
        ControlMode.MIT, ControlMode.MIT, ControlMode.MIT
    ])

    # --- CAN receive tuning, consumed by _read_motor_positions_once() -------------------------
    # Defaults reproduce the historical hard-coded behaviour EXACTLY (8 rounds, 50000us each, no
    # split), so every existing caller is unaffected. Only callers that opt in get the cheaper
    # settings -- see the block comment in openarm_follower._read_motor_positions_once for the
    # measurements (profile_can_read.py, 2026-08-21) and why a long timeout does not buy
    # freshness.
    recv_rounds: int = 8
    recv_first_timeout_us: int = 50_000
    recv_mop_timeout_us: int | None = None   # None -> every round uses recv_first_timeout_us

    # How long a read may WAIT for the motors to answer, as opposed to how many times it polls.
    # These are different things and only this one is under our control: OpenArm::recv_all()'s
    # timeout argument does not produce a wait on this build -- measured 2026-08-26, recv_all()
    # returns in 0.04-0.16 ms whether it is passed 500 us or 200 000 us, on an empty socket. So
    # `recv_rounds` rounds of it cover 2-3 ms of wall clock in total, and cover it by spinning,
    # not by waiting. The motors answer 0.26 ms (J1) to 1.02 ms (J8) after the request on an idle
    # bus and later than that under a control loop's traffic, so the drain window was racing the
    # replies, and the motors that answer LAST -- the highest CAN ids, J5-J8 -- were the ones the
    # window kept closing on. See _read_motor_positions_once().
    #
    # The read now stops as soon as every channel has answered, so this ceiling is only ever paid
    # when something really is missing; a healthy read costs about as much as the old spin did.
    recv_deadline_us: int = 8_000

    # How long a still-silent channel waits before its arm is asked again, inside one read.
    # 0 disables re-asking entirely, which is there to be USED: a re-ask puts another burst on
    # the wire, and if a motor's reply latency ever grew past this interval the re-ask would be
    # landing on top of the very answer it is waiting for -- a self-sustaining outage that looks
    # exactly like a hardware fault. Measured reply latency on this rig is 0.13-0.87 ms with a
    # max of 0.87, so 3 ms has a comfortable margin, but the margin is an assumption about a
    # moving arm made from measurements on a still one. Run once with 0 to settle it.
    recv_retry_us: int = 3_000
    
    gripper_motor_type = MotorType.DM4310
    gripper_motor_send_id = 0x08
    gripper_motor_recv_id = 0x18
    # gripper_motor_mode = ControlMode.POS_VEL
    gripper_motor_mode = ControlMode.MIT
