import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
import pinocchio as pin

from pathlib import Path
from pinocchio.visualize import MeshcatVisualizer

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from teleoperators.umeow_openarm_leader import OpenArmConfig, OpenArmLeader
from robots.umeow_openarm_follower import OpenArmFollowerConfig, OpenArmFollower

camera_config = {}

robot_config = OpenArmFollowerConfig(
    right_port = 'can2',
    left_port  = 'can3',
    
    enable_fd = True,
     
    model_path='/home/csl/lerobot_openarm/model/openarm_description.urdf',
    
    cameras=camera_config # type: ignore
)

teleop_config = OpenArmConfig(
    model_path='/home/csl/lerobot_openarm/model/openarm_description_leader.urdf',
    right_port = 'can0',
    left_port  = 'can1',
    
    enable_fd = True,
)

robot = OpenArmFollower(robot_config)
teleop_device = OpenArmLeader(teleop_config)
robot.connect()
teleop_device.connect()

# model_path = Path("model")
# mesh_dir = Path("model/meshes")
# urdf_filename = "openarm_description.urdf"
# urdf_model_path = model_path / urdf_filename

# model, collision_model, visual_model = pin.buildModelsFromUrdf(
#     urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
# )

# viz = MeshcatVisualizer(model, collision_model, visual_model)
# viz.initViewer(open=True)
# viz.loadViewerModel()

try:
    while True:
        observation = robot.get_observation()
        action = teleop_device.get_action()
        # q = np.array([
        #     0, 0, 0, 0, 0, 0, 0, 
        #     action['LJ1.pos'], action['LJ2.pos'], action['LJ3.pos'], action['LJ4.pos'],
        #     action['LJ5.pos'], action['LJ6.pos'], action['LJ7.pos'], -action['LJ8.pos'] / 25.0, -action['LJ8.pos'] / 25.0,
        #     action['RJ1.pos'], action['RJ2.pos'], action['RJ3.pos'], action['RJ4.pos'],
        #     action['RJ5.pos'], action['RJ6.pos'], action['RJ7.pos'], -action['RJ8.pos'] / 25.0, -action['RJ8.pos'] / 25.0,
        # ], np.float32)
        # viz.display(q=q)
        robot.send_action(action)
except KeyboardInterrupt:
    pass

robot.disconnect()
teleop_device.disconnect()