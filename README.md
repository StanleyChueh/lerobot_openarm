# Run in Real-world

## Activate CAN-FD

```
cd ~/Stanley_ws/openarm_can/setup
```

```
sudo ./my_arm 
```

## Model evaluation on real robot

```
cd ~/Stanley_ws/lerobot_openarm
uv sync
source .venv/bin/activate
env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python deploy_smolvla_pickup_jointspace.py     --checkpoint ethanCSL/openarm_visuomotor_VR_pringles_V14_background_30hz     --body-cam-index rs_body --wrist-cam-index rs_wrist_left --right-wrist-cam-index rs_wrist_right     --calibration calibration.json     --inference-hz 30 --max-joint-speed 1.5 --max-episode-seconds 600 
```

Deploy in async evaluation

```
cd ~/Stanley_ws/lerobot_openarm
uv sync
source .venv/bin/activate
env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python deploy_smolvla_async.py     --checkpoint ethanCSL/openarm_visuomotor_VR_pringles_V14_background_30hz    --body-cam-index rs_body --wrist-cam-index rs_wrist_left --right-wrist-cam-index rs_wrist_right     --calibration calibration.json     --control-hz 30 --max-joint-speed 1.5     --actions-per-chunk 50 --chunk-size-threshold 0.8     --max-episode-seconds 25 --max-episodes 20
```

