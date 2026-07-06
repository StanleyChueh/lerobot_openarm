# lerobot_openarm

Please install uv first. See Installation for details.
 [Installation](https://docs.astral.sh/uv/getting-started/installation/).

Clone this repo, and sync the environments automatically.

```
git clone https://github.com/umeow0716/lerobot_openarm.git
uv sync
source .venv/bin/activate
```

Ensure the robotic arm is powered on and correctly plugged in. Then, run this script to enable the CAN interface:

```
cd ~/openarm_can/setup
sudo ./my_arm
```

### Record an eposide 
```
python record.py
```

### Training
```
lerobot-train \
    --policy.path=lerobot/smolvla_openarm \
    --dataset.repo_id=ethanCSL/0508-6767 \
    --batch_size=32 \
    --steps=30000 \
    --output_dir=outputs/train/svla_multiblock \
    --job_name=my_smolvla_training \
    --policy.device=cuda \
    --wandb.enable=false \
    --policy.repo_id=0508-6767 \
    --rename_map='{"observation.images.right_camera": "observation.images.camera1", "observation.images.left_camera": "observation.images.camera2", "observation.images.body_camera": "observation.images.camera3"}'

``` 

### Deploy the policy
```
python deploy_ACT.py
python deploy_smolvla.py
```

lerobot-train \
    --policy.type=smolvla_openarm \
    --dataset.repo_id=ethanCSL/0409tist \
    --batch_size=32 \
    --steps=30000 \
    --output_dir=outputs/train/0409tist3 \
    --job_name=my_smolvla_training \
    --policy.device=cuda \
    --wandb.enable=false \
    --policy.repo_id=0409tist3 \
    --rename_map='{"observation.images.right_camera": "observation.images.camera1", "observation.images.left_camera": "observation.images.camera2", "observation.images.body_camera": "observation.images.camera3"}'

lerobot-train \
  --dataset.repo_id=ethanCSL/0409tist \
  --policy.type=smolvla_openarm \
  --policy.expert_width_multiplier=0.5 \
  --policy.num_vlm_layers=0 \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Instruct \
  --output_dir=outputs/train/0409tist4 \
  --job_name=smolvla_pick_openarm \
  --batch_size=16 \
  --steps=30000 \
  --optimizer.lr=1e-5 \
  --num_workers=4 \
  --save_freq=2500 \
  --eval_freq=2500 \
  --wandb.enable=false \
  --policy.push_to_hub=true \
  --save_checkpoint=true \
  --policy.repo_id=0409tist4

lerobot-train \
    --policy.type=smolvla \
    --dataset.repo_id=ethanCSL/0422_stanley_red_cube \
    --batch_size=32 \
    --steps=30000 \
    --output_dir=outputs/train/ethanCSL/0422_umeow_red_cube \
    --job_name=my_smolvla_training \
    --policy.device=cuda \
    --policy.repo_id=ethanCSL/0422_umeow_red_cube_test \
    --wandb.enable=false