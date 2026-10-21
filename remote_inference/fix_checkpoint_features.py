"""Copy a checkpoint with its input features taken from the dataset it was trained on.

    python remote_inference/fix_checkpoint_features.py ethanCSL/openarm_pringles_lerobot_real_smolvla_20_v00 \
        outputs/remote_checkpoints/openarm_pringles_lerobot_real_smolvla_20_v00

A SmolVLA fine-tuned from lerobot/smolvla_base keeps the BASE model's input features in config.json
(observation.state [6], camera1/2/3 at 3x256x256): lerobot-train only fills input_features from the dataset when
the pretrained config has none. The model itself is trained on the dataset's state (16 joints here, padded to
max_state_dim), so this is metadata only -- but lerobot-policy-server checks it against the deployment schema and
refuses the checkpoint ("Tensor feature shape differs from checkpoint: observation.state").

The copy gets each input feature's shape from the training dataset's meta/info.json, under the checkpoint's own
names (through the rename_map in train_config.json). Weights, processors and stats are copied unchanged; only
config.json and the normalizer's feature list in policy_preprocessor.json are rewritten. Point the server's
model.repo_or_path at the copy.
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def dataset_shapes(train_cfg: dict) -> dict[str, list[int]]:
    """Feature -> shape in the checkpoint's naming (images as C, H, W), from the training dataset."""
    ds = train_cfg["dataset"]
    info = json.loads(Path(hf_hub_download(ds["repo_id"], "meta/info.json", repo_type="dataset",
                                           revision=ds.get("revision"))).read_text())
    rename = train_cfg.get("rename_map") or {}
    shapes = {}
    for key, ft in info["features"].items():
        if key == "observation.state":
            shapes[key] = list(ft["shape"])
        elif key.startswith("observation.images."):
            h, w, c = ft["shape"]
            shapes[rename.get(key, key)] = [c, h, w]
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help="Hub repo id or local checkpoint directory")
    parser.add_argument("out_dir", help="where to write the corrected copy")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    src = Path(args.checkpoint)
    if not src.is_dir():
        src = Path(snapshot_download(args.checkpoint, revision=args.revision))
    out = Path(args.out_dir)
    if out.exists():
        raise SystemExit(f"{out} exists; remove it or choose another directory")
    shutil.copytree(src, out)  # follows the Hub cache's symlinks: a self-contained copy

    shapes = dataset_shapes(json.loads((out / "train_config.json").read_text()))
    config = json.loads((out / "config.json").read_text())
    changed = []
    for key, ft in config["input_features"].items():
        if key in shapes and ft["shape"] != shapes[key]:
            changed.append(f"{key}: {ft['shape']} -> {shapes[key]}")
            ft["shape"] = shapes[key]
    missing = [k for k in config["input_features"] if k not in shapes]
    if missing:
        raise SystemExit(f"no dataset feature for {missing}; check rename_map in train_config.json")
    (out / "config.json").write_text(json.dumps(config, indent=4))

    pre = out / "policy_preprocessor.json"
    pipeline = json.loads(pre.read_text())
    for step in pipeline["steps"]:
        for key, ft in ((step.get("config") or {}).get("features") or {}).items():
            if key in shapes:
                ft["shape"] = shapes[key]
    pre.write_text(json.dumps(pipeline, indent=2))

    print("\n".join(changed) if changed else "input features already match the dataset")
    print(f"corrected copy: {out.resolve()}")


if __name__ == "__main__":
    main()
