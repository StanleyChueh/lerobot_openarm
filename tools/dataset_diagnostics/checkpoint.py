"""Compare a dataset with a policy checkpoint's declared features, padding limits and normalizer statistics.

Only small files are read (config.json, train_config.json, policy_preprocessor.json and the normalizer
.safetensors); model weights are never downloaded. The SmolVLA rules mirror the code path audited in
research/dataset_diagnostics/SMOLVLA_STATE_AUDIT.md:

  * input_features is copied from the pretrained base when that base already declares it
    (lerobot/policies/factory.py, `if not cfg.input_features`), so a declared state shape that differs from
    the dataset is stale metadata, not proof of an execution error. Whether the run really used the dataset
    shape is decided by the normalizer statistics' shape, which always comes from the training dataset.
  * prepare_state pads with pad_vector(truncate=False): a state wider than max_state_dim is NOT truncated
    but fails in state_proj (Linear(max_state_dim, ...)); the same holds for actions and max_action_dim.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .checks import FAIL, PASS, UNKNOWN, WARN, Check
from .loader import DatasetView

SMALL_FILES = ["config.json", "train_config.json", "policy_preprocessor.json", "policy_postprocessor.json",
               "*normalizer_processor.safetensors", "*unnormalizer_processor.safetensors"]


def load_checkpoint_meta(spec: str, revision: str | None = None, offline: bool = False) -> dict:
    path = Path(spec).expanduser()
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(spec, revision=revision, allow_patterns=SMALL_FILES, local_files_only=offline))
    meta = {"spec": spec, "dir": str(path), "revision": path.name if not Path(spec).is_dir() else None}
    for name in ("config.json", "train_config.json", "policy_preprocessor.json"):
        p = path / name
        meta[name] = json.loads(p.read_text()) if p.exists() else None
    norm = sorted(path.glob("*preprocessor*normalizer_processor.safetensors"))
    meta["normalizer"] = None
    if norm:
        from safetensors.numpy import load_file

        meta["normalizer"] = {k: v for k, v in load_file(str(norm[0])).items()}
        meta["normalizer_file"] = norm[0].name
    return meta


def _norm_stat(norm: dict | None, key: str, stat: str) -> np.ndarray | None:
    if not norm or f"{key}.{stat}" not in norm:
        return None
    return np.asarray(norm[f"{key}.{stat}"], dtype=np.float64).ravel()


def compare_checkpoint(view: DatasetView, meta: dict) -> list[Check]:
    cfg = meta.get("config.json") or {}
    train = meta.get("train_config.json") or {}
    norm = meta.get("normalizer")
    checks: list[Check] = []
    if not cfg:
        return [Check("checkpoint_config", FAIL, f"no config.json in {meta['dir']}", {})]
    ptype = cfg.get("type", "?")
    rename = train.get("rename_map") or {}
    inputs = cfg.get("input_features") or {}
    outputs = cfg.get("output_features") or {}

    # ------------------------------------------------------------ state
    ds_state = view.dim("observation.state") if "observation.state" in view.features else None
    decl_state = int(np.prod(inputs.get("observation.state", {}).get("shape", [0]))) or None
    norm_state = _norm_stat(norm, "observation.state", "mean")
    max_state = cfg.get("max_state_dim")
    ev = {"dataset": ds_state, "declared": decl_state, "normalizer": None if norm_state is None else len(norm_state),
          "max_state_dim": max_state, "policy_type": ptype}
    if ds_state is None:
        checks.append(Check("ckpt_state_dim", UNKNOWN, "dataset has no observation.state", ev))
    elif max_state is not None and ds_state > max_state:
        checks.append(Check("ckpt_state_dim", FAIL, f"state dim {ds_state} > max_state_dim {max_state}: pad_vector "
                                                     "does not truncate, state_proj input size mismatch", ev))
    elif norm_state is not None and len(norm_state) != ds_state:
        checks.append(Check("ckpt_state_dim", FAIL, f"normalizer state stats have {len(norm_state)} dims, dataset "
                                                     f"{ds_state}: the checkpoint was trained on a different state", ev))
    elif decl_state is not None and decl_state != ds_state:
        status = WARN if norm_state is not None else UNKNOWN
        checks.append(Check("ckpt_state_dim", status,
                            f"config declares state [{decl_state}] but normalizer stats are [{ev['normalizer']}] = "
                            f"dataset [{ds_state}]: stale input_features inherited from the pretrained base "
                            "(metadata only; strict servers such as lerobot-policy-server refuse it)"
                            if norm_state is not None else
                            f"config declares state [{decl_state}], dataset has [{ds_state}], no normalizer to tell "
                            "which one training used", ev))
    else:
        checks.append(Check("ckpt_state_dim", PASS, f"state [{ds_state}] matches checkpoint"
                            + (f", padded to {max_state}" if max_state else ""), ev))

    # ------------------------------------------------------------ action
    ds_act = view.dim("action") if "action" in view.features else None
    decl_act = int(np.prod(outputs.get("action", {}).get("shape", [0]))) or None
    norm_act = _norm_stat(norm, "action", "mean")
    max_act = cfg.get("max_action_dim")
    ev = {"dataset": ds_act, "declared": decl_act, "normalizer": None if norm_act is None else len(norm_act),
          "max_action_dim": max_act}
    if ds_act is None:
        checks.append(Check("ckpt_action_dim", UNKNOWN, "dataset has no action", ev))
    elif max_act is not None and ds_act > max_act:
        checks.append(Check("ckpt_action_dim", FAIL, f"action dim {ds_act} > max_action_dim {max_act}", ev))
    elif decl_act is not None and decl_act != ds_act:
        checks.append(Check("ckpt_action_dim", FAIL, f"checkpoint outputs action [{decl_act}], dataset [{ds_act}]: "
                                                      "predictions are unpadded to the declared size", ev))
    elif norm_act is not None and len(norm_act) != ds_act:
        checks.append(Check("ckpt_action_dim", FAIL, f"normalizer action stats [{len(norm_act)}] != dataset", ev))
    else:
        checks.append(Check("ckpt_action_dim", PASS, f"action [{ds_act}] matches checkpoint", ev))

    # ------------------------------------------------------------ cameras
    ds_cams = {rename.get(c, c): c for c in view.camera_keys}
    ck_cams = [k for k, v in inputs.items() if v.get("type") == "VISUAL"]
    missing = [c for c in ck_cams if c not in ds_cams]
    unused = [ds_cams[c] for c in ds_cams if c not in ck_cams]
    empty_ok = int(cfg.get("empty_cameras") or 0)
    ev = {"checkpoint_cameras": ck_cams, "dataset_cameras_after_rename": ds_cams, "rename_map": rename,
          "empty_cameras": empty_ok}
    if missing and len(missing) > empty_ok:
        checks.append(Check("ckpt_cameras", FAIL, f"checkpoint expects {missing}, absent from the dataset "
                                                   f"(after rename_map)", ev))
    elif missing or unused:
        checks.append(Check("ckpt_cameras", WARN, f"missing {missing} (covered by empty_cameras={empty_ok}); "
                                                   f"unused dataset cameras {unused}", ev))
    else:
        checks.append(Check("ckpt_cameras", PASS, f"all {len(ck_cams)} checkpoint cameras present", ev))
    resize = cfg.get("resize_imgs_with_padding")
    shape_diff = {c: (inputs[c].get("shape"), view.features[ds_cams[c]].get("shape")) for c in ck_cams
                  if c in ds_cams}
    checks.append(Check("ckpt_image_shapes", PASS if resize else WARN,
                        f"policy resizes to {resize}; declared image shapes are not used for slicing" if resize else
                        "policy does not resize; declared and dataset image shapes must agree", shape_diff))

    # ------------------------------------------------------------ normalizer vs dataset statistics
    checks.extend(_normalizer_vs_dataset(view, norm))

    # ------------------------------------------------------------ chunking vs episode length
    chunk = cfg.get("chunk_size")
    if chunk and len(view.data):
        lengths = view.data.groupby("episode_index").size()
        short = sorted(int(e) for e, n in lengths.items() if n < chunk)
        checks.append(Check("ckpt_chunk_vs_episode_length", WARN if short else PASS,
                            f"episodes shorter than chunk_size {chunk}: {short}" if short else
                            f"every episode >= chunk_size {chunk}", {"short_episodes": short}))
    return checks


def _normalizer_vs_dataset(view: DatasetView, norm: dict | None) -> list[Check]:
    out = []
    if norm is None:
        return [Check("ckpt_normalizer", UNKNOWN, "checkpoint has no normalizer statistics file", {})]
    for key in ("observation.state", "action"):
        mean, std = _norm_stat(norm, key, "mean"), _norm_stat(norm, key, "std")
        if mean is None or key not in view.data.columns or len(mean) != view.dim(key):
            continue
        arr = np.stack(view.data[key].values).astype(np.float64)
        z = (arr - mean) / (std + 1e-8)
        frac_out = float((np.abs(z) > 4).any(1).mean())
        dmean = np.abs(arr.mean(0) - mean) / np.maximum(std, 1e-6)
        names = view.names(key)
        worst = np.argsort(-dmean)[:3]
        ev = {"max_mean_shift_in_ckpt_std": float(dmean.max()),
              "worst_dims": {names[i]: round(float(dmean[i]), 3) for i in worst},
              "frac_frames_any_dim_|z|>4": round(frac_out, 4),
              "ckpt_count": float(np.asarray(norm.get(f"{key}.count", [np.nan])).ravel()[0])
              if f"{key}.count" in norm else None, "dataset_frames": int(len(arr))}
        if dmean.max() < 1e-4:
            out.append(Check(f"ckpt_normalizer_{key}", PASS, "checkpoint statistics equal this dataset's "
                                                              "(it was trained on exactly these frames)", ev))
        elif dmean.max() < 0.5 and frac_out < 0.01:
            out.append(Check(f"ckpt_normalizer_{key}", PASS, "dataset is within the checkpoint's normalization range",
                             ev))
        else:
            out.append(Check(f"ckpt_normalizer_{key}", WARN,
                             f"dataset shifted vs checkpoint statistics (max {dmean.max():.2f} std in "
                             f"{list(ev['worst_dims'])}); {100 * frac_out:.1f}% frames have a |z|>4 dimension", ev))
    return out


def training_provenance(view: DatasetView, meta: dict) -> dict:
    """What the checkpoint says about its training data, and whether this dataset is that data."""
    train = meta.get("train_config.json") or {}
    ds = train.get("dataset") or {}
    norm = meta.get("normalizer") or {}
    count = float(np.asarray(norm.get("observation.state.count", [np.nan])).ravel()[0]) if norm else None
    ep_max = float(np.asarray(norm.get("episode_index.max", [np.nan])).ravel()[0]) if "episode_index.max" in norm else None
    return {
        "checkpoint": meta.get("spec"), "checkpoint_revision": meta.get("revision"),
        "train_dataset_repo_id": ds.get("repo_id"), "train_dataset_revision": ds.get("revision"),
        "train_episodes_arg": ds.get("episodes"), "steps": train.get("steps"), "batch_size": train.get("batch_size"),
        "seed": train.get("seed"), "pretrained_path": (train.get("policy") or {}).get("pretrained_path"),
        "image_transforms_enabled": ((ds.get("image_transforms") or {}).get("enable")),
        "normalizer_frame_count": count, "normalizer_episode_index_max": ep_max,
        "dataset_frames": int(len(view.data)),
        "frame_count_matches_dataset": None if count is None else bool(int(count) == len(view.data)),
        "same_repo": ds.get("repo_id") == view.repo_id if view.repo_id else None,
        "source_code_sha": "not recorded by lerobot-train",
    }
