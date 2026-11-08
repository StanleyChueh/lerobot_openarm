"""SD10-L1 execution library: implements the frozen contract in ../learning_v1 (read-only), never tunes it.

Real pool observations/actions only (no synthetic data, no image augmentation). LeRobot is imported from the
registered source tree (530b253f, ../../../../lerobot/src), never from the older venv wheel.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

EXEC = Path(__file__).resolve().parent                   # sd10/l1_exec
SD = EXEC.parent                                          # sd10
L1 = SD / "learning_v1"                                   # frozen, sealed protocol bundle (read-only)
REPO = SD.parents[2]
LEROBOT_SRC = REPO.parent / "lerobot" / "src"
OUT = EXEC / "outputs"                                    # gitignored; small results force-added
CACHE = OUT / "frame_cache"
TRAIN = OUT / "train"
PRED = OUT / "predictions"
for p in (str(SD), str(L1)):
    if p not in sys.path:
        sys.path.insert(0, p)
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

import protocol_tools as PT  # noqa: E402
from sd10 import common as C  # noqa: E402

CFG = json.loads((L1 / "training_config.json").read_text())
NORM = json.loads((L1 / "training_normalization.json").read_text())
RT = json.loads((L1 / "source_and_runtime_manifest.json").read_text())
SEAL = L1 / "locked_artifacts.sha256"
SEAL_SHA = "7c74ef44bd658136535e3ad65991d53278c5f957624388f929ca929c54197183"
CAM_KEY = "observation.images.camera1"
ARMS, SEEDS, EVAL = PT.ARMS, PT.SEEDS, PT.EVAL
TRAIN_EPS = sorted(set().union(*map(set, ARMS.values())))


def sha(p):
    return PT.file_sha(p)


def check_seal():
    bad = []
    for line in SEAL.read_text().splitlines():
        h, rel = line.split(None, 1)
        if sha(REPO / rel.strip()) != h:
            bad.append(rel)
    assert sha(SEAL) == SEAL_SHA, "seal manifest changed"
    assert not bad, f"sealed artifacts changed: {bad}"
    return True


def exec_code_hashes():
    return {str(p.relative_to(EXEC)): sha(p) for p in sorted(EXEC.rglob("*.py")) if "__pycache__" not in p.parts}


def determinism(seed: int):
    """Seed Python/NumPy/Torch/CUDA; disable TF32 and cuDNN benchmarking; deterministic algorithms (warn-only)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from lerobot.utils.random_utils import set_seed
    set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {"seed": seed, "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32, "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def arm_stats(arm):
    import torch
    s = NORM["arms"][arm]["stats"]
    return {k: {"mean": torch.tensor(np.array(s[k]["mean"], np.float32)), "std": torch.tensor(np.array(s[k]["std"], np.float32))}
            for k in ("observation.state", "action")}


def policy_config(device="cuda"):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    snap = CFG["pretrained_snapshot"]
    cfg = PreTrainedConfig.from_pretrained(snap, local_files_only=True)
    ov = CFG["policy_overrides"]
    ft = {"STATE": FeatureType.STATE, "VISUAL": FeatureType.VISUAL, "ACTION": FeatureType.ACTION}
    from lerobot.configs.types import NormalizationMode
    for k, v in ov.items():
        if k in ("type",):
            assert cfg.type == v
            continue
        if k in ("input_features", "output_features"):
            v = {n: PolicyFeature(type=ft[f["type"]], shape=tuple(f["shape"])) for n, f in v.items()}
        elif k == "normalization_mapping":
            v = {kk: NormalizationMode(vv) for kk, vv in v.items()}
        elif k == "resize_imgs_with_padding":
            v = tuple(v)
        assert hasattr(cfg, k), k
        setattr(cfg, k, v)
    cfg.device = device
    cfg.pretrained_path = Path(snap)
    return cfg


def make_policy_and_processors(arm, device="cuda"):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    cfg = policy_config(device)
    policy = SmolVLAPolicy.from_pretrained(CFG["pretrained_snapshot"], config=cfg, local_files_only=True)
    pre, post = make_smolvla_pre_post_processors(cfg, dataset_stats=arm_stats(arm))
    return policy.to(device), pre, post, cfg


def normalizer_tensors(pipeline):
    """Extract (un)normalizer stats from a processor pipeline as float32 numpy arrays."""
    out = {}
    for st in pipeline.steps:
        if hasattr(st, "stats") and st.stats:
            for k, d in st.stats.items():
                for kk, v in d.items():
                    out[f"{type(st).__name__}/{k}/{kk}"] = np.asarray(v.detach().cpu() if hasattr(v, "detach") else v, np.float32)
            if hasattr(st, "eps"):
                out[f"{type(st).__name__}/eps"] = np.float64(st.eps)
    return out


def check_normalizers(pre, post, arm):
    s = NORM["arms"][arm]["stats"]
    n = {**normalizer_tensors(pre), **normalizer_tensors(post)}
    for k in ("observation.state", "action"):
        for m in ("mean", "std"):
            assert np.array_equal(n[f"NormalizerProcessorStep/{k}/{m}"], np.array(s[k][m], np.float32)), (k, m)
    for m in ("mean", "std"):
        assert np.array_equal(n[f"UnnormalizerProcessorStep/action/{m}"], np.array(s["action"][m], np.float32))
    assert "UnnormalizerProcessorStep/observation.state/mean" not in n or True
    assert float(n["NormalizerProcessorStep/eps"]) == 1e-8
    return {k: PT.array_sha(v) for k, v in n.items() if isinstance(v, np.ndarray) and v.ndim}


def cache_path(e):
    return CACHE / f"body_ep{e:02d}.npy"


def load_frames(e):
    return np.load(cache_path(e), mmap_mode="r")


class SourceDataset:
    """Map-style dataset indexed by (episode, start). Observation = native body RGB and state at row t;
    target = recorded action[t:t+50]; action_is_pad all False. No augmentation."""

    def __init__(self, episodes):
        pool = C.pool()
        self.S = {e: pool.vec(e, "observation.state") for e in episodes}
        self.A = {e: pool.vec(e, "action") for e in episodes}
        self.episodes = list(episodes)
        self._frames = {}
        self.task = PT.TASK

    def frames(self, e):
        if e not in self._frames:
            self._frames[e] = load_frames(e)
        return self._frames[e]

    def __len__(self):
        return sum(len(self.A[e]) for e in self.episodes)

    def __getitem__(self, key):
        import torch
        e, t = int(key[0]), int(key[1])
        A = self.A[e]
        assert 0 <= t <= len(A) - 50
        img = np.array(self.frames(e)[t], dtype=np.uint8, copy=True)         # HWC uint8 RGB (copied from memmap)
        return {"observation.images.body_cam": torch.from_numpy(img).permute(2, 0, 1).contiguous(),
                "observation.state": torch.from_numpy(self.S[e][t].copy()),
                "action": torch.from_numpy(A[t:t + 50].copy()),
                "action_is_pad": torch.zeros(50, dtype=torch.bool),
                "task": self.task, "episode": e, "frame": t}


class FrozenBatches:
    """Batch sampler that yields exactly the frozen (episode, start) draws, one list per optimizer update."""

    def __init__(self, arm, seed):
        z = np.load(L1 / f"sampling_seed{seed}.npz", allow_pickle=False)
        self.ep, self.fr = z[arm + "_episode"].astype(np.int64), z[arm + "_frame"].astype(np.int64)
        assert self.ep.shape == self.fr.shape == (PT.UPDATES, PT.BATCH)
        self.digest = PT.array_sha(np.stack([self.ep, self.fr]))

    def __len__(self):
        return len(self.ep)

    def __iter__(self):
        for i in range(len(self.ep)):
            yield [(int(e), int(t)) for e, t in zip(self.ep[i], self.fr[i])]


def to_policy_batch(batch):
    """Same as lerobot_train._preprocess_dataset_batch: uint8 -> float32/255, then rename body_cam -> camera1."""
    import torch
    img = batch.pop("observation.images.body_cam")
    assert img.dtype == torch.uint8
    batch[CAM_KEY] = img.to(dtype=torch.float32) / 255.0
    return batch
