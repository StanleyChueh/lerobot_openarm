"""Policy-side fixes so SmolVLA and GR00T N1.7 run in all three evaluation modes on this robot.

  normal  lerobot-rollout (sync inference)        -> GR00T: chunk-wise decoding (below)
  RTC     lerobot-rollout --inference.type=rtc    -> GR00T: bf16 loading (below)
  async   policy server + robot client            -> policy_server.py in this package

1. bf16 loading (lerobot-rollout, and the async policy server). lerobot loads a policy with
   from_pretrained() and moves it to the GPU in fp32. GrootConfig's use_bf16 only autocasts the forward
   pass; the 3B weights stay fp32 (~12 GB) and the first inference ran out of memory on the 16 GB RTX 5080
   (14.7 GB in use). Fix, as deploy_gr00t_async.py did: load on the CPU, cast to bf16 there, only then move
   to the GPU (~7 GB peak). Chosen by --robot.policy_dtype (read off the command line: the policy is loaded
   before the robot): "auto" (default) = bf16 for GR00T only; "bf16" = always; "fp32" = never.

2. Chunk-wise decoding (lerobot-rollout's normal, synchronous mode). With relative actions (our GR00T
   recipe) every action of a chunk is a delta from the observation the chunk was predicted from, and the
   postprocessor can only decode a WHOLE chunk right after preprocessing that observation. lerobot's sync
   engine calls select_action() one step at a time, which GR00T refuses ("use predict_action_chunk and
   postprocess the full chunk before queuing actions"). For relative-action policies only, the sync
   engine here does exactly that: predict a chunk, decode it whole, execute its n_action_steps absolute
   actions one per tick, then predict the next. Every other policy goes through lerobot's code unchanged.
"""

import sys
from collections import deque
from copy import copy


def _argv_value(flag: str) -> str | None:
    for i, a in enumerate(sys.argv):
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def wants_bf16(policy_type: str, choice: str | None = None) -> bool:
    """choice: auto / bf16 / fp32; defaults to --robot.policy_dtype from the command line, else auto."""
    choice = (choice or _argv_value("--robot.policy_dtype") or "auto").lower()
    if choice not in ("auto", "bf16", "fp32"):
        raise ValueError(f"policy dtype must be auto, bf16 or fp32, got {choice!r}")
    return choice == "bf16" or (choice == "auto" and policy_type == "groot")


def load_bf16(load, policy_config):
    """Run load(policy_config) with the policy kept on the CPU, cast it to bf16, and restore the device."""
    import torch

    device = policy_config.device
    policy_config.device = "cpu"  # from_pretrained moves to config.device: keep the fp32 copy off the GPU
    try:
        policy = load(policy_config)
    finally:
        policy_config.device = device
    policy = policy.to(torch.bfloat16)
    policy.config.device = device
    print(f"[openarm] {policy_config.type} policy loaded on the CPU and cast to bf16 before moving to {device}.",
          flush=True)
    return policy


def _install_bf16(context) -> None:
    original = context._load_pretrained_policy

    def load(policy_config):
        if not wants_bf16(policy_config.type):
            return original(policy_config)
        return load_bf16(original, policy_config)

    context._load_pretrained_policy = load


def _install_chunked_sync() -> None:
    import torch

    import lerobot.rollout.inference.sync as sync
    from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference

    engine = sync.SyncInferenceEngine
    original_get, original_reset = engine.get_action, engine.reset

    def get_action(self, obs_frame):
        if not getattr(self._policy.config, "use_relative_actions", False):
            return original_get(self, obs_frame)
        if obs_frame is None:
            return None
        queue = self.__dict__.setdefault("_openarm_chunk", deque())
        task, task_changed = self._take_task()
        if task_changed:
            queue.clear()
        if not queue:
            with torch.inference_mode():
                observation = prepare_observation_for_inference(copy(obs_frame), self._device, task, self._robot_type)
                observation = self._preprocessor(observation)
                chunk = self._policy.predict_action_chunk(observation)
                if chunk.ndim == 2:
                    chunk = chunk.unsqueeze(0)
                chunk = chunk[:, : self._policy.config.n_action_steps]
                decoded = self._postprocessor(chunk)  # the whole chunk, against THIS observation
            queue.extend(decoded.squeeze(0).float().cpu())
        action_dict = make_robot_action(queue.popleft(), self._dataset_features)
        self._set_dispatched_task(task)
        return torch.tensor([action_dict[k] for k in self._ordered_action_keys])

    def reset(self):
        self.__dict__.setdefault("_openarm_chunk", deque()).clear()
        return original_reset(self)

    engine.get_action, engine.reset = get_action, reset


def install() -> None:
    command = sys.argv[0].rsplit("/", 1)[-1] if sys.argv else ""
    if "rollout" not in command:
        return
    import lerobot.rollout.context as context

    if getattr(context, "_openarm_policy_patches", False):
        return
    _install_bf16(context)
    _install_chunked_sync()
    context._openarm_policy_patches = True
