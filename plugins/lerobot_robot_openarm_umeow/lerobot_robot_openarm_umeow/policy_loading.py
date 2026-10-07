"""Load the evaluated policy in bf16 under lerobot-rollout, so GR00T N1.7 fits a 16 GB GPU.

lerobot-rollout loads the policy with from_pretrained() and moves it to the GPU in fp32. GrootConfig's
use_bf16 only autocasts the forward pass; the 3B weights stay fp32 (~12 GB), and the first inference
then runs out of memory on a 16 GB card -- measured here: 14.7 GB in use at the OOM. This is the fix
deploy_gr00t_async.py already used: load on the CPU, cast to bf16 there, and only then move to the
GPU (~6 GB), so the fp32 copy never reaches it.

Chosen with --robot.policy_dtype (read from the command line: the policy is loaded before the robot
config is used): "auto" (default) = bf16 for GR00T, unchanged for every other policy; "bf16" = always;
"fp32" = never. lerobot-rollout only; installed when the plugin is imported, before the policy loads.
"""

import sys


def _argv_value(flag: str) -> str | None:
    for i, a in enumerate(sys.argv):
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def install() -> None:
    command = sys.argv[0].rsplit("/", 1)[-1] if sys.argv else ""
    if "rollout" not in command:
        return
    import lerobot.rollout.context as context

    if getattr(context, "_openarm_policy_dtype", False):
        return
    original = context._load_pretrained_policy

    def load(policy_config):
        choice = (_argv_value("--robot.policy_dtype") or "auto").lower()
        if choice not in ("auto", "bf16", "fp32"):
            raise ValueError(f"--robot.policy_dtype must be auto, bf16 or fp32, got {choice!r}")
        if choice == "fp32" or (choice == "auto" and policy_config.type != "groot"):
            return original(policy_config)
        import torch

        device = policy_config.device
        policy_config.device = "cpu"  # from_pretrained moves to config.device: keep fp32 off the GPU
        try:
            policy = original(policy_config)
        finally:
            policy_config.device = device
        policy = policy.to(torch.bfloat16)
        policy.config.device = device
        print(f"[openarm] {policy_config.type} policy loaded on the CPU and cast to bf16 before moving to"
              f" {device} (--robot.policy_dtype={choice}).", flush=True)
        return policy

    context._load_pretrained_policy = load
    context._openarm_policy_dtype = True
