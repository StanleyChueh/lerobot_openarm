"""lerobot's async-inference policy server, with the fixes our SmolVLA and GR00T N1.7 checkpoints need.

    python -m lerobot_robot_openarm_umeow.policy_server --host=127.0.0.1 --port=8080 [--policy_dtype=auto]
        [--obs_similarity_atol=0.0175]
        [--rtc=true --rtc_execution_horizon=10 --rtc_max_guidance_weight=10.0 --rtc_prefix_attention_schedule=EXP]

Same flags and behaviour as `python -m lerobot.async_inference.policy_server`, which this runs after
patching three things in it (none changes a policy that does not need it):

  - rename map: the robot client never sends one, and the server then OVERRIDES the map stored in the
    checkpoint with an empty one -- so a SmolVLA fine-tuned with --rename_map (smolvla_base's camera1/2/3)
    gets cameras it does not know. An empty map from the client now keeps the checkpoint's own. And the
    server resizes each camera image by looking up the POLICY's feature under the ROBOT's camera name,
    before any renaming (KeyError 'observation.images.body_cam'); those features are now also listed under
    the robot's names, through the same map.
  - whole-chunk decoding: the server postprocesses a chunk one action at a time; GR00T's relative actions
    can only be decoded as a whole chunk against the observation it was predicted from (it raises
    NotImplementedError otherwise). For relative-action policies the chunk is decoded whole, right after
    inference, and the per-step loop then passes the decoded actions through unchanged.
  - bf16 loading for GR00T (--policy_dtype, default auto): see policy_loading.py.
  - "too similar" observations: the server skips an observation whose joint-state vector is within 1.0
    (norm) of the last one it ran -- lerobot's robots report DEGREES, so ~1 degree. This robot reports
    RADIANS, where 1.0 is ~57 degrees: nearly every observation sent mid-chunk was skipped and the server
    only predicted once the client's queue had run EMPTY, i.e. async collapsed to synchronous (and RTC had
    no unexecuted tail to continue). The threshold is now --obs_similarity_atol, default 0.0175 rad (1 deg).

And one addition, async + RTC (--rtc=true; lerobot's server has no RTC). lerobot's RTC engine
(lerobot-rollout --inference.type=rtc) guides each new chunk to continue the unexecuted tail of the previous
one; here the same is done server-side. The server remembers the last chunk it sent, in both model space
and decoded form, with its timesteps. A new observation carries the timestep the client has reached, so the
actions after it are the tail still to be executed. For relative-action policies (GR00T) that tail is
re-anchored to the new observation's state (lerobot's reanchor_relative_rtc_prefix). It is padded or trimmed
to execution_horizon, and predict_action_chunk(inference_delay=..., prev_chunk_left_over=...) is called with the
delay estimated from the measured latency (lerobot's own helpers). The chunk is decoded whole and sent. Pair
it with the client's --aggregate_fn_name=latest_only: the overlap is already blended by RTC.
"""

import sys

from .policy_loading import load_bf16, wants_bf16

_POLICY_DTYPE = "auto"  # --policy_dtype, taken off the command line in main()
_OBS_ATOL = 0.0175  # --obs_similarity_atol (rad): ~1 degree, lerobot's intent for its degree-based robots
_RTC: dict = {"enabled": False, "execution_horizon": 10, "max_guidance_weight": 10.0,
              "prefix_attention_schedule": None}  # --rtc*, taken off the command line in main()


class _PassThrough:
    def __call__(self, x):
        return x


def _patch(server_module) -> None:
    from lerobot.configs.policies import PreTrainedConfig

    # bf16: wrap the policy class the server resolves, so its from_pretrained() loads on the CPU first.
    original_get_class = server_module.get_policy_class

    def get_policy_class(policy_type):
        cls = original_get_class(policy_type)
        if not wants_bf16(policy_type, _POLICY_DTYPE):
            return cls

        class _Bf16Loader:
            @staticmethod
            def from_pretrained(path, **kwargs):
                config = PreTrainedConfig.from_pretrained(path)
                return load_bf16(lambda cfg: cls.from_pretrained(path, config=cfg, **kwargs), config)

        return _Bf16Loader

    server_module.get_policy_class = get_policy_class

    # Rename map: an empty override would replace the checkpoint's own map; drop it instead.
    original_processors = server_module.make_pre_post_processors

    def make_pre_post_processors(*args, preprocessor_overrides=None, **kwargs):
        overrides = dict(preprocessor_overrides or {})
        if not (overrides.get("rename_observations_processor") or {}).get("rename_map"):
            overrides.pop("rename_observations_processor", None)
        return original_processors(*args, preprocessor_overrides=overrides, **kwargs)

    server_module.make_pre_post_processors = make_pre_post_processors

    # "Too similar" threshold in this robot's units (radians), not lerobot's default of 1.0 (degrees).
    original_similar = server_module.observations_similar

    def observations_similar(obs1, obs2, lerobot_features, atol=None):
        return original_similar(obs1, obs2, lerobot_features=lerobot_features, atol=_OBS_ATOL)

    server_module.observations_similar = observations_similar

    server_cls = server_module.PolicyServer

    # Image features under the robot's camera names too (via the checkpoint's rename map), for the resize.
    original_image_features = server_cls.policy_image_features

    def policy_image_features(self):
        features = dict(original_image_features.fget(self))
        for step in getattr(getattr(self, "preprocessor", None), "steps", []):
            for robot_key, policy_key in (getattr(step, "rename_map", None) or {}).items():
                if policy_key in features:
                    features.setdefault(robot_key, features[policy_key])
        return features

    server_cls.policy_image_features = property(policy_image_features)

    # Whole-chunk decoding for relative-action policies, and RTC.
    original_predict, original_get_chunk = server_cls._predict_action_chunk, server_cls._get_action_chunk
    original_reset = server_cls._reset_server

    def relative(self) -> bool:
        return bool(getattr(self.policy.config, "use_relative_actions", False))

    def _get_action_chunk(self, observation):
        if _RTC["enabled"]:
            return _rtc_chunk(self, observation)
        chunk = original_get_chunk(self, observation)
        if relative(self):
            chunk = self._openarm_postprocessor(chunk)  # decode the whole chunk against this observation
        return chunk

    def _predict_action_chunk(self, observation_t):
        if not (relative(self) or _RTC["enabled"]):
            return original_predict(self, observation_t)
        self._openarm_obs_timestep = observation_t.get_timestep()
        self._openarm_postprocessor, self.postprocessor = self.postprocessor, _PassThrough()
        try:
            return original_predict(self, observation_t)
        finally:
            self.postprocessor = self._openarm_postprocessor

    def _reset_server(self):
        self._openarm_prev = None  # a new client: no previous chunk to continue
        self._openarm_latencies = []
        return original_reset(self)

    server_cls._reset_server = _reset_server

    server_cls._get_action_chunk = _get_action_chunk
    server_cls._predict_action_chunk = _predict_action_chunk


def _enable_rtc(server) -> None:
    """Switch RTC on in the loaded policy (once), as lerobot-rollout does for --inference.type=rtc."""
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.processor import NormalizerProcessorStep
    from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep
    from lerobot.rollout.inference.rtc import supports_rtc_inference

    policy = server.policy
    if not supports_rtc_inference(policy):
        raise ValueError(f"policy type {policy.config.type!r} does not support RTC")
    kwargs = {"execution_horizon": _RTC["execution_horizon"], "max_guidance_weight": _RTC["max_guidance_weight"]}
    if _RTC["prefix_attention_schedule"]:
        from lerobot.configs import RTCAttentionSchedule

        kwargs["prefix_attention_schedule"] = RTCAttentionSchedule[_RTC["prefix_attention_schedule"].upper()]
    policy.config.rtc_config = RTCConfig(**kwargs)
    if hasattr(policy, "init_rtc_processor"):
        policy.init_rtc_processor()
    steps = server.preprocessor.steps
    server._openarm_relative = next(
        (s for s in steps if isinstance(s, RelativeActionsProcessorStep) and s.enabled), None)
    server._openarm_normalizer = next((s for s in steps if isinstance(s, NormalizerProcessorStep)), None)
    if server._openarm_relative is not None and server._openarm_relative.action_names is None:
        server._openarm_relative.action_names = list(
            getattr(policy.config, "action_feature_names", None) or server.lerobot_features["observation.state"]["names"])
    server._openarm_rtc_ready = True
    server.logger.info(f"RTC on: {policy.config.rtc_config}")


def _rtc_chunk(server, observation):
    """One RTC chunk: guided by the unexecuted tail of the previous chunk, decoded whole."""
    import math
    import time

    import torch
    from lerobot.policies.rtc.relative import reanchor_relative_rtc_prefix
    from lerobot.rollout.inference.rtc import _normalize_prev_actions_length

    if not getattr(server, "_openarm_rtc_ready", False):
        _enable_rtc(server)
    t_obs = server._openarm_obs_timestep
    prev = getattr(server, "_openarm_prev", None)
    tail = tail_abs = None
    if prev is not None:
        first = t_obs - prev["start"] + 1  # chunk row i is the action for timestep start + i
        if 0 <= first < prev["original"].shape[0]:
            tail, tail_abs = prev["original"][first:], prev["processed"][first:]
    if tail is not None and server._openarm_relative is not None:
        state = server._openarm_relative.get_cached_state()  # cached by the preprocessor run just before
        if state is not None:
            tail = reanchor_relative_rtc_prefix(
                prev_actions_absolute=tail_abs, current_state=state, relative_step=server._openarm_relative,
                normalizer_step=server._openarm_normalizer, policy_device=server.device)
    if tail is not None:
        tail = _normalize_prev_actions_length(tail.to(server.device), target_steps=_RTC["execution_horizon"])
    latencies = getattr(server, "_openarm_latencies", [])
    delay = math.ceil(max(latencies) / server.config.environment_dt) if (latencies and tail is not None) else 0

    t0 = time.perf_counter()
    chunk = server.policy.predict_action_chunk(observation, inference_delay=delay, prev_chunk_left_over=tail)
    if chunk.ndim != 3:
        chunk = chunk.unsqueeze(0)
    chunk = chunk[:, : server.actions_per_chunk, :]
    original = chunk.squeeze(0).clone()
    processed = server._openarm_postprocessor(chunk)  # decoded whole, against this observation
    server._openarm_latencies = (latencies + [time.perf_counter() - t0])[-10:]
    server._openarm_prev = {"start": t_obs, "original": original, "processed": processed.squeeze(0).to(original.device)}
    server.logger.info(f"RTC chunk @ timestep {t_obs}: guided by {0 if tail_abs is None else len(tail_abs)} unexecuted"
                       f" actions, inference_delay {delay}")
    return processed


def main() -> None:
    global _POLICY_DTYPE, _OBS_ATOL
    rest = []
    for arg in sys.argv[1:]:  # --policy_dtype and --rtc* are ours; lerobot's parser must not see them
        if arg.startswith("--policy_dtype="):
            _POLICY_DTYPE = arg.split("=", 1)[1]
        elif arg.startswith("--obs_similarity_atol="):
            _OBS_ATOL = float(arg.split("=", 1)[1])
        elif arg.startswith("--rtc="):
            _RTC["enabled"] = arg.split("=", 1)[1].lower() in ("1", "true", "yes")
        elif arg.startswith("--rtc_execution_horizon="):
            _RTC["execution_horizon"] = int(arg.split("=", 1)[1])
        elif arg.startswith("--rtc_max_guidance_weight="):
            _RTC["max_guidance_weight"] = float(arg.split("=", 1)[1])
        elif arg.startswith("--rtc_prefix_attention_schedule="):
            _RTC["prefix_attention_schedule"] = arg.split("=", 1)[1]
        else:
            rest.append(arg)
    sys.argv = sys.argv[:1] + rest
    import lerobot.async_inference.policy_server as server_module

    _patch(server_module)
    server_module.serve()


if __name__ == "__main__":
    main()
