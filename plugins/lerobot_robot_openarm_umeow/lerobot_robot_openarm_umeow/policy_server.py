"""lerobot's async-inference policy server, with the fixes our SmolVLA and GR00T N1.7 checkpoints need.

    python -m lerobot_robot_openarm_umeow.policy_server --host=127.0.0.1 --port=8080 [--policy_dtype=auto]

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
"""

import sys

from .policy_loading import load_bf16, wants_bf16

_POLICY_DTYPE = "auto"  # --policy_dtype, taken off the command line in main()


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

    # Whole-chunk decoding for relative-action policies.
    original_predict, original_get_chunk = server_cls._predict_action_chunk, server_cls._get_action_chunk

    def relative(self) -> bool:
        return bool(getattr(self.policy.config, "use_relative_actions", False))

    def _get_action_chunk(self, observation):
        chunk = original_get_chunk(self, observation)
        if relative(self):
            chunk = self._openarm_postprocessor(chunk)  # decode the whole chunk against this observation
        return chunk

    def _predict_action_chunk(self, observation_t):
        if not relative(self):
            return original_predict(self, observation_t)
        self._openarm_postprocessor, self.postprocessor = self.postprocessor, _PassThrough()
        try:
            return original_predict(self, observation_t)
        finally:
            self.postprocessor = self._openarm_postprocessor

    server_cls._get_action_chunk = _get_action_chunk
    server_cls._predict_action_chunk = _predict_action_chunk


def main() -> None:
    global _POLICY_DTYPE
    rest = []
    for arg in sys.argv[1:]:  # --policy_dtype is ours; lerobot's parser must not see it
        if arg.startswith("--policy_dtype="):
            _POLICY_DTYPE = arg.split("=", 1)[1]
        else:
            rest.append(arg)
    sys.argv = sys.argv[:1] + rest
    import lerobot.async_inference.policy_server as server_module

    _patch(server_module)
    server_module.serve()


if __name__ == "__main__":
    main()
