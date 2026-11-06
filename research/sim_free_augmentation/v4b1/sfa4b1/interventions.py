"""v4B.1 training interventions on top of the frozen v4B data view and sampler (research/sim_free_augmentation/v4).

The v4B selection, StratifiedBatches schedule, datasets, normalisation and training loop are reused unchanged; an arm
only adds per-item flags in the dataset view and applies them right after the official preprocessing step:

  synth_rgb = "missing"  : synthetic rows get SmolVLA's native missing-camera representation - pixels set to the
                           empty-camera fill (0 in [0,1] -> -1 after SmolVLA's [-1,1] mapping, exactly what
                           prepare_images() uses for absent cameras) AND observation.images.camera1_padding_mask=False,
                           so no image token of a synthetic row enters attention. Real rows untouched.
  target_mask = "phase"  : synthetic rows' action-target step k is excluded from the loss (OR-ed into action_is_pad)
                           iff its target frame f+k is at or after grasp-end = right_close + 15 (the v3.1/v4B
                           per-target-step phase boundary between grasp and carry), with right_close taken from that
                           synthetic episode's own exported gripper commands. All 16 channels share the time mask.
  target_mask = "random" : budget control. Each synthetic target step is excluded independently with probability
                           p_drop (frozen; = 1 - expected valid fraction of the phase mask under the same schedules),
                           from a fixed hash of (seed, global row index). Phase composition of synthetic supervision
                           is unchanged; only the amount is reduced.
Real rows' action_is_pad is never modified (tested). SmolVLA's loss is sum(valid) / count(valid) over the batch, so
masking moves loss weight toward real targets; that shift is logged per step.

AMENDMENT A1 (2026-11-06, after CODEX_SIM_FREE_V4B1_PRETRAIN_AUDIT.md): the pooled-rate "random" control above matches
only aggregate synthetic-target retention, not active rows, per-batch loss denominators or source loss exposure. It is
SUPERSEDED (kept for the record, never run) by
  target_mask = "matched_random" : for each sampled synthetic row, K = number of originally valid target steps the phase
                           mask retains for that row; exactly K positions are drawn uniformly without replacement
                           from the row's originally valid (non-padded) target steps, seeded by
                           (matched_random_seed, seed, global row index). K = 0 -> zero retained targets. Hence every
                           row's retained count, every batch's active-row set and loss denominator, and every source's
                           direct-target exposure equal PhaseMask's exactly; only WHICH target steps inside the same
                           active contexts are supervised differs. Original padding is a subset of the exclusion.
"""

from __future__ import annotations

import atexit
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
V4 = HERE.parent / "v4"
sys.path.insert(0, str(V4))
sys.path.insert(0, str(HERE.parent))
from sfa4b import mixture as mx  # noqa: E402

CFG = json.loads((HERE / "configs/v4b1.json").read_text())
CAM = "observation.images.camera1"
CHUNK = 50


def right_close_events(ds):
    """{episode_index: right-close frame (dataset frame coordinates)} from the dataset's own action column."""
    from sfa.data import command_events  # noqa: PLC0415
    ep = ds.meta.episodes
    A = np.stack(ds.hf_dataset["action"])
    out = {}
    for e, a, b in zip(ep["episode_index"], ep["dataset_from_index"], ep["dataset_to_index"]):
        out[int(e)] = int(command_events(A[int(a):int(b)])["right_close"])
    return out


def phase_mask(frame_index, grasp_end):
    """True = excluded. Target step k of a row at frame f is excluded iff f + k >= grasp_end."""
    return (frame_index + np.arange(CHUNK)) >= grasp_end


def random_mask(seed, global_index, p_drop):
    """SUPERSEDED by amendment A1 (pooled-rate control); retained for the record only."""
    return np.random.default_rng([CFG["random_mask_seed"], seed, int(global_index)]).random(CHUNK) < p_drop


def valid_targets(frame_index, episode_len):
    """original (pre-intervention) valid target steps: target frame inside the episode (LeRobot pads beyond the end)."""
    return (frame_index + np.arange(CHUNK)) < episode_len


def matched_random_retained(seed, global_index, valid, k):
    """exactly k of the originally valid target steps, uniformly without replacement; deterministic in (seed, row)."""
    ids = np.flatnonzero(valid)
    assert 0 <= k <= len(ids), (k, len(ids))
    keep = np.zeros(CHUNK, bool)
    if k:
        keep[np.random.default_rng([CFG["matched_random_seed"], seed, int(global_index)]).choice(ids, size=k, replace=False)] = True
    return keep


def synthetic_exclusion(kind, seed, global_index, frame_index, episode_len, grasp_end):
    """True = target step excluded from the loss for a synthetic row (OR-ed with the original padding)."""
    if kind == "none":
        return np.zeros(CHUNK, bool)
    if kind == "phase":
        return phase_mask(frame_index, grasp_end)
    if kind == "random":
        return random_mask(seed, global_index, CFG["p_drop"])
    if kind == "matched_random":
        valid = valid_targets(frame_index, episode_len)
        k = int((valid & ~phase_mask(frame_index, grasp_end)).sum())
        return ~matched_random_retained(seed, global_index, valid, k)
    raise ValueError(kind)


def arm_spec(arm):
    return CFG["arms"][arm] if arm in CFG["arms"] else CFG["superseded_arms"][arm]


class TwoPartV41(mx.TwoPart):
    def __init__(self, ds_real, ds_syn, arm, seed):
        super().__init__(ds_real, ds_syn)
        self.spec = arm_spec(arm)
        self.seed = seed
        self.syn_len = {e: b - a for e, (a, b) in mx.episode_ranges(ds_syn.meta).items()}
        self.grasp_end = {e: rc + CFG["grasp_end_offset"] for e, rc in right_close_events(ds_syn).items()} \
            if self.spec["target_mask"] in ("phase", "matched_random") else None

    def __getitem__(self, i):
        item = super().__getitem__(i)
        syn = int(i) >= self.nR
        m = np.zeros(CHUNK, bool)
        if syn and self.spec["target_mask"] != "none":
            f, e = int(item["frame_index"]), int(item["episode_index"])
            # fail closed if the dataset's own padding disagrees with the episode-length rule used by the masks
            assert np.array_equal(~item["action_is_pad"].numpy(), valid_targets(f, self.syn_len[e])), (i, f, e)
            m = synthetic_exclusion(self.spec["target_mask"], self.seed, i, f, self.syn_len[e],
                                    self.grasp_end[e] if self.grasp_end else None)
        item["sfa_is_synth"] = torch.tensor(syn)
        item["sfa_target_mask"] = torch.from_numpy(m)
        item["sfa_index"] = torch.tensor(int(i))
        return item


class Applier:
    """Wraps lerobot_train._preprocess_dataset_batch; logs per-step loss-budget bookkeeping."""

    def __init__(self, arm, orig, log_path):
        self.spec, self.orig, self.log_path, self.rows = arm_spec(arm), orig, Path(log_path), []
        self.rec_index, self.rec_valid, self.rec_syn = [], [], []
        atexit.register(self.dump)

    def __call__(self, batch, camera_keys, rename_map, preprocessor):
        is_syn = batch.pop("sfa_is_synth")
        tmask = batch.pop("sfa_target_mask")
        index = batch.pop("sfa_index", None)
        pad_before = batch["action_is_pad"].detach().clone().cpu()
        out = self.orig(batch, camera_keys, rename_map, preprocessor)
        dev = out["observation.state"].device
        is_syn, tmask = is_syn.to(dev).bool(), tmask.to(dev).bool()
        if self.spec["synth_rgb"] == "missing":
            keep = (~is_syn).to(out[CAM].dtype).view(-1, 1, 1, 1)
            out[CAM] = out[CAM] * keep                                   # synthetic rows -> empty-camera fill
            out[f"{CAM}_padding_mask"] = ~is_syn
        if self.spec["target_mask"] != "none":
            out["action_is_pad"] = out["action_is_pad"] | (tmask & is_syn[:, None])
        pad = out["action_is_pad"].detach().cpu()
        syn = is_syn.detach().cpu()
        self.rows.append({"real_rows": int((~syn).sum()), "syn_rows": int(syn.sum()),
                          "valid_real_steps": int((~pad[~syn]).sum()), "valid_syn_steps": int((~pad[syn]).sum()),
                          "real_pad_changed": bool((pad[~syn] != pad_before[~syn]).any()),
                          "syn_rows_fully_masked": int(pad[syn].all(1).sum()),
                          "loss_denominator_elements": int((~pad).sum()) * int(out["action"].shape[-1] if "action" in out else 16)})
        # actual applied supervision, recorded for post-hoc verification against the frozen schedule (A1)
        self.rec_index.append(index.detach().cpu().numpy().astype(np.int64) if index is not None else np.full(len(syn), -1))
        self.rec_valid.append(np.packbits((~pad).numpy(), axis=-1))
        self.rec_syn.append(syn.numpy())
        return out

    def dump(self):
        if not self.rows:
            return
        v_r = np.array([r["valid_real_steps"] for r in self.rows])
        v_s = np.array([r["valid_syn_steps"] for r in self.rows])
        summ = {"steps": len(self.rows), "valid_real_steps_mean": float(v_r.mean()), "valid_syn_steps_mean": float(v_s.mean()),
                "synthetic_loss_share_mean": float(np.mean(v_s / np.maximum(v_r + v_s, 1))),
                "syn_rows_fully_masked_mean": float(np.mean([r["syn_rows_fully_masked"] for r in self.rows])),
                "real_pad_ever_changed": bool(any(r["real_pad_changed"] for r in self.rows))}
        self.log_path.write_text(json.dumps({"summary": summ, "per_step": self.rows}))
        np.savez_compressed(self.log_path.with_name(self.log_path.name.replace("_budget.json", "_masks.npz")),
                            index=np.stack(self.rec_index), valid_packed=np.stack(self.rec_valid), is_syn=np.stack(self.rec_syn))
