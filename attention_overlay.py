"""SmolVLA attention dumps: what the VLM grounded, and what the action expert actually used.

WHY BOTH MAPS
-------------
SmolVLA's prefix holds three kinds of token -- camera patches, the task string, and the robot
state -- and two different attention stages run over them:

  1. VLM prefix self-attention. The language and state tokens attend over the image patches inside
     the SmolVLM layers. This is the PERCEPTION side: what the vision-language model grounded the
     instruction to.
  2. Action-expert cross-attention. The action queries attend over that whole prefix. This is the
     CONTROL side: what the chunk of motion was actually conditioned on.

They can disagree, and the disagreement is the interesting part. A policy that executes a
memorised trajectory regardless of the scene typically still shows a plausible VLM map -- the VLM
does find the object, or finds where it usually is -- while the expert puts most of its mass on
the STATE token and spreads what little it gives the images almost uniformly. In that regime the
cameras are barely steering anything, so removing the target object from the table changes the
policy's output very little. The per-modality shares and the spatial-entropy figures written
alongside each dump are there to make that case decidable rather than a matter of impression.

WHAT IS MEASURED, AND HOW
-------------------------
The lerobot installed here is stock (no attn_records fork), so this attaches to it with instance-
level hooks that shadow bound methods; detach() restores the model exactly.

  * SmolVLAPolicy.prepare_images        -- image keys in prefix order, each one's pre-resize size,
                                           and the padded tensors the model actually saw.
  * SmolVLMWithExpertModel.embed_image  -- tokens produced per image, so the prefix layout is
                                           measured rather than assumed.
  * ...embed_language_tokens            -- language token count, which fixes where the language
                                           block ends and the state token begins.
  * ...eager_attention_forward          -- both attention stages. get_attention_interface() always
                                           returns the eager path, so the probabilities really are
                                           computed; the wrapper recomputes them (same GQA
                                           expansion, same scaling, same masking) and reduces them
                                           to a head-averaged [Q, K] matrix. Prefix self-attention
                                           and expert cross-attention are told apart by shape:
                                           self-attention has query length == key length, the
                                           expert's action queries attending to the prefix do not.

Only the LAST layer's matrices are kept, matching the layer the reference implementation reads.

ALIGNMENT
---------
Displayed images are the model's own preprocessed tensors, so a heatmap cannot be misaligned by a
resize assumption. resize_with_pad() fits the frame in 512x512 preserving aspect ratio and pads on
the LEFT and TOP (128 black rows for a 640x480 frame); image and heatmap are cropped to the same
valid region, so padding never reaches the output.

RESOLUTION AND NORMALISATION
----------------------------
Each camera contributes 64 prefix tokens -- an 8x8 grid over the padded 512x512 input. Maps are
upsampled from that, so they localise to roughly a 64-pixel block of the original frame: read them
as regions, never as pixels. Each panel is contrast-normalised independently (2nd-98th
percentile), so brightness compares only WITHIN a panel. The numbers printed on each panel are
what compare panels against each other.

COST
----
A re-plan forward pass went 90ms -> 113ms with capture on (2026-08-21, this checkpoint, warm), and
steps that pop a cached action are untouched because no forward pass happens on them.
"""

import csv
import math
import os

import numpy as np
import torch

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class AttentionDump:
    """Captures both attention stages and writes them to disk once per policy re-plan.

    Every failure path disables capture and records the reason rather than raising: this runs
    inside a loop driving real hardware, and a visualisation must never be able to stop it.
    """

    PANEL = (320, 240)
    ALPHA = 0.5

    def __init__(self, policy, out_dir, key_labels=None):
        self.policy = policy
        self.out_dir = out_dir
        self.key_labels = key_labels or {}
        self.enabled = False
        self.error = None
        self.chunks = 0
        self.rows = []

        self._img_keys, self._img_sizes, self._img_tensors = [], [], []
        self._token_counts, self._n_lang = [], 0
        self._expert = None      # [Q, K] head-averaged, action queries over the prefix
        self._vlm = None         # [K, K] head-averaged, prefix self-attention

        try:
            self._vlm_model = policy.model.vlm_with_expert
            self._o_prepare = policy.prepare_images
            self._o_embed_img = self._vlm_model.embed_image
            self._o_embed_lang = self._vlm_model.embed_language_tokens
            self._o_eager = self._vlm_model.eager_attention_forward

            policy.prepare_images = self._h_prepare
            self._vlm_model.embed_image = self._h_embed_img
            self._vlm_model.embed_language_tokens = self._h_embed_lang
            self._vlm_model.eager_attention_forward = self._h_eager

            os.makedirs(out_dir, exist_ok=True)
            self.enabled = True
        except Exception as e:
            self.error = f"could not attach to the policy: {e}"

    def detach(self):
        if not self.enabled:
            return
        self.policy.prepare_images = self._o_prepare
        self._vlm_model.embed_image = self._o_embed_img
        self._vlm_model.embed_language_tokens = self._o_embed_lang
        self._vlm_model.eager_attention_forward = self._o_eager
        self.enabled = False

    def _fail(self, msg):
        self.error = msg
        self.detach()

    # ------------------------------------------------------------------ hooks
    def _h_prepare(self, batch):
        images, masks = self._o_prepare(batch)
        try:
            keys = [k for k in self.policy.config.image_features if k in batch]
            self._img_keys = keys
            self._img_sizes = []
            for k in keys:
                t = batch[k]
                t = t[:, -1] if t.ndim == 5 else t
                self._img_sizes.append((int(t.shape[-2]), int(t.shape[-1])))
            self._img_tensors = [im.detach()[0].float().cpu() for im in images]
            self._token_counts = []      # refilled during embed_prefix, which runs next
            self._n_lang = 0
        except Exception as e:
            self._fail(f"prepare_images hook failed: {e}")
        return images, masks

    def _h_embed_img(self, image):
        out = self._o_embed_img(image)
        try:
            self._token_counts.append(int(out.shape[1]))
        except Exception as e:
            self._fail(f"embed_image hook failed: {e}")
        return out

    def _h_embed_lang(self, tokens):
        out = self._o_embed_lang(tokens)
        try:
            # embed_prefix calls this for the image special tokens too when they are enabled; the
            # task string is always the last call, which is the one that sizes the language block.
            self._n_lang = int(out.shape[-2])
        except Exception as e:
            self._fail(f"embed_language_tokens hook failed: {e}")
        return out

    def _h_eager(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):
        out = self._o_eager(attention_mask, batch_size, head_dim, query_states, key_states, value_states)
        try:
            probs = self._probs(attention_mask, batch_size, head_dim, query_states, key_states)
            if query_states.shape[1] == key_states.shape[1]:
                self._vlm = probs        # prefix self-attention
            else:
                self._expert = probs     # action expert cross-attention
        except Exception as e:
            self._fail(f"attention capture failed: {e}")
        return out

    def _probs(self, attention_mask, batch_size, head_dim, query_states, key_states):
        """Mirror eager_attention_forward's own maths, reduced to a head-averaged [Q, K] matrix."""
        n_heads = self._vlm_model.num_attention_heads
        n_kv = self._vlm_model.num_key_value_heads
        groups = n_heads // n_kv
        seq = key_states.shape[1]

        k = key_states[:, :, :, None, :].expand(batch_size, seq, n_kv, groups, head_dim)
        k = k.reshape(batch_size, seq, n_kv * groups, head_dim)

        q = query_states.to(torch.float32).transpose(1, 2)
        k = k.to(torch.float32).transpose(1, 2)
        w = torch.matmul(q, k.transpose(2, 3)) * head_dim**-0.5
        w = torch.where(attention_mask[:, None, :, :], w, torch.finfo(w.dtype).min)
        return torch.softmax(w, dim=-1).mean(dim=1)[0].detach().float().cpu()   # [Q, K]

    # ------------------------------------------------------------------ layout
    def _layout(self, k_len):
        """Column ranges for each image, the language block and the state block."""
        special = bool(getattr(self.policy.config, "add_image_special_tokens", False))
        segs, start = [], 0
        for count in self._token_counts:
            if special:
                start += 1
            segs.append((start, start + count))
            start += count
            if special:
                start += 1
        lang = (start, min(start + self._n_lang, k_len))
        state = (lang[1], k_len)      # state token, plus any prefix padding
        return segs, lang, state

    @staticmethod
    def _entropy(v):
        """Spatial entropy of one camera's attention, normalised so 1.0 == perfectly uniform.

        Kept in the CSV because it is the conventional figure, but NOT used for any verdict: over
        64 tokens it is a badly conditioned discriminator. A map with obvious hot and cold regions
        still scores ~0.98, so any threshold near 1.0 fires on maps that plainly are structured.
        _peak_mass below is what the readouts use instead.
        """
        p = v / (v.sum() + 1e-12)
        h = float(-(p * torch.log(p + 1e-12)).sum())
        return h / math.log(len(p)) if len(p) > 1 else 0.0

    @staticmethod
    def _peak_mass(v, frac=0.125):
        """Share of this camera's attention falling on its top `frac` of patches.

        Directly interpretable against a known baseline: with attention spread perfectly evenly the
        top 12.5% of patches hold 12.5% of the mass, so the ratio peak_mass/frac is "how many times
        more than chance the busiest patches receive". 1.0x is no spatial preference whatsoever;
        the larger it gets, the more the policy is keying on specific regions rather than on the
        image as a whole.
        """
        k = max(1, int(round(frac * len(v))))
        p = v / (v.sum() + 1e-12)
        return float(torch.topk(p, k).values.sum())

    # ------------------------------------------------------------------ rendering
    @staticmethod
    def _to_bgr(t):
        img = ((t.permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _panel(self, idx, heat_1d, caption):
        base = self._to_bgr(self._img_tensors[idx])
        hp, wp = base.shape[:2]
        oh, ow = self._img_sizes[idx]
        ratio = max(ow / wp, oh / hp)
        top = max(0, hp - int(oh / ratio))       # resize_with_pad pads left and top
        left = max(0, wp - int(ow / ratio))

        n = heat_1d.numel()
        side = int(round(n**0.5))
        if side * side == n:
            hm = cv2.resize(heat_1d.reshape(side, side).numpy(), (wp, hp),
                            interpolation=cv2.INTER_CUBIC)[top:, left:]
            lo, hi = np.percentile(hm, [2, 98])
            hm = np.clip((hm - lo) / (hi - lo + 1e-6), 0, 1)
            colour = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
            out = cv2.addWeighted(base[top:, left:], 1.0 - self.ALPHA, colour, self.ALPHA, 0)
        else:
            out = base[top:, left:]

        out = cv2.resize(out, self.PANEL)
        for colour, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(out, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, thick, cv2.LINE_AA)
        return out

    @staticmethod
    def _banner(width, lines, height=None):
        height = height or (18 * len(lines) + 10)
        strip = np.zeros((height, width, 3), np.uint8)
        for i, (text, colour) in enumerate(lines):
            cv2.putText(strip, text, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        colour, 1, cv2.LINE_AA)
        return strip

    def dump(self, step, t_s):
        """Write one PNG for this chunk and record its numbers. Returns the path, or None."""
        if not self.enabled or cv2 is None or self._expert is None or not self._token_counts:
            return None
        try:
            k_len = self._expert.shape[1]
            segs, lang, state = self._layout(k_len)
            if len(segs) != len(self._img_tensors):
                return None

            # ---- action expert: mean over its action queries, then split by modality
            e = self._expert.mean(dim=0)                       # [K]
            e_img = [float(e[a:b].sum()) for a, b in segs]
            e_lang = float(e[lang[0]:lang[1]].sum())
            e_state = float(e[state[0]:state[1]].sum())
            e_total = sum(e_img) + e_lang + e_state + 1e-12

            # ---- VLM: what the language and state tokens attend to in the images. Restricting
            # the queries to the non-image tokens is what makes this the grounding signal rather
            # than image patches attending among themselves.
            v_rows = self._vlm[lang[0]:k_len] if self._vlm is not None and self._vlm.shape[0] >= k_len else None
            v = v_rows.mean(dim=0) if v_rows is not None and v_rows.numel() else None
            v_img = [float(v[a:b].sum()) for a, b in segs] if v is not None else [0.0] * len(segs)
            v_total = sum(v_img) + 1e-12

            rows, row = [], {"chunk": self.chunks, "step": step, "t_s": round(t_s, 2),
                             "expert_image_share": round(sum(e_img) / e_total, 4),
                             "expert_lang_share": round(e_lang / e_total, 4),
                             "expert_state_share": round(e_state / e_total, 4)}

            expert_panels, vlm_panels = [], []
            for i, (a, b) in enumerate(segs):
                key = self._img_keys[i] if i < len(self._img_keys) else f"image{i}"
                name = self.key_labels.get(key, key.removeprefix("observation.images."))

                e_peak = self._peak_mass(e[a:b])
                expert_panels.append(self._panel(
                    i, e[a:b], f"{name}  {100*e_img[i]/e_total:.0f}% of prefix  "
                               f"peak {e_peak/0.125:.1f}x"))
                row[f"expert_share_{name}"] = round(e_img[i] / e_total, 4)
                row[f"expert_peak_x_{name}"] = round(e_peak / 0.125, 3)
                row[f"expert_entropy_{name}"] = round(self._entropy(e[a:b]), 4)

                if v is not None:
                    v_peak = self._peak_mass(v[a:b])
                    vlm_panels.append(self._panel(
                        i, v[a:b], f"{name}  {100*v_img[i]/v_total:.0f}% of images  "
                                   f"peak {v_peak/0.125:.1f}x"))
                    row[f"vlm_share_{name}"] = round(v_img[i] / v_total, 4)
                    row[f"vlm_peak_x_{name}"] = round(v_peak / 0.125, 3)
                    row[f"vlm_entropy_{name}"] = round(self._entropy(v[a:b]), 4)

            blocks = []
            width = self.PANEL[0] * max(1, len(expert_panels))
            blocks.append(self._banner(width, [
                (f"chunk {self.chunks}  step {step}  t={t_s:.1f}s", (255, 255, 255)),
                (f"ACTION EXPERT attention over the prefix:  images "
                 f"{100*sum(e_img)/e_total:.1f}%   task text {100*e_lang/e_total:.1f}%   "
                 f"robot state {100*e_state/e_total:.1f}%", (0, 220, 255)),
            ]))
            blocks.append(np.hstack(expert_panels))
            if vlm_panels:
                blocks.append(self._banner(width, [
                    ("VLM prefix self-attention (task+state queries -> image patches): "
                     "what perception grounded", (0, 255, 120)),
                ], height=28))
                blocks.append(np.hstack(vlm_panels))

            path = os.path.join(self.out_dir, f"chunk_{self.chunks:04d}_step{step:05d}.png")
            cv2.imwrite(path, np.vstack(blocks))
            self.rows.append(row)
            self.chunks += 1
            return path
        except Exception as e:
            self._fail(f"dump failed: {e}")
            return None

    # ------------------------------------------------------------------ summary
    def write_csv(self):
        if not self.rows:
            return None
        path = os.path.join(self.out_dir, "attention_log.csv")
        fields = list(self.rows[0].keys())
        for r in self.rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(self.rows)
        return path

    def summary_lines(self):
        """Aggregate the run into the numbers that bear on 'is this policy using its cameras?'."""
        if not self.rows:
            return ["  no attention was captured (the policy never re-planned)."]

        def mean(key):
            vals = [r[key] for r in self.rows if key in r]
            return sum(vals) / len(vals) if vals else float("nan")

        img, lang, state = (mean("expert_image_share"), mean("expert_lang_share"),
                            mean("expert_state_share"))
        peak_keys = [k for k in self.rows[0] if k.startswith("expert_peak_x_")]
        peaks = {k.removeprefix("expert_peak_x_"): mean(k) for k in peak_keys}
        flattest = min(peaks.values()) if peaks else float("nan")

        out = [
            f"  {len(self.rows)} chunks captured -> {self.out_dir}",
            f"  action expert attention, averaged over the run: "
            f"images {100*img:.1f}%  task text {100*lang:.1f}%  robot state {100*state:.1f}%",
            "  per-camera spatial focus, as a multiple of chance (1.0x = attention spread evenly "
            "over the patches, so the image content localises nothing):",
        ] + [f"      {name:<18} {p:.2f}x" for name, p in peaks.items()]

        # Deliberately phrased as evidence, not a verdict: attention is not causation, and the
        # controlled test below is what would actually settle it.
        if state > img:
            out.append(
                f"  READ: the expert put more attention on the robot's own state ({100*state:.1f}%) "
                f"than on all three cameras combined ({100*img:.1f}%). That is consistent with a "
                f"policy\n        driving from proprioception -- which would explain a trajectory "
                f"that runs whether or not the target object is present.")
        if flattest < 1.6:
            out.append(
                f"  READ: the flattest camera concentrates only {flattest:.2f}x chance on its "
                f"busiest patches, so within that camera the expert is barely preferring any "
                f"region --\n        the share it gives that image carries little spatial "
                f"information.")
        out.append(
            "  These are correlations, not proof. The decisive test is a controlled pair: record "
            "one run with the object present and one with it absent, then compare\n"
            "        both these numbers AND the commanded joint trajectories. If the trajectories "
            "match closely, the cameras are not steering the policy.")
        return out
