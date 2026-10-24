"""python -m unittest discover -s tools/dataset_diagnostics/tests -t .   (from the lerobot_openarm root)"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # tests never touch the network

from tools.dataset_diagnostics.checkpoint import compare_checkpoint, load_checkpoint_meta  # noqa: E402
from tools.dataset_diagnostics.checks import FAIL, PASS, UNKNOWN, WARN  # noqa: E402
from tools.dataset_diagnostics.compat import compare_datasets  # noqa: E402
from tools.dataset_diagnostics.coverage import (WorkspaceConfig, analyze_coverage, kernel_effective_n,  # noqa: E402
                                                near_duplicate_groups)
from tools.dataset_diagnostics.loader import load_dataset  # noqa: E402
from tools.dataset_diagnostics.positions import (AUTO, HUMAN, RECORDED, UNKNOWN_SRC, ImageToWorkspace,  # noqa: E402
                                                 Position, TransformError, fit_homography, fit_validated_transform,
                                                 merge_by_priority, read_annotations, write_annotations)
from tools.dataset_diagnostics.report import coverage_checks  # noqa: E402
from tools.dataset_diagnostics.technical import run_technical_checks  # noqa: E402
from tools.dataset_diagnostics.tests import fixtures as fx  # noqa: E402


def by_name(checks):
    return {c.name: c for c in checks}


def metric_cfg(**kw):
    d = {"name": "t", "frame_type": "metric", "frame_id": "world", "units": "m", "x_range": [0.0, 0.4],
         "y_range": [-0.2, 0.2], "cell_size": 0.02, "coverage_radius": 0.05, "duplicate_radius": 0.01,
         "min_separation": 0.05}
    d.update(kw)
    return WorkspaceConfig.from_json(d)


class Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class TestSchemasAndTechnical(Tmp):
    def test_clean_v30_passes(self):
        view = load_dataset(root=fx.write_v30(self.tmp / "ds"))
        c = by_name(run_technical_checks(view))
        for name in ("feature_schema", "finite_state_action", "empty_episodes", "missing_frames", "timestamps",
                     "camera_streams", "stats_observation.state", "stats_action", "total_episodes"):
            self.assertEqual(c[name].status, PASS, f"{name}: {c[name].detail}")

    def test_v21_is_read_and_flagged_legacy(self):
        view = load_dataset(root=fx.write_v21(self.tmp / "v21"))
        self.assertEqual(view.version, "v2.1")
        self.assertEqual(view.data["episode_index"].nunique(), 2)
        c = by_name(run_technical_checks(view))
        self.assertEqual(c["codebase_version"].status, WARN)
        self.assertEqual(c["stats_observation.state"].status, PASS)  # pooled from episodes_stats.jsonl

    def test_unsupported_version_rejected(self):
        root = fx.write_v30(self.tmp / "ds")
        info = json.loads((root / "meta/info.json").read_text())
        info["codebase_version"] = "v1.6"
        (root / "meta/info.json").write_text(json.dumps(info))
        with self.assertRaises(ValueError):
            load_dataset(root=root)

    def test_empty_episode_fails(self):
        view = load_dataset(root=fx.write_v30(self.tmp / "ds", lengths=(50, 0, 50)))
        c = by_name(run_technical_checks(view))
        self.assertEqual(c["empty_episodes"].status, FAIL)
        self.assertIn(1, c["empty_episodes"].evidence["episodes"])

    def test_missing_frames_and_bad_timestamps_fail(self):
        def drop(df, eps, info, stats):
            df = df.drop(index=df.index[(df.episode_index == 0) & (df.frame_index == 10)])
            df["timestamp"] = df["timestamp"].astype("float64")
            df.loc[df.episode_index == 1, "timestamp"] = df.loc[df.episode_index == 1, "frame_index"] / 20.0
            return df, eps, info, stats

        view = load_dataset(root=fx.write_v30(self.tmp / "ds", mutate=drop))
        c = by_name(run_technical_checks(view))
        self.assertEqual(c["missing_frames"].status, FAIL)
        self.assertEqual(c["timestamps"].status, FAIL)
        self.assertEqual(c["episode_lengths_match_metadata"].status, FAIL)

    def test_nonfinite_values_fail(self):
        def nan(df, eps, info, stats):
            a = df.at[5, "action"].copy()
            a[3] = np.nan
            df.at[5, "action"] = a
            return df, eps, info, stats

        c = by_name(run_technical_checks(load_dataset(root=fx.write_v30(self.tmp / "ds", mutate=nan))))
        self.assertEqual(c["finite_state_action"].status, FAIL)

    def test_stale_stats_fail(self):
        def stale(df, eps, info, stats):
            stats["observation.state"]["mean"] = (np.asarray(stats["observation.state"]["mean"]) + 0.5).tolist()
            return df, eps, info, stats

        c = by_name(run_technical_checks(load_dataset(root=fx.write_v30(self.tmp / "ds", mutate=stale))))
        self.assertEqual(c["stats_observation.state"].status, FAIL)

    def test_varying_schema_without_state(self):
        def no_state(df, eps, info, stats):
            info["features"].pop("observation.state")
            stats.pop("observation.state")
            return df.drop(columns=["observation.state"]), eps, info, stats

        c = by_name(run_technical_checks(load_dataset(root=fx.write_v30(self.tmp / "ds", mutate=no_state))))
        self.assertEqual(c["feature_schema"].status, FAIL)
        self.assertIn("observation.state", c["feature_schema"].detail)

    def test_shape_mismatch_between_info_and_parquet(self):
        def widen(df, eps, info, stats):
            info["features"]["action"]["shape"] = [18]
            return df, eps, info, stats

        c = by_name(run_technical_checks(load_dataset(root=fx.write_v30(self.tmp / "ds", mutate=widen))))
        self.assertEqual(c["feature_schema"].status, FAIL)

    def test_missing_camera_stream_fails(self):
        view = load_dataset(root=fx.write_v30(self.tmp / "ds", make_videos=False))
        self.assertEqual(by_name(run_technical_checks(view))["camera_streams"].status, FAIL)

    def test_camera_metadata_missing_columns_fails(self):
        def strip(df, eps, info, stats):
            return df, eps.drop(columns=[c for c in eps.columns if c.startswith("videos/")]), info, stats

        view = load_dataset(root=fx.write_v30(self.tmp / "ds", mutate=strip))
        self.assertEqual(by_name(run_technical_checks(view))["camera_streams"].status, FAIL)

    def test_source_dataset_is_not_modified(self):
        from tools.dataset_diagnostics.cli import run

        root = fx.write_v30(self.tmp / "ds")
        before = fx.tree_digest(root)
        poses = self.tmp / "poses.csv"
        poses.write_text("episode_index,x,y,frame_id\n0,0.1,0.0,world\n1,0.2,0.05,world\n2,0.3,-0.05,world\n")
        ws = self.tmp / "ws.json"
        ws.write_text(json.dumps(metric_cfg().__dict__ | {"x_range": [0, 0.4], "y_range": [-0.2, 0.2]}))
        out = self.tmp / "out"
        run({"output": str(out), "datasets": [{"label": "syn", "root": str(root), "object_pose_source":
                                                [f"recorded:{poses}"], "workspace_config": [str(ws)]}]})
        self.assertEqual(before, fx.tree_digest(root))
        for f in ("dataset_report.html", "readiness.json", "provenance.json", "syn/episode_quality.csv",
                  "syn/position_annotations.csv", "syn/recommended_positions.csv", "syn/position_coverage_t.png",
                  "syn/position_density_t.png", "syn/nearest_demo_distance_t.png", "syn/readiness.json"):
            self.assertTrue((out / f).exists(), f)
        ready = json.loads((out / "syn" / "readiness.json").read_text())
        self.assertEqual(ready["quality"]["status"], UNKNOWN)  # no adapter -> not silently PASS


class TestCheckpoint(Tmp):
    def setUp(self):
        super().setUp()
        self.view = load_dataset(root=fx.write_v30(self.tmp / "ds"))

    def check(self, **kw):
        meta = load_checkpoint_meta(str(fx.write_checkpoint(self.tmp / "ck", **kw)))
        return by_name(compare_checkpoint(self.view, meta))

    def test_stale_declared_state_is_warn_not_fail(self):
        c = self.check(state_decl=6, state_norm=16)
        self.assertEqual(c["ckpt_state_dim"].status, WARN)
        self.assertIn("stale", c["ckpt_state_dim"].detail)

    def test_state_wider_than_max_state_dim_fails(self):
        c = self.check(state_decl=16, state_norm=16, max_state_dim=8)
        self.assertEqual(c["ckpt_state_dim"].status, FAIL)

    def test_normalizer_trained_on_other_state_fails(self):
        c = self.check(state_decl=14, state_norm=14)
        self.assertEqual(c["ckpt_state_dim"].status, FAIL)

    def test_action_dim_mismatch_fails(self):
        self.assertEqual(self.check(action=14)["ckpt_action_dim"].status, FAIL)

    def test_matching_checkpoint_passes(self):
        c = self.check(state_decl=16, state_norm=16)
        self.assertEqual(c["ckpt_state_dim"].status, PASS)
        self.assertEqual(c["ckpt_cameras"].status, PASS)

    def test_checkpoint_camera_missing_in_dataset_fails(self):
        c = self.check(state_decl=16, cameras=("observation.images.camera3", "observation.images.camera1"))
        self.assertEqual(c["ckpt_cameras"].status, FAIL)


class TestPositionsAndTransforms(Tmp):
    def test_missing_object_poses_are_unknown(self):
        best = merge_by_priority({}, episodes=[0, 1, 2])
        self.assertTrue(all(p.source == UNKNOWN_SRC for p in best.values()))
        cov = analyze_coverage({}, metric_cfg())
        checks = coverage_checks(cov, {}, [0, 1, 2], {}, provisional=False)
        self.assertEqual(checks[0].status, UNKNOWN)

    def test_joint_states_are_not_positions(self):
        # a dataset alone never yields positions: only declared sources do
        from tools.dataset_diagnostics.cli import collect_positions

        view = load_dataset(root=fx.write_v30(self.tmp / "ds"))
        sources, _, _ = collect_positions(view, ["none"], self.tmp)
        self.assertEqual(sources, {})

    def test_priority_recorded_beats_human_beats_auto(self):
        rec = {0: Position(0, RECORDED, x=0.1, y=0.0, frame_id="world")}
        hum = {0: Position(0, HUMAN, u=1, v=2, camera="c"), 1: Position(1, HUMAN, u=3, v=4, camera="c")}
        auto = {1: Position(1, AUTO, u=5, v=6, camera="c"), 2: Position(2, AUTO, u=7, v=8, camera="c")}
        best = merge_by_priority(auto, hum, rec, episodes=[0, 1, 2, 3])
        self.assertEqual([best[e].source for e in range(4)], [RECORDED, HUMAN, AUTO, UNKNOWN_SRC])

    def test_homography_needs_four_non_degenerate_points(self):
        with self.assertRaises(TransformError):
            fit_homography(np.array([[0, 0], [1, 0], [0, 1]]), np.zeros((3, 2)))
        line = np.c_[np.arange(6), np.arange(6)]
        with self.assertRaises(TransformError):
            fit_homography(line, line * 0.01)

    def test_exact_homography_is_validated(self):
        rng = np.random.default_rng(0)
        H = np.array([[0.001, 0.0002, -0.1], [0.0001, -0.0012, 0.3], [1e-6, 2e-6, 1.0]])
        uv = rng.uniform(0, 640, (10, 2))
        p = np.c_[uv, np.ones(10)] @ H.T
        xy = p[:, :2] / p[:, 2:]
        tr = fit_validated_transform(uv, xy, camera="c", frame_id="world", max_allowed_m=0.005)
        self.assertTrue(tr.valid)
        self.assertLess(tr.heldout_max_m, 1e-6)

    def test_noisy_or_small_transform_is_refused(self):
        rng = np.random.default_rng(1)
        uv = rng.uniform(0, 640, (8, 2))
        xy = rng.uniform(0, 0.4, (8, 2))  # no relation to uv at all
        tr = fit_validated_transform(uv, xy, camera="c", frame_id="world", max_allowed_m=0.02)
        self.assertFalse(tr.valid)
        with self.assertRaises(TransformError):
            tr.map({0: Position(0, HUMAN, u=10, v=10, camera="c")})
        tr4 = fit_validated_transform(uv[:5], uv[:5] * 0.001, camera="c", frame_id="world")
        self.assertFalse(tr4.valid)  # < 6 points is never enough evidence

    def test_transform_json_without_evidence_is_refused(self):
        with self.assertRaises(TransformError):
            ImageToWorkspace.from_json({"camera": "c", "frame_id": "w", "H": np.eye(3).tolist()})
        bad = {"camera": "c", "frame_id": "w", "H": np.zeros((3, 3)).tolist(), "n_points": 10,
               "heldout_rmse_m": 0.001, "heldout_max_m": 0.002, "max_allowed_m": 0.02}
        with self.assertRaises(TransformError):
            ImageToWorkspace.from_json(bad)

    def test_transform_rejects_other_image_size(self):
        rng = np.random.default_rng(2)
        uv = rng.uniform(0, 640, (8, 2))
        tr = fit_validated_transform(uv, uv * 0.001, camera="c", frame_id="w", image_size=(640, 480))
        with self.assertRaises(TransformError):
            tr.map({0: Position(0, HUMAN, u=1, v=1, camera="c", image_size=(320, 240))})

    def test_annotation_roundtrip_and_confirmed_proposals(self):
        path = self.tmp / "ann.csv"
        write_annotations(path, "ds", {0: Position(0, AUTO, u=10, v=20, camera="c", image_size=(64, 48)),
                                       1: Position(1, HUMAN, u=30, v=40, camera="c", verified=True)})
        got = read_annotations(path, dataset="ds")
        self.assertEqual(got[0].source, AUTO)
        self.assertFalse(got[0].verified)
        self.assertEqual(got[1].source, HUMAN)
        self.assertEqual(read_annotations(path, dataset="other"), {})


class TestCoverage(unittest.TestCase):
    def test_near_duplicates_grouped(self):
        pts = np.array([[0.1, 0.0], [0.105, 0.0], [0.3, 0.1], [0.1, 0.004]])
        groups = near_duplicate_groups(pts, 0.01)
        self.assertEqual(sorted(map(sorted, groups)), [[0, 1, 3], [2]])
        self.assertAlmostEqual(kernel_effective_n(np.zeros((5, 2)), 0.05), 1.0)
        self.assertAlmostEqual(kernel_effective_n(np.array([[0, 0], [10, 10]]), 0.05), 2.0)

    def test_duplicate_starts_lower_diversity(self):
        cfg = metric_cfg()
        pos = {e: Position(e, RECORDED, x=0.1 + 0.001 * e, y=0.0, frame_id="world") for e in range(6)}
        pos[6] = Position(6, RECORDED, x=0.3, y=0.1, frame_id="world")
        cov = analyze_coverage(pos, cfg)
        self.assertEqual(cov.metrics["n_independent_groups"], 2)
        c = {x.name: x for x in coverage_checks(cov, pos, list(range(7)), {}, provisional=False)}
        self.assertEqual(c["start_diversity"].status, WARN)

    def test_unreachable_cells_are_not_missing(self):
        free = metric_cfg()
        blocked = metric_cfg(keep_out=[[0.2, 0.4, -0.2, 0.2]])  # right half unreachable
        pos = {e: Position(e, RECORDED, x=x, y=y, frame_id="world")
               for e, (x, y) in enumerate([(0.05, -0.15), (0.05, 0.0), (0.05, 0.15), (0.15, -0.15), (0.15, 0.0),
                                           (0.15, 0.15)])}
        a, b = analyze_coverage(pos, free), analyze_coverage(pos, blocked)
        self.assertGreater(b.metrics["target_covered_fraction"], a.metrics["target_covered_fraction"])
        for r in b.recommendations:
            self.assertLess(r["x"], 0.2)

    def test_recommendations_keep_separation_and_explain(self):
        cfg = metric_cfg()
        pos = {0: Position(0, RECORDED, x=0.1, y=0.0, frame_id="world")}
        cov = analyze_coverage(pos, cfg, n_recommend=4)
        pts = np.array([[r["x"], r["y"]] for r in cov.recommendations] + [[0.1, 0.0]])
        d = np.linalg.norm(pts[:, None] - pts[None], axis=-1) + np.eye(len(pts))
        self.assertGreaterEqual(d.min(), cfg.min_separation - 1e-9)
        self.assertTrue(all("nearest demonstration" in r["reason"] for r in cov.recommendations))

    def test_eval_distribution_points_shift_targets(self):
        cfg = metric_cfg(eval_distribution={"type": "points", "points": [[0.35, 0.15, 1.0]], "sigma": 0.03})
        pos = {0: Position(0, RECORDED, x=0.05, y=-0.15, frame_id="world")}
        cov = analyze_coverage(pos, cfg, n_recommend=1)
        r = cov.recommendations[0]
        self.assertLess(np.hypot(r["x"] - 0.35, r["y"] - 0.15), 0.06)
        self.assertLess(cov.metrics["target_covered_fraction"], 0.01)

    def test_positions_in_other_frame_are_ignored(self):
        pos = {0: Position(0, RECORDED, x=0.1, y=0.0, frame_id="camera_frame")}
        self.assertEqual(analyze_coverage(pos, metric_cfg()).metrics["n_with_position"], 0)

    def test_invalid_workspace_config_rejected(self):
        with self.assertRaises(ValueError):
            metric_cfg(x_range=[0.4, 0.1])
        with self.assertRaises(ValueError):
            metric_cfg(frame_type="pixels")
        with self.assertRaises(ValueError):
            analyze_coverage({}, metric_cfg(keep_out=[[-1, 1, -1, 1]]))  # nothing reachable

    def test_provisional_positions_never_pass(self):
        pos = {e: Position(e, AUTO, u=50 + 100 * e, v=240, camera="cam") for e in range(6)}
        cfg = WorkspaceConfig.from_json({"name": "img", "frame_type": "image", "frame_id": "image:cam", "units": "px",
                                         "x_range": [0, 640], "y_range": [0, 480], "cell_size": 32,
                                         "coverage_radius": 400, "duplicate_radius": 5, "min_separation": 40})
        cov = analyze_coverage(pos, cfg)
        c = {x.name: x for x in coverage_checks(cov, pos, list(range(6)), {}, provisional=True)}
        self.assertEqual(c["target_coverage"].status, UNKNOWN)


class TestCompatAndAdapter(Tmp):
    def test_schema_mismatch_fails(self):
        a = load_dataset(root=fx.write_v30(self.tmp / "a"), label="real")
        b = load_dataset(root=fx.write_v30(self.tmp / "b", names=list(reversed(fx.NAMES16))), label="sim")
        c = {x.name: x for x in compare_datasets(a, b)[0]}
        self.assertEqual(c["schema_observation.state"].status, FAIL)

    def test_incompatible_state_dims_fail(self):
        a = load_dataset(root=fx.write_v30(self.tmp / "a"), label="real")
        b = load_dataset(root=fx.write_v30(self.tmp / "b", names=[f"j{i}" for i in range(6)]), label="other")
        c = {x.name: x for x in compare_datasets(a, b)[0]}
        self.assertEqual(c["schema_action"].status, FAIL)

    def test_identical_domains_pass(self):
        a = load_dataset(root=fx.write_v30(self.tmp / "a", seed=1), label="real")
        b = load_dataset(root=fx.write_v30(self.tmp / "b", seed=2), label="sim")
        c = {x.name: x for x in compare_datasets(a, b)[0]}
        self.assertEqual(c["schema_cameras"].status, PASS)
        self.assertEqual(c["joint_speed_per_frame"].status, PASS)

    def test_openarm_quality_reuses_existing_audit(self):
        from tools.dataset_diagnostics.openarm import run_quality

        view = load_dataset(root=fx.write_v30(self.tmp / "ds", lengths=(120, 120, 120, 120)))
        ep, summ, checks, hold = run_quality(view)
        self.assertEqual(len(ep), 4)
        self.assertIn("right_regrasp", ep.columns)  # columns come from analyze_dataset_quality.analyze()
        c = {x.name: x for x in checks}
        self.assertEqual(c["openarm_exclude_flags"].status, PASS, c["openarm_exclude_flags"].detail)
        self.assertAlmostEqual(hold["right_hold_openness_median"], 0.40, delta=0.03)


if __name__ == "__main__":
    unittest.main()
