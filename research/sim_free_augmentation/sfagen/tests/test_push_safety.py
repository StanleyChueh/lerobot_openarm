"""Hub upload safety of tools/push_unified.py, against a fake HfApi (no network)."""
import sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import push_unified as pu


class FakeApi:
    def __init__(self, main="old0", tags=None, files=("data/chunk-000/file-000.parquet", "meta/info.json"), drop_v3=False):
        self.main, self.tags, self.files, self.drop_v3, self.writes = main, dict(tags or {}), list(files), drop_v3, []
    def dataset_info(self, R): return SimpleNamespace(sha=self.main)
    def list_repo_refs(self, R, **k): return SimpleNamespace(tags=[SimpleNamespace(name=n, target_commit=c) for n, c in self.tags.items()])
    def list_repo_files(self, R, **k): return self.files
    def create_tag(self, R, tag, revision, **k):
        self.writes.append(("tag", tag, revision))
        if tag in self.tags: raise RuntimeError("tag exists")
        if not (self.drop_v3 and tag == "v3.0"): self.tags[tag] = revision
    def delete_tag(self, R, tag, **k): self.writes.append(("untag", tag)); self.tags.pop(tag)
    def create_commit(self, R, operations, parent_commit, **k):
        self.writes.append(("commit", parent_commit)); self.main = "new1"; return SimpleNamespace(oid="new1")


def export(root, parts=("meta/info.json", "data/chunk-000/file-000.parquet", "videos/cam/chunk-000/file-000.mp4")):
    for p in parts:
        (root / p).parent.mkdir(parents=True, exist_ok=True); (root / p).write_text("x")
    return root


class PushSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_empty_or_incomplete_export_is_refused(self):
        with self.assertRaises(RuntimeError):
            pu.export_files(self.tmp / "missing")
        with self.assertRaises(RuntimeError):
            pu.export_files(export(self.tmp / "novideo", ("meta/info.json", "data/a.parquet")))

    def test_fresh_publish_tags_old_and_moves_v3(self):
        api = FakeApi(tags={"v3.0": "old0"}); new = pu.export_files(export(self.tmp / "ok"))
        self.assertEqual(pu.publish(api, "r", new, [], "prev"), "new1")
        self.assertEqual(api.tags, {"prev": "old0", "v3.0": "new1"})

    def test_rerun_after_failed_commit_reuses_old_tag(self):
        api = FakeApi(tags={"prev": "old0", "v3.0": "old0"}); new = pu.export_files(export(self.tmp / "ok"))
        pu.publish(api, "r", new, [], "prev")
        self.assertEqual(api.tags, {"prev": "old0", "v3.0": "new1"})
        self.assertNotIn(("tag", "prev", "old0"), api.writes)

    def test_old_tag_elsewhere_stops_before_any_write(self):
        api = FakeApi(main="new1", tags={"prev": "old0", "v3.0": "new1"}); new = pu.export_files(export(self.tmp / "ok"))
        with self.assertRaises(RuntimeError):
            pu.publish(api, "r", new, [], "prev")
        self.assertEqual(api.writes, [])

    def test_missing_v3_after_move_is_reported(self):
        api = FakeApi(tags={"v3.0": "old0"}, drop_v3=True); new = pu.export_files(export(self.tmp / "ok"))
        with self.assertRaises(RuntimeError):
            pu.publish(api, "r", new, [], "prev")

    def test_dry_run_writes_nothing(self):
        api = FakeApi(tags={"v3.0": "old0"}); new = pu.export_files(export(self.tmp / "ok"))
        self.assertIsNone(pu.publish(api, "r", new, [], "prev", dry=True)); self.assertEqual(api.writes, [])


if __name__ == "__main__":
    unittest.main()
