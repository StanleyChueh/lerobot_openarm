# Packaging-only correction after generation

Both synthetic exports completed at 104 episodes / 26,781 frames. The initial real+synthetic merge failed because
LeRobot 0.6.2 compares codec and pixel-format metadata: real AV1/yuv420p versus synthetic H.264/yuv444p.

The correction changes only temporary merge metadata: heterogeneous codec/pixel-format values are null, while
per-source video metadata is retained in the mixture provenance. Real and synthetic MP4 files are copied without
re-encoding. The original failed temporary outputs and logs remain under mixed_tmp_failed_codec/ and build_*.log.

The immutable FREEZE.json and original generation manifests remain intact. PACKAGING_AMENDMENT.json records the
old and new build.py hashes and timestamp. No candidates, renderer, action labels, target assignments or gates
changed. The added unit test verifies heterogeneous metadata and that original feature dictionaries stay unchanged.
The full suite passes 53 tests. Independent export checks verify original/synthetic video byte hashes and all labels.

Both final mixtures contain 114 episodes / 29,356 frames, with flags identifying the 10 original real episodes.
No Hub upload, publication, policy training or hardware operation occurred.
