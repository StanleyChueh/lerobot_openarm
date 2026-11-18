"""Before / after sheet for the same candidates rendered with baseline masks (A) and R05 masks (B): three edited frames
per episode, A left and B right. Usage: r05_compare.py OUT.jpg LABEL:A_BODY_DIR:B_BODY_DIR:EP:KEY ..."""
import json, sys
import numpy as np, cv2
out, rows = sys.argv[1], []
for spec in sys.argv[2:]:
    label, a, b, e, key = spec.split(":"); A = np.load(f"{a}/{e}_{key}.npy", mmap_mode="r"); B = np.load(f"{b}/{e}_{key}.npy", mmap_mode="r")
    ed = [i for i, x in enumerate(json.load(open(f"{a}/{e}_{key}.json"))) if not x.get("copied")]
    tiles = []
    for f in (ed[len(ed) // 4], ed[len(ed) // 2], ed[3 * len(ed) // 4]):
        for im, tag in ((A[f], "A"), (B[f], "B")):
            t = cv2.resize(np.ascontiguousarray(im[..., ::-1]), (320, 240)); cv2.putText(t, f"{tag} {label} ep{e} {key} f{f}", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t)
    rows.append(np.concatenate(tiles, 1)); d = np.abs(A[ed].astype(np.int16) - B[ed].astype(np.int16)).max(-1) > 20
    print(label, e, key, "edited frames", len(ed), "pixels changed A->B per edited frame", int(d.sum() / len(ed)))
cv2.imwrite(out, np.concatenate(rows, 0)); print("wrote", out)
