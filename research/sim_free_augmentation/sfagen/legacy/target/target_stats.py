import json, numpy as np
from scipy.spatial import ConvexHull
r = json.load(open('target_positions.json'))
pp = json.load(open('../3cam_v3/dataset/meta/sfa_provenance.json'))['episodes']
bp = json.load(open('../battery/dataset/meta/sfa_provenance.json'))['episodes']
def stats(task, key, synkey, prov, srcfield):
    src = {int(k): np.array(v[key]) for k, v in r[task]['source'].items()}
    syn = {int(k): np.array(v[key]) for k, v in r[task][synkey].items()}
    pool = np.array([v[key] for v in r[task]['pool'].values()])
    d = [np.linalg.norm(syn[e['export_episode_index']] - src[e[srcfield]]) * 1000 for e in prov]
    S = np.array(list(src.values())); Y = np.concatenate([S, np.array(list(syn.values()))])
    def cov(T):
        dd = np.linalg.norm(pool[:, None, :2] - T[None, :, :2], axis=2).min(1) * 100; return round(float(np.median(dd)), 2), int((dd <= 2).sum())
    print(task, key, '| synthetic target - its source target (mm): max %.2f median %.2f' % (max(d), np.median(d)),
          '| hull xy cm2 src %.0f src+syn %.0f pool %.0f' % (ConvexHull(S[:, :2] * 100).volume, ConvexHull(Y[:, :2] * 100).volume, ConvexHull(pool[:, :2] * 100).volume),
          '| pool cov (median cm, within2cm/%d) src' % len(pool), cov(S), 'src+syn', cov(Y),
          '| z range src %.1f pool %.1f cm' % (np.ptp(S[:, 2]) * 100, np.ptp(pool[:, 2]) * 100))
stats('pringles', 'handover', 'synthetic_v3', pp, 'source_real_episode')
k = [x for x in bp[0] if 'source' in x and 'episode' in x][0]; print('battery source field', k)
stats('battery', 'station_pick', 'synthetic_v00', bp, k)
stats('battery', 'insertion', 'synthetic_v00', bp, k)
