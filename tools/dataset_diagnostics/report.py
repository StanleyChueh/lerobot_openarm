"""Readiness statuses and exports (HTML + JSON + CSV)."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np

from .checks import FAIL, PASS, UNKNOWN, WARN, Check, section_status, worst

SECTIONS = ("technical", "coverage", "quality", "real_sim")
SECTION_TITLES = {
    "technical": "Technical readiness (schema, dimensions, timing, preprocessing compatibility)",
    "coverage": "Coverage readiness (workspace coverage, start-pose diversity, uncertainty)",
    "quality": "Demonstration quality (gripper, handover, timing, trajectory, tracking)",
    "real_sim": "Real/sim compatibility (mappings, conventions, normalization, trajectory statistics)",
}
CAVEAT = ("These statuses describe the dataset, not the policy. Even coverage and quality marked PASS do not "
          "guarantee that a policy will train or succeed. No success prediction is made without "
          "empirical validation.")


def coverage_checks(cov, positions: dict, episodes: list[int], fragility: dict, *, provisional: bool) -> list[Check]:
    """Turn a CoverageResult into statuses. Provisional (unverified) positions never yield PASS/FAIL."""
    th = {"target_covered_pass": 0.8, "target_covered_fail": 0.5, "max_duplicate_fraction": 0.2,
          "min_independent_per_region": 1, **(cov.config.thresholds or {})}
    m = cov.metrics
    n_known = m["n_with_position"]
    checks = []
    unknown_eps = [e for e in episodes if e not in cov.episodes]
    if n_known == 0:
        return [Check("positions_available", UNKNOWN, "no episode has a usable position in frame "
                                                      f"{cov.config.frame_id}", {"missing_episodes": unknown_eps})]
    checks.append(Check("positions_available", PASS if not unknown_eps else WARN,
                        f"{n_known}/{len(episodes)} episodes have a position"
                        + (f"; UNKNOWN for {unknown_eps}" if unknown_eps else ""), {"missing": unknown_eps}))
    if cov.outside:
        checks.append(Check("starts_outside_workspace", WARN,
                            f"episodes {cov.outside} start outside the declared reachable workspace", {}))
    frac = m["target_covered_fraction"]
    st = PASS if frac >= th["target_covered_pass"] else (WARN if frac >= th["target_covered_fail"] else FAIL)
    if provisional:
        st = UNKNOWN
    checks.append(Check("target_coverage", st,
                        f"{100 * frac:.0f}% of the target distribution lies within {cov.config.coverage_radius:g} "
                        f"{cov.config.units} of a demo (pass >= {100 * th['target_covered_pass']:.0f}%); "
                        f"leave-one-out range {fragility.get('loo_min')}-{fragility.get('loo_max')}"
                        + (" [PROVISIONAL: unverified positions]" if provisional else ""),
                        {**{k: m[k] for k in ("target_covered_fraction", "reachable_area_covered_fraction",
                                              "target_mean_nn_distance", "covering_radius_reachable")},
                         "fragility": fragility}))
    dup_eps = sum(len(g) - 1 for g in cov.duplicates)
    dfrac = dup_eps / max(n_known, 1)
    st = PASS if dfrac <= th["max_duplicate_fraction"] else WARN
    checks.append(Check("start_diversity", UNKNOWN if provisional else st,
                        f"{m['n_independent_groups']} independent start positions out of {n_known} "
                        f"(near-duplicate radius {cov.config.duplicate_radius:g} {cov.config.units}); kernel "
                        f"effective n = {m['kernel_effective_n']}", {"duplicate_groups": cov.duplicates}))
    if cov.regions:
        empty = [n for n, r in cov.regions.items() if r["independent"] < th["min_independent_per_region"]]
        checks.append(Check("region_counts", UNKNOWN if provisional else (WARN if empty else PASS),
                            "independent demos per region: " + ", ".join(
                                f"{n}={r['independent']}" for n, r in cov.regions.items())
                            + (f"; under-covered {empty}" if empty else ""), cov.regions))
    return checks


def readiness(sections: dict[str, list[Check]]) -> dict:
    out = {}
    for s in SECTIONS:
        checks = sections.get(s) or []
        out[s] = {"status": section_status(checks) if checks else UNKNOWN,
                  "checks": [c.to_dict() for c in checks]}
    out["caveat"] = CAVEAT
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _img(path: Path) -> str:
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img alt="{html.escape(path.stem)}" src="data:image/png;base64,{b64}"></figure>'


def _badge(status: str) -> str:
    icon = {PASS: "✓", WARN: "!", FAIL: "✕", UNKNOWN: "?"}[status]
    return f'<span class="badge {status.lower()}">{icon} {status}</span>'


def _table(rows: list[dict], max_rows: int = 200) -> str:
    if not rows:
        return "<p class='muted'>none</p>"
    cols = list(rows[0].keys())
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(_fmt(r.get(c)))}</td>" for c in cols) + "</tr>"
                   for r in rows[:max_rows])
    return f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, dict)):
        s = json.dumps(v, default=_json_default)
        return s if len(s) < 160 else s[:157] + "..."
    return "" if v is None else str(v)


CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--card:#ffffff;
--pass:#1a7f37;--warn:#9a6700;--fail:#c62828;--unk:#57606a}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--line:#3a3a37;--card:#232321;--pass:#4ac26b;--warn:#d4a72c;--fail:#ff7b72;--unk:#9ea7b3}}
:root[data-theme="dark"]{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;--card:#232321;
--pass:#4ac26b;--warn:#d4a72c;--fail:#ff7b72;--unk:#9ea7b3}
body{background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:24px 16px}
main{max-width:1180px;margin:0 auto}h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px}
h3{font-size:14px;margin:18px 0 6px}.muted{color:var(--ink2)}.card{background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:12px 14px;margin:10px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}
.badge{display:inline-block;border-radius:4px;padding:1px 7px;font-weight:600;font-size:12px;border:1px solid currentColor}
.pass{color:var(--pass)}.warn{color:var(--warn)}.fail{color:var(--fail)}.unknown{color:var(--unk)}
.tw{overflow-x:auto}table{border-collapse:collapse;font-size:12px;width:100%}th,td{border-bottom:1px solid var(--line);
padding:4px 6px;text-align:left;vertical-align:top}th{color:var(--ink2);font-weight:600}
figure{margin:8px 0}img{max-width:100%;height:auto;background:#fff;border-radius:6px}
.caveat{border-left:3px solid var(--warn);padding:6px 10px;color:var(--ink2)}
"""


def render_html(path: Path, title: str, ready_by_dataset: dict, *, figures: dict[str, list[Path]],
                tables: dict[str, dict[str, list[dict]]], notes: dict[str, list[str]], extra_html: str = "") -> None:
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
             f"content='width=device-width,initial-scale=1'><title>Dataset Readiness Report</title>"
             f"<style>{CSS}</style></head><body><main><h1>{html.escape(title)}</h1>"
             f"<p class='caveat'>{html.escape(CAVEAT)}</p>"]
    # summary grid
    parts.append("<h2>Summary</h2><div class='tw'><table><thead><tr><th>dataset</th>"
                 + "".join(f"<th>{s}</th>" for s in SECTIONS) + "</tr></thead><tbody>")
    for label, ready in ready_by_dataset.items():
        parts.append(f"<tr><td>{html.escape(label)}</td>"
                     + "".join(f"<td>{_badge(ready[s]['status'])}</td>" for s in SECTIONS) + "</tr>")
    parts.append("</tbody></table></div>")
    parts.append(extra_html)
    for label, ready in ready_by_dataset.items():
        parts.append(f"<h2>{html.escape(label)}</h2>")
        for n in notes.get(label, []):
            parts.append(f"<p class='muted'>{html.escape(n)}</p>")
        for s in SECTIONS:
            parts.append(f"<div class='card'><h3>{_badge(ready[s]['status'])} {html.escape(SECTION_TITLES[s])}</h3>")
            rows = [{"check": c["name"], "status": c["status"], "detail": c["detail"]} for c in ready[s]["checks"]]
            parts.append(_table(rows))
            parts.append("</div>")
        for fig in figures.get(label, []):
            parts.append(_img(fig))
        for tname, rows in tables.get(label, {}).items():
            parts.append(f"<h3>{html.escape(tname)}</h3>{_table(rows)}")
    parts.append("</main></body></html>")
    path.write_text("".join(parts))


def overall_line(ready: dict) -> str:
    return " | ".join(f"{s}: {ready[s]['status']}" for s in SECTIONS)


__all__ = ["coverage_checks", "readiness", "render_html", "write_json", "overall_line", "worst", "WARN", "FAIL"]
