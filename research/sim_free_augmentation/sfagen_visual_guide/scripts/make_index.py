"""Builds ../index.html from ../STAGE_BY_STAGE_VISUAL_GUIDE.md (minimal Markdown subset) and adds a playable video
gallery to every stage section (all MP4s in that stage's folder). Open index.html locally; videos are local files."""
import html, re
from pathlib import Path
G = Path(__file__).resolve().parents[1]
md = (G / "STAGE_BY_STAGE_VISUAL_GUIDE.md").read_text().splitlines()
STAGE_DIR = {"Stage 1": "stage_01_segmentation", "Stage 2": "stage_02_tracking", "Stage 3": "stage_03_reconstruction", "Stage 4": "stage_04_trajectories",
             "Stage 5": "stage_05_rendering", "Stage 6": "stage_06_cameras"}
EXTRA_VIDEOS = {"Stage 5": [("../../../../sfa_hf_export/sfagen/configs/work/openarm_pringles_handover/video_highlights.mp4", "Pringles: all-episode highlights (existing, also on the Hub)"),
                            ("../../../../sfa_hf_export/sfagen/configs/work/openarm_battery_insertion/video_highlights.mp4", "Battery: all-episode highlights (existing, also on the Hub)")]}


def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    return s


def videos(stage):
    d = G / STAGE_DIR[stage]; items = [(f"{STAGE_DIR[stage]}/{p.name}", p.stem.replace("_", " ")) for p in sorted(d.glob("*.mp4"))] + EXTRA_VIDEOS.get(stage, [])
    if not items:
        return ""
    cards = "".join(f'<figure class="vid"><video controls preload="metadata" src="{s}"></video><figcaption>{html.escape(c)}</figcaption></figure>' for s, c in items)
    return f'<h3>Videos</h3><div class="vgrid">{cards}</div>'


out, i, stage, toc = [], 0, None, []
while i < len(md):
    l = md[i]
    if l.startswith("```"):
        j = i + 1; buf = []
        while j < len(md) and not md[j].startswith("```"):
            buf.append(md[j]); j += 1
        out.append("<pre>" + html.escape("\n".join(buf)) + "</pre>"); i = j + 1; continue
    if l.startswith("|"):
        rows = []
        while i < len(md) and md[i].startswith("|"):
            rows.append([c.strip() for c in md[i].strip().strip("|").split("|")]); i += 1
        hdr, body = rows[0], [r for r in rows[1:] if not set("".join(r)) <= set("-: ")]
        t = "<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in hdr) + "</tr></thead><tbody>"
        t += "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body) + "</tbody></table>"
        out.append(f'<div class="tw">{t}</div>'); continue
    m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", l)
    if m:
        out.append(f'<figure><a href="{m.group(2)}" target="_blank"><img loading="lazy" src="{m.group(2)}" alt="{html.escape(m.group(1))}"></a><figcaption>{html.escape(m.group(1))}</figcaption></figure>'); i += 1; continue
    if l.startswith("## "):
        if stage:
            out.append(videos(stage)); stage = None
        title = l[3:]; sid = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-"); toc.append((sid, title))
        stage = next((k for k in STAGE_DIR if title.startswith(k)), None)
        out.append(f'</section><section id="{sid}"><h2>{inline(title)}</h2>'); i += 1; continue
    if l.startswith("# "):
        out.append(f"<h1>{inline(l[2:])}</h1>"); i += 1; continue
    if l.strip() == "---":
        i += 1; continue
    if l.startswith("- "):
        items = []
        while i < len(md) and md[i].startswith("- "):
            items.append(f"<li>{inline(md[i][2:])}</li>"); i += 1
        out.append("<ul>" + "".join(items) + "</ul>"); continue
    if l.strip():
        buf = [l]; i += 1
        while i < len(md) and md[i].strip() and not md[i].startswith(("|", "#", "!", "```", "- ", "---")):
            buf.append(md[i]); i += 1
        out.append("<p>" + inline(" ".join(buf)) + "</p>"); continue
    i += 1
if stage:
    out.append(videos(stage))
nav = "".join(f'<a href="#{s}">{html.escape(t)}</a>' for s, t in toc)
CSS = """
:root{--bg:#fcfcfb;--fg:#1b1b1a;--mut:#5c5b57;--line:#e2e1dc;--card:#ffffff;--acc:#2a78d6;--code:#f1f0ec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151514;--fg:#ecebe6;--mut:#a5a39c;--line:#33322f;--card:#1e1e1c;--acc:#6aa7f0;--code:#262624}}
:root[data-theme="dark"]{--bg:#151514;--fg:#ecebe6;--mut:#a5a39c;--line:#33322f;--card:#1e1e1c;--acc:#6aa7f0;--code:#262624}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:14px;overflow-x:auto;white-space:nowrap}
header a{color:var(--mut);text-decoration:none;font-size:13px}header a:hover{color:var(--acc)}
main{max-width:1180px;margin:0 auto;padding:16px}section{padding-top:8px;border-top:1px solid var(--line);margin-top:28px}
h1{font-size:26px;line-height:1.25}h2{font-size:21px;margin:18px 0 8px}h3{font-size:16px;color:var(--mut)}
a{color:var(--acc)}code{background:var(--code);padding:1px 5px;border-radius:4px;font-size:12.5px}pre{background:var(--code);padding:12px;border-radius:8px;overflow-x:auto}
.tw{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;width:100%;background:var(--card);font-size:13.5px}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:left;vertical-align:top}th{background:var(--code)}
td:first-child{white-space:nowrap;font-weight:600}
figure{margin:14px 0}img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff}
figcaption{color:var(--mut);font-size:12.5px;margin-top:4px}
.vgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px}.vid video{width:100%;border-radius:6px;background:#000}
"""
page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>sfagen Visual Walkthrough</title><style>{CSS}</style></head><body><header>{nav}</header><main><section>{''.join(out)}</section>
<p style="color:var(--mut);font-size:12px">Generated by scripts/make_index.py from STAGE_BY_STAGE_VISUAL_GUIDE.md. Videos are local files (not in Git); see ARTIFACT_PROVENANCE.json.</p></main></body></html>"""
(G / "index.html").write_text(page); print("index.html", len(page), "bytes,", len(toc), "sections")
