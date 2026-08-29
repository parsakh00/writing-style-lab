"""Render README.md into a small static site for GitHub Pages, with a page
that runs the checker in the browser.

    python scripts/build_site.py  ->  site/index.html, site/check.html
"""
import html
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".claude/skills/writing-style"
OUT = ROOT / "site"
# The polish service (worker/README.md). Empty until the worker is deployed.
POLISH_URL = "https://writing-style-lab.writing-style.workers.dev"

CSS = """
:root{--bg:#fbfbf9;--fg:#1c1c1c;--muted:#5f6368;--line:#e3e1da;--panel:#ffffff;--accent:#1f5f8b;--code:#f1f0ea}
@media (prefers-color-scheme:dark){:root{--bg:#111214;--fg:#e8e6e1;--muted:#9a9a94;--line:#2a2b2f;--panel:#17181b;--accent:#7ab3dc;--code:#1e1f23}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.65 Charter,"Iowan Old Style",Georgia,serif}
header{border-bottom:1px solid var(--line);background:var(--panel)}
header .in{max-width:52rem;margin:0 auto;padding:.9rem 1.2rem;display:flex;align-items:baseline;gap:1.6rem;flex-wrap:wrap}
header .brand{font:600 1.05rem/1 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif;color:var(--fg);text-decoration:none;letter-spacing:.01em}
nav a{font:500 .92rem/1 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif;color:var(--muted);text-decoration:none;margin-right:1.2rem}
nav a:hover,nav a[aria-current]{color:var(--accent)}
main{max-width:52rem;margin:0 auto;padding:2.2rem 1.2rem 4rem}
h1,h2,h3{font-family:-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;line-height:1.25;letter-spacing:-.01em}
h1{font-size:2.1rem;margin:.2rem 0 1rem}
h2{font-size:1.35rem;margin-top:2.6rem;padding-bottom:.35rem;border-bottom:1px solid var(--line)}
h3{font-size:1.1rem;margin-top:1.8rem}
p{margin:.9rem 0}a{color:var(--accent)}
code{font:.9em/1.4 ui-monospace,Menlo,Consolas,monospace;background:var(--code);padding:.12em .35em;border-radius:4px}
pre{background:var(--code);padding:1rem 1.1rem;overflow-x:auto;border-radius:6px;border:1px solid var(--line);line-height:1.5}
pre code{background:none;padding:0}
table{border-collapse:collapse;margin:1.1rem 0;font-size:.94em;width:100%}
td,th{border:1px solid var(--line);padding:.4rem .65rem;text-align:left;vertical-align:top}
th{background:var(--code);font-family:-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;font-weight:600}
blockquote{border-left:3px solid var(--accent);margin:1.1rem 0;padding:.3rem 1.1rem;color:var(--muted)}
img{max-width:100%}hr{border:0;border-top:1px solid var(--line);margin:2rem 0}
.lede{font-size:1.12rem;color:var(--muted);margin-top:0}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:1.2rem 1.3rem;margin:1.2rem 0}
.controls{display:flex;gap:1.4rem;flex-wrap:wrap;align-items:center;font:.92rem -apple-system,"Segoe UI",Helvetica,Arial,sans-serif;color:var(--muted)}
.controls select{font:inherit;color:var(--fg);background:var(--panel);border:1px solid var(--line);border-radius:5px;padding:.25rem .5rem}
textarea{width:100%;font:15px/1.55 ui-monospace,Menlo,Consolas,monospace;padding:.8rem;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--fg);resize:vertical}
textarea:focus{outline:2px solid var(--accent);outline-offset:1px}
button{font:600 .95rem -apple-system,"Segoe UI",Helvetica,Arial,sans-serif;color:#fff;background:var(--accent);border:0;border-radius:6px;padding:.6rem 1.3rem;cursor:pointer}
button[disabled]{opacity:.55;cursor:default}
.status{font:.88rem -apple-system,"Segoe UI",Helvetica,Arial,sans-serif;color:var(--muted);margin-left:.9rem}
#out{min-height:5rem;white-space:pre-wrap;font-size:.88rem;margin-top:1rem}
.note{color:var(--muted);font-size:.92rem}
footer{max-width:52rem;margin:0 auto;padding:1.5rem 1.2rem 3rem;color:var(--muted);font:.85rem -apple-system,"Segoe UI",Helvetica,Arial,sans-serif;border-top:1px solid var(--line)}
"""

CHECK_BODY = r"""
<h1>Polish a draft</h1>
<p class=lede>Paste a draft and get it back written to the skill's policy.</p>
<div class=panel id=loading>
<p style="margin:0"><strong id=loadmsg>Preparing</strong></p>
<div style="height:6px;background:var(--code);border-radius:3px;margin:.8rem 0 .4rem;overflow:hidden"><div id=bar style="height:100%;width:0;background:var(--accent);transition:width .3s"></div></div>
<p class=note style="margin:0" id=loaddetail>Loading the reference data</p>
</div>
<div class=panel id=tool style="display:none">
<div class=controls>
<label>Register <select id=register><option value=paper>paper</option><option value=letter>letter</option><option value=docs>documentation</option></select></label>
</div>
<p style="margin:1rem 0 .5rem"><textarea id=draft rows=14 placeholder="Paste your draft."></textarea></p>
<p style="margin:.4rem 0 0"><button id=run disabled>Polish</button><span class=status id=status></span></p>
<p class=note style="margin:.6rem 0 0" id=quota>3/3 attempts left</p>
</div>
<pre id=polished style="display:none;white-space:pre-wrap;font-family:inherit;font-size:1rem;line-height:1.6"></pre>
<div class=panel id=fb style="display:none">
<p style="margin:0 0 .5rem"><strong>Did something still read wrong?</strong> Send the passage as feedback. Each report is measured against the corpus, and rules change when the papers agree.</p>
<p style="margin:0"><a id=fblink href="#" target=_blank rel=noopener>Send feedback</a></p>
</div>
<link rel=preload as=fetch crossorigin href="tool/data/trigrams.json">
<script type=module>
import { report } from "./tool/check.js";
const FILES = ["reference.json", "group_reference.json", "combined_reference.json", "vocab.json", "formulas.json", "trigrams.json"];
const data = {};
const status = document.getElementById("status"), run = document.getElementById("run");
const bar = document.getElementById("bar"), detail = document.getElementById("loaddetail");
let done = 0;
const POLISH_URL = "__POLISH_URL__";
async function boot() {
  await Promise.all(FILES.map(async f => {
    data[f] = await (await fetch("tool/data/" + f)).json();
    done += 1; bar.style.width = Math.round(100 * done / FILES.length) + "%"; detail.textContent = "loaded " + f;
  }));
  document.getElementById("loading").style.display = "none";
  document.getElementById("tool").style.display = "block";
  run.disabled = false;
}
const ready = boot().catch(e => { document.getElementById("loadmsg").textContent = "The page could not load"; detail.textContent = String(e); });
let ticker = null;
function busy(on) {
  if (ticker) { clearInterval(ticker); ticker = null; }
  if (!on) { status.textContent = ""; return; }
  let n = 0; status.textContent = "polishing";
  ticker = setInterval(() => { n = (n + 1) % 4; status.textContent = "polishing" + " .".repeat(n); }, 450);
}
function showQuota(left, limit) {
  document.getElementById("quota").textContent = `${left}/${limit} attempt${left === 1 ? "" : "s"} left`;
  if (left <= 0) { run.disabled = true; status.textContent = "come back tomorrow"; }
}
async function loadQuota() {
  try { const q = await (await fetch(POLISH_URL + "/quota")).json(); if (q.limit) showQuota(q.remaining, q.limit); } catch (e) {}
}
ready.then(loadQuota);
run.onclick = async () => {
  await ready;
  const draft = document.getElementById("draft").value, reg = document.getElementById("register").value;
  const pe = document.getElementById("polished");
  if (draft.trim().split(/\s+/).length < 5) { status.textContent = "paste a draft"; return; }
  if (!POLISH_URL) { status.textContent = "the polish service is not set up yet"; return; }
  run.disabled = true; busy(true); pe.style.display = "none";
  // The checker runs here and its report travels with the draft, so the rewrite is
  // aimed at what this draft actually does. The report itself is not shown.
  const rep = report(draft, data, { register: reg, reference: "papers", suggest: true, name: "draft" });
  try {
    const r = await fetch(POLISH_URL, { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ draft, register: reg, report: rep.slice(0, 8000) }) });
    let j = await r.json();
    const left = r.headers.get("x-remaining");
    if (!r.ok) throw new Error(j.error || r.statusText);
    // Second pass: check the polished text itself, and if constructions papers do not
    // use remain, send exactly those back once. Free, and tied to this text by token.
    if (j.text) {
      const again = report(j.text, data, { register: reg, reference: "papers", suggest: true, name: "polished" });
      const flagged = again.split("\n").filter(l => /^\s{2,}(?:\d+x |colon |passive |'|sequences no paper|\[|')/.test(l) || /papers write:/.test(l)).slice(0, 40).join("\n");
      if (flagged.trim()) {
        status.textContent = "polishing, second pass";
        const r2 = await fetch(POLISH_URL, { method: "POST", headers: { "content-type": "application/json" },
          body: JSON.stringify({ draft: j.text, register: reg, report: flagged, pass: 2, token: j.token }) });
        const j2 = await r2.json();
        if (r2.ok && j2.text) j = j2;
      }
    }
    pe.textContent = j.text || ("The service answered without text. Reply shape: " + JSON.stringify(j.shape || j).slice(0, 600));
    pe.style.display = "block"; pe.scrollIntoView({ behavior: "smooth", block: "start" });
    if (left !== null) showQuota(parseInt(left, 10), 3); else loadQuota();
    document.getElementById("fblink").href = "https://github.com/parsakh00/writing-style-lab/issues/new?template=feedback.yml&title=" + encodeURIComponent("Feedback: ") + "&passage=" + encodeURIComponent(j.text.slice(0, 3000));
    document.getElementById("fb").style.display = "block";
    busy(false);
  } catch (e) {
    busy(false);
    pe.textContent = "Could not polish: " + e.message; pe.style.display = "block";
    if (/used its/.test(e.message)) showQuota(0, 3);
  }
  if (!run.disabled || !/come back/.test(status.textContent)) run.disabled = false;
};
</script>
"""


def md_to_html(md: str) -> str:
    import markdown
    return markdown.markdown(md, extensions=["tables", "fenced_code"])


def page(title: str, body: str, current: str = "") -> str:
    links = [("index.html", "Overview"), ("check.html", "Polish a draft"),
             ("https://github.com/parsakh00/writing-style-lab", "GitHub")]
    nav = "".join(f'<a href="{h}"{" aria-current=page" if h == current else ""}>{n}</a>' for h, n in links)
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<meta name=description content='Polishes scientific writing by measuring a draft against published papers.'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
            f"<header><div class=in><a class=brand href=index.html>writing-style</a><nav>{nav}</nav></div></header>"
            f"<main>{body}</main>"
            f"<footer>Measured on 6.4 million words of published papers. MIT license.</footer></body></html>")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    (OUT / "index.html").write_text(page("writing-style", md_to_html(readme), "index.html"), encoding="utf-8")

    # The checker itself, run in the browser with Pyodide. check.py and its data are
    # copied beside the page.
    tool = OUT / "tool"
    if tool.exists():
        shutil.rmtree(tool)
    shutil.copytree(SKILL, tool, ignore=shutil.ignore_patterns(
        "__pycache__", "*.py", "local_preferences.json", "sequences.json", "group_profile.json", "awl_measured.json"))
    body = CHECK_BODY.replace("__POLISH_URL__", POLISH_URL)
    (OUT / "check.html").write_text(page("Polish a draft", body, "check.html"), encoding="utf-8")
    print("site/index.html, site/check.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
