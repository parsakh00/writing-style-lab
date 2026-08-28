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

CHECK_BODY = """
<h1>Check a draft</h1>
<p class=lede>Paste a draft, choose the register, and read where it sits against published papers.</p>
<div class=panel>
<div class=controls>
<label>Register <select id=register><option value=paper>paper</option><option value=letter>letter</option><option value=docs>documentation</option></select></label>
<label>Reference <select id=reference><option value=corpus>corpus</option><option value=group>group</option></select></label>
<label><input type=checkbox id=suggest checked> suggest replacements</label>
</div>
<p style="margin:1rem 0 .5rem"><textarea id=draft rows=14 placeholder="Paste your draft. Three hundred words or more gives stable numbers."></textarea></p>
<p style="margin:.4rem 0 0"><button id=run disabled>Check</button><span class=status id=status></span></p>
</div>
<pre id=out></pre>
<script src="https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js"></script>
<script>
const DATA = "__DATA_FILES__".split(",");
let pyodide = null;
const status = document.getElementById("status"), out = document.getElementById("out"), run = document.getElementById("run");
async function boot() {
  status.textContent = "loading Python";
  pyodide = await loadPyodide();
  pyodide.FS.mkdirTree("/tool/data");
  pyodide.FS.writeFile("/tool/check.py", await (await fetch("tool/check.py")).text());
  for (const f of DATA) {
    status.textContent = "loading " + f;
    pyodide.FS.writeFile("/tool/data/" + f, await (await fetch("tool/data/" + f)).text());
  }
  await pyodide.runPythonAsync("import sys; sys.path.insert(0, '/tool'); import check");
  status.textContent = "ready"; run.disabled = false;
}
const ready = boot().catch(e => { status.textContent = "could not load: " + e; });
run.onclick = async () => {
  await ready; run.disabled = true; status.textContent = "checking";
  pyodide.globals.set("draft_text", document.getElementById("draft").value);
  pyodide.globals.set("reg", document.getElementById("register").value);
  pyodide.globals.set("ref", document.getElementById("reference").value);
  pyodide.globals.set("sug", document.getElementById("suggest").checked);
  try {
    out.textContent = await pyodide.runPythonAsync("check.report(draft_text, register=reg, reference=ref, suggest=sug)");
    status.textContent = "";
  } catch (e) { out.textContent = String(e); status.textContent = "error"; }
  run.disabled = false;
};
</script>
"""


def md_to_html(md: str) -> str:
    import markdown
    return markdown.markdown(md, extensions=["tables", "fenced_code"])


def page(title: str, body: str, current: str = "") -> str:
    links = [("index.html", "Overview"), ("check.html", "Check a draft"),
             ("https://github.com/parsakh00/writing-style-lab", "GitHub")]
    nav = "".join(f'<a href="{h}"{" aria-current=page" if h == current else ""}>{n}</a>' for h, n in links)
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<meta name=description content='Polishes scientific writing by measuring a draft against published papers.'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
            f"<header><div class=in><a class=brand href=index.html>writing-style</a><nav>{nav}</nav></div></header>"
            f"<main>{body}</main>"
            f"<footer>Measured on 6 million words of published papers. MIT license.</footer></body></html>")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    (OUT / "index.html").write_text(page("writing-style", md_to_html(readme), "index.html"), encoding="utf-8")

    # The checker itself, run in the browser with Pyodide. check.py and its data are
    # copied beside the page.
    tool = OUT / "tool"
    if tool.exists():
        shutil.rmtree(tool)
    shutil.copytree(SKILL, tool, ignore=shutil.ignore_patterns("__pycache__", "local_preferences.json"))
    data_files = sorted(f.name for f in (tool / "data").glob("*.json"))
    body = CHECK_BODY.replace("__DATA_FILES__", ",".join(data_files))
    (OUT / "check.html").write_text(page("Check a draft", body, "check.html"), encoding="utf-8")
    print("site/index.html, site/check.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
