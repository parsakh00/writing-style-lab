"""Render README.md and SKILL.md into a small static site for GitHub Pages.

    python scripts/build_site.py  ->  site/index.html, site/skill.html
"""
import html, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site"
CSS = """body{max-width:46rem;margin:3rem auto;padding:0 1.2rem;font:17px/1.6 Georgia,serif;color:#222;background:#fff}
h1,h2,h3{font-family:Helvetica,Arial,sans-serif;line-height:1.25}h1{font-size:2rem}h2{margin-top:2.4rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}
code{font:.92em/1.4 Menlo,Consolas,monospace;background:#f4f4f4;padding:.1em .3em;border-radius:3px}pre{background:#f4f4f4;padding:1rem;overflow-x:auto;border-radius:4px}pre code{background:none;padding:0}
table{border-collapse:collapse;margin:1rem 0;font-size:.95em}td,th{border:1px solid #ddd;padding:.35rem .6rem;text-align:left}th{background:#f4f4f4}
blockquote{border-left:3px solid #ccc;margin:1rem 0;padding:.2rem 1rem;color:#444}nav a{margin-right:1.2rem;font-family:Helvetica,Arial,sans-serif}img{max-width:100%}"""


CHECK_BODY = """
<h1>Check a draft</h1>
<p>Paste a draft and run the same checker that ships in the skill. It runs in your browser;
the text is not uploaded anywhere. The first run downloads Python (about 10 MB) and the
reference data (5 MB), then it is fast.</p>
<p>
<label>Register <select id=register><option value=paper>paper</option><option value=letter>letter</option><option value=docs>docs</option></select></label>
&nbsp; <label>Reference <select id=reference><option value=corpus>corpus</option><option value=group>group</option></select></label>
&nbsp; <label><input type=checkbox id=suggest checked> suggest replacements</label>
</p>
<textarea id=draft rows=14 style="width:100%;font:15px/1.5 Menlo,Consolas,monospace;padding:.6rem" placeholder="Paste your draft here. Three hundred words or more gives stable numbers."></textarea>
<p><button id=run style="font:16px Helvetica,Arial,sans-serif;padding:.5rem 1.2rem">Check</button> <span id=status style="color:#666"></span></p>
<pre id=out style="min-height:6rem;white-space:pre-wrap"></pre>
<script src="https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js"></script>
<script>
const DATA = "__DATA_FILES__".split(",");
let pyodide = null;
async function boot() {
  const status = document.getElementById("status");
  status.textContent = "loading Python...";
  pyodide = await loadPyodide();
  pyodide.FS.mkdirTree("/tool/data");
  const src = await (await fetch("tool/check.py")).text();
  pyodide.FS.writeFile("/tool/check.py", src);
  for (const f of DATA) {
    status.textContent = "loading " + f + "...";
    const t = await (await fetch("tool/data/" + f)).text();
    pyodide.FS.writeFile("/tool/data/" + f, t);
  }
  await pyodide.runPythonAsync("import sys; sys.path.insert(0, '/tool'); import check");
  status.textContent = "ready";
}
const ready = boot();
document.getElementById("run").onclick = async () => {
  const status = document.getElementById("status"), out = document.getElementById("out");
  await ready;
  status.textContent = "checking...";
  const text = document.getElementById("draft").value;
  pyodide.globals.set("draft_text", text);
  pyodide.globals.set("reg", document.getElementById("register").value);
  pyodide.globals.set("ref", document.getElementById("reference").value);
  pyodide.globals.set("sug", document.getElementById("suggest").checked);
  try {
    out.textContent = await pyodide.runPythonAsync("check.report(draft_text, register=reg, reference=ref, suggest=sug)");
    status.textContent = "done";
  } catch (e) { out.textContent = String(e); status.textContent = "error"; }
};
</script>
"""


def md_to_html(md: str) -> str:
    import markdown
    return markdown.markdown(md, extensions=["tables", "fenced_code"])


def page(title: str, body: str) -> str:
    nav = '<nav><a href="index.html">Overview</a><a href="skill.html">The skill</a><a href="check.html">Check a draft</a><a href="https://github.com/parsakh00/writing-style-lab">Repository</a></nav>'
    return f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{CSS}</style></head><body>{nav}{body}</body></html>"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    skill = (ROOT / ".claude/skills/writing-style/SKILL.md").read_text(encoding="utf-8")
    skill = re.sub(r"^---\n.*?\n---\n", "", skill, flags=re.S)  # drop the frontmatter
    (OUT / "index.html").write_text(page("writing-style", md_to_html(readme)), encoding="utf-8")
    (OUT / "skill.html").write_text(page("writing-style: the skill", md_to_html(skill)), encoding="utf-8")
    # The checker itself, run in the browser with Pyodide. check.py and its data are
    # copied beside the page; nothing is sent anywhere.
    import shutil
    tool = OUT / "tool"
    if tool.exists():
        shutil.rmtree(tool)
    shutil.copytree(ROOT / ".claude/skills/writing-style", tool, ignore=shutil.ignore_patterns("__pycache__", "local_preferences.json"))
    data_files = sorted(f.name for f in (tool / "data").glob("*.json"))
    (OUT / "check.html").write_text(page("writing-style: check a draft", CHECK_BODY.replace("__DATA_FILES__", ",".join(data_files))), encoding="utf-8")
    print("site/index.html, site/skill.html, site/check.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
