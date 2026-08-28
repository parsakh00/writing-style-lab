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


def md_to_html(md: str) -> str:
    import markdown
    return markdown.markdown(md, extensions=["tables", "fenced_code"])


def page(title: str, body: str) -> str:
    nav = '<nav><a href="index.html">Overview</a><a href="skill.html">The skill</a><a href="https://github.com/parsakh00/writing-style-lab">Repository</a></nav>'
    return f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{CSS}</style></head><body>{nav}{body}</body></html>"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    skill = (ROOT / ".claude/skills/writing-style/SKILL.md").read_text(encoding="utf-8")
    skill = re.sub(r"^---\n.*?\n---\n", "", skill, flags=re.S)  # drop the frontmatter
    (OUT / "index.html").write_text(page("writing-style", md_to_html(readme)), encoding="utf-8")
    (OUT / "skill.html").write_text(page("writing-style: the skill", md_to_html(skill)), encoding="utf-8")
    print("site/index.html, site/skill.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
