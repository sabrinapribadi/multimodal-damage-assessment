"""At-a-glance brief (portfolio rule R22): one source, docs/brief.yaml, written right after the title of the README,
the results page and the technical walkthrough, between <!-- brief:start --> and <!-- brief:end --> markers.

    python scripts/brief.py            # write
    python scripts/brief.py --check    # CI: fail if any target is out of date
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- brief:start -->", "<!-- brief:end -->"
CSS_START, CSS_END = "/* brief:css:start (rule R22, written by scripts/brief.py) */", "/* brief:css:end */"
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def html_inline(text: str, repo_url: str) -> str:
    def link(m):
        label, href = m.group(1), m.group(2)
        if not href.startswith(("http", "#")):
            href = f"{repo_url}/blob/main/{href}"
        return f'<a href="{html.escape(href)}">{html.escape(label)}</a>'
    out, last = [], 0
    for m in LINK.finditer(text):
        out.append(html.escape(text[last:m.start()]))
        out.append(link(m))
        last = m.end()
    out.append(html.escape(text[last:]))
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", "".join(out))


def markdown(b: dict) -> str:
    lines = ["## At a glance", ""]
    for i, item in enumerate(b["brief"], 1):
        lines.append(f"{i}. **{item['w']}:** {item['lines'][0]}")
        lines += [f"   - {x}" for x in item["lines"][1:]]
    lines += ["", "**Skills applied** (each linked to the file or decision that shows it):", ""]
    for s in b["skills"]:
        lines.append(f"- **{s['group']}:** {s['what']} — " + ", ".join(f"[{label}]({path})" for label, path in s["proof"]))
    return "\n".join(lines)


def page_html(b: dict, repo_url: str, compact: bool = False) -> str:
    items = []
    for i, item in enumerate(b["brief"], 1):
        rest = "".join(f'<span class="brief-sub">{html_inline(x, repo_url)}</span>' for x in item["lines"][1:])
        items.append(f'<li><span class="brief-w">{i} · {html.escape(item["w"])}</span>'
                     f'<span class="brief-main">{html_inline(item["lines"][0], repo_url)}</span>{"" if compact else rest}</li>')
    skills = "".join(
        f'<li><b>{html.escape(s["group"])}</b> {html.escape(s["what"])} '
        + " ".join(f'<a class="brief-proof" href="{html.escape(p if p.startswith("http") else f"{repo_url}/blob/main/{p}")}">{html.escape(label)}</a>'
                   for label, p in s["proof"])
        + "</li>" for s in b["skills"])
    return (f'<section class="brief" id="brief" aria-label="Project at a glance">'
            f'<span class="eyebrow">At a glance</span>'
            f'<ol class="brief-list">{"".join(items)}</ol>'
            f'<details class="brief-skills"{"" if compact else " open"}><summary>Skills applied, with proof</summary><ul>{skills}</ul></details></section>')


CSS = CSS_START + """
.brief { display: grid; gap: 12px; padding-block: 18px 4px; text-align: left; }
.brief a { color: var(--accent); }
.brief-list { list-style: none; margin: 0; padding: 0 !important; display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 10px; }
.brief-list li { display: grid; align-content: start; gap: 4px; padding: 12px 14px; min-width: 0;
  background: var(--surface); border: 1px solid var(--line, var(--border)); border-radius: 8px; }
.brief-w { font-size: 11.5px; letter-spacing: .06em; text-transform: uppercase; color: var(--accent); font-weight: 600; }
.brief-main { font-size: 14.5px; line-height: 1.45; color: var(--fg, var(--text)); }
.brief-sub { font-size: 13px; line-height: 1.45; color: var(--muted); }
.brief-skills summary { cursor: pointer; font-weight: 600; font-size: 14px; }
.brief-skills ul { margin: 8px 0 0; padding-left: 18px; display: grid; gap: 6px; font-size: 13.5px; line-height: 1.45; }
.brief-proof { font-size: 12px; margin-left: 4px; white-space: nowrap; }
""" + CSS_END


def place(text: str, block: str, after: str) -> str:
    """Remove any earlier brief, then put it right after the title, so the brief is the first thing read."""
    text = re.sub(r"\n?" + re.escape(START) + r".*?" + re.escape(END) + r"\n*", "\n", text, flags=re.S)
    i = text.index(after) + len(after)
    return text[:i] + f"\n{START}\n{block}\n{END}\n" + text[i:]


def targets(b: dict) -> dict[Path, str]:
    repo_url = b["repo_url"]
    readme = (ROOT / "README.md").read_text()
    title = readme.split("\n", 1)[0] + "\n"
    out = {ROOT / "README.md": place(readme, "\n" + markdown(b) + "\n", title)}
    for page, compact in (("index.html", False), ("build.html", True)):
        p = ROOT / "docs" / page
        if not p.exists():                                  # pages come later (R3, R14); the README is written now
            continue
        t = p.read_text()
        t = re.sub(re.escape(CSS_START) + r".*?" + re.escape(CSS_END) + r"\n?", "", t, flags=re.S)
        t = t.replace("</style>", CSS + "\n</style>", 1)
        out[p] = place(t, page_html(b, repo_url, compact), "</h1>")
    return out


def main() -> int:
    b = yaml.safe_load((ROOT / "docs" / "brief.yaml").read_text())
    stale = []
    for p, text in targets(b).items():
        if p.read_text() != text:
            stale.append(str(p.relative_to(ROOT)))
            if "--check" not in sys.argv:
                p.write_text(text)
    if "--check" in sys.argv and stale:
        print("brief out of date (run python scripts/brief.py):", *stale)
        return 1
    print("brief written:" if stale else "brief up to date", *stale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
