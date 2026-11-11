"""Evidence table for every cited paper (portfolio rule R2): venue, peer review, year, citations.

    python scripts/evidence.py            # writes data/outputs/references_evidence.json and prints a table per ADR

Citations come from OpenAlex (free, no key) and change over time, so the retrieval date is stored with them.
Counts are evidence, never a gate: a new paper is judged on venue and peer review, not on citations.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.openalex.org/works"


def fetch(ref: dict) -> dict:
    if ref.get("kind") in ("dataset", "regulation"):    # no citation search (a title match could be any paper)
        return {"key": ref["key"], "title": ref["title"], "venue": ref["venue"], "peer_reviewed": False, "year": None, "citations": None,
                "citations_per_year": None, "role": ref.get("role"), "used_in": ref.get("used_in", []), "openalex_id": None, "kind": ref["kind"]}
    if ref.get("doi"):
        url = f"{API}/doi:{ref['doi']}"
    else:
        url = f"{API}?search={urllib.parse.quote(ref['title'])}&per-page=1"
    w = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "portfolio-evidence/1.0"}), timeout=30))
    if "results" in w:
        w = w["results"][0] if w["results"] else {}
    src = (w.get("primary_location") or {}).get("source") or {}
    year = w.get("publication_year")
    cites = w.get("cited_by_count", 0)
    is_preprint = bool(ref.get("arxiv")) or src.get("type") == "repository" or w.get("type") == "preprint"
    # a reference may name its venue when OpenAlex's primary copy is a repository mirror (stated in references.yaml)
    venue = ref.get("venue") or ("arXiv " + ref["arxiv"] if ref.get("arxiv") else src.get("display_name"))
    peer = ref["peer_reviewed"] if "peer_reviewed" in ref else not is_preprint
    return {"key": ref["key"], "title": w.get("title"), "venue": venue,
            "peer_reviewed": peer, "year": year, "citations": cites,
            "citations_per_year": round(cites / max(1, dt.date.today().year - (year or dt.date.today().year) + 1), 1),
            "role": ref.get("role"), "used_in": ref.get("used_in", []), "openalex_id": w.get("id"), "kind": ref.get("kind", "paper")}


def review_label(r: dict) -> str:
    if r["kind"] == "magazine":
        return "no (practitioner journal)"
    if r["kind"] in ("dataset", "book", "report", "regulation"):
        return f"n/a ({r['kind']})" if r["kind"] != "report" else "no (technical report)"
    return "yes" if r["peer_reviewed"] else "no (preprint)"


def main() -> None:
    refs = yaml.safe_load((ROOT / "docs" / "references.yaml").read_text())
    rows = [fetch(r) for r in refs]
    out = {"retrieved": dt.date.today().isoformat(), "source": "OpenAlex", "references": rows}
    (ROOT / "data" / "outputs").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "outputs" / "references_evidence.json").write_text(json.dumps(out, indent=1))
    for adr in sorted({a for r in rows for a in r["used_in"]}):
        print(f"\n### {adr} — evidence (OpenAlex, retrieved {out['retrieved']})\n")
        print("| Reference | Venue | Peer-reviewed | Year | Citations | Per year | Used for |")
        print("|---|---|---|---|---|---|---|")
        for r in rows:
            if adr in r["used_in"]:
                print(f"| {r['title']} | {r['venue']} | {review_label(r)} | {r['year'] or '—'} | {'—' if r['citations'] is None else format(r['citations'], ',')} | {'—' if r['citations_per_year'] is None else r['citations_per_year']} | {r['role']} |")


if __name__ == "__main__":
    sys.exit(main())
