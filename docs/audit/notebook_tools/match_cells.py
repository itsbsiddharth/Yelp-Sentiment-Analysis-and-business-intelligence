"""Label every cell of this repo's notebooks by where it came from.

For each cell, find the most similar cell in ANY of the reference repo's notebooks
(difflib ratio on whitespace-normalised source):
    IDENTICAL  normalised source equals a reference cell
    MODIFIED   similarity >= 0.6 (a reference cell, edited)
    OWN        similarity < 0.6 (not traceable to the reference)
    EMPTY      empty cell
Writes docs/audit/results/cell_provenance.json.

Usage:
    python docs/audit/notebook_tools/match_cells.py --reference-repo PATH [NOTEBOOK ...]
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "docs" / "audit" / "results"


def src(c) -> str:
    s = c["source"]
    return "".join(s) if isinstance(s, list) else s


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-repo", type=Path, required=True)
    ap.add_argument("notebooks", nargs="*", type=Path)
    args = ap.parse_args()
    nbs = args.notebooks or sorted(REPO.glob("*.ipynb"))

    refs = []
    for p in sorted(args.reference_repo.glob("*.ipynb")):
        for i, c in enumerate(json.load(open(p))["cells"]):
            refs.append({"notebook": p.name, "cell": i, "norm": norm(src(c))})
    ref_set = {r["norm"]: r for r in refs if r["norm"]}

    out = {"reference_notebooks": sorted(p.name for p in args.reference_repo.glob("*.ipynb")),
           "thresholds": {"MODIFIED": ">= 0.6", "OWN": "< 0.6"}, "notebooks": {}}
    for p in nbs:
        cells = json.load(open(p))["cells"]
        rows, counts = [], {}
        for i, c in enumerate(cells):
            s = norm(src(c))
            if not s:
                lab, best, score = "EMPTY", None, 0.0
            elif s in ref_set:
                lab, best, score = "IDENTICAL", ref_set[s], 1.0
            else:
                best, score = None, 0.0
                for r in refs:
                    if not r["norm"]:
                        continue
                    sm = difflib.SequenceMatcher(None, s, r["norm"], autojunk=False)
                    if sm.real_quick_ratio() < score or sm.quick_ratio() < score:
                        continue
                    q = sm.ratio()
                    if q > score:
                        best, score = r, q
                lab = "MODIFIED" if score >= 0.6 else "OWN"
            counts[lab] = counts.get(lab, 0) + 1
            rows.append({"cell": i, "type": c["cell_type"], "label": lab, "similarity": round(score, 2),
                         "closest_reference": (f"{best['notebook']} cell {best['cell']}"
                                               if best and lab != "OWN" else None),
                         "first_line": s[:100]})
        out["notebooks"][p.name] = {"counts": counts, "cells": rows}
        print(p.name, counts)
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RESULTS / "cell_provenance.json", "w"), indent=1)
    print("wrote docs/audit/results/cell_provenance.json")


if __name__ == "__main__":
    main()
