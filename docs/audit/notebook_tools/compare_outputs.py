"""Compare the outputs saved in the original notebooks with the outputs of a fresh re-run.

Text outputs are compared after removing lines that legitimately differ between machines
(timings, progress bars, download logs, pip output, memory usage, file paths in warnings).
Images are only checked for presence: pixel-exact plots are not expected across library versions.
Writes docs/audit/results/notebook_rerun.json.

Usage:
    python docs/audit/notebook_tools/compare_outputs.py RERUN_DIR [RUN_SUMMARY_JSON]
RERUN_DIR holds executed copies (same file names) written by run_notebooks.py.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "docs" / "audit" / "results"

NOISE = re.compile(r"CPU times|Wall time|\[nltk_data\]|it/s\]|Processing batches|Requirement already|"
                   r"memory usage|Warning|warn\(|^\s*$|Downloading|Obtaining|Collecting|Successfully|"
                   r"\d+%\|", re.I)


def src(c) -> str:
    s = c["source"]
    return "".join(s) if isinstance(s, list) else s


def text_out(c) -> tuple[str, int]:
    parts, images = [], 0
    for o in c.get("outputs", []) or []:
        if o.get("output_type") == "stream":
            if o.get("name") == "stderr":
                continue
            t = o["text"]
            parts.append("".join(t) if isinstance(t, list) else t)
        elif o.get("output_type") in ("execute_result", "display_data"):
            d = o.get("data", {})
            if "image/png" in d:
                images += 1
            elif "text/plain" in d:
                t = d["text/plain"]
                parts.append("".join(t) if isinstance(t, list) else t)
        elif o.get("output_type") == "error":
            parts.append(f"ERROR {o.get('ename')}: {o.get('evalue')}")
    lines = [ln.rstrip() for ln in "\n".join(parts).splitlines() if not NOISE.search(ln)]
    return "\n".join(lines), images


def main() -> None:
    rerun_dir = Path(sys.argv[1])
    summary = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
    out = {}
    for p in sorted(REPO.glob("*.ipynb")):
        q = rerun_dir / p.name.strip()
        if not q.exists():
            continue
        a, b = json.load(open(p))["cells"], json.load(open(q))["cells"]
        rows = []
        stats = {"code_cells": 0, "with_saved_text_output": 0, "identical": 0, "different": 0,
                 "saved_images": 0, "rerun_images": 0, "rerun_errors": 0, "never_executed_in_original": 0}
        for i, (ca, cb) in enumerate(zip(a, b)):
            if ca["cell_type"] != "code":
                continue
            stats["code_cells"] += 1
            if ca.get("execution_count") is None and src(ca).strip():
                stats["never_executed_in_original"] += 1
            ta, ia = text_out(ca)
            tb, ib = text_out(cb)
            stats["saved_images"] += ia
            stats["rerun_images"] += ib
            if tb.startswith("ERROR") or "\nERROR " in tb:
                stats["rerun_errors"] += 1
            if not ta:
                continue
            stats["with_saved_text_output"] += 1
            if ta == tb:
                stats["identical"] += 1
            else:
                stats["different"] += 1
                rows.append({"cell": i, "source_first_line": src(ca).strip().splitlines()[0][:80],
                             "saved": ta[:600], "rerun": tb[:600]})
        out[p.name] = {"stats": stats, "differences": rows,
                       "run": {k: v for k, v in summary.get(p.name.strip(), {}).items() if k != "patches"},
                       "patches_applied": summary.get(p.name.strip(), {}).get("patches", [])}
        print(p.name, stats)
    json.dump(out, open(RESULTS / "notebook_rerun.json", "w"), indent=1)
    print("wrote docs/audit/results/notebook_rerun.json")


if __name__ == "__main__":
    main()
