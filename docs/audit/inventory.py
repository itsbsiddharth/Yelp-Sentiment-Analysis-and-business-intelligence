"""Inventory of the original project (git tag v0-original).

Writes docs/audit/results/inventory.json: every tracked file with its size in git (for LFS files
that is the pointer) and its real size, LFS status, per-notebook cell/output counts, execution-order
checks, kernel versions, the library versions printed by the notebooks' `!pip install` cells, and
the full commit history.
"""
from __future__ import annotations

import json
import re
import subprocess

from common import REPO, save

TAG = "v0-original"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout


def files() -> list[dict]:
    lfs = {line.split(" ", 2)[2].strip() for line in git("lfs", "ls-files", TAG).splitlines() if line.strip()}
    rows = []
    for line in git("ls-tree", "-r", "-l", TAG).splitlines():
        meta, path = line.split("\t", 1)
        size_in_git = int(meta.split()[3])
        real = (REPO / path)
        rows.append({"path": path, "bytes_in_git": size_in_git,
                     "bytes_on_disk_after_lfs_pull": real.stat().st_size if real.exists() else None,
                     "git_lfs": path in lfs})
    return rows


def notebooks() -> dict:
    out = {}
    for p in sorted(REPO.glob("*.ipynb")):
        nb = json.load(open(p))
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        counts = [c.get("execution_count") for c in code]
        ran = [x for x in counts if x is not None]
        pip_text = ""
        for c in code:
            for o in c.get("outputs", []) or []:
                if o.get("output_type") == "stream":
                    pip_text += "".join(o["text"]) if isinstance(o["text"], list) else o["text"]
        versions = dict(re.findall(r"Requirement already satisfied: ([\w\-]+) in [^\n]*?\(([\d.]+[\w.]*)\)", pip_text))
        out[p.name] = {
            "cells": len(nb["cells"]), "code_cells": len(code), "markdown_cells": len(nb["cells"]) - len(code),
            "code_cells_with_outputs": sum(bool(c.get("outputs")) for c in code),
            "code_cells_never_run": sum(1 for c in code if c.get("execution_count") is None
                                        and "".join(c["source"]).strip()),
            "image_outputs": sum("image/png" in o.get("data", {}) for c in code for o in c.get("outputs", []) or []),
            "execution_counts": counts,
            "executed_top_to_bottom_in_one_run": ran == list(range(1, len(ran) + 1)),
            "kernel_python": nb["metadata"].get("language_info", {}).get("version"),
            "library_versions_printed_by_pip_cells": versions,
            "hardcoded_windows_paths": sum("C:/Users" in "".join(c["source"]) or "C:\\Users" in "".join(c["source"])
                                           for c in code),
        }
    return out


def history() -> list[dict]:
    rows = []
    log = git("log", TAG, "--format=@@%h|%ad|%an|%s", "--date=iso", "--name-status")
    for block in log.split("@@")[1:]:
        head, *changes = [x for x in block.strip().splitlines() if x.strip()]
        h, date, author, subject = head.split("|", 3)
        rows.append({"commit": h, "date": date, "author": author, "message": subject,
                     "changes": [c.replace("\t", " ") for c in changes]})
    return rows


def main() -> None:
    gitattributes = (REPO / ".gitattributes").read_text().strip()
    save("inventory", {"tag": TAG, "commit": git("rev-parse", TAG + "^{commit}").strip(),
                       "gitattributes": gitattributes, "files": files(), "notebooks": notebooks(),
                       "history": history()})


if __name__ == "__main__":
    main()
