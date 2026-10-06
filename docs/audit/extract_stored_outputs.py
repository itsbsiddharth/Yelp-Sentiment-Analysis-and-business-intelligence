"""Extract every number the original notebooks report, straight from their saved outputs.

Writes docs/audit/results/stored_outputs.json. This is the "original value" column of the verdict
table in docs/AUDIT.md, so nothing there is retyped by hand.

Usage:
    python docs/audit/extract_stored_outputs.py [--reference-repo PATH]

With --reference-repo it also extracts the reference notebook 4's outputs and checks whether the
five model results pasted as markdown in this repo's modeling notebook are identical to them.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from common import REPO, metrics_from_confusion, save

NB = {
    "eda": REPO / "Data preprocessing and EDA.ipynb",
    "business": REPO / " Business Case Data Analysis.ipynb",
    "sentiment": REPO / "Sentiment Analysis.ipynb",
    "modeling": REPO / "Modeling and Evaluation.ipynb",
}
FLOAT = r"[-+]?\d*\.\d+"


def cells(path: Path) -> list[dict]:
    return json.load(open(path))["cells"]


def src(cell) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def texts(cell) -> str:
    """All text output of a code cell (stdout + text/plain results)."""
    out = []
    for o in cell.get("outputs", []) or []:
        if o.get("output_type") == "stream":
            out.append("".join(o["text"]) if isinstance(o["text"], list) else o["text"])
        elif "data" in o and "text/plain" in o["data"]:
            t = o["data"]["text/plain"]
            out.append("".join(t) if isinstance(t, list) else t)
    return "\n".join(out)


def parse_report_text(t: str) -> dict:
    """Parse Classification.get_scores' report table (rows precision/recall/f1-score; cols 0,1,2,acc,macro)."""
    rows = {}
    for line in t.splitlines():
        parts = line.split()
        if parts and parts[0] in ("precision", "recall", "f1-score"):
            nums = [float(x) for x in parts[1:6]]
            rows[parts[0]] = nums
    return report_dict(rows)


def parse_report_markdown(t: str) -> dict:
    rows = {}
    for line in t.splitlines():
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if cols and cols[0] in ("precision", "recall", "f1-score"):
            rows[cols[0]] = [float(x) for x in cols[1:6]]
    return report_dict(rows)


def report_dict(rows: dict) -> dict:
    names = ["negative(0)", "neutral(1)", "positive(2)"]
    return {
        "val_accuracy": rows["f1-score"][3],
        "val_macro_f1": rows["f1-score"][4],
        "val_macro_precision": rows["precision"][4],
        "val_macro_recall": rows["recall"][4],
        "per_class": {n: {"precision": rows["precision"][i], "recall": rows["recall"][i], "f1": rows["f1-score"][i]}
                      for i, n in enumerate(names)},
    }


def parse_model_block(text: str) -> dict:
    m = re.search(rf"(Logistic Regression|Multinomial Naive Bayes|Random Forest|Decision Tree|KNN|AdaBoost|XGBoost)"
                  rf"\s*\|?\s*({FLOAT})\s*\|?\s*({FLOAT})", text)
    hp = re.search(r"The best hyperparameters are:\s*(\{[^}]*\})", text)
    cpu = re.search(r"CPU times:[^\n]*", text)
    return {"model": m.group(1), "train_accuracy": float(m.group(2)), "val_accuracy_scores_table": float(m.group(3)),
            "best_params": hp.group(1) if hp else None, "timing_line": cpu.group(0) if cpu else None}


def modeling(nb: list[dict]) -> dict:
    out = {"executed": {}, "pasted_as_markdown": {}}
    for i, c in enumerate(nb):
        s = src(c)
        if c["cell_type"] == "code" and "get_scores(parameters, skf)" in s and not s.lstrip().startswith("#"):
            t = texts(c)
            d = parse_model_block(t)
            d.update(parse_report_text(t))
            d["cell"] = i
            out["executed"][d["model"]] = d
        if c["cell_type"] == "markdown" and s.lstrip().startswith("Out["):
            d = parse_model_block(s)
            d.update(parse_report_markdown(s))
            d["cell"] = i
            out["pasted_as_markdown"][d["model"]] = d
        if c["cell_type"] == "code" and "print(X_train_scaled.shape)" in s:
            out["X_train_scaled_shape_output"] = texts(c).splitlines()[0]
        if c["cell_type"] == "code" and s.strip() == "models":
            out["models_comparison_table_output"] = texts(c)
    return out


def sentiment(nb: list[dict]) -> dict:
    out = {}
    for i, c in enumerate(nb):
        s, t = src(c), texts(c)
        if "accuracy_score(reviews.label, reviews.compound_label)" in s and t:
            out["vader_accuracy"] = float(re.search(FLOAT, t).group(0))
        if s.startswith("print(confusion_matrix(reviews.label, reviews.compound_label))"):
            out["vader_confusion_matrix"] = [[int(x) for x in re.findall(r"\d+", ln)]
                                             for ln in t.strip().splitlines()]
        if "compound_label'] = reviews['compound_score'].apply" in s:
            out["vader_rule_code"] = re.search(r"lambda s: (.*)\)", s.split("compound_label'] =")[1]).group(1)
        if "accuracy_score(reviews.label, reviews.textblob_sentiment)" in s and t:
            out["textblob_accuracy"] = float(re.search(FLOAT, t).group(0))
        if "cm = confusion_matrix(reviews.label, reviews.textblob_sentiment)" in s:
            out["textblob_confusion_matrix"] = [[int(x) for x in re.findall(r"\d+", ln)]
                                                for ln in t.strip().splitlines()[1:]]
        if "disagrees with original sentiment" in s and t:
            out["textblob_disagreement_pct"] = float(re.search(r"([\d.]+)%", t).group(1))
        if "print(reviews.groupby('label')['textblob_polarity'].mean())" in s:
            out["textblob_mean_polarity_by_label"] = {k: float(v) for k, v in re.findall(r"(negative|neutral|positive)\s+([\d.]+)", t)}
        if "classification_report(reviews.label, reviews.compound_label)" in s:
            out["vader_classification_report_cell_has_output"] = bool(t.strip())
    out["vader_from_confusion_matrix"] = metrics_from_confusion(out["vader_confusion_matrix"])
    out["textblob_from_confusion_matrix"] = metrics_from_confusion(out["textblob_confusion_matrix"])
    return out


def eda(nb: list[dict]) -> dict:
    out = {}
    for c in nb:
        s, t = src(c), texts(c)
        if "Mean 'cool' votes per reviewer" in s:
            out["mean_cool_votes_per_reviewer_printed"] = float(re.search(FLOAT, t).group(0))
        if "39,450" in s:
            out["comment_zero_cool_reviewers"] = s.strip().splitlines()[-1]
        if "500 to 2000" in s:
            out["comment_review_length_claim"] = " ".join(x.strip("# ") for x in s.strip().splitlines())
        if "correlation = review['reviewer_cool'].corr(review['stars'])" in s:
            out["corr_reviewer_cool_stars_printed"] = float(re.search(r"-?\d\.\d+(?:e-?\d+)?", t).group(0))
    return out


def business(nb: list[dict]) -> dict:
    out = {"markdown_claims": {}}
    for i, c in enumerate(nb):
        s, t = src(c), texts(c)
        if c["cell_type"] == "markdown" and ("Top three rated" in s or "US Airways" in s or "positive linear climb" in s):
            out["markdown_claims"][f"cell_{i}"] = s
        if "print('Total number of reviews: ', len(review))" in s:
            out["review_counts_printed"] = dict(re.findall(r"(\w[\w ]*?):\s+(\d+)", t))
        if "Sum of number reviews per category" in s:
            out["sum_of_categories_with_1_to_6_occurrences_printed"] = int(re.search(r"(\d+)\s*$", t.strip()).group(1))
        if "Other Categories" in s and c["cell_type"] == "code":
            out[f"hardcoded_other_categories_cell_{i}"] = re.findall(r"loc\[(\d+)\] = \['Other Categories', (\d+)\]", s)
        if "sort_values(by='business_review_count', ascending=False)[:" in s:
            out["top_rated_slice"] = re.search(r"ascending=False\)\[:(\d+)\]", s).group(1)
            out["top_rated_title"] = re.search(r"plt.title\('([^']*)'", s).group(1)
            out["top_rated_xlabel"] = re.search(r"plt.xlabel\('([^']*)'", s).group(1)
        if ".drop([0, 1, 10, 17])" in s:
            out["top_bad_words_rows_dropped_by_position"] = [0, 1, 10, 17]
    return out


def compare_with_reference(mine: dict, reference_repo: Path) -> dict:
    ref_nb = cells(reference_repo / "4. Modeling and Evaluation.ipynb")
    ref = modeling(ref_nb)["executed"]
    out = {}
    for name, d in mine["pasted_as_markdown"].items():
        r = ref[name]
        keys = ["train_accuracy", "val_accuracy", "val_macro_f1", "best_params"]
        out[name] = {"identical_numbers_and_params": all(d[k] == r[k] for k in keys) and d["per_class"] == r["per_class"],
                     "identical_timing_line": d["timing_line"] == r["timing_line"],
                     "timing_line_here": d["timing_line"], "timing_line_reference": r["timing_line"]}
    for name, d in mine["executed"].items():
        r = ref[name]
        out[name] = {"identical_numbers_and_params": d["val_accuracy"] == r["val_accuracy"]
                     and d["per_class"] == r["per_class"] and d["best_params"] == r["best_params"],
                     "timing_line_here": d["timing_line"], "timing_line_reference": r["timing_line"],
                     "note": "executed in this repo's notebook (Windows timing format)"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-repo", type=Path, default=None)
    args = ap.parse_args()
    out = {"modeling": modeling(cells(NB["modeling"])), "sentiment": sentiment(cells(NB["sentiment"])),
           "eda": eda(cells(NB["eda"])), "business": business(cells(NB["business"]))}
    if args.reference_repo:
        out["modeling_vs_reference_notebook4"] = compare_with_reference(out["modeling"], args.reference_repo)
    save("stored_outputs", out)


if __name__ == "__main__":
    main()
