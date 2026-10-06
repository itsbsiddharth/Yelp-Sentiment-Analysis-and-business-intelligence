"""Audit checks on 'Sentiment Analysis.ipynb' (VADER), copied from the reference's notebook 3.

Writes docs/audit/results/vader.json.

The notebook maps VADER's compound score c to a label with
    c < 0 -> negative,  0 <= c < 0.5 -> neutral,  c >= 0.5 -> positive      (cell 20)
VADER's authors recommend
    c <= -0.05 -> negative,  -0.05 < c < 0.05 -> neutral,  c >= 0.05 -> positive
This script scores every review once with NLTK's VADER (the implementation the notebook used),
caches the compound scores, and evaluates both rules on all rows and on the original
validation/test rows, next to the majority-class baseline. It also finds the best possible pair of
thresholds on the training rows (for macro-F1) to give VADER its fairest shot.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import LABELS, env_versions, load_reviews, majority_baseline, metrics, original_split, save

CACHE = Path(os.environ.get("AUDIT_CACHE", Path.home() / ".cache" / "yelp-audit"))


def compound_scores(review: pd.DataFrame) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / "vader_nltk_compound.npy"
    if f.exists():
        c = np.load(f)
        if len(c) == len(review):
            return c
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    t = time.time()
    # str(text) exactly as the notebook did, so the 6 missing texts become the string "nan"
    c = np.array([sid.polarity_scores(str(x))["compound"] for x in review["text"]])
    print(f"VADER scored {len(c)} reviews in {time.time() - t:.0f}s", flush=True)
    np.save(f, c)
    return c


def rule(c: np.ndarray, lo: float, hi: float, lo_inclusive: bool) -> np.ndarray:
    """negative if c < lo (or <= lo), positive if c >= hi, neutral otherwise."""
    neg = c <= lo if lo_inclusive else c < lo
    return np.where(neg, "negative", np.where(c >= hi, "positive", "neutral"))


def original_rule(c):
    return rule(c, 0.0, 0.5, lo_inclusive=False)


def recommended_rule(c):
    return rule(c, -0.05, 0.05, lo_inclusive=True)


def tune_thresholds(c: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import f1_score
    best = (-1, None, None)
    grid = np.round(np.arange(-0.95, 0.96, 0.05), 2)
    for lo in grid:
        for hi in grid[grid > lo]:
            s = f1_score(y, rule(c, lo, hi, lo_inclusive=False), labels=LABELS, average="macro")
            if s > best[0]:
                best = (s, float(lo), float(hi))
    return {"train_macro_f1": float(best[0]), "lo": best[1], "hi": best[2]}


def main() -> None:
    review = load_reviews()
    labels = review["stars"].map(lambda s: "positive" if s >= 4 else ("negative" if s <= 2 else "neutral")).to_numpy()
    c = compound_scores(review)
    split = original_split(review)
    idx = {"val": split["X_val"].index.to_numpy(), "test": split["X_test"].index.to_numpy(),
           "train": split["X_train"].index.to_numpy()}

    out = {"environment": env_versions(), "implementation": "nltk.sentiment.vader.SentimentIntensityAnalyzer",
           "rules": {"original": "c < 0 neg | 0 <= c < 0.5 neutral | c >= 0.5 pos",
                     "recommended": "c <= -0.05 neg | -0.05 < c < 0.05 neutral | c >= 0.05 pos"}}
    out["all_rows"] = {"n": int(len(c)),
                       "original_rule": metrics(labels, original_rule(c)),
                       "recommended_rule": metrics(labels, recommended_rule(c)),
                       "majority_baseline": majority_baseline(labels)}
    for part in ("val", "test"):
        i = idx[part]
        out[part] = {"n": int(len(i)),
                     "original_rule": metrics(labels[i], original_rule(c[i])),
                     "recommended_rule": metrics(labels[i], recommended_rule(c[i])),
                     "majority_baseline": majority_baseline(labels[i])}
    tuned = tune_thresholds(c[idx["train"]], labels[idx["train"]])
    tuned["val"] = metrics(labels[idx["val"]], rule(c[idx["val"]], tuned["lo"], tuned["hi"], False))
    out["thresholds_tuned_on_train_for_macro_f1"] = tuned
    out["mean_compound_by_true_label"] = pd.Series(c).groupby(labels).mean().round(4).to_dict()
    out["share_compound_ge_0.5_by_true_label"] = pd.Series(c >= 0.5).groupby(labels).mean().round(4).to_dict()
    out["n_text_missing_scored_as_string_nan"] = int(review["text"].isna().sum())
    save("vader", out)


if __name__ == "__main__":
    main()
