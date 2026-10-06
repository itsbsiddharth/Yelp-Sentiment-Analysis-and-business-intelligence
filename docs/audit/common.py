"""Shared helpers for the Phase 0 audit scripts.

Every number quoted in docs/AUDIT.md is written to docs/audit/results/*.json by one of the
scripts in this folder. This module holds what they share: paths, data loading, the original
labelling rule, an exact replica of the original train/validation/test split, and metric helpers.

The scripts are meant to run in the reconstructed *original* environment
(docs/audit/requirements-audit.txt), because the goal is to reproduce the original numbers.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

LABELS = ["negative", "neutral", "positive"]  # alphabetical = LabelEncoder order 0, 1, 2
SEED = 42  # the seed the original notebooks used everywhere they set one


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reviews() -> pd.DataFrame:
    return pd.read_csv(DATA / "review_prepared.csv")


def load_business() -> pd.DataFrame:
    return pd.read_csv(DATA / "business_prepared.csv")


def load_users() -> pd.DataFrame:
    return pd.read_csv(DATA / "user_prepared.csv")


def star_to_label(s: int) -> str:
    """The original 3-class rule (Modeling and Evaluation.ipynb, cell 13)."""
    return "positive" if s >= 4 else ("negative" if s <= 2 else "neutral")


def original_split(reviews: pd.DataFrame | None = None) -> dict:
    """Replicate the original split exactly (Modeling and Evaluation.ipynb, cells 5-19 and 38).

    1. keep text + stars, drop the 6 rows with missing text
    2. 67/33 train/test split, random_state=42, NOT stratified
    3. the 67% part is later split 70/30 into train/validation, random_state=42, NOT stratified

    In the original, step 3 happens *after* TF-IDF and scaling were fit on the whole 67% part.
    Here we split the raw text so callers can choose to fit on the 70% part only (no leakage)
    or on the whole 67% part (as the original did). Row order matches the original exactly
    because train_test_split only depends on n_samples and the seed.
    """
    if reviews is None:
        reviews = load_reviews()
    df = reviews[["text", "stars"]].reset_index().drop(columns="index")
    df = df.dropna()
    df["label"] = df["stars"].apply(star_to_label)
    from sklearn.model_selection import train_test_split

    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        df["text"], df["label"], test_size=0.33, random_state=SEED
    )
    # The original split the *vectorised* matrix; splitting the index gives the same rows.
    idx = np.arange(len(X_tr_full))
    i_train, i_val = train_test_split(idx, test_size=0.3, random_state=SEED)
    return {
        "df": df,
        "X_train_full": X_tr_full, "y_train_full": y_tr_full,   # 67% (train + validation)
        "X_test": X_test, "y_test": y_test,                     # 33%
        "i_train": i_train, "i_val": i_val,                     # positions inside the 67%
        "X_train": X_tr_full.iloc[i_train], "y_train": y_tr_full.iloc[i_train],
        "X_val": X_tr_full.iloc[i_val], "y_val": y_tr_full.iloc[i_val],
    }


def metrics(y_true, y_pred) -> dict:
    """Accuracy, macro-F1, weighted-F1, per-class precision/recall/F1 and the confusion matrix."""
    rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "per_class": {c: {k: float(rep[c][k]) for k in ("precision", "recall", "f1-score", "support")}
                      for c in LABELS},
        "confusion_matrix": {"labels": LABELS,
                             "rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=LABELS).tolist()},
    }


def metrics_from_confusion(cm: list[list[int]]) -> dict:
    """Recompute accuracy / macro-F1 from a stored confusion matrix (rows = true, cols = predicted)."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return {"n": int(cm.sum()), "accuracy": float(tp.sum() / cm.sum()), "macro_f1": float(f1.mean()),
            "per_class_f1": dict(zip(LABELS, map(float, f1))),
            "per_class_recall": dict(zip(LABELS, map(float, rec))),
            "per_class_precision": dict(zip(LABELS, map(float, prec)))}


def majority_baseline(y_true) -> dict:
    """Always predict the most common class of y_true ("positive" here)."""
    y_true = pd.Series(y_true)
    maj = y_true.value_counts().idxmax()
    out = metrics(y_true, [maj] * len(y_true))
    out["predicted_class"] = maj
    return out


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(centre - half), float(centre + half))


def save(name: str, obj: dict) -> Path:
    path = RESULTS / f"{name}.json"
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", path.relative_to(REPO))
    return path


def env_versions() -> dict:
    import platform
    import sklearn
    out = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
           "scikit-learn": sklearn.__version__}
    for mod in ("xgboost", "nltk", "vaderSentiment", "textblob", "spacy", "scipy"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - optional dependency
            pass
    return out


def nltk_data_dir() -> str | None:
    return os.environ.get("NLTK_DATA")
