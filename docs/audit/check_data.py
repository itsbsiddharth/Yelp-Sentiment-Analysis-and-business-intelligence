"""Audit checks on the data itself and on the EDA notebook's claims.

Writes docs/audit/results/data.json.

Usage:
    python docs/audit/check_data.py [--reference-repo PATH_TO_adzict/yelp_sentiment_analysis]

--reference-repo is optional. When given (with data/review_prepared.csv unzipped next to the zip),
the script checks whether this repo's CSVs are the reference's CSVs passed through the last cell of
"Data preprocessing and EDA.ipynb".
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA, LABELS, load_business, load_reviews, load_users, original_split, save, sha256


def dataset_scope(review: pd.DataFrame, business: pd.DataFrame, users: pd.DataFrame) -> dict:
    d = pd.to_datetime(review["date"])
    per_year = d.dt.year.value_counts().sort_index()
    stars = review["stars"].value_counts().sort_index()
    labels = review["stars"].map(lambda s: "positive" if s >= 4 else ("negative" if s <= 2 else "neutral"))
    return {
        "n_reviews": int(len(review)),
        "n_reviews_with_text": int(review["text"].notna().sum()),
        "n_reviews_missing_text": int(review["text"].isna().sum()),
        "n_businesses": int(len(business)),
        "n_business_ids_in_reviews": int(review["business_id"].nunique()),
        "n_user_rows": int(len(users)),
        "n_user_ids_in_reviews": int(review["user_id"].nunique()),
        "n_reviews_from_users_missing_from_user_file": int(review["reviewer_blank"].sum()),
        "date_min": str(d.min().date()), "date_max": str(d.max().date()),
        "reviews_per_year": {int(k): int(v) for k, v in per_year.items()},
        "first_review_2005": str(d[d.dt.year == 2005].min().date()),
        "last_review_2013": str(d[d.dt.year == 2013].max().date()),
        "states_reviews": review["business_state"].value_counts().to_dict(),
        "states_businesses": business["business_state"].value_counts().to_dict(),
        "top_cities_businesses": business["business_city"].value_counts().head(8).to_dict(),
        "stars_counts": {int(k): int(v) for k, v in stars.items()},
        "label_counts_all_rows": labels.value_counts().reindex(LABELS).astype(int).to_dict(),
        "label_share_all_rows": labels.value_counts(normalize=True).reindex(LABELS).round(6).to_dict(),
    }


def provenance(reference_repo: Path | None) -> dict:
    """Is data/*.csv = reference CSVs re-saved by the last cell of the EDA notebook?

    That cell is: review['review_length'] = review['text'].str.len() (cell 26), then
    df.to_csv(path, index=False) for all three frames (cell 64), with the reference's unnamed index
    column read back in as 'Unnamed: 0'. We redo exactly that and compare bytes.
    """
    out = {"committed_sha256": {p.name: sha256(p) for p in sorted(DATA.glob("*.csv"))}}
    if reference_repo is None:
        out["reference_check"] = "skipped (no --reference-repo given)"
        return out
    ref = reference_repo / "data"
    tmp = Path(__file__).resolve().parent / "results" / "_tmp_provenance"
    tmp.mkdir(exist_ok=True)
    checks = {}
    for name in ("business_prepared", "user_prepared", "review_prepared"):
        src = ref / f"{name}.csv"
        if not src.exists():
            src = ref / f"{name}.csv.zip"  # the reference ships the review file zipped
        df = pd.read_csv(src)
        if name == "review_prepared":
            df["review_length"] = df["text"].str.len()
        target = tmp / f"{name}.csv"
        # The notebook ran on Windows, where pandas ends rows with os.linesep = "\r\n".
        df.to_csv(target, index=False, lineterminator="\r\n")
        checks[name] = {
            "reference_file": src.name,
            "reference_sha256": sha256(src),
            "recreated_sha256": sha256(target),
            "byte_identical_to_committed": sha256(target) == out["committed_sha256"][f"{name}.csv"],
        }
        target.unlink()
    tmp.rmdir()
    out["reference_check"] = checks
    return out


def _normalise(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def duplicates_across_splits(split: dict) -> dict:
    """Exact and normalised duplicate texts inside the data and across the original splits."""
    df = split["df"]
    text = df["text"]
    norm = text.map(_normalise)
    out = {
        "rows_with_text": int(len(df)),
        "exact_duplicate_rows_beyond_first": int(text.duplicated().sum()),
        "exact_duplicate_text_groups": int((text.value_counts() > 1).sum()),
        "normalised_duplicate_rows_beyond_first": int(norm.duplicated().sum()),
        "rows_that_are_full_row_duplicates": "0 (review_id is unique: "
        + str(load_reviews()["review_id"].is_unique) + ")",
    }
    # Same text with different labels (cannot both be right for a text-only model)
    g = df.groupby("text")["label"].nunique()
    out["duplicate_texts_with_conflicting_labels"] = int((g > 1).sum())
    out["most_common_duplicate_texts"] = text.value_counts().head(5).to_dict()

    tr = set(split["X_train"])
    va = set(split["X_val"])
    te = split["X_test"]
    tr_full = set(split["X_train_full"])
    out["test_rows_whose_exact_text_is_in_train_or_val"] = int(te.isin(tr_full).sum())
    out["val_rows_whose_exact_text_is_in_train"] = int(split["X_val"].isin(tr).sum())
    ntr_full = set(split["X_train_full"].map(_normalise))
    out["test_rows_whose_normalised_text_is_in_train_or_val"] = int(te.map(_normalise).isin(ntr_full).sum())
    out["n_test"] = int(len(te))
    out["n_val"] = int(len(split["X_val"]))
    out["test_share_exact_dup_of_train"] = out["test_rows_whose_exact_text_is_in_train_or_val"] / len(te)
    _ = va
    return out


def near_duplicates_sample(split: dict, n_sample: int = 2000, seed: int = 0) -> dict:
    """Near-duplicates: cosine similarity of word-count vectors, sampled test reviews vs ALL train+val.

    A sample keeps this cheap (2,000 x ~153k sparse dot product). Thresholds 0.9 and 0.8.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    rng = np.random.default_rng(seed)
    te = split["X_test"].sample(n=n_sample, random_state=seed)
    tr = split["X_train_full"]
    vec = TfidfVectorizer(lowercase=True, min_df=2)  # plain unigram TF-IDF, l2-normalised rows
    A = vec.fit_transform(tr)
    B = vec.transform(te)
    best = np.zeros(n_sample)
    step = 250
    for s in range(0, n_sample, step):
        sims = (B[s:s + step] @ A.T).toarray()
        best[s:s + step] = sims.max(axis=1)
    _ = rng
    return {
        "n_test_sampled": n_sample, "sample_seed": seed,
        "share_max_cosine_ge_0.99": float((best >= 0.99).mean()),
        "share_max_cosine_ge_0.9": float((best >= 0.9).mean()),
        "share_max_cosine_ge_0.8": float((best >= 0.8).mean()),
        "median_max_cosine": float(np.median(best)),
    }


def split_balance(split: dict) -> dict:
    def share(y):
        return pd.Series(y).value_counts(normalize=True).reindex(LABELS).round(6).to_dict()

    return {"train_70pct_of_67pct": share(split["y_train"]), "val_30pct_of_67pct": share(split["y_val"]),
            "test_33pct": share(split["y_test"]), "all": share(split["df"]["label"]),
            "sizes": {"train": len(split["y_train"]), "val": len(split["y_val"]), "test": len(split["y_test"])}}


def eda_claims(review: pd.DataFrame, users: pd.DataFrame) -> dict:
    """Claims made in 'Data preprocessing and EDA.ipynb'."""
    first_per_user = review.drop_duplicates("user_id")
    return {
        # cell 22 prints review['reviewer_cool'].mean() and calls it "per reviewer"
        "mean_reviewer_cool_over_review_rows": float(review["reviewer_cool"].mean()),
        "mean_reviewer_cool_over_unique_users_in_reviews": float(first_per_user["reviewer_cool"].mean()),
        "mean_reviewer_cool_over_user_file": float(users["reviewer_cool"].mean()),
        # cell 19 comment says 39,450 reviewers have 0 cool votes; cell 20 shows review rows
        "review_rows_with_reviewer_cool_eq_0": int((review["reviewer_cool"] == 0).sum()),
        "unique_users_with_reviewer_cool_eq_0": int((first_per_user["reviewer_cool"] == 0).sum()),
        # reviewer_average_stars == 0 is impossible for a real 1-5 average: it marks missing users
        "review_rows_with_reviewer_average_stars_eq_0": int((review["reviewer_average_stars"] == 0).sum()),
        "review_rows_with_reviewer_blank_true": int(review["reviewer_blank"].sum()),
        "rows_avg0_and_blank": int(((review["reviewer_average_stars"] == 0) & review["reviewer_blank"]).sum()),
        "rows_avg0_and_not_blank": int(((review["reviewer_average_stars"] == 0) & ~review["reviewer_blank"]).sum()),
        "corr_reviewer_cool_vs_stars": float(review["reviewer_cool"].corr(review["stars"])),
        # cell 27 claims 500-2000 char reviews "get more cool votes" using the *reviewer's* lifetime cool
        # count. The review's own cool votes are in column 'cool':
        "spearman_review_cool_vs_review_length": float(
            review[["cool", "review_length"]].corr(method="spearman").iloc[0, 1]),
        "spearman_reviewer_cool_vs_review_length": float(
            review[["reviewer_cool", "review_length"]].corr(method="spearman").iloc[0, 1]),
        "review_length_definition": "characters (text.str.len()), matches committed column for "
        + str(int((review["text"].str.len() == review["review_length"]).sum())) + " of "
        + str(int(review["review_length"].notna().sum())) + " non-null rows",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-repo", type=Path, default=None)
    args = ap.parse_args()

    review, business, users = load_reviews(), load_business(), load_users()
    split = original_split(review)
    out = {
        "scope": dataset_scope(review, business, users),
        "provenance": provenance(args.reference_repo),
        "split_balance": split_balance(split),
        "duplicates": duplicates_across_splits(split),
        "near_duplicates_test_vs_train": near_duplicates_sample(split),
        "eda_claims": eda_claims(review, users),
    }
    save("data", out)


if __name__ == "__main__":
    main()
