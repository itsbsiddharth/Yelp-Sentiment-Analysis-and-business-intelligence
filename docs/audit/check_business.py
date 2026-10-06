"""Audit checks on ' Business Case Data Analysis.ipynb' (copied from the reference's notebook 2).

Writes docs/audit/results/business.json.

Each function replicates one chart or claim with the notebook's own code logic, then adds the
check that tells us whether the claim holds (base rates, normalisation, confidence intervals).
Section names follow the notebook's headings.
"""
from __future__ import annotations

import re
import time
from collections import Counter

import numpy as np
import pandas as pd

from common import env_versions, load_business, load_reviews, save, wilson_ci


def split_categories(s: pd.Series) -> list[str]:
    """The notebook's category parsing: str.split(';').sum() then strip()."""
    return [x.strip() for x in s.str.split(";").sum()]


def category_counts(s: pd.Series) -> pd.DataFrame:
    """The notebook's dict/list.count logic, done with a Counter, sorted like the notebook.

    Counter keeps first-appearance order, like the notebook's dict. The sort uses pandas' default
    (quicksort, not stable) on purpose: which category lands on a given row among ties decides
    which row the notebook's hard-coded `.loc[8] = ...` overwrites.
    """
    c = Counter(split_categories(s))
    df = pd.DataFrame(list(c.items()), columns=["category", "occurance"])
    return df.sort_values(by="occurance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- "What businesses are getting top reviews?"
def top_rated(business: pd.DataFrame, review: pd.DataFrame) -> dict:
    good = business[(business["business_stars"] >= 4) & (business["business_review_count"] > 300)]
    ranked = good[["business_name", "business_review_count", "business_stars", "business_categories",
                   "business_id"]].sort_values(by="business_review_count", ascending=False)
    out = {"n_businesses_meeting_filter": int(len(good))}
    for k in (30, 50):  # reference used [:30], this repo's notebook uses [:50] but titles it "Top 30"
        t = ranked[:k]
        names_in_order = list(dict.fromkeys(t["business_name"]))   # seaborn's category order
        bar_value = t.groupby("business_name")["business_review_count"].mean()
        labels = list(t["business_review_count"])                 # ax.text(v, i) for i, v in enumerate(...)
        mismatched = [i for i, name in enumerate(names_in_order) if labels[i] != bar_value[name]]
        out[f"top_{k}"] = {
            "slice": f"[:{k}]", "rows": len(t), "unique_names (bars drawn)": len(names_in_order),
            "names_appearing_more_than_once": t["business_name"].value_counts()[lambda v: v > 1].to_dict(),
            "value_labels_not_matching_their_bar": len(mismatched),
            "value_labels_drawn_below_the_last_bar": len(labels) - len(names_in_order),
        }
    top3 = ranked.head(3)
    rv = review.groupby("business_id")["stars"]
    out["top3"] = [{
        "name": r.business_name, "business_stars": float(r.business_stars),
        "business_review_count (plotted, labelled 'Number of Positive Reviews')": int(r.business_review_count),
        "reviews_in_this_dataset": int(rv.size().get(r.business_id, 0)),
        "positive_4_5_star_reviews_in_this_dataset": int((review.loc[review.business_id == r.business_id, "stars"] >= 4).sum()),
    } for r in top3.itertuples()]
    out["n_businesses_with_5_star_average"] = int((business["business_stars"] == 5).sum())
    out["median_review_count_of_5_star_businesses"] = float(
        business.loc[business["business_stars"] == 5, "business_review_count"].median())
    return out


# ---------------------------------------------------------------- category base rates
def category_base_rates(business: pd.DataFrame, review: pd.DataFrame) -> dict:
    is_rest_b = business["business_categories"].str.contains(r"(?:^|;\s*)Restaurants(?:;|$)", regex=True)
    is_rest_r = review["business_categories"].str.contains(r"(?:^|;\s*)Restaurants(?:;|$)", regex=True)
    all_cats = category_counts(business["business_categories"])
    good = business[(business["business_stars"] >= 4) & (business["business_review_count"] > 300)]
    t30 = good.sort_values(by="business_review_count", ascending=False)[:30]
    c30 = category_counts(t30["business_categories"])
    return {
        "share_of_businesses_tagged_restaurants": float(is_rest_b.mean()),
        "share_of_reviews_for_restaurants": float(is_rest_r.mean()),
        "top10_categories_all_businesses": all_cats.head(10).set_index("category")["occurance"].to_dict(),
        "restaurants_share_of_category_tags_in_top30": float(
            c30.loc[c30.category == "Restaurants", "occurance"].sum() / c30["occurance"].sum()),
        "restaurants_share_of_category_tags_all_businesses": float(
            all_cats.loc[all_cats.category == "Restaurants", "occurance"].sum() / all_cats["occurance"].sum()),
    }


# ---------------------------------------------------------------- "How often do businesses get reviewed over time?"
def reviews_over_time(review: pd.DataFrame) -> dict:
    year = pd.to_datetime(review["date"]).dt.year
    per_year = year.value_counts().sort_index()
    out = {"reviews_per_year": {int(k): int(v) for k, v in per_year.items()}}
    picks = {"Pizzeria Bianco": [0, 8], "Postino Arcadia": [0, 7],
             "Phoenix Sky Harbor International Airport": [7], "Joe's Farm Grill": [6]}  # notebook's drop() lists
    for name, drop in picks.items():
        counts = review.loc[review["business_name"] == name].groupby(year).size()
        years = [int(y) for y in counts.index]
        dropped = [years[i] for i in drop if i < len(years)]
        kept = counts.drop(index=dropped)
        share = (kept / per_year.loc[kept.index] * 1000).round(3)
        out[name] = {
            "n_business_ids_with_this_name": int(review.loc[review.business_name == name, "business_id"].nunique()),
            "years_present": years, "years_dropped_by_positional_drop": dropped,
            "full_years_dropped_by_mistake": [y for y in dropped if y not in (2005, 2013)],
            "counts_plotted": {int(k): int(v) for k, v in kept.items()},
            "per_1000_reviews_that_year": {int(k): float(v) for k, v in share.items()},
        }
    tot = per_year.loc[2006:2012]
    out["growth_2006_to_2012_all_reviews_x"] = float(tot.loc[2012] / tot.loc[2006])
    return out


# ---------------------------------------------------------------- "trending vs top reviewed"
def trending(review: pd.DataFrame, business: pd.DataFrame) -> dict:
    review = review.assign(year=pd.to_datetime(review["date"]).dt.year)
    cond = (review["business_stars"] > 4) & (review["business_review_count"] > 200)
    with_year = review[cond & (review["year"] >= 2010)].drop_duplicates(subset=["business_name"])
    without_year = review[cond].drop_duplicates(subset=["business_name"])
    cat_trend = category_counts(with_year["business_categories"])
    good = business[(business["business_stars"] >= 4) & (business["business_review_count"] > 300)]
    out = {"n_trending_businesses": int(len(with_year)),
           "n_if_the_year_condition_is_removed": int(len(without_year)),
           "trending_names": sorted(with_year["business_name"])}
    # Hard-coded 'Other Categories' rows: cat_top_trend.loc[8] = [..., 28]; cat_top_rated.loc[10] = [..., 29]
    out["trend_other_categories_true_sum"] = int(cat_trend.loc[cat_trend.occurance.isin([1, 2]), "occurance"].sum())
    out["trend_row_overwritten_by_loc8"] = cat_trend.loc[8].to_dict() if len(cat_trend) > 8 else None
    for k in (30, 50):
        cr = category_counts(good.sort_values(by="business_review_count", ascending=False)[:k]["business_categories"])
        kept = cr[~cr.occurance.isin([1, 2])]
        out[f"top{k}_other_categories_true_sum"] = int(cr.loc[cr.occurance.isin([1, 2]), "occurance"].sum())
        out[f"top{k}_row_overwritten_by_loc10"] = kept.loc[10].to_dict() if 10 in kept.index else "none (appended)"
        out[f"top{k}_restaurants_count"] = int(cr.loc[cr.category == "Restaurants", "occurance"].sum())
    out["trend_restaurants_count"] = int(cat_trend.loc[cat_trend.category == "Restaurants", "occurance"].sum())
    return out


# ---------------------------------------------------------------- "Which business categories get bad reviews?"
def bad_review_categories(review: pd.DataFrame) -> dict:
    # Notebook logic: 1-star reviews (stars < 2), ONE ROW PER BUSINESS NAME, count category tags.
    first = review[review["stars"] < 2].drop_duplicates(subset=["business_name"])
    cats = category_counts(first["business_categories"])
    top7 = cats.nlargest(7, "occurance")
    pie = (top7.set_index("category")["occurance"] / top7["occurance"].sum() * 100).round(1)
    out = {"replication": {"rows_after_drop_duplicates_by_name": int(len(first)),
                           "what_is_counted": "business names with >= 1 one-star review, per category tag",
                           "top7_counts": top7.set_index("category")["occurance"].to_dict(),
                           "pie_percentages": pie.to_dict()}}

    # Proper version: per-category rates over ALL reviews, Wilson 95% CIs, minimum sample size.
    ex = review[["business_categories", "stars"]].copy()
    ex["cat"] = ex["business_categories"].str.split(";")
    ex = ex.explode("cat")
    ex["cat"] = ex["cat"].str.strip()
    g = ex.groupby("cat")["stars"]
    tab = pd.DataFrame({"n_reviews": g.size(), "n_1star": g.apply(lambda s: int((s == 1).sum())),
                        "n_negative_1_2": g.apply(lambda s: int((s <= 2).sum()))})
    MIN_N = 1000
    tab = tab[tab.n_reviews >= MIN_N].copy()
    tab["neg_rate"] = tab.n_negative_1_2 / tab.n_reviews
    ci = [wilson_ci(k, n) for k, n in zip(tab.n_negative_1_2, tab.n_reviews)]
    tab["neg_rate_ci_low"], tab["neg_rate_ci_high"] = zip(*ci)
    tab = tab.sort_values("neg_rate", ascending=False)
    overall = float((review["stars"] <= 2).mean())
    rest = tab.loc["Restaurants"]
    is_rest = review["business_categories"].str.contains(r"(?:^|;\s*)Restaurants(?:;|$)", regex=True)
    out["rates"] = {
        "min_reviews_per_category": MIN_N, "n_categories_kept": int(len(tab)),
        "overall_negative_rate": overall, "overall_1star_rate": float((review["stars"] == 1).mean()),
        "restaurants": {"n_reviews": int(rest.n_reviews), "negative_rate": float(rest.neg_rate),
                        "ci": [float(rest.neg_rate_ci_low), float(rest.neg_rate_ci_high)],
                        "rank_by_negative_rate": int(list(tab.index).index("Restaurants") + 1)},
        "restaurants_share_of_all_reviews": float(is_rest.mean()),
        "restaurants_share_of_1star_reviews": float(is_rest[review["stars"] == 1].mean()),
        "top10_by_negative_rate": tab.head(10).round(4).reset_index().to_dict(orient="records"),
        "bottom5_by_negative_rate": tab.tail(5).round(4).reset_index().to_dict(orient="records"),
    }
    return out


def bad_review_businesses(review: pd.DataFrame) -> dict:
    """'The highest number of bad reviews is US Airways with 95 1-star review, whereas the mean ... is 43.'"""
    one = review[review["stars"] < 2]
    by_name = one["business_name"].value_counts()
    by_id = one.groupby(["business_id", "business_name"]).size().sort_values(ascending=False)
    locations = review.groupby("business_name")["business_id"].nunique()
    return {
        "us_airways_1star_by_name": int(by_name.get("US Airways", 0)),
        "us_airways_business_ids": int(locations.get("US Airways", 0)),
        "top15_by_name": by_name.head(15).to_dict(),
        "top15_locations_per_name": {n: int(locations[n]) for n in by_name.head(15).index},
        "mean_1star_top15_by_name": float(by_name.head(15).mean()),
        "mean_1star_top30_by_name": float(by_name.head(30).mean()),
        "mean_1star_over_names_with_any": float(by_name.mean()),
        "top5_by_single_business_id": [{"name": n, "n": int(v)} for (i, n), v in by_id.head(5).items()],
    }


# ---------------------------------------------------------------- "What are the most common words in bad reviews?"
def bad_words(review: pd.DataFrame) -> dict:
    import spacy

    nlp = spacy.load("en_core_web_sm")
    texts = review.loc[review["stars"] == 1, "text"].reset_index(drop=True)
    t = time.time()
    lemma_counts = Counter()
    good_total = good_negated = 0
    for doc in nlp.pipe(texts, batch_size=1000, disable=["parser", "ner"]):
        toks = list(doc)
        for i, tok in enumerate(toks):
            if not tok.is_stop and not tok.is_punct:
                lemma_counts[tok.lemma_] += 1
            if tok.lemma_.lower() == "good":
                good_total += 1
                prev = {p.lower_ for p in toks[max(0, i - 3):i]}
                if prev & {"not", "n't", "never", "no", "nothing"}:
                    good_negated += 1
    top19 = lemma_counts.most_common(19)
    dropped_positions = [0, 1, 10, 17]   # notebook: .head(19).reset_index().drop([0, 1, 10, 17])
    word_rate = {}
    for w in ("good", "place", "food", "service", "time", "order", "like"):
        pat = re.compile(rf"\b{w}\b", re.I)
        for s in (1, 5):
            tx = review.loc[review["stars"] == s, "text"].dropna()
            word_rate.setdefault(w, {})[f"share_of_{s}star_reviews_containing"] = float(tx.str.contains(pat).mean())
    return {
        "n_one_star_reviews": int(len(texts)), "spacy_seconds": round(time.time() - t),
        "top19_lemmas": [{"pos": i, "lemma": repr(l), "count": c} for i, (l, c) in enumerate(top19)],
        "rows_dropped_by_hand": [{"pos": i, "lemma": repr(top19[i][0]), "count": top19[i][1]}
                                 for i in dropped_positions],
        "lemma_good_occurrences": good_total,
        "lemma_good_with_negation_in_previous_3_tokens": good_negated,
        "word_presence_1star_vs_5star": word_rate,
    }


def main() -> None:
    review, business = load_reviews(), load_business()
    out = {"environment": env_versions(),
           "top_rated": top_rated(business, review),
           "category_base_rates": category_base_rates(business, review),
           "reviews_over_time": reviews_over_time(review),
           "trending": trending(review, business),
           "bad_review_categories": bad_review_categories(review),
           "bad_review_businesses": bad_review_businesses(review)}
    save("business", out)          # save early: the spaCy step below takes minutes
    out["bad_words"] = bad_words(review)
    save("business", out)


if __name__ == "__main__":
    main()
