"""Build every table and figure in docs/AUDIT.md from docs/audit/results/*.json.

Tables are written between markers in AUDIT.md:
    <!-- BEGIN GENERATED: name -->  ...  <!-- END GENERATED: name -->
so re-running this script after any check changes keeps the document consistent.
Quotes from the original notebooks in the verdict table go through `quote()`, which fails if
the quoted text is not actually in that notebook cell.

Usage:  python docs/audit/make_report.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from common import REPO, RESULTS

AUDIT_MD = REPO / "docs" / "AUDIT.md"
FIG_DIR = Path(__file__).resolve().parent / "figures"
NB = {"eda": "Data preprocessing and EDA.ipynb", "business": " Business Case Data Analysis.ipynb",
      "sentiment": "Sentiment Analysis.ipynb", "modeling": "Modeling and Evaluation.ipynb"}


def load(name: str) -> dict:
    p = RESULTS / f"{name}.json"
    return json.load(open(p)) if p.exists() else {}


R = {n: load(n) for n in ("inventory", "cell_provenance", "notebook_rerun", "stored_outputs", "data",
                          "modeling", "vader", "business", "reference_models_rerun")}


def quote(nb: str, cell: int, text: str) -> str:
    c = json.load(open(REPO / NB[nb]))["cells"][cell]
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    if text not in src:
        raise AssertionError(f"{text!r} not found in {NB[nb]} cell {cell}")
    return f"“{text}” ({NB[nb].strip()}, cell {cell})"


def f4(x): return "n/a" if x is None else f"{x:.4f}"
def f3(x): return "n/a" if x is None else f"{x:.3f}"
def pct(x, d=1): return "n/a" if x is None else f"{100 * x:.{d}f}%"
def n(x): return f"{x:,}"


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------ tables
def t_files() -> str:
    rows = []
    for f in R["inventory"]["files"]:
        rows.append([f"`{f['path']}`", n(f["bytes_in_git"]), n(f["bytes_on_disk_after_lfs_pull"]),
                     "yes (pointer in git)" if f["git_lfs"] else "no"])
    return table(["File", "Bytes in git", "Bytes after `git lfs pull`", "Git LFS"], rows)


def t_notebooks() -> str:
    rows = []
    for name, v in R["inventory"]["notebooks"].items():
        prov = R["cell_provenance"]["notebooks"][name]["counts"]
        rows.append([f"`{name.strip()}`", v["cells"], v["code_cells"], v["code_cells_with_outputs"],
                     v["code_cells_never_run"], v["image_outputs"],
                     "no" if not v["executed_top_to_bottom_in_one_run"] else "yes", v["kernel_python"],
                     prov.get("IDENTICAL", 0), prov.get("MODIFIED", 0), prov.get("OWN", 0)])
    return table(["Notebook", "Cells", "Code cells", "Code cells with saved output", "Code cells never run",
                  "Saved charts", "Run top-to-bottom?", "Kernel Python", "Cells identical to reference",
                  "Cells modified from reference", "Cells of your own"], rows)


def t_history() -> str:
    rows = [[f"`{h['commit']}`", h["date"][:16], h["author"], h["message"], "; ".join(h["changes"])]
            for h in R["inventory"]["history"]]
    return table(["Commit", "Date", "Author", "Message", "Files (A=added, D=deleted, M=modified)"], rows)


def model_rows() -> list[dict]:
    st = R["stored_outputs"]["modeling"]
    rr = R["reference_models_rerun"]
    rows = []
    for name in ("Logistic Regression", "Multinomial Naive Bayes", "XGBoost", "Random Forest",
                 "Decision Tree", "AdaBoost", "KNN"):
        ran = name in st["executed"]
        s = st["executed"][name] if ran else st["pasted_as_markdown"][name]
        if ran:
            rep_acc = R["modeling"]["reproduce_logreg_val" if name == "Logistic Regression"
                                    else "reproduce_mnb_val"]["accuracy"]
            rep_f1 = R["modeling"]["reproduce_logreg_val" if name == "Logistic Regression"
                                   else "reproduce_mnb_val"]["macro_f1"]
        else:
            rep_acc = rr.get(name, {}).get("val_accuracy")
            rep_f1 = rr.get(name, {}).get("val_macro_f1")
        rows.append({"name": name, "ran": ran, "acc": s["val_accuracy"], "f1": s["val_macro_f1"],
                     "neutral_recall": s["per_class"]["neutral(1)"]["recall"], "rep_acc": rep_acc,
                     "rep_f1": rep_f1})
    return rows


def t_baselines() -> str:
    maj = R["modeling"]["baselines"]["majority_val"]
    v = R["vader"]["val"]["original_rule"]
    rows = [["Always predict “positive” (majority baseline)", "computed in this audit", f4(maj["accuracy"]),
             "0", f3(maj["macro_f1"]), pct(maj["per_class"]["neutral"]["recall"]), "", ""]]
    for r in model_rows():
        rows.append([r["name"], "ran in your notebook" if r["ran"] else "pasted from the reference, not run",
                     f4(r["acc"]), f"{r['acc'] - maj['accuracy']:+.4f}", f3(r["f1"]), pct(r["neutral_recall"]),
                     f4(r["rep_acc"]) if r["rep_acc"] is not None else "not re-run",
                     f3(r["rep_f1"]) if r["rep_f1"] is not None else "not re-run"])
    rows.append(["VADER, original thresholds (same validation rows)", "computed in this audit",
                 f4(v["accuracy"]), f"{v['accuracy'] - maj['accuracy']:+.4f}", f3(v["macro_f1"]),
                 pct(v["per_class"]["neutral"]["recall"]), "", ""])
    return table(["Model", "Where the number comes from", "Validation accuracy (stored)",
                  "Accuracy minus baseline", "Macro-F1 (stored)", "Neutral recall (stored)",
                  "Accuracy re-run", "Macro-F1 re-run"], rows)


def t_vader() -> str:
    V = R["vader"]
    rows = []
    for part, label in (("all_rows", "all 229,130 rows (what the notebook scored)"), ("val", "validation rows"),
                        ("test", "test rows")):
        for rule, rl in (("original_rule", "original: <0 / <0.5 / ≥0.5"),
                         ("recommended_rule", "VADER authors: ≤−0.05 / <0.05 / ≥0.05"),
                         ("majority_baseline", "always “positive”")):
            m = V[part][rule]
            rows.append([label, rl, f4(m["accuracy"]), f3(m["macro_f1"]),
                         pct(m["per_class"]["negative"]["recall"]), pct(m["per_class"]["neutral"]["recall"]),
                         pct(m["per_class"]["positive"]["recall"])])
    t = V["thresholds_tuned_on_train_for_macro_f1"]
    m = t["val"]
    rows.append(["validation rows", f"best thresholds found on train rows: <{t['lo']} / ≥{t['hi']}",
                 f4(m["accuracy"]), f3(m["macro_f1"]), pct(m["per_class"]["negative"]["recall"]),
                 pct(m["per_class"]["neutral"]["recall"]), pct(m["per_class"]["positive"]["recall"])])
    return table(["Rows", "Rule (negative / neutral / positive)", "Accuracy", "Macro-F1", "Negative recall",
                  "Neutral recall", "Positive recall"], rows)


def t_scaling() -> str:
    S = R["modeling"]["scaling_ablation_val"]
    rows = []
    for k, v in S.items():
        if k == "note":
            continue
        rows.append([k, json.dumps(v["best_params_by_cv_macro_f1_on_train"]), f3(v["cv_macro_f1"]),
                     f4(v["val_accuracy"]), f3(v["val_macro_f1"]), f3(v["val_per_class_f1"]["neutral"])])
    return table(["Variant", "Regulariser picked by CV on train", "CV macro-F1 (train)", "Validation accuracy",
                  "Validation macro-F1", "Validation neutral F1"], rows)


def t_rerun() -> str:
    reasons = {
        "Data preprocessing and EDA.ipynb": "pip log removed; 3 cells show 28 columns instead of 27 because the "
        "saved run read the reference CSVs and this notebook's last cell then added `review_length` to them; "
        "one correlation differs in the 17th significant digit",
        "Modeling and Evaluation.ipynb": "`type(stopwords)` printout depends on hidden kernel state",
        "Sentiment Analysis.ipynb": "`.sample(5)` has no seed, so different example rows are shown",
        " Business Case Data Analysis.ipynb": "",
    }
    rows = []
    tot_same = tot = 0
    for name, v in R["notebook_rerun"].items():
        s = v["stats"]
        tot_same += s["identical"]
        tot += s["with_saved_text_output"]
        rows.append([f"`{name.strip()}`", v["run"].get("status"), f"{v['run'].get('seconds')} s", s["rerun_errors"],
                     s["with_saved_text_output"], s["identical"], s["different"], reasons.get(name, "")])
    rows.append(["**Total**", "", "", "", tot, tot_same, tot - tot_same, ""])
    return table(["Notebook", "Run", "Time", "Cells with errors", "Cells with saved text output",
                  "Identical after re-run", "Different", "Why different"], rows)


def t_categories() -> str:
    B = R["business"]["bad_review_categories"]["rates"]
    rows = [[r["cat"], n(r["n_reviews"]), pct(r["neg_rate"]), f"{pct(r['neg_rate_ci_low'])} – {pct(r['neg_rate_ci_high'])}"]
            for r in B["top10_by_negative_rate"]]
    r = B["restaurants"]
    rows.append([f"Restaurants (rank {r['rank_by_negative_rate']} of {B['n_categories_kept']})", n(r["n_reviews"]),
                 pct(r["negative_rate"]), f"{pct(r['ci'][0])} – {pct(r['ci'][1])}"])
    rows.append(["All reviews", n(R["data"]["scope"]["n_reviews"]), pct(B["overall_negative_rate"]), ""])
    return table(["Category (≥ 1,000 reviews)", "Reviews", "Share rated 1–2★", "95% Wilson interval"], rows)


def t_verdict() -> str:
    D, M, V, B, S = R["data"], R["modeling"], R["vader"], R["business"], R["stored_outputs"]
    E = D["eda_claims"]
    maj_all = V["all_rows"]["majority_baseline"]["accuracy"]
    maj_val = M["baselines"]["majority_val"]["accuracy"]
    rr = R["reference_models_rerun"]
    tr = B["top_rated"]
    ot = B["reviews_over_time"]
    bc = B["bad_review_categories"]
    bb = B["bad_review_businesses"]
    bw = B["bad_words"]
    tg = B["trending"]
    rerun = R["notebook_rerun"]
    same = sum(v["stats"]["identical"] for v in rerun.values())
    tot = sum(v["stats"]["with_saved_text_output"] for v in rerun.values())
    st = S["modeling"]
    rows = []

    def add(claim, orig, rep, ok, note):
        rows.append([claim, orig, rep, ok, note])

    sc = D["scope"]
    add("Review counts: total / positive / neutral / negative",
        " / ".join(S["business"]["review_counts_printed"][k] for k in
                   ("Total number of reviews", "Positive reviews", "Neutral reviews", "Negative reviews")),
        f"{n(sc['n_reviews'])} / {n(sc['label_counts_all_rows']['positive'])} / "
        f"{n(sc['label_counts_all_rows']['neutral'])} / {n(sc['label_counts_all_rows']['negative'])}",
        "Yes", "Phoenix-area reviews, " + sc["date_min"] + " to " + sc["date_max"])
    dup = D["duplicates"]
    add(quote("eda", 51, "There were no duplicate rows detected in the dataset"), "0 duplicate rows",
        f"0 duplicate rows, but {dup['exact_duplicate_rows_beyond_first']} repeated review texts "
        f"({dup['duplicate_texts_with_conflicting_labels']} with conflicting labels)", "Yes, incomplete",
        f"{dup['test_rows_whose_exact_text_is_in_train_or_val']} test reviews have an exact copy in train/validation "
        f"({pct(dup['test_share_exact_dup_of_train'], 2)}): negligible")
    add(quote("eda", 22, "Mean 'cool' votes per reviewer"), f"{S['eda']['mean_cool_votes_per_reviewer_printed']:.1f}",
        f"{E['mean_reviewer_cool_over_review_rows']:.1f} per review row; "
        f"{E['mean_reviewer_cool_over_unique_users_in_reviews']:.1f} per unique reviewer",
        "No", "averaged over review rows, so prolific reviewers count many times")
    add(quote("eda", 19, "39,450 reviewers have received 0 \"cool\" votes"), "39,450",
        f"{n(E['review_rows_with_reviewer_cool_eq_0'])} review rows; {n(E['unique_users_with_reviewer_cool_eq_0'])} unique reviewers",
        "No", "number not produced by any cell; and it counts rows, not reviewers")
    add(quote("eda", 27, "500 to 2000 length reviews"), "read off a scatter plot",
        f"Spearman correlation of a review's own cool votes with its length: {E['spearman_review_cool_vs_review_length']:.2f}",
        "Unsupported", "the plot used the reviewer's lifetime cool votes, not the review's")
    add("Top three rated: " + quote("business", 10, "Pizzeria Bianco (stars: 4.0) with 803 reviews") + ", Four Peaks 735, Matt's 689",
        "803 / 735 / 689", " / ".join(str(x["reviews_in_this_dataset"]) for x in tr["top3"]) +
        " reviews in the data (" + " / ".join(str(x["positive_4_5_star_reviews_in_this_dataset"]) for x in tr["top3"]) +
        " of them 4–5★)", "Numbers yes, label no",
        f"ranked by Yelp's total review count among the {tr['n_businesses_meeting_filter']} businesses with ≥4★ and >300 reviews, "
        f"not by rating ({n(tr['n_businesses_with_5_star_average'])} businesses average 5★)")
    t50 = tr["top_50"]
    add("Chart " + quote("business", 9, "Top 30 Rated Businesses") + " with x-axis “Number of Positive Reviews”",
        "chart", f"slice [:50] gives {t50['rows']} rows and {t50['unique_names (bars drawn)']} bars; x is total reviews; "
        f"{t50['value_labels_not_matching_their_bar']} bars show another bar's value label; "
        f"{t50['value_labels_drawn_below_the_last_bar']} labels fall below the last bar", "No",
        "chains sharing a name are averaged into one bar (the black error bars)")
    add("Reference README: “Almost the third of total categories in the top 30 reviewed businesses belong to Restaurants”",
        "≈1/3", f"{pct(B['category_base_rates']['restaurants_share_of_category_tags_in_top30'])} of category tags",
        "Yes, but base rate", f"Restaurants are {pct(B['category_base_rates']['share_of_businesses_tagged_restaurants'])} "
        f"of businesses and {pct(B['category_base_rates']['share_of_reviews_for_restaurants'])} of reviews")
    pyear = ot["reviews_per_year"]
    add(quote("business", 15, "Number of reviews per year show a positive linear climb"), "line chart",
        f"{n(pyear['2006'])} (2006) → {n(pyear['2012'])} (2012), ×{ot['growth_2006_to_2012_all_reviews_x']:.1f}",
        "Growth yes, “linear” no", "this is growth of Yelp / of this dataset sample, not of any business")
    pb = ot["Pizzeria Bianco"]["per_1000_reviews_that_year"]
    add(quote("business", 15, "Highly reviewed businesses such as the Phoenix Sky Airport, and Pizzeria Bianco show a positive trend"),
        "bar charts", f"Pizzeria Bianco per 1,000 reviews that year: {pb['2006']:.1f} (2006) → {pb['2012']:.1f} (2012)",
        "Misleading", "raw counts rise only because the platform grew; its share fell")
    pa = ot["Postino Arcadia"]
    add(quote("business", 15, "The randomly selected two businesses"), "bar charts",
        f"Postino Arcadia chart drops {pa['full_years_dropped_by_mistake'][0]} by mistake (positional `.drop([0,7])`)",
        "Unverifiable", "no code or seed shows they were random; counts themselves reproduce")
    add(quote("business", 15, "these businesses value their customer's feedback"),
        "—", "no analysis supports it", "No", "causal language from a count chart")
    add("“Top trending businesses” (reviews since 2010, >4★, >200 reviews)", "chart",
        f"{tg['n_trending_businesses']} businesses with the year filter, {tg['n_if_the_year_condition_is_removed']} without it",
        "No", "the year filter does nothing; `business_stars` is a single 2013 snapshot, so no trend is measured")
    hc = S["business"]["hardcoded_other_categories_cell_29"]
    add("“Other Categories” bars typed in as " + " and ".join(v for _, v in hc), "28 / 29",
        f"true sums {tg['trend_other_categories_true_sum']} (trending) / {tg['top50_other_categories_true_sum']} "
        f"(your top-50 list; {tg['top30_other_categories_true_sum']} for the reference's top 30)",
        "Partly", f"`.loc[10] = ...` overwrote the real “{tg['top50_row_overwritten_by_loc10']['category']}” row; "
        "the chart's axis labels are also swapped")
    pie = bc["replication"]["pie_percentages"]
    rest = bc["rates"]["restaurants"]
    add(quote("business", 31, "the highest number of these reviews, almost 50%, goes to Restaurants"),
        "49.1% (pie)", f"{pie['Restaurants']}% reproduced, but it is a share of business names with ≥1 one-star review, "
        "among the top 7 categories only", "No",
        f"Restaurants' 1–2★ rate is {pct(rest['negative_rate'])} vs {pct(bc['rates']['overall_negative_rate'])} overall "
        f"(rank {rest['rank_by_negative_rate']} of {bc['rates']['n_categories_kept']}); see the category table")
    add(quote("business", 31, "US Airways with 95 1-star review"), "95",
        f"{bb['us_airways_1star_by_name']} by name (your own chart shows this); 95 is one of its "
        f"{bb['us_airways_business_ids']} locations", "No", "text contradicts the chart next to it")
    add(quote("business", 31, "the mean number of 1-star reviews is 43"), "43",
        f"mean of top 15: {bb['mean_1star_top15_by_name']:.1f}; top 30: {bb['mean_1star_top30_by_name']:.1f}; "
        f"all names: {bb['mean_1star_over_names_with_any']:.1f}", "Unverifiable", "no cell computes a mean of 43")
    add(quote("business", 31, "except the presence of the Automotive category"), "—", "no code in either notebook",
        "Unverifiable", "")
    top = {x["lemma"].strip("'"): x["count"] for x in bw["top19_lemmas"]}
    wp = bw["word_presence_1star_vs_5star"]["good"]
    add("Top 15 words in 1★ reviews (chart)", "place, food, like, go, …",
        f"reproduced exactly (place {n(top['place'])}, food {n(top['food'])}, …) after rows ' ', '\\n\\n', '\\n', '$' "
        "are dropped by position", "Counts yes, insight weak",
        f"“good” is in {pct(wp['share_of_1star_reviews_containing'])} of 1★ and {pct(wp['share_of_5star_reviews_containing'])} "
        "of 5★ reviews: frequent ≠ characteristic")
    va = V["all_rows"]["original_rule"]
    add("VADER accuracy (Sentiment Analysis.ipynb, cell 22)",
        f"{S['sentiment']['vader_accuracy']:.4f}", f"{va['accuracy']:.4f} (confusion matrix identical)",
        "Computation yes, conclusion no", f"always-“positive” scores {maj_all:.4f}; VADER macro-F1 {va['macro_f1']:.3f}, "
        f"neutral recall {pct(va['per_class']['neutral']['recall'])}")
    add("Reference README: “negative being less than 0.5, neutral if the score is less than 0.5”", "as written",
        "code: " + S["sentiment"]["vader_rule_code"], "No (README)", "the code uses 0 and 0.5; VADER's authors recommend ±0.05")
    tb = S["sentiment"]["textblob_from_confusion_matrix"]
    add("TextBlob accuracy (your addition, Sentiment Analysis.ipynb cell 33)", f"{S['sentiment']['textblob_accuracy']:.4f}",
        f"{tb['accuracy']:.4f} (re-run identical)", "Computation yes, conclusion no",
        f"below always-“positive” ({maj_all:.4f}); macro-F1 {tb['macro_f1']:.3f}")
    f = M["features"]
    add("TF-IDF matrix shape", st["X_train_scaled_shape_output"], str(tuple(f["X_train_full_shape"])), "Yes",
        "only 1,095 features because `min_df=0.01`")
    lr = st["executed"]["Logistic Regression"]
    add("Logistic regression, validation accuracy / macro-F1", f"{lr['val_accuracy']:.4f} / {lr['val_macro_f1']:.3f}",
        f"{M['reproduce_logreg_val']['accuracy']:.4f} / {M['reproduce_logreg_val']['macro_f1']:.3f}", "Yes",
        f"{lr['val_accuracy'] - maj_val:+.4f} over always-“positive”; neutral recall {pct(lr['per_class']['neutral(1)']['recall'])}")
    g = M["logreg_grid_search"]["mean_cv_accuracy_by_params"]
    l2 = [v for k, v in g.items() if v is not None]
    add("Logistic regression best hyperparameters", lr["best_params"], json.dumps(M["logreg_grid_search"]["best_params"]),
        "Yes, but meaningless", f"all {M['logreg_grid_search']['n_param_combinations_that_failed']} l1 settings crashed; "
        f"the l2 settings differ by {max(l2) - min(l2):.5f} CV accuracy")
    mnb = st["executed"]["Multinomial Naive Bayes"]
    add("Multinomial NB, validation accuracy / macro-F1", f"{mnb['val_accuracy']:.4f} / {mnb['val_macro_f1']:.3f}",
        f"{M['reproduce_mnb_val']['accuracy']:.4f} / {M['reproduce_mnb_val']['macro_f1']:.3f}", "Yes",
        f"{mnb['val_accuracy'] - maj_val:+.4f} over always-“positive”")
    for name in ("XGBoost", "Random Forest", "Decision Tree", "AdaBoost", "KNN"):
        p = st["pasted_as_markdown"][name]
        r = rr.get(name)
        rep = (f"{r['val_accuracy']:.4f} / {r['val_macro_f1']:.3f} (params {json.dumps(r['best_params'])})"
               if r else "not re-run")
        add(f"{name}, validation accuracy / macro-F1 (cell {p['cell']}, markdown)",
            f"{p['val_accuracy']:.4f} / {p['val_macro_f1']:.3f}", rep, "Copied, not run",
            f"identical to the reference's output, timing line included; {p['val_accuracy'] - maj_val:+.4f} over always-“positive”")
    add("Reference README: “XGBoost and Logistic Regression displayed best results”", "—",
        "logistic regression is best on accuracy and macro-F1; XGBoost is second on accuracy but below Naive Bayes on macro-F1",
        "Partly", "")
    tv = M["test_vectorizer_bug"]
    te = M["original_logreg_on_test_correct_transform"]
    add(quote("modeling", 0, "Best models will be then tested again."), "section 5.2 is empty",
        f"test features were built with a re-fitted vectorizer ({tv['test_refit_vocab_size']} columns vs "
        f"{tv['train_vocab_size']}), so any test evaluation would have crashed", "No",
        f"evaluated once in this audit: logistic regression test accuracy {te['accuracy']:.4f}, macro-F1 {te['macro_f1']:.3f}")
    add("The notebooks reproduce the stored results", "—",
        f"as written: 2 of 4 fail on load (Windows paths, missing `Classification_py`); with minimal patches: "
        f"all 4 run, {same} of {tot} saved text outputs identical", "Yes, after patches",
        "none was run top-to-bottom originally (execution counts out of order)")
    return table(["Claim", "Original value", "Reproduced value", "Correct?", "Notes"], rows)


# ------------------------------------------------------------------------------------------ figure
def figure() -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(exist_ok=True)
    INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
    RAN, PASTED, LEX = "#2a78d6", "#86b6ef", "#eb6834"
    maj = R["modeling"]["baselines"]["majority_val"]
    vad = R["vader"]["val"]["original_rule"]
    rows = model_rows() + [{"name": "VADER (lexicon)", "ran": None, "acc": vad["accuracy"], "f1": vad["macro_f1"]}]
    rows.sort(key=lambda r: r["f1"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor=SURF)
    for ax, key, title, base in ((axes[0], "acc", "Accuracy", maj["accuracy"]),
                                 (axes[1], "f1", "Macro-F1 (each class counts equally)", maj["macro_f1"])):
        ax.set_facecolor(SURF)
        for i, r in enumerate(rows):
            color = LEX if r["ran"] is None else (RAN if r["ran"] else PASTED)
            hatch = "////" if r["ran"] is False else None
            ax.barh(i, r[key], height=0.55, color=color, hatch=hatch, edgecolor=SURF if hatch is None else RAN,
                    linewidth=0)
            ax.text(r[key] + 0.012, i, f"{r[key]:.3f}", va="center", fontsize=9, color=INK2)
        ax.axvline(base, color=INK, linewidth=1.5)
        ax.text(base, len(rows) - 0.35, f" always “positive” = {base:.3f}", fontsize=9, color=INK, va="bottom")
        ax.set_xlim(0, 1.0)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=22)
        ax.grid(axis="x", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, length=0)
    axes[0].set_yticks(range(len(rows)), [r["name"] for r in rows], color=INK)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=RAN, label="ran in your notebook"),
                        Patch(facecolor=PASTED, edgecolor=RAN, hatch="////", label="pasted from the reference, never run"),
                        Patch(color=LEX, label="computed in this audit (same validation rows)")],
               loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Accuracy makes every model look close to good; macro-F1 shows how far most are from it",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.89, f"Validation set of the original split, n = {maj['n']:,}. Stored notebook values; "
             "VADER computed here. Black line: the majority-class baseline.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.88))
    out = FIG_DIR / "accuracy_vs_macro_f1.png"
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)
    return f"![Accuracy versus macro-F1 for every model](audit/figures/{out.name})"


# ------------------------------------------------------------------------------------------ splice
def splice(md: str, name: str, body: str) -> str:
    pat = re.compile(rf"(<!-- BEGIN GENERATED: {name} -->\n).*?(<!-- END GENERATED: {name} -->)", re.S)
    if not pat.search(md):
        raise KeyError(f"marker for {name!r} not found in AUDIT.md")
    return pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), md)


def main() -> None:
    md = AUDIT_MD.read_text()
    parts = {"files": t_files(), "notebooks": t_notebooks(), "history": t_history(), "baselines": t_baselines(),
             "vader": t_vader(), "scaling": t_scaling(), "rerun": t_rerun(), "categories": t_categories(),
             "verdict": t_verdict(), "figure": figure()}
    for k, v in parts.items():
        md = splice(md, k, v)
    AUDIT_MD.write_text(md)
    print("updated", AUDIT_MD.relative_to(REPO))


if __name__ == "__main__":
    main()
