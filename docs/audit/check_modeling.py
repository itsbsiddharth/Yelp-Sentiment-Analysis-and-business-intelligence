"""Audit checks on 'Modeling and Evaluation.ipynb' (copied from the reference's notebook 4).

Writes docs/audit/results/modeling.json.

What it does, in order:
  1. Rebuilds the original features exactly (TF-IDF fit on the 67% "train" part, StandardScaler on top)
     and checks the shape matches the stored output (153513 x 1095).
  2. Majority-class baseline on the validation and test sets.
  3. Refits the two models the notebook actually ran (logistic regression C=0.01, Multinomial NB
     alpha=0.001) and checks validation metrics match the stored outputs.
  4. Re-runs the logistic-regression grid search to show what it really searched.
  5. Leakage: TF-IDF/scaler were fit on train+validation. Refit on train only and compare.
  6. The test-set bug: the notebook calls vectorizer.fit_transform(X_test) and scaler.fit_transform on
     the test set. Measures how different the test "features" are from the train features.
  7. Evaluates the original logistic regression on the original test set ONCE (the notebook never did),
     with the correct transform.
  8. Stop words: which negations the NLTK list removes; which sentiment words min_df=0.01 drops.
  9. Scaling ablation: logistic regression and Multinomial NB with and without StandardScaler, each with
     its regularisation re-tuned by 5-fold CV on the train part only (validation untouched until the end).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler

from common import LABELS, SEED, env_versions, majority_baseline, metrics, original_split, save

LABEL_TO_INT = {c: i for i, c in enumerate(LABELS)}  # LabelEncoder order


def original_vectorizer() -> TfidfVectorizer:
    """Exactly the notebook's vectorizer (cell 25), with the stop-word set turned into a list (cell 24)."""
    return TfidfVectorizer(lowercase=True, stop_words=list(set(nltk_stopwords.words("english"))),
                           ngram_range=(1, 2), min_df=0.01)


def build_original_features(split: dict) -> dict:
    vec = original_vectorizer()
    Xtr_full = vec.fit_transform(split["X_train_full"])           # fit on 67% (train + validation)
    scaler = StandardScaler(with_mean=False)
    Xtr_full_s = scaler.fit_transform(Xtr_full)
    y = split["y_train_full"].map(LABEL_TO_INT).to_numpy()
    i_tr, i_va = split["i_train"], split["i_val"]
    return {"vec": vec, "scaler": scaler, "X_full_raw": Xtr_full, "X_full_scaled": Xtr_full_s,
            "X_tr": Xtr_full_s[i_tr], "X_va": Xtr_full_s[i_va], "y_tr": y[i_tr], "y_va": y[i_va],
            "X_tr_raw": Xtr_full[i_tr], "X_va_raw": Xtr_full[i_va]}


def to_labels(y_int) -> list[str]:
    return [LABELS[i] for i in y_int]


def fit_eval(model, X_tr, y_tr, X_va, y_va) -> dict:
    t = time.time()
    model.fit(X_tr, y_tr)
    out = metrics(to_labels(y_va), to_labels(model.predict(X_va)))
    out["train_accuracy"] = float((model.predict(X_tr) == y_tr).mean())
    out["fit_seconds"] = round(time.time() - t, 1)
    return out


def lr_grid_search(F: dict) -> dict:
    """The notebook's grid (cell 41) inside Classification.get_scores: GridSearchCV(accuracy, skf)."""
    skf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
    gs = GridSearchCV(LogisticRegression(solver="lbfgs", max_iter=1000),
                      {"penalty": ["l1", "l2"], "C": [0.01, 0.05, 0.5, 5]},
                      cv=skf, scoring="accuracy", n_jobs=-1)
    gs.fit(F["X_tr"], F["y_tr"])
    res = pd.DataFrame(gs.cv_results_)
    return {
        "best_params": gs.best_params_,
        "mean_cv_accuracy_by_params": {f"{r['param_penalty']}, C={r['param_C']}":
                                       (None if pd.isna(r["mean_test_score"]) else float(r["mean_test_score"]))
                                       for _, r in res.iterrows()},
        "n_param_combinations_that_failed": int(res["mean_test_score"].isna().sum()),
        "why": "solver='lbfgs' does not support penalty='l1', so every l1 fit raises and scores NaN",
    }


def leakage_refit_on_train_only(split: dict) -> dict:
    """Fit TF-IDF and scaler on the 70% train part only, then the same LR (C=0.01) as the notebook."""
    vec = original_vectorizer()
    Xtr = vec.fit_transform(split["X_train"])
    Xva = vec.transform(split["X_val"])
    sc = StandardScaler(with_mean=False)
    Xtr, Xva = sc.fit_transform(Xtr), sc.transform(Xva)
    ytr = split["y_train"].map(LABEL_TO_INT).to_numpy()
    yva = split["y_val"].map(LABEL_TO_INT).to_numpy()
    out = fit_eval(LogisticRegression(C=0.01, solver="lbfgs", max_iter=1000), Xtr, ytr, Xva, yva)
    out["n_features"] = int(Xtr.shape[1])
    return out


def test_vectorizer_bug(split: dict, F: dict, lr_model) -> dict:
    """What the notebook's test features would have been (cells 28 and 32) vs the correct ones."""
    vec_test = original_vectorizer()
    X_test_wrong = vec_test.fit_transform(split["X_test"])         # the notebook's cell 28
    X_test_wrong = StandardScaler(with_mean=False).fit_transform(X_test_wrong)  # cell 32
    train_vocab = F["vec"].get_feature_names_out()
    test_vocab = vec_test.get_feature_names_out()
    same_size = len(train_vocab) == len(test_vocab)
    n_same_position = int(sum(a == b for a, b in zip(train_vocab, test_vocab))) if same_size else None
    first_mismatch = None
    if same_size:
        for i, (a, b) in enumerate(zip(train_vocab, test_vocab)):
            if a != b:
                first_mismatch = {"column": i, "train_term": a, "test_term": b}
                break
    out = {
        "train_vocab_size": int(len(train_vocab)), "test_refit_vocab_size": int(len(test_vocab)),
        "terms_in_both": int(len(set(train_vocab) & set(test_vocab))),
        "same_number_of_columns": bool(same_size),
        "columns_with_same_term_at_same_position": n_same_position,
        "first_mismatched_column": first_mismatch,
    }
    y_test = split["y_test"].tolist()
    if same_size:
        out["lr_on_buggy_test_features"] = metrics(y_test, to_labels(lr_model.predict(X_test_wrong)))
    else:
        out["lr_on_buggy_test_features"] = (
            f"would crash: model expects {len(train_vocab)} features, buggy test matrix has {len(test_vocab)}")
    return out


def stopword_and_vocab_checks(split: dict, F: dict) -> dict:
    sw = set(nltk_stopwords.words("english"))
    negations = sorted(w for w in sw if w in {"not", "no", "nor", "never"} or w.endswith("n't")
                       or w in {"don", "didn", "doesn", "isn", "wasn", "weren", "aren", "couldn", "shouldn",
                                "wouldn", "hasn", "haven", "hadn", "won", "mightn", "mustn", "needn", "shan",
                                "ain"})
    vocab = set(F["vec"].get_feature_names_out())
    x = split["X_train_full"].str.lower()
    has_neg = x.str.contains(r"\bnot\b|n't\b|\bno\b|\bnever\b", regex=True)
    probe = ["terrible", "horrible", "awful", "worst", "disgusting", "rude", "mediocre", "bland", "amazing",
             "excellent", "delicious", "great", "bad", "ok", "okay", "not good", "never", "not"]
    # document frequency share in the 67% part, using the same tokenisation (no stop words)
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(lowercase=True, ngram_range=(1, 2), vocabulary=[p for p in probe], binary=True)
    df_share = np.asarray(cv.transform(split["X_train_full"]).mean(axis=0)).ravel()
    return {
        "nltk_stopword_count": len(sw),
        "negation_words_in_nltk_stoplist": negations,
        "n_negation_words_removed": len(negations),
        "share_of_reviews_with_not_nt_no_never": float(has_neg.mean()),
        "min_df_rule": "min_df=0.01 keeps only terms in >= 1% of the 153,513 training documents",
        "probe_terms": {p: {"in_vocabulary": p in vocab, "document_share": float(s)}
                        for p, s in zip(probe, df_share)},
        "note": "'not good' can never be a feature: 'not' is removed as a stop word before bigrams are built",
    }


def scaling_ablation(F: dict, split: dict) -> dict:
    """Is StandardScaler on TF-IDF helping? Re-tune the regulariser by CV on train only for each variant.

    Two extra Naive Bayes variants test *why* scaling changes MNB: on raw counts (what MNB is designed
    for) and on raw TF-IDF with a uniform class prior (fit_prior=False). If the uniform-prior variant
    recovers, the scaler was only amplifying the word evidence relative to the 68%-positive prior.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer(lowercase=True, stop_words=list(set(nltk_stopwords.words("english"))),
                         ngram_range=(1, 2), min_df=0.01)
    C_full = cv.fit_transform(split["X_train_full"])
    C_tr, C_va = C_full[split["i_train"]], C_full[split["i_val"]]
    skf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
    out = {}
    variants = {
        "mnb_raw_counts": (MultinomialNB(), {"alpha": [0.01, 0.1, 1.0, 10.0]}, C_tr, C_va),
        "mnb_raw_tfidf_uniform_prior": (MultinomialNB(fit_prior=False), {"alpha": [0.001, 0.01, 0.1, 1.0]},
                                        F["X_tr_raw"], F["X_va_raw"]),
        "logreg_scaled (original)": (LogisticRegression(solver="lbfgs", max_iter=1000),
                                     {"C": [0.001, 0.003, 0.01, 0.03, 0.1]}, F["X_tr"], F["X_va"]),
        "logreg_raw_tfidf": (LogisticRegression(solver="lbfgs", max_iter=1000),
                             {"C": [0.3, 1, 3, 10, 30]}, F["X_tr_raw"], F["X_va_raw"]),
        "mnb_scaled (original)": (MultinomialNB(), {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]}, F["X_tr"], F["X_va"]),
        "mnb_raw_tfidf": (MultinomialNB(), {"alpha": [0.001, 0.01, 0.1, 1.0]}, F["X_tr_raw"], F["X_va_raw"]),
    }
    for name, (est, grid, Xtr, Xva) in variants.items():
        gs = GridSearchCV(est, grid, cv=skf, scoring="f1_macro", n_jobs=-1)
        gs.fit(Xtr, F["y_tr"])
        m = metrics(to_labels(F["y_va"]), to_labels(gs.best_estimator_.predict(Xva)))
        out[name] = {"best_params_by_cv_macro_f1_on_train": gs.best_params_,
                     "cv_macro_f1": float(gs.best_score_),
                     "val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"],
                     "val_per_class_f1": {c: m["per_class"][c]["f1-score"] for c in LABELS}}
    out["note"] = ("Diagnostic only: regulariser chosen by 5-fold CV (macro-F1) on the train part, "
                   "validation used once per variant. Grids differ because scaled features need much "
                   "stronger regularisation than raw TF-IDF.")
    return out


def main() -> None:
    t0 = time.time()
    split = original_split()
    F = build_original_features(split)
    out = {"environment": env_versions()}
    out["features"] = {"X_train_full_shape": list(F["X_full_scaled"].shape),
                       "matches_stored_153513x1095": list(F["X_full_scaled"].shape) == [153513, 1095],
                       "train_rows": int(F["X_tr"].shape[0]), "val_rows": int(F["X_va"].shape[0]),
                       "test_rows": int(len(split["X_test"]))}
    out["baselines"] = {"majority_val": majority_baseline(split["y_val"]),
                        "majority_test": majority_baseline(split["y_test"]),
                        "majority_all": majority_baseline(split["df"]["label"])}
    print("features + baselines", round(time.time() - t0), "s", flush=True)

    lr = LogisticRegression(C=0.01, penalty="l2", solver="lbfgs", max_iter=1000)
    out["reproduce_logreg_val"] = fit_eval(lr, F["X_tr"], F["y_tr"], F["X_va"], F["y_va"])
    out["reproduce_mnb_val"] = fit_eval(MultinomialNB(alpha=0.001), F["X_tr"], F["y_tr"], F["X_va"], F["y_va"])
    print("reproduced LR/MNB", round(time.time() - t0), "s", flush=True)

    out["logreg_grid_search"] = lr_grid_search(F)
    print("grid", round(time.time() - t0), "s", flush=True)

    out["leakage_refit_train_only_logreg_val"] = leakage_refit_on_train_only(split)
    out["leakage_effect_val_accuracy"] = (out["reproduce_logreg_val"]["accuracy"]
                                          - out["leakage_refit_train_only_logreg_val"]["accuracy"])
    out["leakage_effect_val_macro_f1"] = (out["reproduce_logreg_val"]["macro_f1"]
                                          - out["leakage_refit_train_only_logreg_val"]["macro_f1"])
    print("leakage", round(time.time() - t0), "s", flush=True)

    out["test_vectorizer_bug"] = test_vectorizer_bug(split, F, lr)
    # Correct one-off test evaluation of the original LR (TF-IDF + scaler fit on 67%, transform test).
    X_test = F["scaler"].transform(F["vec"].transform(split["X_test"]))
    out["original_logreg_on_test_correct_transform"] = metrics(split["y_test"].tolist(),
                                                               to_labels(lr.predict(X_test)))
    print("test", round(time.time() - t0), "s", flush=True)

    out["stopwords_and_vocabulary"] = stopword_and_vocab_checks(split, F)
    out["scaling_ablation_val"] = scaling_ablation(F, split)
    out["seconds"] = round(time.time() - t0)
    save("modeling", out)


if __name__ == "__main__":
    main()
