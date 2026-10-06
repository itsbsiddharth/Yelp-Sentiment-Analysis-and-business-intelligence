"""Re-run the five models whose results were pasted (not run) in 'Modeling and Evaluation.ipynb'.

Writes docs/audit/results/reference_models_rerun.json, updated after each model finishes.

The notebook imports `Classification_py`, which is not in this repo; it is the reference repo's
Classification.py. This script loads that file from --reference-repo unchanged and calls it exactly
like the notebook: same features (TF-IDF fit on 67%, StandardScaler), same 70/30 split, same grids,
same StratifiedKFold(5, shuffle, random_state=42), GridSearchCV(scoring='accuracy').

Usage:
    python docs/audit/rerun_reference_models.py --reference-repo PATH [--models "Decision Tree" ...]
"""
from __future__ import annotations

import argparse
import builtins
import importlib.util
import json
import time
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

from check_modeling import build_original_features
from common import LABELS, RESULTS, SEED, env_versions, majority_baseline, metrics, original_split, save

GRIDS = {  # copied from the notebook cells 49, 54, 59, 65, 70
    "Decision Tree": {"min_samples_leaf": [3, 15, 50, 100], "max_depth": [3, 5, 7, 10]},
    "Random Forest": {"min_samples_leaf": [1, 3, 15, 50], "max_depth": [5, 10, 15, 20]},
    "AdaBoost": {"learning_rate": [0.1, 1, 10]},
    "XGBoost": {"eta": [0.001, 0.005, 0.1, 0.5], "min_child_weight": [1, 5, 10]},
    "KNN": {"n_neighbors": [5, 10, 50, 150, 300]},
}


def load_classification(reference_repo: Path):
    spec = importlib.util.spec_from_file_location("Classification_py", reference_repo / "Classification.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Classification


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-repo", type=Path, required=True)
    ap.add_argument("--models", nargs="*", default=list(GRIDS))
    args = ap.parse_args()

    builtins.display = lambda obj: None  # Classification.get_scores calls IPython's display()
    Classification = load_classification(args.reference_repo)
    split = original_split()
    F = build_original_features(split)
    skf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)

    path = RESULTS / "reference_models_rerun.json"
    out = json.load(open(path)) if path.exists() else {}
    out["environment"] = env_versions()
    out["majority_baseline_val"] = majority_baseline(split["y_val"])
    for name in args.models:
        t = time.time()
        clf = Classification(name, F["X_tr"], F["X_va"], F["y_tr"], F["y_va"])
        clf.get_scores(GRIDS[name], skf)
        y_pred = clf.best_model.predict(F["X_va"])
        m = metrics([LABELS[i] for i in F["y_va"]], [LABELS[i] for i in y_pred])
        out[name] = {"best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in clf.best_params.items()},
                     "train_accuracy": float(clf.acc_train), "val_accuracy": float(clf.acc_val),
                     "val_macro_f1": m["macro_f1"], "val": m, "seconds": round(time.time() - t)}
        save("reference_models_rerun", out)
        print(name, out[name]["val_accuracy"], out[name]["val_macro_f1"], out[name]["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
