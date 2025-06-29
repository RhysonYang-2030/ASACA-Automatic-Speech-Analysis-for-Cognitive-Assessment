"""
Nested LOSO grid-search for Logistic Regression.
Keeps the same 17-D feature vector input.
Run:  python -m asaca_cognition.model_training_improved  speech_vectors.xlsx  --out_dir improved/
"""

from pathlib import Path
import argparse, json, joblib, itertools, numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from tqdm import tqdm

# --------- helper to load vectors -----------
def load_vectors(xlsx):
    df = pd.read_excel(xlsx)
    X = np.vstack(df["vector"].apply(json.loads).values)
    y = df["label"].values
    g = df["subject_id"].values
    return X, y, g, df

# --------- grid definition ------------------
C_grid        = [0.01, 0.1, 1, 3, 10]
penalty_grid  = ["l2", "elasticnet"]
l1_ratio_grid = [0.1, 0.5]           # only used when penalty == "elasticnet"
# Balanced the dataset distribution
weight_grid   = [
    "balanced",
    {0: 1, 1: 1.3, 2: 1},            # boost AD (=1) 30 %
    {0: 1, 1: 1.5, 2: 1.2},          # stronger boost
    None                             # no weighting
]


def make_clf(C, penalty, l1_ratio, weight):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="saga", max_iter=1000,
            C=C, penalty=penalty, l1_ratio=(None if penalty=="l2" else l1_ratio),
            class_weight=weight, n_jobs=-1
        ))
    ])

# ------------- training ---------------------
def nested_train(X, y, groups):
    outer = LeaveOneGroupOut()
    best_params = None
    y_pred_full = np.zeros_like(y)

    outer_loop = tqdm(list(outer.split(X, y, groups)), desc="Outer-fold")
    for train_idx, test_idx in outer_loop:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        g_tr       = groups[train_idx]

        # --- inner grid
        inner = LeaveOneGroupOut()
        best_f1, best_clf = -1, None

        for C, pen, l1r, w in itertools.product(C_grid, penalty_grid, l1_ratio_grid, weight_grid):
            if pen == "l2" and l1r != l1_ratio_grid[0]:
                continue  # skip redundant combos
            f1_scores = []
            for tr_idx, val_idx in inner.split(X_tr, y_tr, g_tr):
                pipe = make_clf(C, pen, l1r, w)
                pipe.fit(X_tr[tr_idx], y_tr[tr_idx])
                y_val = pipe.predict(X_tr[val_idx])
                f1_scores.append(f1_score(y_tr[val_idx], y_val, average="macro"))
            f1_mean = np.mean(f1_scores)
            if f1_mean > best_f1:
                best_f1, best_clf = f1_mean, make_clf(C, pen, l1r, w)

        best_clf.fit(X_tr, y_tr)          # refit on full training of this outer fold
        y_pred_full[test_idx] = best_clf.predict(X_te)

        outer_loop.set_postfix({"inner-F1": f"{best_f1:.3f}"})
        if best_params is None:   # store params from first fold for later refit
            best_params = best_clf.get_params()["clf"].get_params()

    macro_f1 = f1_score(y, y_pred_full, average="macro")
    print(f"Nested LOSO Macro-F1: {macro_f1:.3f}")

    # ------- final refit on entire dataset with best_params ----------
    final_model = make_clf(
        best_params["C"],
        best_params["penalty"],
        best_params.get("l1_ratio"),
        best_params["class_weight"]
    ).fit(X, y)
    return final_model, macro_f1, best_params

# ------------- main -------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("feature_xlsx")
    ap.add_argument("--out_dir", default="cognition_training_improved/")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    X, y, g, df = load_vectors(args.feature_xlsx)

    model, f1, params = nested_train(X, y, g)
    joblib.dump(model, out_dir / "classifier.pkl")
    pd.Series(params).to_json(out_dir / "best_params.json", indent=2)

    print("[✓] Saved improved classifier to", out_dir / "classifier.pkl")
# python -m asaca_cognition.model_training_improved cognition_training/TUH_FV.xlsx --out_dir improved/
