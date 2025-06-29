"""
Train a classifier (SVM vs LogisticRegression) with Leave-Subject-Out CV,
visualise results, and save the best model.
"""
import argparse, json, joblib
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# F1 LR : 0.859
# ----------------------------------------------------------------------
def load_vectors(xlsx: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_excel(xlsx)
    X = np.vstack(df["vector"].apply(lambda s: json.loads(s)).values)
    y = df["label"].values
    groups = df["subject_id"].values
    return X, y, groups

# ----------------------------------------------------------------------
def train_and_evaluate(X, y, groups, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    logo = LeaveOneGroupOut()

    cand_models = {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "SVM":    SVC(kernel="rbf", probability=True, class_weight="balanced"),
    }

    results = {}
    for name, clf in cand_models.items():
        y_pred = np.empty_like(y)
        for train_idx, test_idx in logo.split(X, y, groups):
            pipe = Pipeline(
                [("scaler", StandardScaler()), ("clf", clf.__class__(**clf.get_params()))]
            )
            pipe.fit(X[train_idx], y[train_idx])
            y_pred[test_idx] = pipe.predict(X[test_idx])
        f1 = f1_score(y, y_pred, average="macro")
        results[name] = (f1, y_pred)
        print(f"{name}: macro-F1 = {f1:.3f}")

    best_name = max(results, key=lambda k: results[k][0])
    best_pipe = Pipeline(
        [("scaler", StandardScaler()), ("clf", cand_models[best_name])]
    ).fit(X, y)

    # ---- visualisation ----
    disp = ConfusionMatrixDisplay(
        confusion_matrix(y, results[best_name][1], labels=[0,1,2]),
        display_labels=["HC","AD","MCI"]
    )
    disp.plot()
    plt.title(f"Leave-Subject-Out Confusion Matrix ({best_name})")
    fig_path = out_dir / "confusion_matrix.png"
    plt.savefig(fig_path, dpi=300); plt.close()

    # bar chart per-fold F1
    plt.figure()
    plt.bar(results.keys(), [v[0] for v in results.values()])
    plt.ylabel("Macro-F1")
    plt.title("Model comparison")
    plt.ylim(0,1)
    plt.savefig(out_dir / "model_comparison.png", dpi=300); plt.close()

    joblib.dump(best_pipe, out_dir / "classifier.pkl")
    print(f"[✓] Best model “{best_name}” saved → classifier.pkl")

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# CLI & GUI entry
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys, tkinter as tk
    from tkinter import filedialog, messagebox

    ap = argparse.ArgumentParser(
        description="Train cognition classifier (SVM / LR) with LOSO CV")
    ap.add_argument("feature_xlsx", nargs="?", help="speech_vectors.xlsx")
    ap.add_argument("--out_dir", type=str, help="output folder")
    ap.add_argument("--gui", action="store_true",
                    help="Select paths via dialogs (no other flags needed)")
    args = ap.parse_args()

    # ---------------- GUI branch -----------------
    if args.gui or args.feature_xlsx is None:
        tk.Tk().withdraw()                          # 隐藏根窗口
        try:
            feature_xlsx = filedialog.askopenfilename(
                title="选择特征向量 Excel (speech_vectors.xlsx)",
                filetypes=[("Excel", "*.xlsx *.xls")])
            if not feature_xlsx:
                sys.exit("已取消")

            out_dir = filedialog.askdirectory(
                title="选择 / 创建 输出文件夹 (out_dir)")
            if not out_dir:
                sys.exit("已取消")

        except Exception as e:
            messagebox.showerror("错误", str(e))
            sys.exit(1)

    # ---------------- CLI branch -----------------
    else:
        feature_xlsx = args.feature_xlsx
        out_dir      = args.out_dir or "cognition_training"

    # ---------------- Run training ---------------
    try:
        X, y, g = load_vectors(feature_xlsx)
        train_and_evaluate(X, y, g, Path(out_dir))
        print("[✓] 训练完成")
        if args.gui or args.feature_xlsx is None:
            messagebox.showinfo("完成", f"模型已保存到 {out_dir}")
    except Exception as exc:
        if args.gui or args.feature_xlsx is None:
            messagebox.showerror("错误", str(exc))
        raise
