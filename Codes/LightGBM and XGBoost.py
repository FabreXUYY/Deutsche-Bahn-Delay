# multi_model_same_split.py
"""
Train multiple models using the exact same train/test split (random split).
Target is 'is_punctual' (1 = on-time). If original file has 'has_delay' (1=delay),
the script creates is_punctual = 1 - has_delay.

LogisticRegression uses OneHotEncoder (fit on X_train only).
Tree models (RandomForest, LightGBM, optional XGBoost) use integer label mapping.
All models use the same train/test rows to ensure fair comparison.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Optional boosters
try:
    from lightgbm import LGBMClassifier
except Exception as e:
    raise ImportError("Please install lightgbm: pip install lightgbm") from e

try:
    import xgboost as xgb
except Exception:
    xgb = None

print("scikit-learn version:", sklearn.__version__)

# ---------------- CONFIG ----------------
FILE_PATH = r"E:\3model.xlsx"   # change to your path
RANDOM_STATE = 42
TEST_SIZE = 0.2
FEATURES = ["Hbf", "arrive_station", "train_category", "depart_hour_bucket"]
ORIG_TARGET = "has_delay"   # if present, 1=delay
TARGET = "is_punctual"      # 1 = on-time
# ----------------------------------------

# 1) load data
df = pd.read_excel(FILE_PATH)
df.columns = df.columns.str.strip()

# 2) build target
if TARGET not in df.columns:
    if ORIG_TARGET in df.columns:
        df[TARGET] = 1 - df[ORIG_TARGET].astype(int)
        print("Created 'is_punctual' from 'has_delay'")
    else:
        raise KeyError("Provide 'has_delay' (1=delay) or 'is_punctual' in the input file.")

# 3) basic cleaning for categorical columns
for c in FEATURES:
    if c not in df.columns:
        raise ValueError(f"Missing feature column: {c}")
    df[c] = df[c].astype(str).str.strip().fillna("MISSING")

# 4) Compose dataset to use for splitting
#    If you want only July+Sept rows, filter first; otherwise use full df.
#    Example: filter to July+Sept (uncomment and adapt column name if you have a date column)
# date_col = "date"  # change if applicable
# df = df[df[date_col].dt.month.isin([7,9])]

X_full = df[FEATURES].copy()
y_full = df[TARGET].astype(int).copy()

# 5) single random split used for all models
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_full, y_full, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_full
)
print(f"Train rows: {len(X_train_raw)}, Test rows: {len(X_test_raw)}")

# 6) Prepare OneHot for Logistic Regression (fit on training rows only)
ohe_kwargs = {"handle_unknown": "ignore"}
try:
    encoder = OneHotEncoder(**ohe_kwargs, sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(**ohe_kwargs, sparse=False)

ohe_cols = FEATURES  # treat all features as categorical for logistic model
encoder.fit(X_train_raw[ohe_cols])
X_train_ohe = encoder.transform(X_train_raw[ohe_cols])
X_test_ohe  = encoder.transform(X_test_raw[ohe_cols])

# 7) Prepare integer label mapping for tree models (fit mapping on train only)
label_maps = {}
X_train_tree = pd.DataFrame(index=X_train_raw.index)
X_test_tree  = pd.DataFrame(index=X_test_raw.index)

for col in FEATURES:
    uniques = X_train_raw[col].unique().tolist()
    mapping = {v: i for i, v in enumerate(uniques)}
    unknown_idx = len(mapping)
    # apply mapping
    X_train_tree[col + "_idx"] = X_train_raw[col].map(mapping).fillna(unknown_idx).astype(int)
    X_test_tree[col + "_idx"]  = X_test_raw[col].map(lambda v: mapping.get(v, unknown_idx)).astype(int)
    label_maps[col] = {"mapping": mapping, "unknown_index": unknown_idx}

# 8) Train models on the same train split
print("\nTraining models on identical train/test split...")

# Logistic Regression (One-Hot input)
logreg = LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')
logreg.fit(X_train_ohe, y_train)

# Random Forest (integer-coded input)
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced')
rf.fit(X_train_tree, y_train)

# LightGBM (integer-coded input)
lgbm = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=RANDOM_STATE, class_weight='balanced')
lgbm.fit(X_train_tree, y_train)

# XGBoost (optional)
xgb_model = None
if xgb is not None:
    try:
        xgb_model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
        xgb_model.fit(X_train_tree, y_train, eval_set=[(X_test_tree, y_test)], verbose=False)
    except Exception as e:
        print("Warning: XGBoost training failed:", e)
        xgb_model = None

# 9) Evaluate helper
def evaluate(name, model, X, y_true, is_prob_model=True):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:,1] if is_prob_model and hasattr(model, "predict_proba") else None
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan
    cm   = confusion_matrix(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")
    print("Confusion matrix:\n", cm)
    return {"name": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc, "cm": cm, "proba": y_proba}

results = []
results.append(evaluate("Logistic Regression", logreg, X_test_ohe, y_test, is_prob_model=True))
results.append(evaluate("Random Forest", rf, X_test_tree, y_test, is_prob_model=True))
results.append(evaluate("LightGBM", lgbm, X_test_tree, y_test, is_prob_model=True))
if xgb_model is not None:
    results.append(evaluate("XGBoost", xgb_model, X_test_tree, y_test, is_prob_model=True))

# 10) ROC plot (if probabilities exist)
plt.figure(figsize=(8,6))
plotted = False
for res in results:
    proba = res["proba"]
    if proba is not None and not np.all(np.isnan(proba)):
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_val = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, label=f"{res['name']} (AUC={auc_val:.3f})")
        plotted = True
if plotted:
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - All models (same random split)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_same_split.png", dpi=300)
    plt.close()
    print("Saved ROC plot -> roc_same_split.png")
else:
    print("No probability outputs available; skipped ROC plot.")

# 11) Summary table and save
summary = pd.DataFrame(results).set_index("name")[["accuracy","precision","recall","f1","auc"]]
print("\nModel comparison (on test set):")
print(summary)
summary.to_csv("model_comparison_same_split.csv", index=True)
print("Saved summary -> model_comparison_same_split.csv")

# 12) Save confusion matrix for Random Forest
cm_rf = results[1]["cm"]
plt.figure(figsize=(6,5))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", xticklabels=["Delay(0)","On-time(1)"], yticklabels=["Delay(0)","On-time(1)"], cbar=False)
plt.title("Random Forest - Confusion Matrix (same split)")
plt.tight_layout()
plt.savefig("rf_confusion_same_split.png", dpi=300)
plt.close()
print("Saved rf_confusion_same_split.png")

# 13) Save feature importances for tree models (they are integer-coded columns)
rf_feats = X_train_tree.columns.tolist()
imp_df = pd.DataFrame({"feature": rf_feats, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
imp_df.to_csv("rf_importances_same_split.csv", index=False)
print("Saved rf_importances_same_split.csv")

print("All done.")
