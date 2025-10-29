# random_forest_punctuality.py
# This script trains a RandomForestClassifier to predict punctuality (is_punctual = 1 for on-time)
# It uses the same train/test split (random_state=42, stratify by target) so results are directly
# comparable to a Logistic Regression baseline trained with the same split.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sklearn

# ---------- CONFIG ----------
INPUT_PATH = r"E:\3model.xlsx"      # change to your file path
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 200
OUTPUT_CM = "./rf_confusion_matrix.png"
OUTPUT_IMPORTANCES = "./rf_feature_importances.csv"
# ----------------------------

print("scikit-learn version:", sklearn.__version__)

# 1) Load data
df = pd.read_excel(INPUT_PATH)

# 2) clean column names
df.columns = df.columns.str.strip()

# 3) ensure punctuality target column exists
# If you have only 'has_delay' (1 = delayed, 0 = on-time), create is_punctual = 1 - has_delay
if "is_punctual" not in df.columns:
    if "has_delay" in df.columns:
        df["is_punctual"] = 1 - df["has_delay"]
    else:
        raise KeyError("No 'is_punctual' or 'has_delay' column found in the input file.")

# 4) Define features and target
features = ["Hbf", "arrive_station", "train_category", "depart_hour_bucket"]
target = "is_punctual"

# sanity check
for col in features + [target]:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col} in the input file")

X = df[features].copy()
y = df[target].copy()

# 5) Create train/test split ONCE so it matches the one used for logistic regression baseline
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# 6) Build OneHotEncoder with compatibility for different sklearn versions
# newer sklearn (>=1.2) uses sparse_output, older uses sparse
ohe_kwargs = {"handle_unknown": "ignore"}
try:
    # try the newer parameter name
    enc = OneHotEncoder(**ohe_kwargs, sparse_output=False)
except TypeError:
    # fall back to older parameter name
    enc = OneHotEncoder(**ohe_kwargs, sparse=False)

column_transformer = ColumnTransformer(
    transformers=[("cat", enc, features)],
    remainder="drop",
    verbose_feature_names_out=False  # keep feature names clean if supported
)

# 7) Fit encoder on training raw data and transform both train and test
# We fit the encoder only on train to avoid leakage
column_transformer.fit(X_train_raw)
X_train = column_transformer.transform(X_train_raw)
X_test = column_transformer.transform(X_test_raw)

# 8) Train Random Forest on encoded features
rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

# 9) Predict and evaluate
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 10) Plot and save confusion matrix (labels: Delay=0, On-time=1)
labels = ["Delay (0)", "On-time (1)"]
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.title("Random Forest Confusion Matrix (Punctuality Prediction)")
plt.tight_layout()
plt.savefig(OUTPUT_CM, dpi=300)
plt.close()
print(f"Saved confusion matrix to {os.path.abspath(OUTPUT_CM)}")

# 11) Map feature importances back to one-hot feature names
# get feature names from the fitted OneHotEncoder
ohe = column_transformer.named_transformers_["cat"]
try:
    feature_names = ohe.get_feature_names_out(features)
except Exception:
    # fallback: manually build feature names
    categories = ohe.categories_
    feature_names = []
    for col, cats in zip(features, categories):
        feature_names += [f"{col}__{str(c)}" for c in cats]

importances = rf.feature_importances_
if len(feature_names) != len(importances):
    print("Warning: feature names length does not match importances length.")
# create dataframe of importances
imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
imp_df.to_csv(OUTPUT_IMPORTANCES, index=False)
print(f"Saved feature importances to {os.path.abspath(OUTPUT_IMPORTANCES)}")

# 12) Optional: plot top 30 importances
top_k = 30
plt.figure(figsize=(8, max(4, 0.25 * min(top_k, len(imp_df)))))
imp_df.head(top_k).sort_values("importance").plot.barh(x="feature", y="importance", legend=False, color="skyblue")
plt.title("Top feature importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("./rf_top_importances.png", dpi=300)
plt.close()
print(f"Saved top feature importances plot to {os.path.abspath('./rf_top_importances.png')}")
