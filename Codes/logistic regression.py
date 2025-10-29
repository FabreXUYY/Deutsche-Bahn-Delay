
# logreg_punctuality.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------- CONFIG ----------
INPUT_PATH = r"E:\3model.xlsx"
OUTPUT_IMAGE = r"./confusion_matrix.png"
OUTPUT_COEF_CSV = r"./logreg_feature_coefs.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
# ----------------------------

# 1. Load data
df = pd.read_excel(INPUT_PATH)

# 2. Ensure target column 'is_punctual' exists (1 = on-time, 0 = delay)
# If your file only has 'has_delay' (1 = delay, 0 = on-time), create is_punctual = 1 - has_delay
if "is_punctual" not in df.columns:
    if "has_delay" in df.columns:
        df["is_punctual"] = 1 - df["has_delay"]
    else:
        raise KeyError("No 'is_punctual' or 'has_delay' column found in the input file.")

# 3. Define features and target
features = ["Hbf", "arrive_station", "train_category", "depart_hour_bucket"]
target = "is_punctual"

X = df[features].copy()
y = df[target].copy()

# 4. Build column transformer with OneHotEncoder for categorical columns
# handle_unknown='ignore' prevents errors when test set contains unseen categories
cat_transformer = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features)
    ],
    remainder="drop",
    verbose_feature_names_out=False  # available in newer sklearn; keeps names clean
)

# 5. Build pipeline: encoder -> classifier
pipeline = Pipeline([
    ("encoder", cat_transformer),
    ("clf", LogisticRegression(max_iter=1000, solver="lbfgs"))
])

# 6. Train-test split (randomly mix July and September data to reduce seasonality bias)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# 7. Fit model
pipeline.fit(X_train, y_train)

# 8. Predict and evaluate
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 9. Plot and save confusion matrix
labels = ["Delay (0)", "On-time (1)"]
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.title("Confusion Matrix - Logistic Regression (Punctuality Prediction)")
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=300)
plt.close()
print(f"\nSaved confusion matrix to {os.path.abspath(OUTPUT_IMAGE)}")

# 10. Extract one-hot feature names and coefficients, then save to CSV
# Get the fitted OneHotEncoder from the pipeline
ohe = pipeline.named_steps["encoder"].named_transformers_["cat"]
# sklearn >=1.0: use get_feature_names_out
try:
    feat_names = ohe.get_feature_names_out(features)
except AttributeError:
    # fallback: construct names manually
    categories = ohe.categories_
    feat_names = []
    for col, cats in zip(features, categories):
        feat_names += [f"{col}__{str(c)}" for c in cats]

# Get coefficients from the logistic regression step
coefs = pipeline.named_steps["clf"].coef_[0]
coef_df = pd.DataFrame({
    "feature": feat_names,
    "coefficient": coefs
}).sort_values(by="coefficient", key=abs, ascending=False)

coef_df.to_csv(OUTPUT_COEF_CSV, index=False)
print(f"Saved logistic regression feature coefficients to {os.path.abspath(OUTPUT_COEF_CSV)}")

