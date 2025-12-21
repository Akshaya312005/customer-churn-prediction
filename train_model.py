import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

import joblib


# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# -----------------------------
# 2. Drop unnecessary column
# -----------------------------
df.drop("customerID", axis=1, inplace=True)


# -----------------------------
# 3. Fix TotalCharges column
# -----------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)


# -----------------------------
# 4. Separate target and features
# -----------------------------
y = df["Churn"].map({"Yes": 1, "No": 0})
X = df.drop("Churn", axis=1)


# -----------------------------
# 5. Identify column types
# -----------------------------
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns


# -----------------------------
# 6. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# -----------------------------
# 7. Preprocessing pipelines
# -----------------------------
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ]
)


# -----------------------------
# 8. Model pipeline
# -----------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ]
)


# -----------------------------
# 9. Train model
# -----------------------------
model.fit(X_train, y_train)


# -----------------------------
# 10. Evaluate model
# -----------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# -----------------------------
# 11. Save model
# -----------------------------
joblib.dump(model, "churn_model.pkl")

print("\nModel saved as churn_model.pkl")

# -----------------------------
# 12. Feature importance
# -----------------------------
feature_names_num = num_cols
feature_names_cat = model.named_steps["preprocessor"] \
    .named_transformers_["cat"] \
    .named_steps["encoder"] \
    .get_feature_names_out(cat_cols)

all_feature_names = np.concatenate([feature_names_num, feature_names_cat])

importances = model.named_steps["classifier"].feature_importances_

feature_importance_df = (
    pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    })
    .sort_values(by="importance", ascending=False)
)

print("\nTop 10 Important Features:")
print(feature_importance_df.head(10))
