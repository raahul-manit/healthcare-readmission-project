import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("data/diabetic_data.csv")

# Cleaning
df = df.replace("?", np.nan)
df = df.drop(columns=["weight", "payer_code", "medical_specialty"])
df = df[df["gender"] != "Unknown/Invalid"]

# Binary target
df["readmitted_binary"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)
df = df.drop(columns=["readmitted", "encounter_id", "patient_nbr"])

# Features and target
X = df.drop("readmitted_binary", axis=1)
y = df["readmitted_binary"]

# Column groups
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Final pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42
    ))
])

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
metrics = {
    "Accuracy": round(accuracy_score(y_test, y_pred), 4),
    "Precision": round(precision_score(y_test, y_pred), 4),
    "Recall": round(recall_score(y_test, y_pred), 4),
    "F1 Score": round(f1_score(y_test, y_pred), 4),
    "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4)
}

print("===== FINAL XGBOOST METRICS =====")
for key, value in metrics.items():
    print(f"{key}: {value}")

# Save metrics
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv("outputs/final_xgboost_metrics.csv", index=False)

# Save model
with open("outputs/final_xgboost_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nSaved:")
print("1. outputs/final_xgboost_metrics.csv")
print("2. outputs/final_xgboost_model.pkl")