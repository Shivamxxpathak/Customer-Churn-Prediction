from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

TARGET = "Churn"


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset.")
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric_features),
            ("categorical", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


def build_models(deployment: bool = False) -> dict[str, Any]:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=9),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "SVM": SVC(probability=True, random_state=42, class_weight="balanced"),
        "AdaBoost": AdaBoostClassifier(n_estimators=150, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    return {
        name: estimator
        for name, estimator in models.items()
        if not deployment or name in {
            "Logistic Regression",
            "Random Forest",
            "SVM",
            "Gradient Boosting",
        }
    }


def make_pipeline(X: pd.DataFrame, estimator: Any) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(X)),
            ("model", estimator),
        ]
    )


def evaluate_models(X_train, X_test, y_train, y_test, deployment: bool = False):
    results = []
    fitted = {}

    for name, estimator in build_models(deployment=deployment).items():
        pipeline = make_pipeline(X_train, estimator)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)[:, 1]
        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1": f1_score(y_test, pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, proba),
            }
        )
        fitted[name] = pipeline

    return pd.DataFrame(results).sort_values("ROC-AUC", ascending=False), fitted


def train_and_save(
    dataset_path: str | Path,
    output_dir: str | Path = "outputs",
    model_dir: str | Path = "models",
    deployment: bool = False,
):
    df = load_data(dataset_path)
    y = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results, fitted = evaluate_models(X_train, X_test, y_train, y_test, deployment=deployment)
    best_name = str(results.iloc[0]["Model"])
    best_pipeline = fitted[best_name]

    cv_mean, cv_std = (None, None)
    if not deployment:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(best_pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_mean, cv_std = float(scores.mean()), float(scores.std())
        results.loc[results["Model"] == best_name, "CV ROC-AUC Mean"] = cv_mean
        results.loc[results["Model"] == best_name, "CV ROC-AUC Std"] = cv_std

    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_dir / "model_comparison.csv", index=False)
    model_path = model_dir / "best_churn_model.joblib"
    joblib.dump(best_pipeline, model_path)

    predictions = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": best_pipeline.predict(X_test),
            "Churn_Probability": best_pipeline.predict_proba(X_test)[:, 1],
        }
    )
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    best_row = results.iloc[0].to_dict()
    metadata = {
        "best_model": best_name,
        "training_mode": "deployment" if deployment else "full",
        "dataset_rows": int(len(df)),
        "test_rows": int(len(X_test)),
        "metrics": {
            key: round(float(value), 4)
            for key, value in best_row.items()
            if key != "Model" and pd.notna(value)
        },
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    return results, model_path
