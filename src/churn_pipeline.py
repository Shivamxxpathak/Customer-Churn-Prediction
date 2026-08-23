from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

TARGET = "Churn"


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the Telco churn CSV and normalize common formatting issues."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset.")
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("categorical", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


def build_models() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=8),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ),
        "SVM": SVC(probability=True, random_state=42, class_weight="balanced"),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    results = []
    fitted: dict[str, Pipeline] = {}

    for name, estimator in build_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", estimator),
            ]
        )
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

    voting_estimators = [
        ("lr", fitted["Logistic Regression"]),
        ("rf", fitted["Random Forest"]),
        ("svm", fitted["SVM"]),
    ]
    # VotingClassifier expects base estimators that are not already fitted.
    voting = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=2000)),
            ("rf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
            ("svm", SVC(probability=True, random_state=42, class_weight="balanced")),
        ],
        voting="soft",
    )
    voting_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", voting),
        ]
    )
    voting_pipeline.fit(X_train, y_train)
    voting_pred = voting_pipeline.predict(X_test)
    voting_proba = voting_pipeline.predict_proba(X_test)[:, 1]
    results.append(
        {
            "Model": "Voting Ensemble",
            "Accuracy": accuracy_score(y_test, voting_pred),
            "Precision": precision_score(y_test, voting_pred, zero_division=0),
            "Recall": recall_score(y_test, voting_pred, zero_division=0),
            "F1": f1_score(y_test, voting_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, voting_proba),
        }
    )
    fitted["Voting Ensemble"] = voting_pipeline

    return pd.DataFrame(results).sort_values("ROC-AUC", ascending=False), fitted


def cross_validate_best(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    folds: int = 5,
) -> tuple[float, float]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())


def train_and_save(
    dataset_path: str | Path,
    output_dir: str | Path = "outputs",
    model_dir: str | Path = "models",
) -> tuple[pd.DataFrame, Path]:
    df = load_data(dataset_path)
    y = df[TARGET].map({"Yes": 1, "No": 0}) if df[TARGET].dtype == "object" else df[TARGET]
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results, fitted = evaluate_models(X_train, X_test, y_train, y_test)
    best_name = results.iloc[0]["Model"]
    best_pipeline = fitted[best_name]

    cv_mean, cv_std = cross_validate_best(best_pipeline, X_train, y_train)
    results.loc[results["Model"] == best_name, "CV ROC-AUC Mean"] = cv_mean
    results.loc[results["Model"] == best_name, "CV ROC-AUC Std"] = cv_std

    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_dir / "model_comparison.csv", index=False)
    joblib.dump(best_pipeline, model_dir / "best_churn_model.joblib")

    predictions = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": best_pipeline.predict(X_test),
            "Churn_Probability": best_pipeline.predict_proba(X_test)[:, 1],
        }
    )
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    return results, model_dir / "best_churn_model.joblib"
