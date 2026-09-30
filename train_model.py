"""Train and evaluate customer churn classifiers."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "customer_churn.csv"
MODEL_DIR = ROOT_DIR / "models"
STATIC_DIR = ROOT_DIR / "static"
TARGET_COLUMN = "Churn"
RANDOM_STATE = 42


def load_and_clean_data(data_path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load the Telco CSV, validate its target, and prepare features."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. See README.md to download it."
        )

    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip()
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    required_columns = {TARGET_COLUMN, "customerID"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    print(f"Dataset shape: {data.shape[0]} rows x {data.shape[1]} columns")
    print("\nData types:\n", data.dtypes.to_string())
    print("\nMissing values:\n", data.isna().sum().to_string())
    print(f"\nDuplicate records: {data.duplicated().sum()}")
    print("\nDescriptive statistics:\n", data.describe(include="all").to_string())

    data = data.drop_duplicates().copy()
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype("string").str.strip()
    target = data[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    valid_target = target.notna()
    data = data.loc[valid_target].copy()
    target = target.loc[valid_target].astype(int)

    data = data.drop(columns=[TARGET_COLUMN, "customerID"])
    return data, target


def create_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build numeric and categorical preprocessing from the training schema."""
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def save_eda_plots(features: pd.DataFrame, target: pd.Series) -> None:
    """Save the requested exploratory plots in the Flask static directory."""
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    eda_data = features.copy()
    eda_data[TARGET_COLUMN] = target.map({1: "Yes", 0: "No"})
    plot_specs = [
        (TARGET_COLUMN, "Churn distribution", "churn_distribution.png"),
        ("Contract", "Churn by contract type", "churn_by_contract.png"),
        ("InternetService", "Churn by internet service", "churn_by_internet.png"),
        ("PaymentMethod", "Churn by payment method", "churn_by_payment.png"),
    ]
    for column, title, filename in plot_specs:
        if column not in eda_data.columns:
            continue
        figure, axis = plt.subplots(figsize=(8, 5))
        if column == TARGET_COLUMN:
            sns.countplot(data=eda_data, x=column, hue=column, legend=False, ax=axis)
        else:
            sns.countplot(data=eda_data, x=column, hue=TARGET_COLUMN, ax=axis)
            axis.legend(title="Churn")
        axis.set_title(title)
        axis.set_ylabel("Customers")
        axis.tick_params(axis="x", rotation=20)
        figure.tight_layout()
        figure.savefig(STATIC_DIR / filename, dpi=140)
        plt.close(figure)

    for column, title, filename in [
        ("MonthlyCharges", "Monthly charges distribution", "monthly_charges.png"),
        ("tenure", "Tenure distribution", "tenure_distribution.png"),
    ]:
        if column not in eda_data.columns:
            continue
        figure, axis = plt.subplots(figsize=(8, 5))
        sns.histplot(data=eda_data, x=column, hue=TARGET_COLUMN, kde=True, ax=axis)
        axis.set_title(title)
        figure.tight_layout()
        figure.savefig(STATIC_DIR / filename, dpi=140)
        plt.close(figure)

    numeric_data = features.select_dtypes(include=[np.number]).copy()
    if not numeric_data.empty:
        numeric_data[TARGET_COLUMN] = target
        figure, axis = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="vlag", center=0, fmt=".2f", ax=axis)
        axis.set_title("Correlation heatmap: numeric features and churn")
        figure.tight_layout()
        figure.savefig(STATIC_DIR / "numeric_correlation.png", dpi=140)
        plt.close(figure)


def train_and_evaluate() -> dict[str, object]:
    """Train both models, select by ROC-AUC then F1, and serialize artifacts."""
    features, target = load_and_clean_data()
    save_eda_plots(features, target)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    results: list[dict[str, object]] = []
    for name, classifier in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor(x_train)),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        results.append(
            {
                "name": name,
                "pipeline": pipeline,
                "predictions": predictions,
                "probabilities": probabilities,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1_score": f1_score(y_test, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probabilities),
            }
        )

    metrics = ("accuracy", "precision", "recall", "f1_score", "roc_auc")
    comparison = pd.DataFrame(
        [{"name": result["name"], **{key: result[key] for key in metrics}} for result in results]
    ).set_index("name")
    print("\nModel comparison (test set):\n", comparison.to_string(float_format=lambda value: f"{value:.4f}"))
    best_result = max(results, key=lambda result: (result["roc_auc"], result["f1_score"]))
    best_pipeline = best_result["pipeline"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_DIR / "churn_model.pkl")
    joblib.dump(best_pipeline.named_steps["preprocessor"], MODEL_DIR / "preprocessor.pkl")

    confusion_figure, confusion_axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        confusion_matrix(y_test, best_result["predictions"]),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No churn", "Churn"],
        yticklabels=["No churn", "Churn"],
        ax=confusion_axis,
    )
    confusion_axis.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix: {best_result['name']}")
    confusion_figure.tight_layout()
    confusion_figure.savefig(STATIC_DIR / "confusion_matrix.png", dpi=140)
    plt.close(confusion_figure)

    roc_figure, roc_axis = plt.subplots(figsize=(6, 5))
    for result in results:
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, result["probabilities"])
        roc_axis.plot(false_positive_rate, true_positive_rate, label=f"{result['name']} (AUC {result['roc_auc']:.3f})")
    roc_axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    roc_axis.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    roc_axis.legend(loc="lower right")
    roc_figure.tight_layout()
    roc_figure.savefig(STATIC_DIR / "roc_curve.png", dpi=140)
    plt.close(roc_figure)

    print(f"\nSelected-model classification report ({best_result['name']}):\n")
    print(classification_report(y_test, best_result["predictions"], target_names=["No", "Yes"], zero_division=0))
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)
    print(f"Selected Model: {best_result['name']}")
    for metric in metrics:
        label = "F1 Score" if metric == "f1_score" else metric.replace("_", " ").title()
        print(f"{label}: {best_result[metric] * 100:.2f}%")
    print(f"\nModel saved to:\n{MODEL_DIR / 'churn_model.pkl'}")
    print("Preprocessor saved to:", MODEL_DIR / "preprocessor.pkl")
    print("Evaluation and EDA plots saved to:", STATIC_DIR)
    return best_result


if __name__ == "__main__":
    train_and_evaluate()