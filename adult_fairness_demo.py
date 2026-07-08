"""
Adult Income Fairness Demonstration Study
Author: Gulsum Cigdem Cavdaroglu 'Algorithmic Fairness and Inclusive Innovation'

This script:
1. Loads the UCI Adult Income dataset from adult.zip or an extracted folder.
2. Cleans missing values marked as '?'.
3. Trains two models: Logistic Regression and Random Forest.
4. Computes predictive performance metrics.
5. Computes group fairness metrics for Sex = Female vs Male:
   - Statistical Parity Difference (SPD)
   - Equal Opportunity Difference (EOD)
   - Disparate Impact Ratio (DIR)
6. Produces SHAP-based explainability output for the Random Forest model.
7. Saves publication-ready tables and figures under an output folder.

Run:
    python adult_fairness_demo.py --data_zip adult.zip --output_dir adult_fairness_outputs
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income"
]

NUMERIC_FEATURES = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_FEATURES = [
    "workclass", "education", "marital_status", "occupation", "relationship",
    "race", "sex", "native_country"
]
TARGET = "income_binary"
PROTECTED = "sex"
PRIVILEGED_GROUP = "Male"
UNPRIVILEGED_GROUP = "Female"


def extract_zip_if_needed(data_zip: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    if data_zip.exists():
        with zipfile.ZipFile(data_zip, "r") as zf:
            zf.extractall(extract_dir)
    return extract_dir


def load_adult_dataset(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "adult.data"
    test_path = data_dir / "adult.test"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"adult.data and adult.test must be present in {data_dir}")

    train = pd.read_csv(train_path, header=None, names=COLUMNS, na_values="?", skipinitialspace=True)
    # adult.test has a first line note, so skip it
    test = pd.read_csv(test_path, header=None, names=COLUMNS, na_values="?", skipinitialspace=True, skiprows=1)

    for df in (train, test):
        # Strip whitespace and trailing periods in test labels
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip().str.replace(r"\.$", "", regex=True)
        df.replace("?", np.nan, inplace=True)
        df[TARGET] = (df["income"] == ">50K").astype(int)

    # Drop rows with missing values for demonstration clarity.
    train = train.dropna().reset_index(drop=True)
    test = test.dropna().reset_index(drop=True)
    return train, test


def build_preprocessor() -> ColumnTransformer:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older scikit-learn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe)
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES)
    ])


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names = []
    names.extend(NUMERIC_FEATURES)
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    try:
        cat_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    except Exception:
        cat_names = ohe.get_feature_names(CATEGORICAL_FEATURES).tolist()
    names.extend(cat_names)
    return names


def performance_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_proba),
    }


def fairness_metrics(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    group = df[PROTECTED].values
    priv = group == PRIVILEGED_GROUP
    unpriv = group == UNPRIVILEGED_GROUP

    def positive_rate(mask: np.ndarray) -> float:
        return float(np.mean(y_pred[mask] == 1)) if np.sum(mask) else np.nan

    def tpr(mask: np.ndarray) -> float:
        denom = np.sum((y_true == 1) & mask)
        if denom == 0:
            return np.nan
        return float(np.sum((y_pred == 1) & (y_true == 1) & mask) / denom)

    def fpr(mask: np.ndarray) -> float:
        denom = np.sum((y_true == 0) & mask)
        if denom == 0:
            return np.nan
        return float(np.sum((y_pred == 1) & (y_true == 0) & mask) / denom)

    pr_priv = positive_rate(priv)
    pr_unpriv = positive_rate(unpriv)
    tpr_priv = tpr(priv)
    tpr_unpriv = tpr(unpriv)
    fpr_priv = fpr(priv)
    fpr_unpriv = fpr(unpriv)

    return {
        "PositiveRate_Female": pr_unpriv,
        "PositiveRate_Male": pr_priv,
        "SPD_Female_minus_Male": pr_unpriv - pr_priv,
        "Abs_SPD": abs(pr_unpriv - pr_priv),
        "TPR_Female": tpr_unpriv,
        "TPR_Male": tpr_priv,
        "EOD_Female_minus_Male": tpr_unpriv - tpr_priv,
        "Abs_EOD": abs(tpr_unpriv - tpr_priv),
        "FPR_Female": fpr_unpriv,
        "FPR_Male": fpr_priv,
        "Disparate_Impact_Ratio_Female_over_Male": pr_unpriv / pr_priv if pr_priv else np.nan,
    }


def data_audit(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)
    rows = []
    for split_name, sub in combined.groupby("split"):
        for sex, g in sub.groupby(PROTECTED):
            rows.append({
                "Split": split_name,
                "Group": sex,
                "N": len(g),
                "Share": len(g) / len(sub),
                "Observed_Positive_Income_Rate": g[TARGET].mean(),
            })
    return pd.DataFrame(rows)


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Pipeline]:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=None, class_weight=None, solver="lbfgs"),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, random_state=42, n_jobs=-1, min_samples_leaf=3, class_weight=None
        ),
    }
    pipelines = {}
    for name, clf in models.items():
        pipe = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", clf)
        ])
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe
    return pipelines


def save_bar_plot(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(df[x], df[y])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def create_shap_outputs(model_pipe: Pipeline, X_test: pd.DataFrame, output_dir: Path, model_label: str = "Logistic Regression") -> pd.DataFrame:

    import shap

    preprocessor = model_pipe.named_steps["preprocess"]
    model = model_pipe.named_steps["model"]
    X_trans = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)

    rng = np.random.RandomState(42)
    sample_n = min(500, X_trans.shape[0])
    bg_n = min(500, X_trans.shape[0])
    sample_idx = rng.choice(X_trans.shape[0], size=sample_n, replace=False)
    bg_idx = rng.choice(X_trans.shape[0], size=bg_n, replace=False)

    explainer = shap.LinearExplainer(model, X_trans[bg_idx])
    shap_values = explainer.shap_values(X_trans[sample_idx])
    sv = np.asarray(shap_values)

    mean_abs = np.abs(sv).mean(axis=0)
    imp = pd.DataFrame({"Feature": feature_names, "MeanAbsSHAP": mean_abs}).sort_values("MeanAbsSHAP", ascending=False)
    imp.head(20).to_csv(output_dir / "shap_top20_logistic_regression.csv", index=False)

    plt.figure(figsize=(9, 6))
    top = imp.head(15).iloc[::-1]
    plt.barh(top["Feature"], top["MeanAbsSHAP"])
    plt.xlabel("Mean |SHAP value|")
    plt.title("Logistic Regression SHAP Feature Importance (Top 15)")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_shap_top15_logistic_regression.png", dpi=300)
    plt.close()
    return imp

def write_markdown_report(output_dir: Path, perf_df: pd.DataFrame, fair_df: pd.DataFrame, audit_df: pd.DataFrame, shap_df: pd.DataFrame) -> None:
    top_features = ", ".join(shap_df.head(5)["Feature"].tolist())
    best_acc_model = perf_df.sort_values("Accuracy", ascending=False).iloc[0]["Model"]
    lowest_spd_model = fair_df.sort_values("Abs_SPD").iloc[0]["Model"]
    lowest_eod_model = fair_df.sort_values("Abs_EOD").iloc[0]["Model"]

    md = []
    md.append("# Adult Income Fairness Demonstration Study Results\n")
    md.append("## Dataset and task\n")
    md.append("The demonstration uses the UCI Adult Income dataset. The prediction task is binary classification of whether income is >50K. The protected attribute used for the fairness audit is sex, with Female treated as the unprivileged group and Male as the privileged comparison group. Rows with missing values marked as '?' were removed for this compact demonstration.\n")
    md.append("## Data audit by sex\n")
    md.append(audit_df.to_markdown(index=False, floatfmt=".4f"))
    md.append("\n\n## Predictive performance\n")
    md.append(perf_df.to_markdown(index=False, floatfmt=".4f"))
    md.append("\n\n## Fairness metrics\n")
    md.append(fair_df.to_markdown(index=False, floatfmt=".4f"))
    (output_dir / "adult_fairness_report.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_zip", type=str, default="/mnt/data/adult.zip")
    parser.add_argument("--data_dir", type=str, default="/mnt/data/adult")
    parser.add_argument("--output_dir", type=str, default="/mnt/data/adult_fairness_outputs")
    args = parser.parse_args()

    data_zip = Path(args.data_zip)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if data_zip.exists():
        extract_zip_if_needed(data_zip, data_dir)

    train, test = load_adult_dataset(data_dir)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train, y_train = train[features], train[TARGET]
    X_test, y_test = test[features], test[TARGET]

    audit_df = data_audit(train, test)
    audit_df.to_csv(output_dir / "data_audit_by_sex.csv", index=False)

    pipelines = train_models(X_train, y_train)

    perf_rows = []
    fair_rows = []
    for name, pipe in pipelines.items():
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        perf = performance_metrics(y_test.values, y_pred, y_proba)
        fair = fairness_metrics(test, y_test.values, y_pred)
        perf_rows.append({"Model": name, **perf})
        fair_rows.append({"Model": name, **fair})

    perf_df = pd.DataFrame(perf_rows)
    fair_df = pd.DataFrame(fair_rows)
    perf_df.to_csv(output_dir / "model_performance_metrics.csv", index=False)
    fair_df.to_csv(output_dir / "fairness_metrics_by_sex.csv", index=False)

    # Combined compact publication table
    combined = perf_df.merge(fair_df[["Model", "SPD_Female_minus_Male", "Abs_SPD", "EOD_Female_minus_Male", "Abs_EOD", "Disparate_Impact_Ratio_Female_over_Male"]], on="Model")
    combined.to_csv(output_dir / "publication_metrics_table.csv", index=False)

    # Figures
    save_bar_plot(perf_df, "Model", "Accuracy", "Model Accuracy Comparison", "Accuracy", output_dir / "figure_accuracy_comparison.png")
    save_bar_plot(fair_df, "Model", "Abs_SPD", "Absolute Statistical Parity Difference", "Abs(SPD)", output_dir / "figure_abs_spd_comparison.png")
    save_bar_plot(fair_df, "Model", "Abs_EOD", "Absolute Equal Opportunity Difference", "Abs(EOD)", output_dir / "figure_abs_eod_comparison.png")

    shap_df = create_shap_outputs(pipelines["Logistic Regression"], X_test, output_dir, model_label="Logistic Regression")
    write_markdown_report(output_dir, perf_df, fair_df, audit_df, shap_df)

    print("Outputs saved to:", output_dir)
    print("\nPublication metrics table:")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
