"""EDA utilities for the Steel Plate Defect Prediction project.

This script generates summary tables and exploratory figures and saves them
under the `reports/` directory so they can be embedded in the written report.

Run it directly (``python src/visualization/eda.py``) or import the public
functions inside the 01_eda.ipynb notebook.
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.autolayout": True})

# ---------------------------------------------------------------------------
# Paths helpers ----------------------------------------------------------------

def _project_root() -> Path:
    """Return the root directory ``tfg-steel-plate-defects`` regardless of cwd."""
    return Path(__file__).resolve().parents[2]


def _reports_dir(sub: str) -> Path:
    root = _project_root() / "reports" / sub
    root.mkdir(parents=True, exist_ok=True)
    return root

# ---------------------------------------------------------------------------
# Core EDA routines -----------------------------------------------------------

def load_data(csv_path: Path | str) -> pd.DataFrame:
    """Load CSV and ensure expected targets are treated as floats."""
    df = pd.read_csv(csv_path)
    target_cols = [
        "Pastry",
        "Z_Scratch",
        "K_Scatch",
        "Stains",
        "Dirtiness",
        "Bumps",
        "Other_Faults",
    ]
    df[target_cols] = df[target_cols].astype(float)
    return df


# ---------------------------------------------------------------------------
# 4.1 Summary statistics ------------------------------------------------------

def summary_statistics(df: pd.DataFrame, *, output_csv: bool = True) -> pd.DataFrame:
    """Return DataFrame with numeric summary and optionally write CSV."""
    numeric_stats = df.describe().T  # count, mean, std, min, 25%, 50%, 75%, max
    numeric_stats["iqr"] = numeric_stats["75%"] - numeric_stats["25%"]

    if output_csv:
        out_path = _reports_dir("tables") / "summary_statistics.csv"
        numeric_stats.to_csv(out_path, float_format="%.4f")
    return numeric_stats


def class_balance(df: pd.DataFrame, *, output_csv: bool = True) -> pd.DataFrame:
    """Return balance of the 7 target classes and save as CSV."""
    targets = [
        "Pastry",
        "Z_Scratch",
        "K_Scatch",
        "Stains",
        "Dirtiness",
        "Bumps",
        "Other_Faults",
    ]
    counts = df[targets].sum().rename("positives")
    total = len(df)
    balance = pd.DataFrame({"positives": counts, "percent": counts / total * 100})

    if output_csv:
        out_path = _reports_dir("tables") / "class_balance.csv"
        balance.to_csv(out_path, float_format="%.2f")
    return balance

# ---------------------------------------------------------------------------
# 4.2 Distributions & correlations -------------------------------------------

def plot_histograms(df: pd.DataFrame, cols: List[str] | None = None) -> None:
    """Histogram & KDE for each numeric column (or subset)."""
    cols = cols or df.select_dtypes(include=np.number).columns.tolist()
    figs_path = _reports_dir("figures") / "histograms"
    figs_path.mkdir(parents=True, exist_ok=True)

    for col in cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        fig.savefig(figs_path / f"hist_{col}.png", dpi=300)
        plt.close(fig)


def boxplots_by_target(df: pd.DataFrame, feature_cols: List[str] | None = None, log_scale: bool = False,) -> None:
    """Box/violin plots stratified by each defect label (one vs rest)."""
    targets = [
        "Pastry",
        "Z_Scratch",
        "K_Scatch",
        "Stains",
        "Dirtiness",
        "Bumps",
        "Other_Faults",
    ]
    feature_cols = feature_cols or [c for c in df.columns if c not in targets + ["id"]]
    figs_path = _reports_dir("figures") / "boxplots"
    figs_path.mkdir(parents=True, exist_ok=True)

    for target in targets:
        melt_df = df[[target] + feature_cols].melt(id_vars=target, var_name="feature", value_name="value")
        melt_df[target] = melt_df[target].astype(bool)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=melt_df, x="feature", y="value", hue=target, ax=ax)
        """nuevo"""
        if log_scale:
            ax.set_yscale("symlog")  # o "log" si no hay ceros/negativos
        
        ax.tick_params(axis="x", rotation=90)
        ax.set_title(f"Feature distributions vs {target}")
        fig.savefig(figs_path / f"box_{target}.png", dpi=300)
        plt.close(fig)


def correlation_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of Pearson correlations for numeric features."""
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmax=1, vmin=-1, annot=False, ax=ax)
    ax.set_title("Correlation matrix (Pearson)")
    fig.savefig(_reports_dir("figures") / "correlation_heatmap.png", dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------------------
# 4.3 Missing values ----------------------------------------------------------

def missing_values_table(df: pd.DataFrame, *, output_csv: bool = True) -> pd.DataFrame:
    missing_series = df.isna().sum()
    percent = missing_series / len(df) * 100
    miss_df = pd.DataFrame({"n_missing": missing_series, "percent": percent})

    if output_csv:
        miss_df.to_csv(_reports_dir("tables") / "missing_values.csv", float_format="%.2f")
    return miss_df


def missing_heatmap(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isna(), cbar=False)
    plt.title("Missing values heatmap")
    plt.savefig(_reports_dir("figures") / "missing_heatmap.png", dpi=300)
    plt.close()

# ---------------------------------------------------------------------------
# 4.4 Outlier detection & data quality ---------------------------------------

def detect_outliers_iqr(df: pd.DataFrame, cols: List[str] | None = None, *, output_csv: bool = True) -> pd.DataFrame:
    """Detect outliers via 1.5*IQR rule and return counts per column."""
    cols = cols or df.select_dtypes(include=np.number).columns.tolist()
    outlier_counts = {}
    for col in cols:
        q1, q3 = np.percentile(df[col].dropna(), [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_counts[col] = mask.sum()

    out_df = pd.DataFrame.from_dict(outlier_counts, orient="index", columns=["n_outliers"])

    if output_csv:
        out_df.to_csv(_reports_dir("tables") / "outliers_iqr.csv")
    return out_df


# ---------------------------------------------------------------------------
# Runner ----------------------------------------------------------------------

def run_eda(train_csv: Path | str | None = None) -> None:
    """Convenience wrapper to execute the full EDA pipeline."""
    if train_csv is None:
        train_csv = _project_root() / "data" / "raw" / "playground-series-s4e3" / "train.csv"
    df = load_data(train_csv)

    # 4.1 Summary tables
    summary_statistics(df)
    class_balance(df)

    # 4.2 Visualisations
    plot_histograms(df)
    boxplots_by_target(df, log_scale = True)
    correlation_heatmap(df)

    # 4.3 Missing values
    missing_values_table(df)
    missing_heatmap(df)

    # 4.4 Outliers
    detect_outliers_iqr(df)

    print("EDA artefacts saved under reports/figures and reports/tables.")


if __name__ == "__main__":
    run_eda()
