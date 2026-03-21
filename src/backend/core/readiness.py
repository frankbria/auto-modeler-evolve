"""readiness.py

Data Readiness Assessment — scores a DataFrame on its suitability for ML
modeling before the analyst clicks Train.

Five components (weighted, total 100 pts):
  1. Row count        (25 pts) — enough data to train on?
  2. Missing values   (25 pts) — how complete is the data?
  3. Duplicate rows   (20 pts) — is data deduplicated?
  4. Feature diversity (15 pts) — mix of numeric + categorical columns?
  5. Data type quality (15 pts) — no all-null columns, usable types?

Each component produces a score, a status ("good"/"warning"/"critical"),
a plain-English detail line, and (optionally) a recommendation.

The overall score maps to a letter grade:
  90-100 → A  (ready to train)
  75-89  → B  (minor issues, safe to train)
  60-74  → C  (notable issues, consider fixing)
  45-59  → D  (significant issues)
  0-44   → F  (not ready)
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_data_readiness(
    df: pd.DataFrame,
    target_col: str | None = None,
) -> dict[str, Any]:
    """Compute a data-readiness report for *df*.

    Args:
        df:         The dataset as a pandas DataFrame.
        target_col: Optional; if set, adds a class-imbalance check
                    when the target column is categorical.

    Returns a dict with keys:
        score           int 0-100
        grade           str "A"|"B"|"C"|"D"|"F"
        status          str "ready"|"needs_attention"|"not_ready"
        summary         str plain-English headline
        components      list[ComponentResult]
        recommendations list[str]
    """
    components: list[dict[str, Any]] = []
    recommendations: list[str] = []

    # 1. Row count -------------------------------------------------------
    comp_rows = _score_row_count(df)
    components.append(comp_rows)
    if comp_rows.get("recommendation"):
        recommendations.append(comp_rows["recommendation"])

    # 2. Missing values --------------------------------------------------
    comp_missing = _score_missing_values(df)
    components.append(comp_missing)
    if comp_missing.get("recommendation"):
        recommendations.append(comp_missing["recommendation"])

    # 3. Duplicate rows --------------------------------------------------
    comp_dupes = _score_duplicates(df)
    components.append(comp_dupes)
    if comp_dupes.get("recommendation"):
        recommendations.append(comp_dupes["recommendation"])

    # 4. Feature diversity -----------------------------------------------
    comp_diversity = _score_feature_diversity(df)
    components.append(comp_diversity)
    if comp_diversity.get("recommendation"):
        recommendations.append(comp_diversity["recommendation"])

    # 5. Data type quality -----------------------------------------------
    comp_types = _score_data_type_quality(df)
    components.append(comp_types)
    if comp_types.get("recommendation"):
        recommendations.append(comp_types["recommendation"])

    # 6. Optional: class imbalance (only when target_col is categorical) -
    if target_col and target_col in df.columns:
        comp_balance = _score_class_balance(df, target_col)
        if comp_balance is not None:
            components.append(comp_balance)
            if comp_balance.get("recommendation"):
                recommendations.append(comp_balance["recommendation"])

    # Overall score = weighted average of the 5 core components ----------
    # (class balance is advisory — not included in weighted score)
    core = components[:5]
    score = sum(c["score"] for c in core)

    # Clamp to valid range (defensive)
    score = max(0, min(100, score))

    grade = _score_to_grade(score)
    status = _score_to_status(score)

    # Plain-English summary
    critical_count = sum(1 for c in components if c["status"] == "critical")
    warning_count = sum(1 for c in components if c["status"] == "warning")
    if critical_count:
        summary = (
            f"Your data has {critical_count} critical issue"
            f"{'s' if critical_count > 1 else ''} that should be fixed before training."
        )
    elif warning_count:
        summary = (
            f"Your data looks mostly good but has {warning_count} minor issue"
            f"{'s' if warning_count > 1 else ''} worth reviewing."
        )
    else:
        summary = "Your data looks great — ready for modeling!"

    return {
        "score": score,
        "grade": grade,
        "status": status,
        "summary": summary,
        "components": components,
        "recommendations": recommendations[:5],  # cap at 5
    }


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------


def _score_row_count(df: pd.DataFrame) -> dict[str, Any]:
    """Score the dataset on number of rows. Max 25 pts."""
    n = len(df)
    if n < 50:
        score = 0
        status = "critical"
        detail = f"{n} rows — needs at least 50 rows to train reliably."
        rec = "Collect more data or use a template dataset to experiment."
    elif n < 200:
        score = 12
        status = "warning"
        detail = f"{n} rows — usable but models may overfit. 200+ rows recommended."
        rec = None
    elif n < 1000:
        score = 20
        status = "good"
        detail = f"{n} rows — good dataset size for most algorithms."
        rec = None
    else:
        score = 25
        status = "good"
        detail = f"{n:,} rows — large dataset; all algorithms will perform well."
        rec = None

    result: dict[str, Any] = {
        "name": "Row Count",
        "score": score,
        "max_score": 25,
        "status": status,
        "detail": detail,
    }
    if rec:
        result["recommendation"] = rec
    return result


def _score_missing_values(df: pd.DataFrame) -> dict[str, Any]:
    """Score based on missing value rate. Max 25 pts."""
    if df.empty:
        return {
            "name": "Missing Values",
            "score": 0,
            "max_score": 25,
            "status": "critical",
            "detail": "Dataset is empty.",
        }

    missing_pct = df.isnull().mean()
    worst_col_pct = float(missing_pct.max()) * 100
    cols_with_any = int((missing_pct > 0).sum())
    total_missing_pct = float(df.isnull().values.mean()) * 100

    if worst_col_pct >= 50:
        score = 5
        status = "critical"
        worst_col = missing_pct.idxmax()
        detail = (
            f"{cols_with_any} column{'s' if cols_with_any != 1 else ''} have missing "
            f"values; '{worst_col}' is {worst_col_pct:.0f}% empty."
        )
        rec = f"Consider dropping '{worst_col}' or filling missing values with median/mode."
    elif worst_col_pct >= 20:
        score = 15
        status = "warning"
        worst_col = missing_pct.idxmax()
        detail = (
            f"{cols_with_any} column{'s' if cols_with_any != 1 else ''} have missing "
            f"values; worst is '{worst_col}' at {worst_col_pct:.0f}%."
        )
        rec = f"Fill missing values in '{worst_col}' with median or mode to improve accuracy."
    elif cols_with_any > 0:
        score = 20
        status = "warning"
        detail = (
            f"{cols_with_any} column{'s' if cols_with_any != 1 else ''} have some "
            f"missing values (overall {total_missing_pct:.1f}% missing)."
        )
        rec = "Fill minor missing values to maximize training data usage."
    else:
        score = 25
        status = "good"
        detail = "No missing values — perfectly complete dataset."
        rec = None

    result: dict[str, Any] = {
        "name": "Missing Values",
        "score": score,
        "max_score": 25,
        "status": status,
        "detail": detail,
    }
    if rec:
        result["recommendation"] = rec
    return result


def _score_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """Score based on duplicate row ratio. Max 20 pts."""
    if df.empty:
        return {
            "name": "Duplicate Rows",
            "score": 20,
            "max_score": 20,
            "status": "good",
            "detail": "Dataset is empty — no duplicates.",
        }

    n_dupes = int(df.duplicated().sum())
    dupe_pct = n_dupes / len(df) * 100

    if dupe_pct >= 20:
        score = 5
        status = "critical"
        detail = f"{n_dupes} duplicate rows ({dupe_pct:.1f}%) — may skew model training."
        rec = "Remove duplicate rows to prevent bias in model training."
    elif dupe_pct >= 5:
        score = 12
        status = "warning"
        detail = f"{n_dupes} duplicate rows ({dupe_pct:.1f}%) detected."
        rec = "Consider removing duplicate rows for cleaner training data."
    elif n_dupes > 0:
        score = 17
        status = "warning"
        detail = f"{n_dupes} duplicate row{'s' if n_dupes != 1 else ''} found ({dupe_pct:.1f}%)."
        rec = None
    else:
        score = 20
        status = "good"
        detail = "No duplicate rows detected."
        rec = None

    result: dict[str, Any] = {
        "name": "Duplicate Rows",
        "score": score,
        "max_score": 20,
        "status": status,
        "detail": detail,
    }
    if rec:
        result["recommendation"] = rec
    return result


def _score_feature_diversity(df: pd.DataFrame) -> dict[str, Any]:
    """Score based on presence of varied column types. Max 15 pts."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    total_cols = len(df.columns)

    n_numeric = len(numeric_cols)
    n_categorical = len(categorical_cols)

    if total_cols < 2:
        score = 0
        status = "critical"
        detail = "Only 1 column — need at least a feature and a target column."
        rec = "Add more columns: features to predict from, and a target column to predict."
    elif n_numeric == 0:
        score = 5
        status = "critical"
        detail = f"No numeric columns ({total_cols} text-only columns). Most ML algorithms need numbers."
        rec = "Ensure your dataset has at least one numeric column for ML training."
    elif n_numeric < 2 and n_categorical == 0:
        score = 8
        status = "warning"
        detail = f"Only {n_numeric} numeric column. More features give the model more to learn from."
        rec = "Add more feature columns or use computed columns to derive new features."
    elif n_numeric >= 2 and n_categorical >= 1:
        score = 15
        status = "good"
        detail = (
            f"Good mix: {n_numeric} numeric + {n_categorical} categorical "
            f"column{'s' if n_categorical != 1 else ''} ({total_cols} total)."
        )
        rec = None
    elif n_numeric >= 3:
        score = 12
        status = "good"
        detail = f"{n_numeric} numeric columns ({total_cols} total) — solid for regression."
        rec = None
    else:
        score = 10
        status = "warning"
        detail = f"{n_numeric} numeric + {n_categorical} text columns. Consider encoding text columns."
        rec = "Use feature engineering to encode categorical columns for better model accuracy."

    result: dict[str, Any] = {
        "name": "Feature Diversity",
        "score": score,
        "max_score": 15,
        "status": status,
        "detail": detail,
    }
    if rec:
        result["recommendation"] = rec
    return result


def _score_data_type_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Score based on data type integrity. Max 15 pts."""
    if df.empty:
        return {
            "name": "Data Type Quality",
            "score": 0,
            "max_score": 15,
            "status": "critical",
            "detail": "Dataset is empty.",
        }

    all_null_cols = [col for col in df.columns if df[col].isnull().all()]
    high_cardinality_text = [
        col
        for col in df.select_dtypes(include="object").columns
        if df[col].nunique() > 0.9 * len(df) and len(df) >= 10
    ]
    # Columns that might be IDs (all unique, non-numeric)
    id_like_cols = [
        col
        for col in df.select_dtypes(include="object").columns
        if df[col].nunique() == len(df)
    ]

    issues = []
    if all_null_cols:
        issues.append(f"{len(all_null_cols)} all-null column(s): {', '.join(all_null_cols[:3])}")
    if high_cardinality_text:
        issues.append(
            f"{len(high_cardinality_text)} high-cardinality text column(s) "
            f"that may not help: {', '.join(high_cardinality_text[:2])}"
        )

    critical_issues = len(all_null_cols)
    warning_issues = len(high_cardinality_text) - len(all_null_cols)

    if critical_issues:
        score = 5
        status = "critical"
        detail = "; ".join(issues) + "."
        rec = f"Drop all-null columns: {', '.join(all_null_cols[:3])}."
    elif len(high_cardinality_text) >= 2:
        score = 8
        status = "warning"
        detail = "; ".join(issues) + "."
        rec = f"Consider dropping high-cardinality columns like '{high_cardinality_text[0]}' (too many unique values to be useful as features)."
    elif warning_issues > 0 or id_like_cols:
        score = 11
        status = "warning"
        detail = (
            "; ".join(issues) if issues
            else f"Possible ID columns detected: {', '.join(id_like_cols[:2])} — unlikely to help prediction."
        )
        rec = None
    else:
        score = 15
        status = "good"
        detail = "Column types look clean and usable for modeling."
        rec = None

    result: dict[str, Any] = {
        "name": "Data Type Quality",
        "score": score,
        "max_score": 15,
        "status": status,
        "detail": detail,
    }
    if rec:
        result["recommendation"] = rec
    return result


def _score_class_balance(
    df: pd.DataFrame,
    target_col: str,
) -> dict[str, Any] | None:
    """Advisory check for classification targets. Not included in weighted score."""
    if target_col not in df.columns:
        return None

    col = df[target_col].dropna()
    # Only check if this looks like a classification target (few unique values)
    if col.nunique() > 20 or col.nunique() < 2:
        return None

    value_counts = col.value_counts(normalize=True)
    majority_pct = float(value_counts.iloc[0]) * 100
    minority_pct = float(value_counts.iloc[-1]) * 100

    if majority_pct >= 95:
        score = 0
        status = "critical"
        detail = f"Severe class imbalance in '{target_col}': {majority_pct:.0f}% / {minority_pct:.0f}%. Model will predict majority class always."
        rec = "Use oversampling (SMOTE) or collect more minority class examples."
    elif majority_pct >= 80:
        score = 0  # advisory only
        status = "warning"
        detail = f"Class imbalance in '{target_col}': {majority_pct:.0f}% / {minority_pct:.0f}%. Model may be biased."
        rec = "Consider balancing classes for better minority-class recall."
    else:
        score = 0  # advisory only
        status = "good"
        detail = f"Class balance in '{target_col}' looks reasonable ({majority_pct:.0f}% / {minority_pct:.0f}%)."
        rec = None

    result: dict[str, Any] = {
        "name": "Class Balance (Advisory)",
        "score": score,  # advisory — not included in weighted total
        "max_score": 0,
        "status": status,
        "detail": detail,
        "advisory": True,
    }
    if rec:
        result["recommendation"] = rec
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score_to_grade(score: int) -> str:
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 45:
        return "D"
    return "F"


def _score_to_status(score: int) -> str:
    if score >= 75:
        return "ready"
    if score >= 45:
        return "needs_attention"
    return "not_ready"
