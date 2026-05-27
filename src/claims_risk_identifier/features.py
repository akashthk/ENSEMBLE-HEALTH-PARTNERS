import numpy as np
import pandas as pd

from claims_risk_identifier.config import CATEGORICAL_COLS, LEAKAGE_AND_ID_COLS


def load_data(history_path, current_path):
    """Load historical labeled claims and current unlabeled claims."""
    history_df = pd.read_csv(history_path)
    current_df = pd.read_csv(current_path)
    return history_df, current_df


def get_feature_columns(history_df):
    """Select valid model input columns and exclude identifiers/leakage columns."""
    return [col for col in history_df.columns if col not in LEAKAGE_AND_ID_COLS]


def coerce_feature_types(df, feature_cols):
    """Force non-categorical feature columns to numeric."""
    df = df.copy()
    for col in feature_cols:
        if col in df.columns and col not in CATEGORICAL_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_engineered_features(df):
    """Add business-oriented claim-level engineered features."""
    df = df.copy()

    df["auth_missing_when_required"] = (
        (df["prior_auth_required"] == 1) & (df["has_prior_auth"] == 0)
    ).astype(int)
    df["referral_missing_when_required"] = (
        (df["referral_required"] == 1) & (df["referral_present"] == 0)
    ).astype(int)
    df["payment_to_billed_ratio"] = (
        df["expected_payment"] / df["total_billed"].replace(0, np.nan)
    )
    df["billed_minus_expected"] = df["total_billed"] - df["expected_payment"]
    df["late_submission_30d"] = (df["days_to_submit"] >= 30).astype(int)
    df["complex_claim"] = (
        (df["num_procedures"] >= 6) | (df["num_diagnoses"] >= 7)
    ).astype(int)
    df["admin_issue_count"] = (
        df["auth_missing_when_required"]
        + df["referral_missing_when_required"]
        + (df["missing_documentation_flag"] == 1).astype(int)
        + (df["eligibility_verified"] == 0).astype(int)
        + (df["is_in_network"] == 0).astype(int)
        + df["late_submission_30d"]
    )
    return df


def split_history_data(history_df, feature_cols, target_col="is_denied"):
    """Use the assignment-provided train/validation/test split exactly as given."""
    train_df = history_df[history_df["split"] == "train"].copy()
    valid_df = history_df[history_df["split"] == "validation"].copy()
    test_df = history_df[history_df["split"] == "test"].copy()

    return (
        train_df[feature_cols],
        train_df[target_col],
        valid_df[feature_cols],
        valid_df[target_col],
        test_df[feature_cols],
        test_df[target_col],
    )


def identify_column_types(feature_cols):
    """Use explicit column lists instead of relying only on pandas dtypes."""
    categorical_cols = [col for col in feature_cols if col in CATEGORICAL_COLS]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    return numeric_cols, categorical_cols
