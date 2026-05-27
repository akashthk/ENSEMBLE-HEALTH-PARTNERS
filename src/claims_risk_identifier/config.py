TARGET_COL = "is_denied"

LEAKAGE_AND_ID_COLS = {
    "claim_id",
    "split",
    "is_denied",
    "denial_reason",
}

CATEGORICAL_COLS = [
    "payer_id",
    "payer_type",
    "visit_type",
    "service_month",
]

ENGINEERED_FEATURE_COLS = [
    "auth_missing_when_required",
    "referral_missing_when_required",
    "payment_to_billed_ratio",
    "billed_minus_expected",
    "late_submission_30d",
    "complex_claim",
    "admin_issue_count",
]
