import unittest

import pandas as pd

from claims_risk_identifier.features import add_engineered_features, get_feature_columns


class FeatureTests(unittest.TestCase):
    def test_get_feature_columns_excludes_ids_target_and_leakage(self):
        df = pd.DataFrame(
            columns=[
                "claim_id",
                "payer_type",
                "split",
                "is_denied",
                "denial_reason",
                "total_billed",
            ]
        )
        self.assertEqual(get_feature_columns(df), ["payer_type", "total_billed"])

    def test_add_engineered_features_creates_expected_admin_counts(self):
        df = pd.DataFrame(
            [
                {
                    "prior_auth_required": 1,
                    "has_prior_auth": 0,
                    "referral_required": 1,
                    "referral_present": 0,
                    "expected_payment": 100,
                    "total_billed": 250,
                    "days_to_submit": 31,
                    "num_procedures": 6,
                    "num_diagnoses": 2,
                    "missing_documentation_flag": 1,
                    "eligibility_verified": 0,
                    "is_in_network": 0,
                }
            ]
        )

        result = add_engineered_features(df).iloc[0]

        self.assertEqual(result["auth_missing_when_required"], 1)
        self.assertEqual(result["referral_missing_when_required"], 1)
        self.assertEqual(result["late_submission_30d"], 1)
        self.assertEqual(result["complex_claim"], 1)
        self.assertEqual(result["payment_to_billed_ratio"], 0.4)
        self.assertEqual(result["admin_issue_count"], 6)


if __name__ == "__main__":
    unittest.main()
