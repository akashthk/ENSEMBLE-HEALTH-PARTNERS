import unittest

import pandas as pd

from claims_risk_identifier.models import (
    assign_risk_tier,
    calculate_top_quartile_capture,
    top25_capture_score,
)


class MetricTests(unittest.TestCase):
    def test_top25_capture_score_uses_highest_scores(self):
        y_true = [1, 0, 1, 0]
        y_score = [0.9, 0.8, 0.2, 0.1]
        self.assertEqual(top25_capture_score(y_true, y_score), 0.5)

    def test_calculate_top_quartile_capture_returns_business_metrics(self):
        y_true = pd.Series([1, 0, 1, 0])
        y_proba = [0.9, 0.8, 0.2, 0.1]

        metrics = calculate_top_quartile_capture(y_true, y_proba, review_fraction=0.25)

        self.assertEqual(metrics["claims_reviewed"], 1)
        self.assertEqual(metrics["total_denials"], 2)
        self.assertEqual(metrics["captured_denials"], 1)
        self.assertEqual(metrics["denial_capture_rate"], 0.5)

    def test_assign_risk_tier(self):
        self.assertEqual(
            assign_risk_tier(0.8, high_threshold=0.7, medium_threshold=0.4),
            "High",
        )
        self.assertEqual(
            assign_risk_tier(0.5, high_threshold=0.7, medium_threshold=0.4),
            "Medium",
        )
        self.assertEqual(
            assign_risk_tier(0.3, high_threshold=0.7, medium_threshold=0.4),
            "Low",
        )


if __name__ == "__main__":
    unittest.main()
