import unittest
from pathlib import Path

import pandas as pd

from claims_risk_identifier.workflow import run_full_workflow


class WorkflowSmokeTests(unittest.TestCase):
    def test_workflow_scores_current_claims(self):
        project_root = Path(__file__).resolve().parents[1]
        output_path = project_root / "outputs" / "test_predictions.csv"

        results = run_full_workflow(
            history_path=project_root / "data" / "claims_history.csv",
            current_path=project_root / "data" / "current_claims.csv",
            output_path=output_path,
            n_iter=1,
            cv_splits=2,
        )

        predictions = pd.read_csv(output_path)
        expected_columns = {
            "claim_id",
            "denial_probability",
            "predicted_denial",
            "risk_tier",
            "top_risk_factors",
            "explanation",
        }

        self.assertTrue(output_path.exists())
        self.assertEqual(len(predictions), 500)
        self.assertTrue(expected_columns.issubset(predictions.columns))
        self.assertIn(
            results["best_model_name"],
            results["model_selection_comparison"].index,
        )


if __name__ == "__main__":
    unittest.main()
