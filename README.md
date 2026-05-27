# Claims Risk Identifier


The project trains denial-risk models on historical claims, selects the best model
using validation-set top-quartile denial capture, scores current claims, assigns
risk tiers, and writes analyst-friendly explanations.

## Project Layout

```text
claims-risk-identifier-project/
  data/
    claims_history.csv
    current_claims.csv
  outputs/
    predictions_current_claims.csv
  src/claims_risk_identifier/
    cli.py
    config.py
    explanations.py
    features.py
    models.py
    workflow.py
  tests/
    test_features.py
    test_metrics.py
    test_workflow_smoke.py
  requirements.txt
  README.md
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run End to End

From the project root:

```powershell
$env:PYTHONPATH = "src"
python -m claims_risk_identifier.cli `
  --history-path data/claims_history.csv `
  --current-path data/current_claims.csv `
  --output-path outputs/predictions_current_claims.csv
```

The default run uses a compact hyperparameter search so it is practical to run
as a script. To run the wider notebook-style search:

```powershell
$env:PYTHONPATH = "src"
python -m claims_risk_identifier.cli --full-search
```

## Tests

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```

## Output Columns

The scoring output includes:

- `claim_id`
- `denial_probability`
- `predicted_denial`
- `risk_tier`
- `top_risk_factors`
- `explanation`

OpenAI-based explanations are supported by the module code, but the CLI defaults
to deterministic manual explanations so the workflow runs without API access.

## Threshold
Operational threshold: 0.276365
High risk threshold: 0.276365
Medium risk threshold: 0.159841
