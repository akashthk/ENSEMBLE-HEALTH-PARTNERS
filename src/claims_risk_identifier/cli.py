import argparse
from pathlib import Path

from claims_risk_identifier.workflow import run_full_workflow


def parse_args():
    parser = argparse.ArgumentParser(description="Train and score claim denial risk.")
    parser.add_argument(
        "--history-path",
        default="data/claims_history.csv",
        help="Path to historical labeled claims CSV.",
    )
    parser.add_argument(
        "--current-path",
        default="data/current_claims.csv",
        help="Path to current unlabeled claims CSV.",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/predictions_current_claims.csv",
        help="Where to write scored current claims.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=3,
        help="RandomizedSearchCV iterations per candidate model.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=3,
        help="Cross-validation folds.",
    )
    parser.add_argument(
        "--full-search",
        action="store_true",
        help="Use the wider notebook-style hyperparameter search.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    n_iter = 25 if args.full_search else args.n_iter
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_full_workflow(
        history_path=args.history_path,
        current_path=args.current_path,
        output_path=args.output_path,
        n_iter=n_iter,
        cv_splits=args.cv_splits,
    )


if __name__ == "__main__":
    main()
