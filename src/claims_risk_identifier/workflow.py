import pandas as pd

from claims_risk_identifier.explanations import score_current_claims
from claims_risk_identifier.features import (
    add_engineered_features,
    coerce_feature_types,
    get_feature_columns,
    identify_column_types,
    load_data,
    split_history_data,
)
from claims_risk_identifier.models import (
    build_candidate_models,
    build_preprocessor,
    choose_operational_threshold,
    choose_risk_tier_thresholds,
    evaluate_model,
)


def _format_table(df):
    return df.to_string(float_format=lambda value: f"{value:.6f}")


def run_full_workflow(
    history_path,
    current_path,
    output_path,
    openai_api_key=None,
    openai_model="gpt-5.5-medium",
    n_iter=3,
    cv_splits=3,
    random_state=42,
):
    """Run the end-to-end training, model selection, scoring, and export workflow."""
    history_df, current_df = load_data(history_path, current_path)

    feature_cols = get_feature_columns(history_df)
    history_df = coerce_feature_types(history_df, feature_cols)
    current_df = coerce_feature_types(current_df, feature_cols)

    history_df = add_engineered_features(history_df)
    current_df = add_engineered_features(current_df)
    feature_cols = get_feature_columns(history_df)

    X_train, y_train, X_valid, y_valid, X_test, y_test = split_history_data(
        history_df=history_df,
        feature_cols=feature_cols,
    )
    numeric_cols, categorical_cols = identify_column_types(feature_cols)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    models, search_summary = build_candidate_models(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        n_iter=n_iter,
        cv_splits=cv_splits,
        random_state=random_state,
    )

    trained_results = {}
    train_comparison = {}
    validation_comparison = {}
    test_comparison = {}
    model_selection_rows = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        model_valid_threshold = choose_operational_threshold(
            model=model,
            X_valid=X_valid,
            review_fraction=0.25,
        )
        train_metrics, _, _ = evaluate_model(
            model, X_train, y_train, model_valid_threshold, 0.25
        )
        validation_metrics, _, _ = evaluate_model(
            model, X_valid, y_valid, model_valid_threshold, 0.25
        )
        test_metrics, _, _ = evaluate_model(
            model, X_test, y_test, model_valid_threshold, 0.25
        )
        train_capture = train_metrics["denial_capture_rate"]
        validation_capture = validation_metrics["denial_capture_rate"]
        generalization_gap = abs(train_capture - validation_capture)

        trained_results[model_name] = {
            "model": model,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "train_validation_generalization_gap": generalization_gap,
        }
        train_comparison[model_name] = train_metrics
        validation_comparison[model_name] = validation_metrics
        test_comparison[model_name] = test_metrics
        model_selection_rows[model_name] = {
            **validation_metrics,
            "train_denial_capture_rate": train_capture,
            "validation_denial_capture_rate": validation_capture,
            "train_validation_generalization_gap": generalization_gap,
        }

    train_comparison = pd.DataFrame(train_comparison).T
    validation_comparison = pd.DataFrame(validation_comparison).T
    test_comparison = pd.DataFrame(test_comparison).T
    model_selection_comparison = pd.DataFrame(model_selection_rows).T
    model_selection_comparison = model_selection_comparison.sort_values(
        by=["train_validation_generalization_gap",
            "denial_capture_rate",
            "pr_auc",
            "roc_auc",
        ],
        ascending=[True, False, False, False],
    )

    ordered_models = model_selection_comparison.index
    validation_comparison = validation_comparison.loc[ordered_models]
    train_comparison = train_comparison.loc[ordered_models]
    test_comparison = test_comparison.loc[ordered_models]

    best_model_name = model_selection_comparison.index[0]
    best_model = trained_results[best_model_name]["model"]
    threshold = choose_operational_threshold(best_model, X_valid, review_fraction=0.25)
    high_threshold, medium_threshold = choose_risk_tier_thresholds(best_model, X_valid)
    final_test_metrics, _, _ = evaluate_model(
        best_model, X_test, y_test, threshold, review_fraction=0.25
    )

    predictions_df, scored_full_df = score_current_claims(
        model=best_model,
        current_df=current_df,
        feature_cols=feature_cols,
        threshold=threshold,
        high_threshold=high_threshold,
        medium_threshold=medium_threshold,
        use_openai_explanations=False,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        llm_explanation_top_n=10,
    )
    predictions_df.to_csv(output_path, index=False)

    print(f"Best model: {best_model_name}")
    print(f"Operational threshold: {threshold:.6f}")
    print(f"High risk threshold: {high_threshold:.6f}")
    print(f"Medium risk threshold: {medium_threshold:.6f}")
    print("\nModel selection comparison:")
    print(_format_table(model_selection_comparison))
    print("\nTest comparison:")
    print(_format_table(test_comparison))
    print(f"\nPredictions written to: {output_path}")

    return {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "search_summary": search_summary,
        "model_selection_comparison": model_selection_comparison,
        "validation_comparison": validation_comparison,
        "train_comparison": train_comparison,
        "test_comparison": test_comparison,
        "test_metrics": final_test_metrics,
        "threshold": threshold,
        "high_threshold": high_threshold,
        "medium_threshold": medium_threshold,
        "predictions": predictions_df,
        "scored_full": scored_full_df,
    }
