import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def build_preprocessor(numeric_cols, categorical_cols):
    """Build preprocessing pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ]
    )


def top25_capture_score(y_true, y_score):
    """Measure actual denials captured in the top 25% highest-risk claims."""
    y_score = np.asarray(y_score)
    if len(y_score.shape) == 2:
        y_score = y_score[:, 1]

    eval_df = pd.DataFrame({"actual": y_true, "score": y_score}).sort_values(
        "score", ascending=False
    )
    n_review = int(np.ceil(len(eval_df) * 0.25))
    total_denials = eval_df["actual"].sum()
    if total_denials == 0:
        return 0.0
    return eval_df.head(n_review)["actual"].sum() / total_denials


def _candidate_search_spaces(preprocessor, y_train, random_state):
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count

    spaces = {
        "logistic_regression": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
                    ("model", LogisticRegression(
                        max_iter=2000,
                        random_state=random_state,
                    )),
                ]
            ),
            "params": {
                "model__C": loguniform(0.03, 5.0),
                "model__class_weight": [None, "balanced"],
            },
        },

        "random_forest": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
                    ("model", RandomForestClassifier(
                        random_state=random_state,
                        n_jobs=-1,
                    )),
                ]
            ),
            "params": {
                "model__n_estimators": randint(200, 700),
                "model__max_depth": [5, 8, 12, 16, None],
                "model__min_samples_leaf": randint(5, 50),
                "model__min_samples_split": randint(2, 30),
                "model__class_weight": ["balanced", "balanced_subsample", None],
            },
        },

        "hist_gradient_boosting": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
                    ("model", HistGradientBoostingClassifier(
                        random_state=random_state,
                    )),
                ]
            ),
            "params": {
                "model__max_iter": randint(75, 400),
                "model__learning_rate": loguniform(0.02, 0.12),
                "model__max_leaf_nodes": randint(8, 40),
                "model__l2_regularization": loguniform(0.01, 5.0),
                "model__min_samples_leaf": randint(10, 80),
            },
        },

        "xgboost": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
                    ("model", XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=random_state,
                        n_jobs=-1,
                    )),
                ]
            ),
            "params": {
                "model__n_estimators": randint(100, 500),
                "model__max_depth": randint(2, 6),
                "model__learning_rate": loguniform(0.02, 0.12),
                "model__min_child_weight": randint(1, 8),
                "model__subsample": uniform(0.7, 0.3),
                "model__colsample_bytree": uniform(0.7, 0.3),
                "model__reg_lambda": loguniform(0.5, 8.0),
                "model__scale_pos_weight": [1.0, scale_pos_weight],
            },
        },
    }

    if XGBClassifier is not None:
        spaces["xgboost"] = {
            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
                    (
                        "model",
                        XGBClassifier(
                            objective="binary:logistic",
                            eval_metric="logloss",
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "params": {
                "model__n_estimators": randint(100, 350),
                "model__max_depth": randint(2, 6),
                "model__learning_rate": loguniform(0.02, 0.12),
                "model__min_child_weight": randint(1, 8),
                "model__subsample": uniform(0.7, 0.3),
                "model__colsample_bytree": uniform(0.7, 0.3),
                "model__reg_lambda": loguniform(0.5, 8.0),
                "model__scale_pos_weight": [1.0, scale_pos_weight],
            },
        }

    return spaces


def build_candidate_models(
    preprocessor,
    X_train,
    y_train,
    n_iter=3,
    cv_splits=3,
    random_state=42,
):
    """Tune candidate algorithms and return fitted best estimators."""
    scoring = {
        "top25_capture": make_scorer(
            top25_capture_score,
            response_method="predict_proba",
        ),
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
    }
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    best_models = {}
    search_summaries = []

    for model_name, search_config in _candidate_search_spaces(
        preprocessor, y_train, random_state
    ).items():
        print(f"Running RandomizedSearchCV for {model_name}...")
        search = RandomizedSearchCV(
            estimator=search_config["pipeline"],
            param_distributions=search_config["params"],
            n_iter=n_iter,
            scoring=scoring,
            refit="top25_capture",
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            verbose=0,
            return_train_score=True,
        )
        search.fit(X_train, y_train)
        best_models[model_name] = search.best_estimator_
        search_summaries.append(
            {
                "model": model_name,
                "best_cv_top25_capture": search.best_score_,
                "best_params": search.best_params_,
            }
        )
        print(f"  best CV top-25 capture: {search.best_score_:.4f}")

    return best_models, pd.DataFrame(search_summaries)


def calculate_top_quartile_capture(y_true, y_proba, review_fraction=0.25):
    """Calculate denial capture among the highest-risk claims."""
    eval_df = pd.DataFrame({"actual": y_true.values, "proba": y_proba}).sort_values(
        "proba", ascending=False
    )
    n_review = int(np.ceil(len(eval_df) * review_fraction))
    reviewed_df = eval_df.head(n_review)
    total_denials = eval_df["actual"].sum()
    captured_denials = reviewed_df["actual"].sum()
    capture_rate = captured_denials / total_denials if total_denials > 0 else 0
    precision_at_review = captured_denials / n_review if n_review > 0 else 0
    base_denial_rate = eval_df["actual"].mean()
    lift = precision_at_review / base_denial_rate if base_denial_rate > 0 else 0

    return {
        "review_fraction": review_fraction,
        "claims_reviewed": n_review,
        "total_denials": int(total_denials),
        "captured_denials": int(captured_denials),
        "denial_capture_rate": capture_rate,
        "precision_in_reviewed_claims": precision_at_review,
        "base_denial_rate": base_denial_rate,
        "lift_vs_base_rate": lift,
    }


def evaluate_model(model, X, y, threshold=0.5, review_fraction=0.25):
    """Evaluate with standard ML metrics and ranking-focused business metrics."""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y, y_proba),
        "pr_auc": average_precision_score(y, y_proba),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "threshold": threshold,
    }
    metrics.update(calculate_top_quartile_capture(y, y_proba, review_fraction))
    return metrics, y_proba, y_pred


def choose_operational_threshold(model, X_valid, review_fraction=0.25):
    """Use the validation quantile matching the analyst review capacity."""
    valid_proba = model.predict_proba(X_valid)[:, 1]
    return np.quantile(valid_proba, 1 - review_fraction)


def assign_risk_tier(probability, high_threshold, medium_threshold):
    """Convert probability into an analyst-friendly risk tier."""
    if probability >= high_threshold:
        return "High"
    if probability >= medium_threshold:
        return "Medium"
    return "Low"


def choose_risk_tier_thresholds(model, X_valid):
    """High is top 25%, medium is next 25%, low is bottom 50%."""
    valid_proba = model.predict_proba(X_valid)[:, 1]
    return np.quantile(valid_proba, 0.75), np.quantile(valid_proba, 0.50)
