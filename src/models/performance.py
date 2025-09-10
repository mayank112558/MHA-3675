# src/my_krml_25506751/models/performance.py
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

__all__ = [
    "print_regressor_scores",
    "print_more_regressor_scores",
    "feature_importance",
    "cross_validate_model",
]

def print_regressor_scores(y_preds, y_actuals, set_name=None):
    print(f"RMSE {set_name}: {rmse(y_actuals, y_preds)}")
    print(f"MAE {set_name}: {mae(y_actuals, y_preds)}")

def print_more_regressor_scores(y_preds: np.ndarray, y_actuals: np.ndarray, set_name: str | None = None):
    print(f"R2 Score {set_name}: {r2_score(y_actuals, y_preds)}")
    print(f"MSE {set_name}: {mean_squared_error(y_actuals, y_preds)}")
    print(f"Explained Variance {set_name}: {explained_variance_score(y_actuals, y_preds)}")

def feature_importance(model: object, feature_names: list) -> "pd.DataFrame":
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)
    raise ValueError("Model does not have feature_importances_ attribute.")

def cross_validate_model(model: object, X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
    scorer = make_scorer(rmse, greater_is_better=False)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
    return {"mean_rmse": -scores.mean(), "std_rmse": scores.std()}
