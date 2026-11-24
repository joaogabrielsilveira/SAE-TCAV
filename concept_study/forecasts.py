"""Chronological forecasts of future subgroup performance change.

For a horizon h and a target year T, the forecast is made at origin year T - h. A model is
trained only on rows whose future outcome year is at most the origin (outcomes known at
that time) and tested on rows observed at the origin. Ridge regression with train-only
standardization and an unpenalized intercept, as in the November chronological analysis.
Models: no change (predict 0), history only (current level and last change of the target),
and history plus one block of concept measures. Every model uses the same rows: those
complete on the target, the history and every block.
"""
import numpy as np
import pandas as pd

from concept_study import paths


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list, target: str, alpha: float) -> np.ndarray:
    """Ridge fit on train only (standardized with train mean/SD); returns test predictions."""
    X = train[features].to_numpy(float)
    Z = test[features].to_numpy(float)
    y = train[target].to_numpy(float)
    means, scales = X.mean(axis=0), X.std(axis=0)
    scales = np.where(scales > 1e-14, scales, 1.0)
    Xs = (X - means) / scales
    coef = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(len(features)), Xs.T @ (y - y.mean()))
    return y.mean() + ((Z - means) / scales) @ coef


def chronological_forecasts(panel: pd.DataFrame, target: str, history: list, blocks: dict,
                            horizon: int, alphas=(paths.RIDGE_ALPHA,) + paths.RIDGE_ALPHA_CHECKS) -> pd.DataFrame:
    """Forecast errors for each target year, model and alpha. panel: one horizon_panel output."""
    rows = []
    every_feature = history + [name for block in blocks.values() for name in block if name not in history]
    data = panel.dropna(subset=every_feature + [target])
    for block_name, block in [("history only", [])] + list(blocks.items()):
        features = history + block
        for target_year in range(2011, paths.LAST_OUTCOME_YEAR + 1):
            origin = target_year - horizon
            train = data[data.future_year <= origin]
            test = data[data.year == origin]
            base = {"horizon": horizon, "target": target, "model": block_name, "target_year": target_year,
                    "train_rows": len(train), "test_rows": len(test)}
            if len(train) < paths.MIN_TRAINING_ROWS or not len(test):
                rows.append({**base, "status": "too few training rows" if len(test) else "no test rows"})
                continue
            if train.year.max() >= origin:
                raise AssertionError("Training rows overlap the forecast origin")
            truth = test[target].to_numpy(float)
            for alpha in alphas:
                prediction = fit_predict(train, test, features, target, alpha)
                rows.append({**base, "status": "fitted", "alpha": alpha,
                             "mse": float(np.mean((prediction - truth) ** 2)),
                             "mse_no_change": float(np.mean(truth ** 2)),
                             "concepts": test.concept.nunique()})
    return pd.DataFrame(rows)


def compare_to_history(scores: pd.DataFrame, alpha: float = paths.RIDGE_ALPHA) -> pd.DataFrame:
    """Per target year: error of each block model relative to history only (negative = better).

    Matched on horizon, target and target year, and on variant and group when the table has them.
    """
    fitted = scores[(scores.status == "fitted") & (scores.alpha == alpha)]
    keys = [k for k in ("variant", "group") if k in fitted] + ["horizon", "target", "target_year"]
    history = fitted[fitted.model == "history only"][keys + ["mse"]]
    merged = fitted.merge(history, on=keys, suffixes=("", "_history"), validate="many_to_one")
    merged["mse_change_vs_history_percent"] = 100 * (merged.mse / merged.mse_history - 1)
    merged["mse_change_vs_no_change_percent"] = 100 * (merged.mse / merged.mse_no_change - 1)
    return merged
