"""Output fidelity of one predictor: how well a frozen linear readout of concept scores reproduces its teacher probabilities.

The teacher is the predictor's own death probability, given as two columns (class 0, class 1). A readout is a ridge
regression of the teacher log-odds on standardised input columns, fitted on reference-year fit rows only. Its
probability is compared with the teacher probability on held-out rows. Labels (deaths) never enter these functions.
"""
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


# ---------------------------------------------------------------- teacher scale

def teacher_log_odds(teacher_probabilities: np.ndarray) -> np.ndarray:
    """Log-odds log(p1 / p0) of the death class from (rows, 2) teacher probabilities."""
    probabilities = np.asarray(teacher_probabilities, dtype=np.float64)
    return np.log(probabilities[:, 1]) - np.log(probabilities[:, 0])


def _logit(probability: np.ndarray) -> np.ndarray:
    return np.log(probability) - np.log1p(-probability)


# ---------------------------------------------------------------- readout fit and prediction

def patient_folds(patient_ids: np.ndarray, folds: int) -> list:
    """(train rows, test rows) index pairs of a patient-grouped split: all rows of a patient share one test fold."""
    splitter = GroupKFold(n_splits=folds)
    return list(splitter.split(patient_ids, groups=patient_ids))


def _ridge_fit(features: np.ndarray, target: np.ndarray, alpha: float) -> dict:
    """Ridge fit on columns standardised with their own mean and SD; the intercept is not penalised."""
    column_mean = features.mean(axis=0)
    column_scale = features.std(axis=0)
    if np.any(column_scale == 0):
        raise ValueError("a readout column has zero spread on the fit rows")
    model = Ridge(alpha=alpha).fit((features - column_mean) / column_scale, target)
    return {"column_mean": column_mean, "column_scale": column_scale,
            "coefficient": model.coef_, "intercept": float(model.intercept_), "alpha": alpha}


def _cross_validated_mse(features: np.ndarray, target: np.ndarray, teacher_one: np.ndarray,
                         splits: list, alpha: float) -> float:
    """Probability MSE against the teacher on held-out patients, pooled over folds. Each fold is standardised on its training rows."""
    squared_errors = []
    for train, test in splits:
        readout = _ridge_fit(features[train], target[train], alpha)
        squared_errors.append((predict_probability(readout, features[test]) - teacher_one[test]) ** 2)
    return float(np.mean(np.concatenate(squared_errors)))


def fit_readout(features: np.ndarray, teacher_probabilities: np.ndarray, patient_ids: np.ndarray,
                alphas: np.ndarray, folds: int) -> dict:
    """Ridge readout of the teacher log-odds. The strength is the grid value with the lowest patient-grouped
    cross-validated probability MSE. Returns the frozen readout with its cross-validation curve."""
    features = np.asarray(features, dtype=np.float64)
    target = teacher_log_odds(teacher_probabilities)
    teacher_one = np.asarray(teacher_probabilities, dtype=np.float64)[:, 1]
    splits = patient_folds(patient_ids, folds)
    curve = np.array([_cross_validated_mse(features, target, teacher_one, splits, alpha) for alpha in alphas])
    best = int(np.argmin(curve))
    readout = _ridge_fit(features, target, alphas[best])
    readout.update({"alphas": np.asarray(alphas), "cv_mse": curve,
                    "alpha_on_grid_edge": best in (0, len(alphas) - 1)})
    return readout


def predict_probability(readout: dict, features: np.ndarray) -> np.ndarray:
    """Predicted death probability (rows,) of a frozen readout: the sigmoid of its linear log-odds."""
    standardised = (np.asarray(features, dtype=np.float64) - readout["column_mean"]) / readout["column_scale"]
    return expit(standardised @ readout["coefficient"] + readout["intercept"])


# ---------------------------------------------------------------- errors and rank agreement under the patient bootstrap

def _weighted_mean(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    return weights @ values / weights.sum(axis=1)


def bootstrap_errors(predicted: np.ndarray, teacher_probabilities: np.ndarray, weights: np.ndarray) -> dict:
    """Weighted probability MSE, probability MAE and log-odds MSE for every replicate.

    weights: (replicates, rows) from concept_study.bootstrap; its row 0 is all ones and gives the point estimate.
    """
    weights = np.asarray(weights, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(teacher_probabilities, dtype=np.float64)[:, 1]
    difference = predicted - truth
    log_odds_difference = _logit(predicted) - teacher_log_odds(teacher_probabilities)
    return {
        "probability_mse": _weighted_mean(weights, difference ** 2),
        "probability_mae": _weighted_mean(weights, np.abs(difference)),
        "log_odds_mse": _weighted_mean(weights, log_odds_difference ** 2),
    }


def bootstrap_spearman(predicted: np.ndarray, teacher_probabilities: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Spearman rank correlation of predicted and teacher probability for every replicate.
    A row with weight w enters w times, as in the patient bootstrap."""
    truth = np.asarray(teacher_probabilities, dtype=np.float64)[:, 1]
    predicted = np.asarray(predicted, dtype=np.float64)
    counts = np.asarray(weights).astype(int)
    return np.array([spearmanr(np.repeat(predicted, count), np.repeat(truth, count))[0] for count in counts])


# ---------------------------------------------------------------- agreement at fixed cuts

def stratum_agreement(teacher_probabilities: np.ndarray, predicted: np.ndarray, cuts: list) -> pd.DataFrame:
    """Rows above each cut by teacher, by readout and by both, with the share of each side that the other side also flags.
    A cut is flagged when the probability is strictly above it. Shares are NaN when the denominator has no rows."""
    truth = np.asarray(teacher_probabilities, dtype=np.float64)[:, 1]
    predicted = np.asarray(predicted, dtype=np.float64)
    rows = []
    for cut in cuts:
        teacher_above = truth > cut
        readout_above = predicted > cut
        rows.append({"cut": cut, "rows": len(truth), "teacher_above": int(teacher_above.sum()),
                     "readout_above": int(readout_above.sum()), "both_above": int((teacher_above & readout_above).sum())})
    table = pd.DataFrame(rows)
    table["share_of_teacher_above_also_readout_above"] = table.both_above / table.teacher_above.where(table.teacher_above > 0)
    table["share_of_readout_above_also_teacher_above"] = table.both_above / table.readout_above.where(table.readout_above > 0)
    return table
