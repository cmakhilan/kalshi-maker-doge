from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import rankdata, t


EPSILON = 1e-6


@dataclass(frozen=True)
class LinkFit:
    intercept: float
    scale: float
    degrees_of_freedom: float | None = None


def _log_loss(y: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(probability, EPSILON, 1.0 - EPSILON)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def fit_probit(z_score: np.ndarray, outcome: np.ndarray) -> LinkFit:
    z = np.clip(np.asarray(z_score, dtype=float), -12.0, 12.0)
    y = np.asarray(outcome, dtype=float)

    def objective(parameters: np.ndarray) -> float:
        intercept, log_scale = parameters
        return _log_loss(y, ndtr(intercept + np.exp(log_scale) * z))

    result = minimize(objective, np.array([0.0, 0.0]), method="L-BFGS-B")
    return LinkFit(float(result.x[0]), float(np.exp(result.x[1])))


def fit_student_t(z_score: np.ndarray, outcome: np.ndarray) -> LinkFit:
    z = np.clip(np.asarray(z_score, dtype=float), -12.0, 12.0)
    y = np.asarray(outcome, dtype=float)
    best_loss = float("inf")
    best_fit = LinkFit(0.0, 1.0, 8.0)
    for degrees_of_freedom in (3.0, 5.0, 8.0, 12.0, 20.0, 50.0, 200.0):
        def objective(parameters: np.ndarray) -> float:
            intercept, log_scale = parameters
            probability = t.cdf(
                intercept + np.exp(log_scale) * z, df=degrees_of_freedom
            )
            return _log_loss(y, probability)

        result = minimize(objective, np.array([0.0, 0.0]), method="L-BFGS-B")
        loss = float(result.fun)
        if loss < best_loss:
            best_loss = loss
            best_fit = LinkFit(
                float(result.x[0]),
                float(np.exp(result.x[1])),
                degrees_of_freedom,
            )
    return best_fit


def empirical_probability(
    training_residuals: np.ndarray, test_z_scores: np.ndarray
) -> np.ndarray:
    residuals = np.sort(np.asarray(training_residuals, dtype=float))
    thresholds = -np.asarray(test_z_scores, dtype=float)
    ranks = np.searchsorted(residuals, thresholds, side="right")
    return np.clip(
        (len(residuals) - ranks + 0.5) / (len(residuals) + 1.0),
        EPSILON,
        1.0 - EPSILON,
    )


def walk_forward_calibration(
    features: pd.DataFrame,
    minimum_training_days: int = 7,
    minimum_training_rows: int = 400,
) -> pd.DataFrame:
    calibrated_groups: list[pd.DataFrame] = []
    for _, horizon_frame in features.groupby("horizon_seconds", sort=True):
        horizon_frame = horizon_frame.sort_values("decision_time").copy()
        horizon_frame["test_day"] = horizon_frame["decision_time"].dt.floor("D")
        first_day = horizon_frame["test_day"].min()
        day_outputs: list[pd.DataFrame] = []
        for test_day, test in horizon_frame.groupby("test_day", sort=True):
            elapsed_days = int((test_day - first_day).days)
            training = horizon_frame[horizon_frame["close_time"] < test_day]
            if elapsed_days < minimum_training_days or len(training) < minimum_training_rows:
                continue
            z_train = training["z_score"].to_numpy(dtype=float)
            y_train = training["outcome"].to_numpy(dtype=float)
            z_test = test["z_score"].to_numpy(dtype=float)

            probit_fit = fit_probit(z_train, y_train)
            student_fit = fit_student_t(z_train, y_train)
            test = test.copy()
            test["p_probit"] = ndtr(
                probit_fit.intercept + probit_fit.scale * z_test
            )
            test["p_student_t"] = t.cdf(
                student_fit.intercept + student_fit.scale * z_test,
                df=float(student_fit.degrees_of_freedom),
            )
            test["p_empirical"] = empirical_probability(
                training["standardized_residual"].to_numpy(dtype=float), z_test
            )
            test["training_rows"] = len(training)
            test["probit_intercept"] = probit_fit.intercept
            test["probit_scale"] = probit_fit.scale
            test["student_intercept"] = student_fit.intercept
            test["student_scale"] = student_fit.scale
            test["student_df"] = student_fit.degrees_of_freedom
            day_outputs.append(test)
        if day_outputs:
            calibrated_groups.append(pd.concat(day_outputs, ignore_index=True))
    if not calibrated_groups:
        raise RuntimeError("Not enough history for walk-forward calibration")
    return pd.concat(calibrated_groups, ignore_index=True).sort_values(
        ["decision_time", "horizon_seconds"]
    )


def _auc(y: np.ndarray, probability: np.ndarray) -> float:
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(probability)
    return float(
        (ranks[y == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def _expected_calibration_error(
    y: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probability, edges) - 1, bins - 1)
    total = len(y)
    error = 0.0
    for bin_index in range(bins):
        mask = assignments == bin_index
        if mask.any():
            error += mask.mean() * abs(probability[mask].mean() - y[mask].mean())
    return float(error)


def score_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    model_columns = [
        "p_gaussian",
        "p_probit",
        "p_student_t",
        "p_empirical",
        "market_mid",
    ]
    rows: list[dict[str, float | int | str]] = []
    for horizon, group in predictions.groupby("horizon_seconds", sort=False):
        for model in model_columns:
            if model not in group:
                continue
            usable = group[["outcome", model]].dropna()
            if usable.empty:
                continue
            y = usable["outcome"].to_numpy(dtype=int)
            p = np.clip(usable[model].to_numpy(dtype=float), EPSILON, 1.0 - EPSILON)
            rows.append(
                {
                    "horizon_seconds": int(horizon),
                    "model": model.removeprefix("p_"),
                    "observations": len(usable),
                    "yes_rate": float(y.mean()),
                    "log_loss": _log_loss(y, p),
                    "brier_score": float(np.mean(np.square(p - y))),
                    "ece_10": _expected_calibration_error(y, p),
                    "auc": _auc(y, p),
                    "accuracy": float(np.mean((p >= 0.5) == y)),
                }
            )
    return pd.DataFrame(rows).sort_values(["horizon_seconds", "log_loss"])


def calibration_bins(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for horizon, group in predictions.groupby("horizon_seconds", sort=False):
        for model in ("p_gaussian", "p_probit", "p_student_t", "p_empirical", "market_mid"):
            if model not in group:
                continue
            usable = group[["outcome", model]].dropna().copy()
            if usable.empty:
                continue
            usable["bin"] = pd.cut(
                usable[model], np.linspace(0.0, 1.0, bins + 1), include_lowest=True
            )
            for probability_bin, values in usable.groupby("bin", observed=True):
                rows.append(
                    {
                        "horizon_seconds": int(horizon),
                        "model": model.removeprefix("p_"),
                        "probability_bin": str(probability_bin),
                        "observations": len(values),
                        "mean_probability": float(values[model].mean()),
                        "observed_rate": float(values["outcome"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def paired_score_comparisons(
    predictions: pd.DataFrame, bootstrap_samples: int = 2_000, seed: int = 17
) -> pd.DataFrame:
    comparisons = (
        ("p_empirical", "p_gaussian"),
        ("p_student_t", "p_gaussian"),
        ("p_probit", "p_gaussian"),
        ("p_empirical", "market_mid"),
        ("p_gaussian", "market_mid"),
    )
    generator = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for horizon, group in predictions.groupby("horizon_seconds", sort=True):
        for challenger, benchmark in comparisons:
            if challenger not in group or benchmark not in group:
                continue
            usable = group[["outcome", "decision_time", challenger, benchmark]].dropna()
            if usable.empty:
                continue
            y = usable["outcome"].to_numpy(dtype=float)
            challenger_p = np.clip(
                usable[challenger].to_numpy(dtype=float), EPSILON, 1.0 - EPSILON
            )
            benchmark_p = np.clip(
                usable[benchmark].to_numpy(dtype=float), EPSILON, 1.0 - EPSILON
            )
            challenger_loss = -(y * np.log(challenger_p) + (1.0 - y) * np.log1p(-challenger_p))
            benchmark_loss = -(y * np.log(benchmark_p) + (1.0 - y) * np.log1p(-benchmark_p))
            log_improvement = benchmark_loss - challenger_loss
            brier_improvement = np.square(benchmark_p - y) - np.square(challenger_p - y)
            days = usable["decision_time"].dt.floor("D").to_numpy()
            unique_days = np.unique(days)

            def interval(values: np.ndarray) -> tuple[float, float]:
                day_values = [values[days == day] for day in unique_days]
                samples = np.empty(bootstrap_samples, dtype=float)
                for index in range(bootstrap_samples):
                    chosen = generator.integers(0, len(day_values), len(day_values))
                    numerator = sum(float(day_values[item].sum()) for item in chosen)
                    denominator = sum(len(day_values[item]) for item in chosen)
                    samples[index] = numerator / denominator
                lower, upper = np.quantile(samples, [0.025, 0.975])
                return float(lower), float(upper)

            log_lower, log_upper = interval(log_improvement)
            brier_lower, brier_upper = interval(brier_improvement)
            rows.append(
                {
                    "horizon_seconds": int(horizon),
                    "challenger": challenger.removeprefix("p_"),
                    "benchmark": benchmark.removeprefix("p_"),
                    "paired_observations": len(usable),
                    "days": len(unique_days),
                    "log_loss_improvement": float(log_improvement.mean()),
                    "log_loss_ci_low": log_lower,
                    "log_loss_ci_high": log_upper,
                    "brier_improvement": float(brier_improvement.mean()),
                    "brier_ci_low": brier_lower,
                    "brier_ci_high": brier_upper,
                }
            )
    return pd.DataFrame(rows)
