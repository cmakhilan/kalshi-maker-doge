from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from scipy.stats import qmc


SETTLEMENT_SECONDS = 60
PRICE_QUANTUM = Decimal("0.0000001")


@dataclass(frozen=True)
class ReplaySummary:
    markets_seen: int
    markets_valid: int
    recorded_strikes: int
    reconstructed_strikes: int
    exchange_recorded_settlements: int
    benchmark_reconstructed_settlements: int
    exact_rounded_strike_matches: int
    strike_match_rate: float
    median_strike_error: float
    maximum_strike_error: float
    yes_rate: float


@dataclass(frozen=True)
class ArithmeticAsianState:
    observed_sum: float
    remaining_threshold: float
    future_samples: int
    future_sum_mean: float
    future_sum_variance: float
    lognormal_mean: float
    lognormal_sd: float
    z_score: float
    probability: float


class MonteCarloAsianCDF:
    """Deterministic quasi-Monte Carlo CDFs for the remaining GBM price sum.

    The simulated sum is normalized by spot, so one set of paths can be reused
    for every market.  Sorted samples are cached by integer seconds to expiry
    and volatility-grid point; CDFs between volatility points are interpolated.
    """

    def __init__(
        self,
        paths: int = 8_192,
        seed: int = 17_031,
        sigma_grid_size: int = 64,
        sigma_min: float = 1e-6,
        sigma_max: float = 4e-3,
    ) -> None:
        if paths < 256 or paths & (paths - 1):
            raise ValueError("Monte Carlo paths must be a power of two >= 256")
        if sigma_grid_size < 2 or sigma_min <= 0.0 or sigma_max <= sigma_min:
            raise ValueError("Invalid Monte Carlo volatility grid")
        self.paths = paths
        self.seed = seed
        self.sigma_grid = np.geomspace(sigma_min, sigma_max, sigma_grid_size)
        uniforms = qmc.Sobol(d=SETTLEMENT_SECONDS, scramble=True, seed=seed).random_base2(
            int(np.log2(paths))
        )
        self.standard_normals = np.asarray(
            ndtri(np.clip(uniforms, 1e-12, 1.0 - 1e-12)), dtype=np.float64
        )
        self._cache: dict[int, np.ndarray] = {}

    @staticmethod
    def future_offsets(horizon_seconds: int) -> np.ndarray:
        horizon = int(horizon_seconds)
        grid_offsets = horizon - SETTLEMENT_SECONDS + np.arange(
            SETTLEMENT_SECONDS, dtype=float
        )
        return grid_offsets[grid_offsets > 0.0]

    def _sorted_normalized_sums(self, horizon_seconds: int) -> np.ndarray:
        horizon = int(horizon_seconds)
        if horizon not in self._cache:
            offsets = self.future_offsets(horizon)
            if not len(offsets):
                self._cache[horizon] = np.empty((len(self.sigma_grid), 0))
                return self._cache[horizon]
            gaps = np.diff(offsets, prepend=0.0)
            brownian = np.cumsum(
                self.standard_normals[:, : len(offsets)] * np.sqrt(gaps), axis=1
            )
            samples = np.empty((len(self.sigma_grid), self.paths), dtype=np.float64)
            for index, sigma in enumerate(self.sigma_grid):
                samples[index] = np.sort(
                    np.exp(
                        sigma * brownian - 0.5 * np.square(sigma) * offsets
                    ).sum(axis=1)
                )
            self._cache[horizon] = samples
        return self._cache[horizon]

    def cdf(
        self, horizon_seconds: int, sigma_per_sqrt_second: float, normalized_sum: float
    ) -> float:
        """Return P(sum future prices / current spot <= normalized_sum)."""
        samples = self._sorted_normalized_sums(int(horizon_seconds))
        if samples.shape[1] == 0:
            return float(normalized_sum >= 0.0)
        sigma = float(np.clip(sigma_per_sqrt_second, self.sigma_grid[0], self.sigma_grid[-1]))
        upper = int(np.searchsorted(self.sigma_grid, sigma, side="right"))
        if upper == 0:
            lower = upper = 0
            weight = 0.0
        elif upper == len(self.sigma_grid):
            lower = upper = len(self.sigma_grid) - 1
            weight = 0.0
        else:
            lower = upper - 1
            weight = (
                np.log(sigma) - np.log(self.sigma_grid[lower])
            ) / (
                np.log(self.sigma_grid[upper]) - np.log(self.sigma_grid[lower])
            )

        def empirical_cdf(row: np.ndarray) -> float:
            rank = int(np.searchsorted(row, normalized_sum, side="right"))
            return (rank + 0.5) / (self.paths + 1.0)

        lower_cdf = empirical_cdf(samples[lower])
        if lower == upper:
            return lower_cdf
        upper_cdf = empirical_cdf(samples[upper])
        return float(lower_cdf * (1.0 - weight) + upper_cdf * weight)


def round_price(value: float) -> float:
    return float(Decimal(str(float(value))).quantize(PRICE_QUANTUM, ROUND_HALF_UP))


def _timestamp_ms(series: pd.Series) -> np.ndarray:
    # Pandas 3 may preserve database timestamps as datetime64[us]. Timestamp.value
    # is always nanoseconds, so normalize explicitly before converting to epoch ms.
    return np.fromiter(
        (pd.Timestamp(value).value // 1_000_000 for value in series),
        dtype=np.int64,
        count=len(series),
    )


def _at_or_before(
    tick_times_ms: np.ndarray, target_times_ms: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(tick_times_ms, target_times_ms, side="right") - 1
    safe_indices = np.clip(indices, 0, len(tick_times_ms) - 1)
    gaps_ms = target_times_ms - tick_times_ms[safe_indices]
    return indices, gaps_ms


def reconstruct_settlements(
    markets: pd.DataFrame,
    tick_times_ms: np.ndarray,
    rolling_average: np.ndarray,
    max_tick_gap_ms: int = 2_500,
) -> tuple[pd.DataFrame, ReplaySummary]:
    replay = markets.copy()
    open_ms = _timestamp_ms(replay["open_time"])
    close_ms = _timestamp_ms(replay["close_time"])
    open_indices, open_gaps = _at_or_before(tick_times_ms, open_ms)
    close_indices, close_gaps = _at_or_before(tick_times_ms, close_ms)

    valid_open = (
        (open_indices >= 0)
        & (open_gaps >= 0)
        & (open_gaps <= max_tick_gap_ms)
        & np.isfinite(rolling_average[np.clip(open_indices, 0, len(rolling_average) - 1)])
    )
    valid_close = (
        (close_indices >= 0)
        & (close_gaps >= 0)
        & (close_gaps <= max_tick_gap_ms)
        & np.isfinite(rolling_average[np.clip(close_indices, 0, len(rolling_average) - 1)])
    )
    replay["open_tick_gap_ms"] = open_gaps
    replay["close_tick_gap_ms"] = close_gaps
    replay["reconstructed_strike"] = np.where(
        valid_open, rolling_average[np.clip(open_indices, 0, len(rolling_average) - 1)], np.nan
    )
    replay["benchmark_settlement_average"] = np.where(
        valid_close,
        rolling_average[np.clip(close_indices, 0, len(rolling_average) - 1)],
        np.nan,
    )
    replay["reconstructed_strike_rounded"] = replay["reconstructed_strike"].map(
        lambda value: round_price(value) if np.isfinite(value) else np.nan
    )
    replay["benchmark_settlement_rounded"] = replay[
        "benchmark_settlement_average"
    ].map(
        lambda value: round_price(value) if np.isfinite(value) else np.nan
    )
    replay["strike"] = replay["recorded_strike"].fillna(
        replay["reconstructed_strike_rounded"]
    )
    replay["strike_error"] = (
        replay["recorded_strike"] - replay["reconstructed_strike_rounded"]
    ).abs()
    replay["strike_round_match"] = np.isclose(
        replay["recorded_strike"],
        replay["reconstructed_strike_rounded"],
        rtol=0.0,
        atol=0.5e-7,
        equal_nan=False,
    )
    # A contiguous market's opening target is the exchange-recorded official
    # closing average for the preceding market. Prefer it to a locally rebuilt
    # average, which can differ when individual RTI samples were missed.
    recorded_by_open = (
        replay.loc[replay["recorded_strike"].notna()]
        .drop_duplicates("open_time", keep="last")
        .set_index("open_time")["recorded_strike"]
    )
    replay["next_market_recorded_settlement"] = replay["close_time"].map(
        recorded_by_open
    )
    replay["settlement_average"] = replay[
        "next_market_recorded_settlement"
    ].fillna(replay["benchmark_settlement_rounded"])
    replay["settlement_rounded"] = replay["settlement_average"].map(
        lambda value: round_price(value) if np.isfinite(value) else np.nan
    )
    replay["settlement_source"] = np.where(
        replay["next_market_recorded_settlement"].notna(),
        "next_market_target",
        "benchmark_reconstruction",
    )
    valid = replay["strike"].notna() & replay["settlement_rounded"].notna()
    replay = replay.loc[valid].copy()
    replay["outcome"] = (
        replay["settlement_rounded"] >= replay["strike"]
    ).astype(int)

    recorded = replay["recorded_strike"].notna()
    errors = replay.loc[recorded, "strike_error"].dropna()
    match_count = int(replay.loc[recorded, "strike_round_match"].sum())
    summary = ReplaySummary(
        markets_seen=len(markets),
        markets_valid=len(replay),
        recorded_strikes=int(recorded.sum()),
        reconstructed_strikes=int((~recorded).sum()),
        exchange_recorded_settlements=int(
            replay["next_market_recorded_settlement"].notna().sum()
        ),
        benchmark_reconstructed_settlements=int(
            replay["next_market_recorded_settlement"].isna().sum()
        ),
        exact_rounded_strike_matches=match_count,
        strike_match_rate=match_count / max(int(recorded.sum()), 1),
        median_strike_error=float(errors.median()) if len(errors) else float("nan"),
        maximum_strike_error=float(errors.max()) if len(errors) else float("nan"),
        yes_rate=float(replay["outcome"].mean()),
    )
    return replay, summary


def effective_log_average_variance_seconds(horizon_seconds: float) -> float:
    if horizon_seconds >= SETTLEMENT_SECONDS:
        return horizon_seconds - 2.0 * SETTLEMENT_SECONDS / 3.0
    return horizon_seconds**3 / (3.0 * SETTLEMENT_SECONDS**2)


def arithmetic_asian_state(
    decision_ms: int,
    close_ms: int,
    strike: float,
    spot: float,
    sigma_per_sqrt_second: float,
    tick_times_ms: np.ndarray,
    tick_values: np.ndarray,
) -> ArithmeticAsianState:
    """Moment-match the unknown discrete RTI settlement sum to a lognormal.

    The contract uses the 60 one-second RTI observations in [T-60, T), so the
    eligible timestamps are T-60, ..., T-1. Split that fixed grid according to
    which observations are available at the decision time.
    """
    settlement_grid = close_ms - SETTLEMENT_SECONDS * 1_000 + np.arange(
        SETTLEMENT_SECONDS, dtype=np.int64
    ) * 1_000
    observed_grid = settlement_grid[settlement_grid <= decision_ms]
    future_grid = settlement_grid[settlement_grid > decision_ms]
    observed_sum = (
        float(np.interp(observed_grid, tick_times_ms, tick_values).sum())
        if len(observed_grid)
        else 0.0
    )
    future_offsets = (future_grid - decision_ms).astype(float) / 1_000.0

    future_samples = len(future_offsets)
    remaining_threshold = SETTLEMENT_SECONDS * strike - observed_sum
    if remaining_threshold <= 0.0:
        return ArithmeticAsianState(
            observed_sum,
            remaining_threshold,
            future_samples,
            future_samples * spot,
            0.0,
            float("nan"),
            0.0,
            12.0,
            0.999,
        )
    if future_samples == 0:
        return ArithmeticAsianState(
            observed_sum,
            remaining_threshold,
            0,
            0.0,
            0.0,
            float("nan"),
            0.0,
            -12.0,
            0.001,
        )

    future_mean = future_samples * spot
    # For zero-drift GBM, Cov(S_i,S_j) = S^2(exp(sigma^2 min(u_i,u_j))-1).
    # In sorted offsets, each u_i is the minimum in 2(n-i)-1 matrix cells.
    weights = 2.0 * (future_samples - np.arange(future_samples)) - 1.0
    variance_factor = float(
        np.sum(weights * np.expm1(np.square(sigma_per_sqrt_second) * future_offsets))
    )
    future_variance = float(np.square(spot) * variance_factor)
    lognormal_variance = float(
        np.log1p(future_variance / np.square(future_mean))
    )
    lognormal_sd = float(np.sqrt(max(lognormal_variance, 1e-20)))
    lognormal_mean = float(np.log(future_mean) - 0.5 * lognormal_variance)
    z_score = float(
        (lognormal_mean - np.log(remaining_threshold)) / lognormal_sd
    )
    probability = float(ndtr(z_score))
    return ArithmeticAsianState(
        observed_sum,
        remaining_threshold,
        future_samples,
        future_mean,
        future_variance,
        lognormal_mean,
        lognormal_sd,
        z_score,
        probability,
    )


def _local_variance_rate(
    decision_ms: int,
    tick_times_ms: np.ndarray,
    log_values: np.ndarray,
    lookback_seconds: int,
) -> tuple[float, float, int]:
    end = int(np.searchsorted(tick_times_ms, decision_ms, side="right"))
    start = int(
        np.searchsorted(
            tick_times_ms, decision_ms - lookback_seconds * 1_000, side="left"
        )
    )
    if end - start < 30:
        return float("nan"), 0.0, end - start
    times = tick_times_ms[start:end].astype(np.float64) / 1_000.0
    values = log_values[start:end]
    dt = np.diff(times)
    dx = np.diff(values)
    usable = (dt > 0.0) & (dt <= 10.0) & np.isfinite(dx)
    elapsed = float(times[-1] - times[0])
    coverage = min(elapsed / lookback_seconds, 1.0)
    if usable.sum() < 20 or dt[usable].sum() <= 0:
        return float("nan"), coverage, int(usable.sum())
    variance_rate = float(np.square(dx[usable]).sum() / dt[usable].sum())
    return variance_rate, coverage, int(usable.sum())


def _conditional_log_average_mean(
    close_ms: int,
    decision_ms: int,
    spot_log: float,
    tick_times_ms: np.ndarray,
    log_values: np.ndarray,
) -> float:
    horizon = (close_ms - decision_ms) / 1_000.0
    if horizon >= SETTLEMENT_SECONDS:
        return spot_log
    past_seconds = SETTLEMENT_SECONDS - int(round(horizon))
    if past_seconds <= 0:
        return spot_log
    grid = close_ms - SETTLEMENT_SECONDS * 1_000 + np.arange(past_seconds) * 1_000
    observed = np.interp(grid, tick_times_ms, log_values)
    return float((observed.sum() + horizon * spot_log) / SETTLEMENT_SECONDS)


def build_diffusion_features(
    replay: pd.DataFrame,
    tick_times_ms: np.ndarray,
    tick_values: np.ndarray,
    horizons_seconds: Sequence[int],
    volatility_lookback_seconds: int = 300,
    minimum_coverage: float = 0.80,
    max_spot_gap_ms: int = 2_500,
) -> pd.DataFrame:
    log_values = np.log(tick_values)
    rows: list[dict[str, object]] = []
    for market in replay.itertuples(index=False):
        close_ms = int(pd.Timestamp(market.close_time).value // 1_000_000)
        for horizon in horizons_seconds:
            decision_ms = close_ms - int(horizon) * 1_000
            index = int(np.searchsorted(tick_times_ms, decision_ms, side="right") - 1)
            if index < 0:
                continue
            spot_gap_ms = decision_ms - int(tick_times_ms[index])
            if spot_gap_ms < 0 or spot_gap_ms > max_spot_gap_ms:
                continue
            variance_rate, coverage, return_count = _local_variance_rate(
                decision_ms,
                tick_times_ms,
                log_values,
                volatility_lookback_seconds,
            )
            if not np.isfinite(variance_rate) or coverage < minimum_coverage:
                continue
            sigma = max(float(np.sqrt(variance_rate)), 1e-10)
            mean_log_average = _conditional_log_average_mean(
                close_ms,
                decision_ms,
                float(log_values[index]),
                tick_times_ms,
                log_values,
            )
            variance_seconds = effective_log_average_variance_seconds(float(horizon))
            forecast_sd = sigma * np.sqrt(variance_seconds)
            if forecast_sd <= 0.0:
                continue
            log_strike = float(np.log(market.strike))
            z_score = (mean_log_average - log_strike) / forecast_sd
            baseline_probability = float(ndtr(z_score))
            realized_log_average = float(np.log(market.settlement_average))
            standardized_residual = (
                realized_log_average - mean_log_average
            ) / forecast_sd
            rows.append(
                {
                    "market_ticker": market.market_ticker,
                    "open_time": market.open_time,
                    "close_time": market.close_time,
                    "decision_time": pd.Timestamp(decision_ms, unit="ms", tz="UTC"),
                    "horizon_seconds": int(horizon),
                    "strike": float(market.strike),
                    "settlement_average": float(market.settlement_average),
                    "outcome": int(market.outcome),
                    "spot": float(tick_values[index]),
                    "spot_gap_ms": spot_gap_ms,
                    "volatility_per_sqrt_second": sigma,
                    "volatility_coverage": coverage,
                    "volatility_return_count": return_count,
                    "forecast_log_average": mean_log_average,
                    "forecast_sd": forecast_sd,
                    "z_score": z_score,
                    "standardized_residual": standardized_residual,
                    "p_gaussian": baseline_probability,
                }
            )
    return pd.DataFrame(rows)


def build_arithmetic_asian_features(
    replay: pd.DataFrame,
    tick_times_ms: np.ndarray,
    tick_values: np.ndarray,
    horizons_seconds: Sequence[int],
    volatility_lookback_seconds: int = 300,
    minimum_coverage: float = 0.80,
    max_spot_gap_ms: int = 2_500,
) -> pd.DataFrame:
    log_values = np.log(tick_values)
    rows: list[dict[str, object]] = []
    for market in replay.itertuples(index=False):
        close_ms = int(pd.Timestamp(market.close_time).value // 1_000_000)
        for horizon in horizons_seconds:
            decision_ms = close_ms - int(horizon) * 1_000
            index = int(np.searchsorted(tick_times_ms, decision_ms, side="right") - 1)
            if index < 0:
                continue
            spot_gap_ms = decision_ms - int(tick_times_ms[index])
            if spot_gap_ms < 0 or spot_gap_ms > max_spot_gap_ms:
                continue
            variance_rate, coverage, return_count = _local_variance_rate(
                decision_ms,
                tick_times_ms,
                log_values,
                volatility_lookback_seconds,
            )
            if not np.isfinite(variance_rate) or coverage < minimum_coverage:
                continue
            sigma = max(float(np.sqrt(variance_rate)), 1e-10)
            state = arithmetic_asian_state(
                decision_ms,
                close_ms,
                float(market.strike),
                float(tick_values[index]),
                sigma,
                tick_times_ms,
                tick_values,
            )
            if not np.isfinite(state.z_score):
                continue
            realized_remaining_sum = (
                SETTLEMENT_SECONDS * float(market.settlement_average)
                - state.observed_sum
            )
            if (
                realized_remaining_sum > 0.0
                and state.lognormal_sd > 0.0
                and np.isfinite(state.lognormal_mean)
            ):
                residual = (
                    np.log(realized_remaining_sum) - state.lognormal_mean
                ) / state.lognormal_sd
            else:
                residual = np.sign(int(market.outcome) - 0.5) * 12.0
            rows.append(
                {
                    "market_ticker": market.market_ticker,
                    "open_time": market.open_time,
                    "close_time": market.close_time,
                    "decision_time": pd.Timestamp(decision_ms, unit="ms", tz="UTC"),
                    "horizon_seconds": int(horizon),
                    "strike": float(market.strike),
                    "settlement_average": float(market.settlement_average),
                    "outcome": int(market.outcome),
                    "spot": float(tick_values[index]),
                    "spot_gap_ms": spot_gap_ms,
                    "volatility_per_sqrt_second": sigma,
                    "volatility_coverage": coverage,
                    "volatility_return_count": return_count,
                    "observed_settlement_sum": state.observed_sum,
                    "remaining_threshold": state.remaining_threshold,
                    "future_samples": state.future_samples,
                    "future_sum_mean": state.future_sum_mean,
                    "future_sum_variance": state.future_sum_variance,
                    "forecast_log_average": state.lognormal_mean,
                    "forecast_sd": state.lognormal_sd,
                    "z_score": state.z_score,
                    "standardized_residual": float(residual),
                    # Retain the shared baseline column name for calibration code.
                    "p_gaussian": float(np.clip(state.probability, 0.001, 0.999)),
                }
            )
    return pd.DataFrame(rows)
