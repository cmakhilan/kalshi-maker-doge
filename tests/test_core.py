import unittest

import numpy as np
import pandas as pd

from kxdoge_backtest.calibration import empirical_probability
from kxdoge_backtest.core import (
    MonteCarloAsianCDF,
    _timestamp_ms,
    arithmetic_asian_state,
    effective_log_average_variance_seconds,
    round_price,
)
from kxdoge_backtest.maker import (
    EventFairValues,
    MakerScenario,
    _desired_prices,
    ceil_allowed,
    floor_allowed,
    price_float,
    simulate_market,
)


class CoreModelTests(unittest.TestCase):
    def test_round_price_uses_contract_precision_and_half_up(self) -> None:
        self.assertEqual(round_price(0.06979525), 0.0697953)
        self.assertEqual(round_price(0.06979524), 0.0697952)

    def test_variance_is_continuous_at_start_of_settlement_window(self) -> None:
        left = effective_log_average_variance_seconds(60.0)
        right = 60.0**3 / (3.0 * 60.0**2)
        self.assertAlmostEqual(left, 20.0)
        self.assertAlmostEqual(left, right)

    def test_variance_before_average_has_shared_brownian_component(self) -> None:
        self.assertAlmostEqual(effective_log_average_variance_seconds(300.0), 260.0)

    def test_arithmetic_asian_uses_exact_remaining_sample_count(self) -> None:
        close_ms = 100_000
        times = np.arange(0, close_ms + 1, 1_000, dtype=np.int64)
        values = np.full(len(times), 2.0)
        state = arithmetic_asian_state(
            close_ms - 30_000, close_ms, 2.0, 2.0, 0.01, times, values
        )
        self.assertEqual(state.future_samples, 29)
        self.assertAlmostEqual(state.observed_sum, 62.0)
        self.assertAlmostEqual(state.remaining_threshold, 58.0)

    def test_arithmetic_asian_covariance_sum_matches_brute_force(self) -> None:
        close_ms = 100_000
        times = np.arange(0, close_ms + 1, 1_000, dtype=np.int64)
        values = np.full(len(times), 2.0)
        sigma = 0.02
        state = arithmetic_asian_state(
            close_ms - 4_000, close_ms, 2.0, 2.0, sigma, times, values
        )
        offsets = np.arange(1.0, 4.0)
        covariance = 4.0 * np.expm1(
            sigma**2 * np.minimum.outer(offsets, offsets)
        )
        self.assertAlmostEqual(state.future_sum_variance, covariance.sum())

    def test_arithmetic_asian_excludes_expiration_boundary_tick(self) -> None:
        close_ms = 100_000
        times = np.arange(0, close_ms + 1, 1_000, dtype=np.int64)
        values = np.full(len(times), 2.0)
        values[-1] = 100.0
        state = arithmetic_asian_state(
            close_ms, close_ms, 2.0, 100.0, 0.01, times, values
        )
        self.assertEqual(state.future_samples, 0)
        self.assertAlmostEqual(state.observed_sum, 120.0)
        self.assertGreater(state.probability, 0.5)

    def test_monte_carlo_sum_matches_gbm_moments(self) -> None:
        engine = MonteCarloAsianCDF(
            paths=4_096, seed=123, sigma_grid_size=3, sigma_min=0.01, sigma_max=0.04
        )
        sigma = float(engine.sigma_grid[1])
        samples = engine._sorted_normalized_sums(4)[1]
        offsets = np.arange(1.0, 4.0)
        expected_variance = np.expm1(
            sigma**2 * np.minimum.outer(offsets, offsets)
        ).sum()
        self.assertAlmostEqual(float(samples.mean()), 3.0, delta=0.002)
        self.assertAlmostEqual(float(samples.var()), expected_variance, delta=2e-5)

    def test_monte_carlo_cdf_is_monotone_in_threshold(self) -> None:
        engine = MonteCarloAsianCDF(
            paths=1_024, seed=123, sigma_grid_size=3, sigma_min=0.01, sigma_max=0.04
        )
        probabilities = [engine.cdf(4, 0.02, value) for value in (2.9, 3.0, 3.1)]
        self.assertTrue(np.all(np.diff(probabilities) >= 0.0))

    def test_empirical_probability_is_monotone_in_z_score(self) -> None:
        residuals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        probabilities = empirical_probability(residuals, np.array([-1.0, 0.0, 1.0]))
        self.assertTrue(np.all(np.diff(probabilities) >= 0.0))

    def test_direct_empirical_tail_is_monotone_in_required_average(self) -> None:
        samples = np.linspace(0.95, 1.05, 400)
        probabilities = [
            EventFairValues._empirical_tail(samples, threshold)
            for threshold in (0.98, 1.00, 1.02)
        ]
        self.assertTrue(np.all(np.diff(probabilities) <= 0.0))

    def test_timestamp_conversion_is_independent_of_pandas_storage_unit(self) -> None:
        expected = 1_786_733_819_000
        microseconds = pd.Series(
            pd.array(["2026-08-14T18:56:59Z"], dtype="datetime64[us, UTC]")
        )
        self.assertEqual(_timestamp_ms(microseconds)[0], expected)

    def test_tapered_price_grid(self) -> None:
        self.assertEqual(price_float(floor_allowed(0.0964)), 0.096)
        self.assertEqual(price_float(ceil_allowed(0.0964)), 0.097)
        self.assertEqual(price_float(floor_allowed(0.526)), 0.52)
        self.assertEqual(price_float(ceil_allowed(0.526)), 0.53)
        self.assertEqual(price_float(floor_allowed(0.9436)), 0.943)
        self.assertEqual(price_float(ceil_allowed(0.9436)), 0.944)

    def test_inverted_signal_quotes_opposite_side_of_model_discrepancy(self) -> None:
        yes_book = {6_000: 10.0}
        no_book = {3_800: 10.0}  # 62-cent YES ask.
        normal = MakerScenario("normal", edge=0.01, latency_ms=0, fill_model="queue")
        inverted = MakerScenario(
            "inverted",
            edge=0.01,
            latency_ms=0,
            fill_model="queue",
            invert_signal=True,
        )

        self.assertEqual(
            _desired_prices(0.50, 0.0, yes_book, no_book, normal),
            (None, 6_200),
        )
        self.assertEqual(
            _desired_prices(0.50, 0.0, yes_book, no_book, inverted),
            (6_000, None),
        )

    def test_inverted_inventory_skew_discourages_larger_long_position(self) -> None:
        yes_book = {5_100: 10.0}
        scenario = MakerScenario(
            "inverted",
            edge=0.01,
            latency_ms=0,
            fill_model="queue",
            invert_signal=True,
        )

        self.assertEqual(
            _desired_prices(0.50, 0.0, yes_book, {}, scenario),
            (5_100, None),
        )
        self.assertEqual(
            _desired_prices(0.50, 5.0, yes_book, {}, scenario),
            (None, None),
        )

    def test_trade_through_fill_is_capped_by_observed_volume(self) -> None:
        close_ms = 1_000_000
        snapshot_payload = {
            "reset_state": True,
            "book_patch": '{"yes":{"0.4800":"10.00"},"no":{"0.4800":"10.00"}}',
        }
        events = [
            (
                close_ms - 60_000,
                1,
                "snapshot",
                "book_frame",
                None,
                None,
                None,
                None,
                None,
                snapshot_payload,
                "connection",
            ),
            (
                close_ms - 59_000,
                2,
                "trade",
                "trade",
                None,
                None,
                0.20,
                0.47,
                "no",
                None,
                None,
            ),
        ]
        fair_times = np.arange(close_ms - 60_000, close_ms - 14_000, 1_000)
        fair_values = np.full(len(fair_times), 0.50)
        result, fills = simulate_market(
            events,
            "TEST",
            close_ms,
            1,
            fair_times,
            fair_values,
            MakerScenario(
                "test", edge=0.01, latency_ms=0, fill_model="cross", start_seconds=60
            ),
        )
        self.assertEqual(len(fills), 1)
        self.assertAlmostEqual(fills[0].quantity, 0.20)
        self.assertAlmostEqual(result["filled_quantity"], 0.20)


if __name__ == "__main__":
    unittest.main()
