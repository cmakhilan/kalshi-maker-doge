from __future__ import annotations

import heapq
import json
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import ndtri

from .core import (
    MonteCarloAsianCDF,
    SETTLEMENT_SECONDS,
    arithmetic_asian_state,
    effective_log_average_variance_seconds,
)


PRICE_SCALE = 10_000
ALLOWED_PRICES = np.array(
    [*range(10, 1_000, 10), *range(1_000, 9_001, 100), *range(9_010, 10_000, 10)],
    dtype=np.int64,
)


def price_units(price: float) -> int:
    return int(round(float(price) * PRICE_SCALE))


def price_float(units: int) -> float:
    return units / PRICE_SCALE


def floor_allowed(price: float) -> int | None:
    index = int(np.searchsorted(ALLOWED_PRICES, price_units(price), side="right") - 1)
    return int(ALLOWED_PRICES[index]) if index >= 0 else None


def ceil_allowed(price: float) -> int | None:
    index = int(np.searchsorted(ALLOWED_PRICES, price_units(price), side="left"))
    return int(ALLOWED_PRICES[index]) if index < len(ALLOWED_PRICES) else None


@dataclass(frozen=True)
class MakerScenario:
    name: str
    edge: float
    latency_ms: int
    fill_model: str
    order_size: float = 1.0
    max_inventory: float = 10.0
    inventory_skew: float = 0.002
    start_seconds: int = 300
    stop_seconds: int = 15
    maker_fee_multiplier: float = 0.0
    invert_signal: bool = False


@dataclass
class RestingOrder:
    side: str
    price_units: int
    remaining: float
    queue_ahead: float
    placed_ms: int


@dataclass
class Fill:
    scenario: str
    market_ticker: str
    event_time_ms: int
    side: str
    price: float
    quantity: float
    fair_at_fill: float
    fee: float
    fill_reason: str
    inventory_after: float
    markout_1s: float = np.nan
    markout_5s: float = np.nan
    settlement_pnl: float = np.nan


class EventFairValues:
    """Point-in-time structural forecast with expanding empirical residuals."""

    def __init__(
        self,
        tick_times_ms: np.ndarray,
        tick_values: np.ndarray,
        historical_features: pd.DataFrame,
        lookback_seconds: int = 300,
        model_kind: str = "geometric",
        monte_carlo_paths: int = 8_192,
        monte_carlo_seed: int = 17_031,
    ) -> None:
        if model_kind not in {
            "geometric",
            "arithmetic",
            "monte_carlo",
            "empirical",
        }:
            raise ValueError(f"Unknown fair-value model: {model_kind}")
        self.times = tick_times_ms
        self.values = tick_values
        self.log_values = np.log(tick_values)
        self.lookback_seconds = lookback_seconds
        self.model_kind = model_kind
        self.monte_carlo = (
            MonteCarloAsianCDF(paths=monte_carlo_paths, seed=monte_carlo_seed)
            if model_kind == "monte_carlo"
            else None
        )
        dt = np.diff(self.times, prepend=self.times[0]).astype(float) / 1_000.0
        dx = np.diff(self.log_values, prepend=self.log_values[0])
        usable = (dt > 0.0) & (dt <= 10.0) & np.isfinite(dx)
        squared = np.where(usable, np.square(dx), 0.0)
        elapsed = np.where(usable, dt, 0.0)
        self.squared_prefix = np.concatenate([[0.0], np.cumsum(squared)])
        self.elapsed_prefix = np.concatenate([[0.0], np.cumsum(elapsed)])

        features = historical_features.copy()
        features["close_time"] = pd.to_datetime(features["close_time"], utc=True)
        if self.monte_carlo is not None:
            mc_residuals: list[float] = []
            for row in features.itertuples(index=False):
                realized_sum = (
                    SETTLEMENT_SECONDS * float(row.settlement_average)
                    - float(row.observed_settlement_sum)
                )
                if realized_sum <= 0.0 or float(row.spot) <= 0.0:
                    residual = np.sign(int(row.outcome) - 0.5) * 12.0
                else:
                    cdf = self.monte_carlo.cdf(
                        int(row.horizon_seconds),
                        float(row.volatility_per_sqrt_second),
                        realized_sum / float(row.spot),
                    )
                    residual = float(ndtri(np.clip(cdf, 1e-9, 1.0 - 1e-9)))
                mc_residuals.append(residual)
            features["mc_standardized_residual"] = mc_residuals
        self.features = features
        self.canonical_horizons = np.array(
            sorted(features["horizon_seconds"].unique()), dtype=float
        )
        self._residual_cache: dict[tuple[pd.Timestamp, int], np.ndarray] = {}
        self._empirical_average_cache: dict[tuple[pd.Timestamp, int], np.ndarray] = {}
        self._direct_empirical_history = (
            self._build_direct_empirical_history()
            if self.model_kind == "empirical"
            else {}
        )

    def _variance_rate(self, event_ms: int, index: int) -> float:
        start = int(
            np.searchsorted(
                self.times,
                event_ms - self.lookback_seconds * 1_000,
                side="left",
            )
        )
        if index - start < int(self.lookback_seconds * 0.8):
            return float("nan")
        numerator = self.squared_prefix[index + 1] - self.squared_prefix[start + 1]
        denominator = self.elapsed_prefix[index + 1] - self.elapsed_prefix[start + 1]
        if denominator < self.lookback_seconds * 0.8:
            return float("nan")
        continuity_start = int(
            np.searchsorted(self.times, event_ms - 60_000, side="left")
        )
        if index - continuity_start < 58:
            return float("nan")
        return float(numerator / denominator)

    def _mean_log_average(
        self, event_ms: int, close_ms: int, index: int, horizon: float
    ) -> float:
        spot_log = float(self.log_values[index])
        if horizon >= SETTLEMENT_SECONDS:
            return spot_log
        past_seconds = SETTLEMENT_SECONDS - int(np.ceil(max(horizon, 0.0)))
        if past_seconds <= 0:
            return spot_log
        grid = close_ms - SETTLEMENT_SECONDS * 1_000 + np.arange(past_seconds) * 1_000
        observed = np.interp(grid, self.times, self.log_values)
        return float((observed.sum() + horizon * spot_log) / SETTLEMENT_SECONDS)

    def _residuals(self, test_day: pd.Timestamp, horizon: int) -> np.ndarray:
        key = (test_day, horizon)
        if key not in self._residual_cache:
            residual_column = (
                "mc_standardized_residual"
                if self.model_kind == "monte_carlo"
                else "standardized_residual"
            )
            values = self.features.loc[
                (self.features["horizon_seconds"] == horizon)
                & (self.features["close_time"] < test_day),
                residual_column,
            ].dropna()
            self._residual_cache[key] = np.sort(values.to_numpy(dtype=float))
        return self._residual_cache[key]

    @staticmethod
    def _empirical(residuals: np.ndarray, z_score: float) -> float:
        if len(residuals) < 400:
            from scipy.special import ndtr

            return float(ndtr(z_score))
        rank = int(np.searchsorted(residuals, -z_score, side="right"))
        return float((len(residuals) - rank + 0.5) / (len(residuals) + 1.0))

    def _empirical_remaining_averages(
        self, test_day: pd.Timestamp, horizon: int
    ) -> np.ndarray:
        """Prior realized future-average / decision-spot ratios."""
        key = (test_day, horizon)
        if key not in self._empirical_average_cache:
            closes_ms, ratios = self._direct_empirical_history[int(horizon)]
            cutoff_ms = int(test_day.value // 1_000_000)
            self._empirical_average_cache[key] = np.sort(
                ratios[closes_ms < cutoff_ms]
            )
        return self._empirical_average_cache[key]

    def _build_direct_empirical_history(
        self,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Build exact-second empirical targets without future leakage."""
        markets = self.features[
            ["market_ticker", "close_time", "settlement_average"]
        ].drop_duplicates("market_ticker", keep="last")
        history: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for horizon in range(15, SETTLEMENT_SECONDS + 1):
            closes: list[int] = []
            ratios: list[float] = []
            future_samples = horizon - 1
            for market in markets.itertuples(index=False):
                close_ms = int(pd.Timestamp(market.close_time).value // 1_000_000)
                decision_ms = close_ms - horizon * 1_000
                index = int(np.searchsorted(self.times, decision_ms, side="right") - 1)
                if index < 0 or decision_ms - int(self.times[index]) > 2_500:
                    continue
                if not np.isfinite(self._variance_rate(decision_ms, index)):
                    continue
                observed_grid = close_ms - SETTLEMENT_SECONDS * 1_000 + np.arange(
                    SETTLEMENT_SECONDS - future_samples, dtype=np.int64
                ) * 1_000
                observed_sum = float(
                    np.interp(observed_grid, self.times, self.values).sum()
                )
                realized_sum = (
                    SETTLEMENT_SECONDS * float(market.settlement_average)
                    - observed_sum
                )
                spot = float(self.values[index])
                ratio = realized_sum / (future_samples * spot)
                if np.isfinite(ratio) and ratio > 0.0:
                    closes.append(close_ms)
                    ratios.append(ratio)
            history[horizon] = (
                np.asarray(closes, dtype=np.int64),
                np.asarray(ratios, dtype=float),
            )
        return history

    @staticmethod
    def _empirical_tail(samples: np.ndarray, threshold: float) -> float:
        if len(samples) < 400:
            return float("nan")
        rank = int(np.searchsorted(samples, threshold, side="left"))
        return float((len(samples) - rank + 0.5) / (len(samples) + 1.0))

    def _horizon_probability(
        self,
        horizon: float,
        probability_at_horizon,
    ) -> float:
        horizons = self.canonical_horizons
        if horizon <= horizons[0]:
            lower = upper = int(horizons[0])
            weight = 0.0
        elif horizon >= horizons[-1]:
            lower = upper = int(horizons[-1])
            weight = 0.0
        else:
            upper_index = int(np.searchsorted(horizons, horizon, side="right"))
            lower = int(horizons[upper_index - 1])
            upper = int(horizons[upper_index])
            weight = (np.log(horizon) - np.log(lower)) / (
                np.log(upper) - np.log(lower)
            )
        lower_p = probability_at_horizon(lower)
        if not np.isfinite(lower_p) or lower == upper:
            return lower_p
        upper_p = probability_at_horizon(upper)
        if not np.isfinite(upper_p):
            return float("nan")
        lower_p = float(np.clip(lower_p, 0.001, 0.999))
        upper_p = float(np.clip(upper_p, 0.001, 0.999))
        lower_logit = np.log(lower_p / (1.0 - lower_p))
        upper_logit = np.log(upper_p / (1.0 - upper_p))
        return float(
            1.0
            / (1.0 + np.exp(-(lower_logit * (1.0 - weight) + upper_logit * weight)))
        )

    def probability(
        self,
        event_ms: int,
        close_ms: int,
        strike: float,
        test_day: pd.Timestamp,
    ) -> float:
        index = int(np.searchsorted(self.times, event_ms, side="right") - 1)
        if index < 0 or event_ms - int(self.times[index]) > 2_500:
            return float("nan")
        horizon = max((close_ms - event_ms) / 1_000.0, 0.001)
        variance_rate = self._variance_rate(event_ms, index)
        if not np.isfinite(variance_rate):
            return float("nan")
        if self.model_kind in {"arithmetic", "monte_carlo", "empirical"}:
            state = arithmetic_asian_state(
                event_ms,
                close_ms,
                strike,
                float(self.values[index]),
                float(np.sqrt(variance_rate)),
                self.times,
                self.values,
            )
            if self.model_kind == "empirical":
                if state.remaining_threshold <= 0.0:
                    return 0.999
                if state.future_samples == 0:
                    return 0.001
                required_average_ratio = state.remaining_threshold / (
                    state.future_samples * float(self.values[index])
                )
                probability = self._empirical_tail(
                    self._empirical_remaining_averages(
                        test_day, int(round(horizon))
                    ),
                    required_average_ratio,
                )
                return (
                    float(np.clip(probability, 0.001, 0.999))
                    if np.isfinite(probability)
                    else float("nan")
                )
            if (
                self.model_kind == "monte_carlo"
                and state.future_samples > 0
                and state.remaining_threshold > 0.0
            ):
                assert self.monte_carlo is not None
                threshold_cdf = self.monte_carlo.cdf(
                    int(round(horizon)),
                    float(np.sqrt(variance_rate)),
                    state.remaining_threshold / float(self.values[index]),
                )
                z_score = -float(
                    ndtri(np.clip(threshold_cdf, 1e-9, 1.0 - 1e-9))
                )
            else:
                z_score = state.z_score
        else:
            mean = self._mean_log_average(event_ms, close_ms, index, horizon)
            forecast_sd = np.sqrt(
                variance_rate * effective_log_average_variance_seconds(horizon)
            )
            if forecast_sd <= 0.0:
                return float("nan")
            z_score = (mean - np.log(strike)) / forecast_sd

        probability = self._horizon_probability(
            horizon,
            lambda historical_horizon: self._empirical(
                self._residuals(test_day, historical_horizon), z_score
            ),
        )
        return float(np.clip(probability, 0.001, 0.999))


def precompute_market_fair_values(
    model: EventFairValues,
    close_ms: int,
    strike: float,
    test_day: pd.Timestamp,
    start_seconds: int,
    stop_seconds: int,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(
        close_ms - start_seconds * 1_000,
        close_ms - stop_seconds * 1_000 + 1,
        1_000,
        dtype=np.int64,
    )
    probabilities = np.array(
        [model.probability(int(time), close_ms, strike, test_day) for time in times],
        dtype=float,
    )
    return times, probabilities


def _fair_at(time_ms: int, fair_times: np.ndarray, fair_values: np.ndarray) -> float:
    index = int(np.searchsorted(fair_times, time_ms, side="right") - 1)
    if index < 0 or index >= len(fair_values):
        return float("nan")
    return float(fair_values[index])


def _best_book(yes_book: dict[int, float], no_book: dict[int, float]) -> tuple[int | None, int | None]:
    yes_bid = max(yes_book, default=None)
    no_bid = max(no_book, default=None)
    yes_ask = PRICE_SCALE - no_bid if no_bid is not None else None
    return yes_bid, yes_ask


def _desired_prices(
    fair: float,
    inventory: float,
    yes_book: dict[int, float],
    no_book: dict[int, float],
    scenario: MakerScenario,
) -> tuple[int | None, int | None]:
    if not np.isfinite(fair):
        return None, None
    inventory_adjustment = scenario.inventory_skew * inventory
    reservation = float(
        np.clip(
            fair + inventory_adjustment
            if scenario.invert_signal
            else fair - inventory_adjustment,
            0.001,
            0.999,
        )
    )
    best_bid, best_ask = _best_book(yes_book, no_book)

    if scenario.invert_signal:
        # Trade against the model-to-market discrepancy while remaining a maker:
        # buy only when the displayed bid is above model value and sell only
        # when the displayed ask is below model value. The reversed inventory
        # adjustment still discourages adding to an existing position.
        minimum_bid = ceil_allowed(reservation + scenario.edge)
        maximum_ask = floor_allowed(reservation - scenario.edge)
        bid = None
        ask = None
        if (
            inventory + scenario.order_size <= scenario.max_inventory
            and minimum_bid is not None
            and best_bid is not None
            and best_bid >= minimum_bid
        ):
            bid = best_bid
        if (
            inventory - scenario.order_size >= -scenario.max_inventory
            and maximum_ask is not None
            and best_ask is not None
            and best_ask <= maximum_ask
        ):
            ask = best_ask
        if bid is not None and ask is not None and bid >= ask:
            return None, None
        return bid, ask

    maximum_bid = floor_allowed(reservation - scenario.edge)
    minimum_ask = ceil_allowed(reservation + scenario.edge)
    bid = None
    ask = None
    if inventory + scenario.order_size <= scenario.max_inventory and maximum_bid is not None:
        if best_bid is None:
            bid = maximum_bid
        elif best_bid <= maximum_bid:
            bid = best_bid
    if inventory - scenario.order_size >= -scenario.max_inventory and minimum_ask is not None:
        if best_ask is None:
            ask = minimum_ask
        elif best_ask >= minimum_ask:
            ask = best_ask
    if bid is not None and ask is not None and bid >= ask:
        return None, None
    return bid, ask


def simulate_market(
    events: Iterable[tuple],
    market_ticker: str,
    close_ms: int,
    outcome: int,
    fair_times: np.ndarray,
    fair_values: np.ndarray,
    scenario: MakerScenario,
) -> tuple[dict[str, object], list[Fill]]:
    yes_book: dict[int, float] = {}
    no_book: dict[int, float] = {}
    active: dict[str, RestingOrder | None] = {"bid": None, "ask": None}
    pending_desired: dict[str, int | None | object] = {"bid": object(), "ask": object()}
    generation = {"bid": 0, "ask": 0}
    actions: list[tuple[int, int, str, int | None]] = []
    inventory = 0.0
    cash = 0.0
    fees = 0.0
    maximum_inventory = 0.0
    current_book_connection: str | None = None
    quoted = False
    fills: list[Fill] = []

    event_rows = events if isinstance(events, list) else list(events)
    if not event_rows:
        return {
            "scenario": scenario.name,
            "market_ticker": market_ticker,
            "close_ms": close_ms,
            "outcome": outcome,
            "quoted": False,
            "fills": 0,
            "filled_quantity": 0.0,
            "fees": 0.0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "ending_inventory": 0.0,
            "maximum_abs_inventory": 0.0,
        }, []

    clock_start = close_ms - scenario.start_seconds * 1_000
    clock_stop = close_ms - scenario.stop_seconds * 1_000
    clocks = [(int(time), 0, "clock") for time in np.arange(clock_start, clock_stop + 1, 1_000)]
    # Start at the most recent full snapshot available before quoting begins.
    # Earlier events cannot affect the public book after that reset.
    start_index = 0
    for index, row in enumerate(event_rows):
        payload = row[9]
        if row[0] > clock_start:
            break
        if row[3] == "book_frame" and payload and payload.get("reset_state"):
            start_index = index
    event_items = (
        (int(row[0]), int(row[1]), row)
        for row in event_rows[start_index:]
        if row[0] <= close_ms
    )
    merged = heapq.merge(event_items, clocks, key=lambda item: (item[0], item[1]))

    def queue_at(side: str, units: int) -> float:
        if scenario.fill_model == "touch":
            return 0.0
        if scenario.fill_model == "cross":
            return float("inf")
        if side == "bid":
            return float(yes_book.get(units, 0.0))
        return float(no_book.get(PRICE_SCALE - units, 0.0))

    def execute_actions(time_ms: int) -> None:
        nonlocal quoted
        while actions and actions[0][0] <= time_ms:
            _, action_generation, side, desired = heapq.heappop(actions)
            if action_generation != generation[side]:
                continue
            pending_desired[side] = object()
            active[side] = None
            if desired is None:
                continue
            best_bid, best_ask = _best_book(yes_book, no_book)
            if side == "bid" and best_ask is not None and desired >= best_ask:
                continue
            if side == "ask" and best_bid is not None and desired <= best_bid:
                continue
            active[side] = RestingOrder(
                side=side,
                price_units=desired,
                remaining=scenario.order_size,
                queue_ahead=queue_at(side, desired),
                placed_ms=time_ms,
            )
            quoted = True

    def request_quotes(time_ms: int) -> None:
        horizon = (close_ms - time_ms) / 1_000.0
        fair = _fair_at(time_ms, fair_times, fair_values)
        if horizon < scenario.stop_seconds or horizon > scenario.start_seconds:
            desired_bid, desired_ask = None, None
        else:
            desired_bid, desired_ask = _desired_prices(
                fair, inventory, yes_book, no_book, scenario
            )
        for side, desired in (("bid", desired_bid), ("ask", desired_ask)):
            current = active[side].price_units if active[side] is not None else None
            if current == desired and not isinstance(pending_desired[side], int):
                continue
            if pending_desired[side] == desired:
                continue
            generation[side] += 1
            pending_desired[side] = desired
            heapq.heappush(
                actions,
                (time_ms + scenario.latency_ms, generation[side], side, desired),
            )

    def apply_fill(order: RestingOrder, quantity: float, time_ms: int, reason: str) -> None:
        nonlocal inventory, cash, fees, maximum_inventory
        quantity = min(quantity, order.remaining)
        if quantity <= 0:
            return
        price = price_float(order.price_units)
        fee = scenario.maker_fee_multiplier * quantity * price * (1.0 - price)
        if order.side == "bid":
            inventory += quantity
            cash -= price * quantity
        else:
            inventory -= quantity
            cash += price * quantity
        fees += fee
        order.remaining -= quantity
        maximum_inventory = max(maximum_inventory, abs(inventory))
        fills.append(
            Fill(
                scenario=scenario.name,
                market_ticker=market_ticker,
                event_time_ms=time_ms,
                side=order.side,
                price=price,
                quantity=quantity,
                fair_at_fill=_fair_at(time_ms, fair_times, fair_values),
                fee=fee,
                fill_reason=reason,
                inventory_after=inventory,
            )
        )
        if order.remaining <= 1e-9:
            active[order.side] = None

    for time_ms, _, event in merged:
        execute_actions(time_ms)
        if event == "clock":
            request_quotes(time_ms)
            continue
        (
            _, _, _, kind, side, price, quantity, yes_price, taker_side, payload,
            connection_id,
        ) = event
        if kind == "book_frame":
            reset_state = bool(payload.get("reset_state")) if payload else False
            if reset_state:
                yes_book = {}
                no_book = {}
                active = {"bid": None, "ask": None}
                current_book_connection = connection_id
            if connection_id != current_book_connection:
                continue
            updates = payload.get("book_updates") if payload else None
            if updates is None:
                patch_value = payload.get("book_patch") if payload else None
                patch = (
                    patch_value
                    if isinstance(patch_value, dict)
                    else json.loads(patch_value or "{}")
                )
                updates = tuple(
                    (
                        outcome_side,
                        price_units(price),
                        None if size is None else float(size),
                    )
                    for outcome_side, changes in patch.items()
                    for price, size in changes.items()
                )
            for outcome_side, units, size in updates:
                book = yes_book if outcome_side == "yes" else no_book
                if size is None or size <= 1e-9:
                    book.pop(units, None)
                else:
                    book[units] = size
            request_quotes(time_ms)
        elif kind == "trade" and yes_price is not None:
            trade_units = price_units(yes_price)
            order_side = "ask" if taker_side == "yes" else "bid"
            order = active[order_side]
            if order is not None:
                crossed = (
                    order_side == "bid" and trade_units < order.price_units
                ) or (
                    order_side == "ask" and trade_units > order.price_units
                )
                if crossed:
                    # The print beyond our price proves only that this much
                    # aggressive volume remained after consuming the historical
                    # queue. Our counterfactual order could have stopped the
                    # market order before it reached that print, so never grant
                    # more than the observed beyond-price quantity.
                    apply_fill(
                        order,
                        min(order.remaining, float(quantity or 0.0)),
                        time_ms,
                        "trade_through",
                    )
                elif trade_units == order.price_units and scenario.fill_model != "cross":
                    available = float(quantity or 0.0)
                    consumed_queue = min(order.queue_ahead, available)
                    order.queue_ahead -= consumed_queue
                    available -= consumed_queue
                    if available > 0:
                        apply_fill(order, available, time_ms, "queue_trade")
            request_quotes(time_ms)

    execute_actions(close_ms)
    gross_pnl = cash + inventory * outcome
    net_pnl = gross_pnl - fees
    for fill in fills:
        direction = 1.0 if fill.side == "bid" else -1.0
        fair_1s = _fair_at(fill.event_time_ms + 1_000, fair_times, fair_values)
        fair_5s = _fair_at(fill.event_time_ms + 5_000, fair_times, fair_values)
        if np.isfinite(fair_1s):
            fill.markout_1s = direction * (fair_1s - fill.price) * fill.quantity
        if np.isfinite(fair_5s):
            fill.markout_5s = direction * (fair_5s - fill.price) * fill.quantity
        fill.settlement_pnl = direction * (outcome - fill.price) * fill.quantity - fill.fee
    return {
        "scenario": scenario.name,
        "market_ticker": market_ticker,
        "close_ms": close_ms,
        "outcome": outcome,
        "quoted": quoted,
        "fills": len(fills),
        "filled_quantity": float(sum(fill.quantity for fill in fills)),
        "fees": fees,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "ending_inventory": inventory,
        "maximum_abs_inventory": maximum_inventory,
    }, fills


def summarize_maker_results(
    market_results: pd.DataFrame, fills: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario, group in market_results.groupby("scenario", sort=False):
        scenario_fills = fills[fills["scenario"] == scenario] if len(fills) else fills
        ordered = group.sort_values("close_ms")
        cumulative = ordered["net_pnl"].cumsum()
        drawdown = cumulative - cumulative.cummax()
        quantity = float(group["filled_quantity"].sum())
        fill_quantity = scenario_fills["quantity"].sum() if len(scenario_fills) else 0.0
        rows.append(
            {
                "scenario": scenario,
                "markets": len(group),
                "quoted_markets": int(group["quoted"].sum()),
                "markets_with_fills": int((group["fills"] > 0).sum()),
                "fill_events": int(group["fills"].sum()),
                "filled_quantity": quantity,
                "gross_pnl": float(group["gross_pnl"].sum()),
                "fees": float(group["fees"].sum()),
                "net_pnl": float(group["net_pnl"].sum()),
                "pnl_per_contract": float(group["net_pnl"].sum() / quantity)
                if quantity
                else np.nan,
                "profitable_market_rate": float((group["net_pnl"] > 0).mean()),
                "profitable_filled_market_rate": float(
                    (group.loc[group["fills"] > 0, "net_pnl"] > 0).mean()
                ) if (group["fills"] > 0).any() else np.nan,
                "maximum_drawdown": float(drawdown.min()),
                "mean_markout_1s": float(scenario_fills["markout_1s"].sum() / fill_quantity)
                if fill_quantity
                else np.nan,
                "mean_markout_5s": float(scenario_fills["markout_5s"].sum() / fill_quantity)
                if fill_quantity
                else np.nan,
                "maximum_abs_inventory": float(group["maximum_abs_inventory"].max()),
            }
        )
    return pd.DataFrame(rows)


def daily_block_bootstrap(
    market_results: pd.DataFrame,
    samples: int = 10_000,
    seed: int = 71,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = market_results.copy()
    daily["day"] = pd.to_datetime(daily["close_ms"], unit="ms", utc=True).dt.floor("D")
    daily = daily.groupby(["scenario", "day"], as_index=False).agg(
        net_pnl=("net_pnl", "sum"),
        filled_quantity=("filled_quantity", "sum"),
        fills=("fills", "sum"),
    )
    generator = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for scenario, group in daily.groupby("scenario", sort=False):
        pnl = group["net_pnl"].to_numpy(dtype=float)
        quantity = group["filled_quantity"].to_numpy(dtype=float)
        boot_total = np.empty(samples, dtype=float)
        boot_per_contract = np.empty(samples, dtype=float)
        for sample in range(samples):
            indices = generator.integers(0, len(group), len(group))
            total = float(pnl[indices].sum())
            total_quantity = float(quantity[indices].sum())
            boot_total[sample] = total
            boot_per_contract[sample] = (
                total / total_quantity if total_quantity else np.nan
            )
        total_interval = np.quantile(boot_total, [0.025, 0.975])
        contract_interval = np.nanquantile(boot_per_contract, [0.025, 0.975])
        rows.append(
            {
                "scenario": scenario,
                "days": len(group),
                "positive_day_rate": float((pnl > 0).mean()),
                "mean_daily_pnl": float(pnl.mean()),
                "daily_pnl_sd": float(pnl.std(ddof=1)),
                "minimum_daily_pnl": float(pnl.min()),
                "maximum_daily_pnl": float(pnl.max()),
                "bootstrap_total_ci_low": float(total_interval[0]),
                "bootstrap_total_ci_high": float(total_interval[1]),
                "bootstrap_ppc_ci_low": float(contract_interval[0]),
                "bootstrap_ppc_ci_high": float(contract_interval[1]),
            }
        )
    return daily, pd.DataFrame(rows)


def fill_horizon_summary(
    market_results: pd.DataFrame, fills: pd.DataFrame
) -> pd.DataFrame:
    if fills.empty:
        return pd.DataFrame()
    closes = market_results[["scenario", "market_ticker", "close_ms"]]
    enriched = fills.merge(closes, on=["scenario", "market_ticker"], how="left")
    enriched["seconds_to_close"] = (
        enriched["close_ms"] - enriched["event_time_ms"]
    ) / 1_000.0
    enriched["horizon_bucket"] = pd.cut(
        enriched["seconds_to_close"], [0, 15, 30, 60, 120, 300, 600, 900], right=True
    )
    result = enriched.groupby(
        ["scenario", "horizon_bucket"], observed=True, as_index=False
    ).agg(
        filled_quantity=("quantity", "sum"),
        settlement_pnl=("settlement_pnl", "sum"),
        markout_1s=("markout_1s", "sum"),
        markout_5s=("markout_5s", "sum"),
    )
    result["pnl_per_contract"] = (
        result["settlement_pnl"] / result["filled_quantity"]
    )
    result["markout_5s_per_contract"] = (
        result["markout_5s"] / result["filled_quantity"]
    )
    return result


def scenario_dicts(scenarios: Iterable[MakerScenario]) -> list[dict[str, object]]:
    return [asdict(scenario) for scenario in scenarios]
