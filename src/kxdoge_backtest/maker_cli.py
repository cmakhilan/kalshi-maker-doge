from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .database import load_benchmark_ticks, load_market_events_batch, readonly_connection
from .maker import (
    EventFairValues,
    MakerScenario,
    precompute_market_fair_values,
    daily_block_bootstrap,
    fill_horizon_summary,
    scenario_dicts,
    simulate_market,
    summarize_maker_results,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Event-time KXDOGE15M maker backtest")
    parser.add_argument(
        "--probability-artifacts", type=Path, default=Path("artifacts/backtest_steps_1_3")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-markets", type=int)
    parser.add_argument(
        "--fair-model",
        choices=("geometric", "arithmetic", "monte_carlo", "empirical"),
        default="geometric",
        help="Structural probability model used by the maker",
    )
    parser.add_argument("--monte-carlo-paths", type=int, default=8_192)
    parser.add_argument("--monte-carlo-seed", type=int, default=17_031)
    parser.add_argument("--event-batch-size", type=int, default=100)
    parser.add_argument(
        "--scenarios",
        help="Optional comma-separated scenario names to run",
    )
    return parser.parse_args()


def _scenarios() -> list[MakerScenario]:
    scenarios = [
        MakerScenario("cross_edge1c_250ms", edge=0.01, latency_ms=250, fill_model="cross"),
        MakerScenario("queue_edge0p5c_250ms", edge=0.005, latency_ms=250, fill_model="queue"),
        MakerScenario("queue_edge1c_250ms", edge=0.01, latency_ms=250, fill_model="queue"),
        MakerScenario("queue_edge2c_250ms", edge=0.02, latency_ms=250, fill_model="queue"),
        MakerScenario("queue_edge1c_50ms", edge=0.01, latency_ms=50, fill_model="queue"),
        MakerScenario("touch_edge1c_100ms", edge=0.01, latency_ms=100, fill_model="touch"),
        MakerScenario(
            "queue_edge1c_hypothetical_fee",
            edge=0.01,
            latency_ms=250,
            fill_model="queue",
            maker_fee_multiplier=0.0175,
        ),
        MakerScenario(
            "cross_edge1c_last60_250ms",
            edge=0.01,
            latency_ms=250,
            fill_model="cross",
            start_seconds=60,
        ),
        MakerScenario(
            "cross_edge2c_last60_250ms",
            edge=0.02,
            latency_ms=250,
            fill_model="cross",
            start_seconds=60,
        ),
        MakerScenario(
            "cross_edge1c_last60_size10_250ms",
            edge=0.01,
            latency_ms=250,
            fill_model="cross",
            order_size=10.0,
            max_inventory=100.0,
            inventory_skew=0.0002,
            start_seconds=60,
        ),
        MakerScenario(
            "cross_edge2c_last60_size10_250ms",
            edge=0.02,
            latency_ms=250,
            fill_model="cross",
            order_size=10.0,
            max_inventory=100.0,
            inventory_skew=0.0002,
            start_seconds=60,
        ),
        MakerScenario(
            "queue_edge1c_last60_250ms",
            edge=0.01,
            latency_ms=250,
            fill_model="queue",
            start_seconds=60,
        ),
        MakerScenario(
            "queue_edge2c_last60_250ms",
            edge=0.02,
            latency_ms=250,
            fill_model="queue",
            start_seconds=60,
        ),
        MakerScenario(
            "queue_edge1c_last60_size10_250ms",
            edge=0.01,
            latency_ms=250,
            fill_model="queue",
            order_size=10.0,
            max_inventory=100.0,
            inventory_skew=0.0002,
            start_seconds=60,
        ),
        MakerScenario(
            "queue_edge2c_last60_size10_250ms",
            edge=0.02,
            latency_ms=250,
            fill_model="queue",
            order_size=10.0,
            max_inventory=100.0,
            inventory_skew=0.0002,
            start_seconds=60,
        ),
        MakerScenario(
            "queue_edge1c_last60_hypothetical_fee",
            edge=0.01,
            latency_ms=250,
            fill_model="queue",
            start_seconds=60,
            maker_fee_multiplier=0.0175,
        ),
        MakerScenario(
            "cross_edge1c_full900_250ms",
            edge=0.01,
            latency_ms=250,
            fill_model="cross",
            start_seconds=900,
        ),
        MakerScenario(
            "cross_edge2c_full900_250ms",
            edge=0.02,
            latency_ms=250,
            fill_model="cross",
            start_seconds=900,
        ),
        MakerScenario(
            "cross_edge1c_full900_size10_250ms",
            edge=0.01,
            latency_ms=250,
            fill_model="cross",
            order_size=10.0,
            max_inventory=100.0,
            inventory_skew=0.0002,
            start_seconds=900,
        ),
        MakerScenario(
            "cross_edge2c_full900_size10_250ms",
            edge=0.02,
            latency_ms=250,
            fill_model="cross",
            order_size=10.0,
            max_inventory=100.0,
            inventory_skew=0.0002,
            start_seconds=900,
        ),
        MakerScenario(
            "queue_edge1c_full900_250ms",
            edge=0.01,
            latency_ms=250,
            fill_model="queue",
            start_seconds=900,
        ),
        MakerScenario(
            "queue_edge2c_full900_250ms",
            edge=0.02,
            latency_ms=250,
            fill_model="queue",
            start_seconds=900,
        ),
        MakerScenario(
            "queue_edge1c_full900_size10_250ms",
            edge=0.01,
            latency_ms=250,
            fill_model="queue",
            order_size=10.0,
            max_inventory=100.0,
            inventory_skew=0.0002,
            start_seconds=900,
        ),
        MakerScenario(
            "queue_edge2c_full900_size10_250ms",
            edge=0.02,
            latency_ms=250,
            fill_model="queue",
            order_size=10.0,
            max_inventory=100.0,
            inventory_skew=0.0002,
            start_seconds=900,
        ),
    ]
    for start_seconds, stop_seconds in ((300, 120), (120, 60)):
        for fill_model in ("cross", "queue"):
            for edge_cents in (1, 2):
                for size, order_size, max_inventory, inventory_skew in (
                    ("1x", 1.0, 10.0, 0.002),
                    ("10x", 10.0, 100.0, 0.0002),
                ):
                    size_suffix = "" if size == "1x" else "_size10"
                    for mode, invert_signal in (("normal", False), ("inverse", True)):
                        scenarios.append(
                            MakerScenario(
                                f"{fill_model}_{mode}_edge{edge_cents}c_t{start_seconds}_{stop_seconds}{size_suffix}_250ms",
                                edge=edge_cents / 100.0,
                                latency_ms=250,
                                fill_model=fill_model,
                                order_size=order_size,
                                max_inventory=max_inventory,
                                inventory_skew=inventory_skew,
                                start_seconds=start_seconds,
                                stop_seconds=stop_seconds,
                                invert_signal=invert_signal,
                            )
                        )
    return scenarios


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    headers = list(frame.columns)
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    rows.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return rows


def main() -> None:
    args = _arguments()
    source = args.probability_artifacts
    replay = pd.read_csv(source / "settlement_replay.csv", parse_dates=["open_time", "close_time"])
    features = pd.read_csv(source / "diffusion_features.csv", parse_dates=["close_time"])
    predictions = pd.read_csv(
        source / "walk_forward_predictions.csv", parse_dates=["decision_time", "close_time"]
    )
    test_tickers = set(predictions["market_ticker"].unique())
    markets = replay[
        replay["market_ticker"].isin(test_tickers)
        & replay["recorded_strike"].notna()
        & (replay["settlement_source"] == "next_market_target")
    ].sort_values("close_time")
    if args.max_markets:
        markets = markets.tail(args.max_markets)
    if markets.empty:
        raise RuntimeError("No exact-label test markets found in probability artifacts")

    scenarios = _scenarios()
    if args.scenarios:
        requested = {name.strip() for name in args.scenarios.split(",") if name.strip()}
        scenarios = [scenario for scenario in scenarios if scenario.name in requested]
        missing = requested - {scenario.name for scenario in scenarios}
        if missing:
            raise ValueError(f"Unknown scenarios: {sorted(missing)}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or Path("artifacts") / f"maker_backtest_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)

    maximum_start_seconds = max(s.start_seconds for s in scenarios)
    start_ms = int(markets["close_time"].min().value // 1_000_000) - (
        maximum_start_seconds + 301
    ) * 1_000
    end_ms = int(markets["close_time"].max().value // 1_000_000) + 2_000
    market_results: list[dict[str, object]] = []
    fill_rows: list[dict[str, object]] = []
    print(f"Loading benchmark state for {len(markets):,} exact-label test markets...")
    with readonly_connection() as connection:
        tick_times, tick_values, _ = load_benchmark_ticks(connection, start_ms, end_ms)
        fair_model = EventFairValues(
            tick_times,
            tick_values,
            features,
            model_kind=args.fair_model,
            monte_carlo_paths=args.monte_carlo_paths,
            monte_carlo_seed=args.monte_carlo_seed,
        )
        market_rows = list(markets.itertuples(index=False))
        for batch_start in range(0, len(market_rows), args.event_batch_size):
            batch = market_rows[batch_start : batch_start + args.event_batch_size]
            events_by_ticker = load_market_events_batch(
                connection, [market.market_ticker for market in batch]
            )
            for offset, market in enumerate(batch, start=1):
                number = batch_start + offset
                close_ms = int(pd.Timestamp(market.close_time).value // 1_000_000)
                test_day = pd.Timestamp(market.close_time).floor("D")
                fair_times, fair_values = precompute_market_fair_values(
                    fair_model,
                    close_ms,
                    float(market.strike),
                    test_day,
                    maximum_start_seconds,
                    min(s.stop_seconds for s in scenarios),
                )
                events = events_by_ticker.get(market.market_ticker, [])
                for scenario in scenarios:
                    result, fills = simulate_market(
                        events,
                        market.market_ticker,
                        close_ms,
                        int(market.outcome),
                        fair_times,
                        fair_values,
                        scenario,
                    )
                    market_results.append(result)
                    fill_rows.extend(fill.__dict__ for fill in fills)
                if number % 100 == 0 or number == len(markets):
                    print(f"Simulated {number:,}/{len(markets):,} markets")

    market_frame = pd.DataFrame(market_results)
    fill_frame = pd.DataFrame(fill_rows)
    if fill_frame.empty:
        fill_frame = pd.DataFrame(
            columns=[
                "scenario", "market_ticker", "event_time_ms", "side", "price",
                "quantity", "fair_at_fill", "fee", "fill_reason", "inventory_after",
                "markout_1s", "markout_5s", "settlement_pnl",
            ]
        )
    summary = summarize_maker_results(market_frame, fill_frame)
    daily, bootstrap = daily_block_bootstrap(market_frame)
    horizon_summary = fill_horizon_summary(market_frame, fill_frame)
    market_frame.to_csv(output / "market_results.csv", index=False)
    fill_frame.to_csv(output / "fills.csv", index=False)
    summary.to_csv(output / "scenario_summary.csv", index=False)
    daily.to_csv(output / "daily_results.csv", index=False)
    bootstrap.to_csv(output / "bootstrap_intervals.csv", index=False)
    horizon_summary.to_csv(output / "fill_horizon_summary.csv", index=False)
    configuration = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_probability_artifacts": str(source),
        "fair_model": args.fair_model,
        "monte_carlo_paths": args.monte_carlo_paths,
        "monte_carlo_seed": args.monte_carlo_seed,
        "event_batch_size": args.event_batch_size,
        "markets": len(markets),
        "scenarios": scenario_dicts(scenarios),
        "fee_note": "KXDOGE15M reported quadratic fee type and no historical fee changes; base maker fee is zero. The fee scenario applies an unrounded 0.0175*p*(1-p) stress cost.",
    }
    (output / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")

    display = summary.copy()
    display = display.merge(
        bootstrap[
            [
                "scenario",
                "positive_day_rate",
                "bootstrap_total_ci_low",
                "bootstrap_total_ci_high",
                "bootstrap_ppc_ci_low",
                "bootstrap_ppc_ci_high",
            ]
        ],
        on="scenario",
        how="left",
    )
    float_columns = display.select_dtypes(include="number").columns
    for column in float_columns:
        if column not in {"markets", "quoted_markets", "markets_with_fills", "fill_events"}:
            display[column] = display[column].map(lambda value: f"{value:.5f}")
    report_lines = [
        "# KXDOGE15M event-time maker backtest",
        "",
        f"Generated: {configuration['generated_at']}",
        "",
        "The simulator joins the historical best price, uses a 1-second RTI-aligned fair value, maintains visible queue ahead, leaves stale orders active through configured cancel latency, settles residual inventory, and uses exchange-recorded targets on both ends.",
        "",
        *_markdown_table(display),
        "",
        "`cross` fills only on trade-through; `queue` requires historical volume to consume visible queue ahead; `touch` assumes no queue and is intentionally optimistic.",
        "",
        "This remains a counterfactual replay: our orders could have changed other participants' behavior, and private order acknowledgements are unavailable.",
    ]
    report = "\n".join(report_lines) + "\n"
    (output / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
