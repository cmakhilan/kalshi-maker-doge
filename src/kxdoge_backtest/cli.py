from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .calibration import (
    calibration_bins,
    paired_score_comparisons,
    score_predictions,
    walk_forward_calibration,
)
from .core import (
    build_arithmetic_asian_features,
    build_diffusion_features,
    reconstruct_settlements,
)
from .database import (
    load_benchmark_ticks,
    load_market_midpoints,
    load_markets,
    readonly_connection,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest settlement-aware KXDOGE15M probability models"
    )
    parser.add_argument(
        "--horizons",
        default="600,300,120,60,30",
        help="Comma-separated seconds to settlement",
    )
    parser.add_argument("--vol-lookback", type=int, default=300)
    parser.add_argument("--min-training-days", type=int, default=7)
    parser.add_argument("--min-training-rows", type=int, default=400)
    parser.add_argument("--quote-tolerance", type=int, default=30)
    parser.add_argument("--without-market-mid", action="store_true")
    parser.add_argument(
        "--model",
        choices=("geometric", "arithmetic"),
        default="geometric",
        help="Settlement distribution approximation",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _report(
    summary: dict[str, object],
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    output_directory: Path,
) -> str:
    display = metrics.copy()
    for column in ("yes_rate", "log_loss", "brier_score", "ece_10", "auc", "accuracy"):
        display[column] = display[column].map(lambda value: f"{value:.5f}")
    headers = list(display.columns)
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    markdown_rows.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    paired_display = paired[
        [
            "horizon_seconds",
            "challenger",
            "benchmark",
            "paired_observations",
            "log_loss_improvement",
            "log_loss_ci_low",
            "log_loss_ci_high",
        ]
    ].copy()
    for column in ("log_loss_improvement", "log_loss_ci_low", "log_loss_ci_high"):
        paired_display[column] = paired_display[column].map(lambda value: f"{value:.5f}")
    paired_headers = list(paired_display.columns)
    paired_markdown = [
        "| " + " | ".join(paired_headers) + " |",
        "| " + " | ".join(["---"] * len(paired_headers)) + " |",
    ]
    paired_markdown.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in paired_display.itertuples(index=False, name=None)
    )
    lines = [
        "# KXDOGE15M settlement-aware probability backtest",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Settlement replay",
        "",
        f"- Markets discovered: {summary['settlement']['markets_seen']}",
        f"- Markets with valid settlement reconstruction: {summary['settlement']['markets_valid']}",
        f"- Exact exchange-recorded markets used for model scores: {summary['exact_exchange_markets_used']}",
        f"- Recorded targets: {summary['settlement']['recorded_strikes']}",
        f"- Targets reconstructed from the opening 60-second average: {summary['settlement']['reconstructed_strikes']}",
        f"- Settlements sourced from the next exchange-recorded target: {summary['settlement']['exchange_recorded_settlements']}",
        f"- Settlements reconstructed from benchmark ticks: {summary['settlement']['benchmark_reconstructed_settlements']}",
        f"- Recorded/reconstructed seven-decimal target match rate: {summary['settlement']['strike_match_rate']:.4%}",
        f"- Median absolute target discrepancy: {summary['settlement']['median_strike_error']:.10f}",
        f"- Maximum absolute target discrepancy: {summary['settlement']['maximum_strike_error']:.10f}",
        f"- Reconstructed YES rate: {summary['settlement']['yes_rate']:.4%}",
        "",
        "## Walk-forward scores",
        "",
        *markdown_rows,
        "",
        "## Paired score comparisons",
        "",
        "Positive improvement favors the challenger. Intervals are 95% UTC-day block-bootstrap intervals.",
        "",
        *paired_markdown,
        "",
        "Probabilities are evaluated out of sample by UTC day. Calibrators only use markets that closed before the test day. Market midpoint is included as a scoring benchmark, not as an executable P&L simulation.",
        "",
        f"Raw artifacts are in `{output_directory}`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _arguments()
    horizons = sorted(
        {int(value.strip()) for value in args.horizons.split(",") if value.strip()},
        reverse=True,
    )
    if not horizons or min(horizons) <= 0:
        raise ValueError("Horizons must contain positive integer seconds")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_directory = args.output_dir or Path("artifacts") / f"backtest_{timestamp}"
    output_directory.mkdir(parents=True, exist_ok=False)

    print("Loading KXDOGE15M markets and benchmark ticks...")
    with readonly_connection() as connection:
        markets = load_markets(connection)
        start_ms = int(markets["open_time"].min().value // 1_000_000) - max(
            args.vol_lookback, 60
        ) * 1_000
        end_ms = int(markets["close_time"].max().value // 1_000_000) + 2_000
        tick_times, tick_values, rolling_average = load_benchmark_ticks(
            connection, start_ms, end_ms
        )
        midpoints = None
        if not args.without_market_mid:
            print("Loading historical Kalshi midpoints...")
            midpoints = load_market_midpoints(
                connection, horizons, tolerance_seconds=args.quote_tolerance
            )

    print("Reconstructing targets and settlement outcomes...")
    replay, replay_summary = reconstruct_settlements(
        markets, tick_times, rolling_average
    )
    print(
        f"Settlement replay retained {len(replay):,} of {len(markets):,} markets "
        f"({replay_summary.strike_match_rate:.2%} recorded-target match rate)."
    )
    exact_replay = replay[
        replay["recorded_strike"].notna()
        & replay["next_market_recorded_settlement"].notna()
    ].copy()
    print(
        f"Using {len(exact_replay):,} markets with exchange-recorded targets on "
        "both ends for model scoring."
    )
    print(f"Building {args.model} settlement-aware diffusion forecasts...")
    feature_builder = (
        build_arithmetic_asian_features
        if args.model == "arithmetic"
        else build_diffusion_features
    )
    features = feature_builder(
        exact_replay,
        tick_times,
        tick_values,
        horizons,
        volatility_lookback_seconds=args.vol_lookback,
    )
    if features.empty:
        raise RuntimeError(
            "No diffusion features were produced; check benchmark timestamp coverage"
        )
    print(f"Built {len(features):,} point-in-time forecasts.")
    if midpoints is not None:
        features = features.merge(
            midpoints[
                [
                    "market_ticker",
                    "horizon_seconds",
                    "quote_time",
                    "yes_bid",
                    "yes_ask",
                    "market_mid",
                ]
            ],
            on=["market_ticker", "horizon_seconds"],
            how="left",
            validate="one_to_one",
        )

    print("Fitting expanding walk-forward Gaussian, Student-t, and empirical calibrators...")
    predictions = walk_forward_calibration(
        features,
        minimum_training_days=args.min_training_days,
        minimum_training_rows=args.min_training_rows,
    )
    metrics = score_predictions(predictions)
    reliability = calibration_bins(predictions)
    paired = paired_score_comparisons(predictions)

    settlement_columns = [
        "market_ticker",
        "open_time",
        "close_time",
        "recorded_strike",
        "reconstructed_strike_rounded",
        "strike",
        "strike_error",
        "settlement_average",
        "settlement_source",
        "benchmark_settlement_rounded",
        "settlement_rounded",
        "outcome",
        "open_tick_gap_ms",
        "close_tick_gap_ms",
    ]
    replay[settlement_columns].to_csv(
        output_directory / "settlement_replay.csv", index=False
    )
    features.to_csv(output_directory / "diffusion_features.csv", index=False)
    predictions.to_csv(output_directory / "walk_forward_predictions.csv", index=False)
    metrics.to_csv(output_directory / "metrics.csv", index=False)
    reliability.to_csv(output_directory / "calibration_bins.csv", index=False)
    paired.to_csv(output_directory / "paired_comparisons.csv", index=False)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizons_seconds": horizons,
        "volatility_lookback_seconds": args.vol_lookback,
        "model": args.model,
        "minimum_training_days": args.min_training_days,
        "minimum_training_rows": args.min_training_rows,
        "feature_rows": len(features),
        "out_of_sample_rows": len(predictions),
        "exact_exchange_markets_used": len(exact_replay),
        "settlement": asdict(replay_summary),
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n"
    )
    report = _report(summary, metrics, paired, output_directory)
    (output_directory / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
