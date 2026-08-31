from __future__ import annotations

import os
import json
from contextlib import contextmanager
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv


MARKETS_CTE = """
WITH markets AS (
    SELECT
        market_ticker,
        min(captured_at) AS first_seen,
        max(captured_at) AS last_seen,
        max((raw_payload->'market'->>'open_time')::timestamptz) AS open_time,
        max((raw_payload->'market'->>'close_time')::timestamptz) AS close_time,
        max(NULLIF(raw_payload->'market'->'custom_strike'->>'floor_strike', ''))
            AS custom_strike,
        max((raw_payload->'market'->>'floor_strike')::numeric) AS fallback_strike,
        max(NULLIF(raw_payload->'market'->>'strike_type', '')) AS strike_type
    FROM market_quotes
    WHERE series_ticker = 'KXDOGE15M'
    GROUP BY market_ticker
)
"""


@contextmanager
def readonly_connection() -> Iterator[psycopg.Connection]:
    load_dotenv()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")
    connection = psycopg.connect(
        database_url,
        connect_timeout=15,
        options="-c default_transaction_read_only=on -c statement_timeout=300000",
    )
    try:
        yield connection
    finally:
        connection.close()


def load_markets(connection: psycopg.Connection) -> pd.DataFrame:
    query = MARKETS_CTE + """
SELECT
    market_ticker,
    first_seen,
    last_seen,
    open_time,
    close_time,
    custom_strike,
    fallback_strike::double precision,
    strike_type
FROM markets
WHERE open_time IS NOT NULL AND close_time IS NOT NULL
ORDER BY close_time
"""
    with connection.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [column.name for column in cursor.description]
    frame = pd.DataFrame(rows, columns=columns)
    for column in ("first_seen", "last_seen", "open_time", "close_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    frame["recorded_strike"] = pd.to_numeric(
        frame["custom_strike"].fillna(frame["fallback_strike"]), errors="coerce"
    )
    return frame.drop(columns=["custom_strike", "fallback_strike"])


def load_benchmark_ticks(
    connection: psycopg.Connection, start_ms: int, end_ms: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = """
SELECT
    source_time_ms,
    value::double precision,
    avg_60s_value::double precision
FROM benchmark_ticks
WHERE index_id = 'DOGEUSD_RTI'
  AND source_time_ms BETWEEN %s AND %s
ORDER BY source_time_ms
"""
    time_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []
    average_chunks: list[np.ndarray] = []
    with connection.cursor(name="kxdoge_benchmark_stream") as cursor:
        cursor.itersize = 100_000
        cursor.execute(query, (start_ms, end_ms))
        while rows := cursor.fetchmany(100_000):
            time_chunks.append(np.fromiter((row[0] for row in rows), dtype=np.int64))
            value_chunks.append(
                np.fromiter((float(row[1]) for row in rows), dtype=np.float64)
            )
            average_chunks.append(
                np.fromiter(
                    (float(row[2]) if row[2] is not None else np.nan for row in rows),
                    dtype=np.float64,
                )
            )
    if not time_chunks:
        raise RuntimeError("No DOGEUSD_RTI benchmark ticks found in the requested range")
    return (
        np.concatenate(time_chunks),
        np.concatenate(value_chunks),
        np.concatenate(average_chunks),
    )


def load_market_midpoints(
    connection: psycopg.Connection,
    horizons_seconds: Sequence[int],
    tolerance_seconds: int = 30,
) -> pd.DataFrame:
    values_sql = ",".join(["(%s)"] * len(horizons_seconds))
    query = MARKETS_CTE + f"""
, horizons(seconds) AS (VALUES {values_sql})
SELECT
    m.market_ticker,
    h.seconds AS horizon_seconds,
    q.captured_at AS quote_time,
    q.yes_bid::double precision AS yes_bid,
    q.yes_ask::double precision AS yes_ask
FROM markets m
CROSS JOIN horizons h
LEFT JOIN LATERAL (
    SELECT captured_at, yes_bid, yes_ask
    FROM market_quotes mq
    WHERE mq.market_ticker = m.market_ticker
      AND mq.captured_at <= m.close_time - make_interval(secs => h.seconds)
      AND mq.captured_at >= m.close_time
          - make_interval(secs => h.seconds + %s)
    ORDER BY mq.captured_at DESC
    LIMIT 1
) q ON true
ORDER BY m.close_time, h.seconds DESC
"""
    parameters = [*map(int, horizons_seconds), int(tolerance_seconds)]
    with connection.cursor() as cursor:
        cursor.execute(query, parameters)
        rows = cursor.fetchall()
        columns = [column.name for column in cursor.description]
    frame = pd.DataFrame(rows, columns=columns)
    frame["quote_time"] = pd.to_datetime(frame["quote_time"], utc=True)
    frame["market_mid"] = (frame["yes_bid"] + frame["yes_ask"]) / 2.0
    return frame


def load_market_events(
    connection: psycopg.Connection, market_ticker: str
) -> list[tuple]:
    """Load book and trade events for one market in exchange-time order.

    Trades sort before the corresponding book patch at identical exchange
    timestamps so simulated resting orders see the aggressive execution before
    the historical queue is reduced.
    """
    query = """
SELECT *
FROM (
    SELECT
        COALESCE(source_time_ms, (extract(epoch FROM captured_at) * 1000)::bigint)
            AS event_time_ms,
        CASE WHEN reset_state THEN 1 ELSE 3 END AS priority,
        id::text AS event_id,
        'book_frame' AS kind,
        NULL::text AS side,
        NULL::double precision AS price,
        NULL::double precision AS quantity,
        NULL::double precision AS yes_price,
        NULL::text AS taker_side,
        jsonb_build_object(
            'reset_state', reset_state,
            'book_patch', book_patch
        ) AS payload,
        connection_id
    FROM quote_frames
    WHERE market_ticker = %s

    UNION ALL

    SELECT
        COALESCE(
            exchange_time_ms,
            (extract(epoch FROM captured_at) * 1000)::bigint
        ) AS event_time_ms,
        2 AS priority,
        trade_id AS event_id,
        'trade' AS kind,
        NULL::text AS side,
        NULL::double precision AS price,
        count::double precision AS quantity,
        yes_price::double precision AS yes_price,
        taker_side,
        NULL::jsonb AS payload
        , NULL::text AS connection_id
    FROM public_trades
    WHERE market_ticker = %s
) events
ORDER BY event_time_ms, priority, event_id
"""
    with connection.cursor() as cursor:
        cursor.execute(query, (market_ticker, market_ticker))
        return cursor.fetchall()


def load_market_events_batch(
    connection: psycopg.Connection, market_tickers: Sequence[str]
) -> dict[str, list[tuple]]:
    """Load exchange-time-ordered events for many markets in one round trip."""
    tickers = list(market_tickers)
    if not tickers:
        return {}
    query = """
SELECT *
FROM (
    SELECT
        market_ticker,
        COALESCE(source_time_ms, (extract(epoch FROM captured_at) * 1000)::bigint)
            AS event_time_ms,
        CASE WHEN reset_state THEN 1 ELSE 3 END AS priority,
        id::text AS event_id,
        'book_frame' AS kind,
        NULL::text AS side,
        NULL::double precision AS price,
        NULL::double precision AS quantity,
        NULL::double precision AS yes_price,
        NULL::text AS taker_side,
        reset_state,
        book_patch::text,
        connection_id
    FROM quote_frames
    WHERE market_ticker = ANY(%s)

    UNION ALL

    SELECT
        market_ticker,
        COALESCE(
            exchange_time_ms,
            (extract(epoch FROM captured_at) * 1000)::bigint
        ) AS event_time_ms,
        2 AS priority,
        trade_id AS event_id,
        'trade' AS kind,
        NULL::text AS side,
        NULL::double precision AS price,
        count::double precision AS quantity,
        yes_price::double precision AS yes_price,
        taker_side,
        NULL::boolean AS reset_state,
        NULL::text AS book_patch,
        NULL::text AS connection_id
    FROM public_trades
    WHERE market_ticker = ANY(%s)
) events
ORDER BY market_ticker, event_time_ms, priority, event_id
"""
    grouped = {ticker: [] for ticker in tickers}
    with connection.cursor() as cursor:
        cursor.execute(query, (tickers, tickers))
        for row in cursor:
            payload = None
            if row[4] == "book_frame":
                patch_value = row[11]
                patch = (
                    patch_value
                    if isinstance(patch_value, dict)
                    else json.loads(patch_value or "{}")
                )
                payload = {
                    "reset_state": bool(row[10]),
                    "book_updates": tuple(
                        (
                            outcome_side,
                            int(round(float(price) * 10_000)),
                            None if size is None else float(size),
                        )
                        for outcome_side, changes in patch.items()
                        for price, size in changes.items()
                    ),
                }
            grouped[row[0]].append(
                (
                    row[1], row[2], row[3], row[4], row[5], row[6], row[7],
                    row[8], row[9], payload, row[12],
                )
            )
    return grouped
