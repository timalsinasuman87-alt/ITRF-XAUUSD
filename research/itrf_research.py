

"""
ITRF-XAUUSD Research Engine v0.2

Purpose:
    Analyze real historical XAU/USD OHLCV data.

Important:
    This engine does NOT claim to know the exact number of buyers
    and sellers because standard OHLCV data does not contain
    trade-by-trade bid/ask information.

    "Delta" and buy/sell pressure are therefore volume-based proxies.

Input:
    data/XAUUSD.csv

Required columns:
    time, open, high, low, close, volume

Output:
    database/itrf_research.db
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


# MAIN
# ============================================================# ============================================================
# PROJECT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "XAUUSD.csv"

DATABASE_FILE = (
    PROJECT_ROOT
    / "database"
    / "itrf_research.db"
)

TIMEFRAME = "15m"

ATR_LENGTH = 14

VOLUME_LOOKBACK = 50

LIQUIDITY_LOOKBACK = 20

FORWARD_BARS = 32

RISK_ATR = 1.5

DEFAULT_OOS_START = "2025-07-28 17:00:00"


# ============================================================
# DATA LOADING
# ============================================================

def resolve_oos_split(df, requested_start, analysis_name):
    """Return a valid chronological split or stop an invalid OOS report safely."""
    split_time = pd.Timestamp(requested_start)
    first_time = df["timestamp"].min()
    last_time = df["timestamp"].max()
    if not first_time < split_time <= last_time:
        print(
            f"{analysis_name} skipped: OOS start {split_time} is outside the available "
            f"range {first_time} to {last_time}."
        )
        print("Choose a chronological --oos-start inside this dataset before interpreting OOS results.")
        return None
    return split_time

def load_market_data(data_file=DATA_FILE):

    data_file = Path(data_file)

    if not data_file.exists():

        raise FileNotFoundError(
            f"Missing market data: {data_file}"
        )

    df = pd.read_csv(data_file)

    df.columns = [
        column.strip().lower()
        for column in df.columns
    ]

    required = {
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = required - set(df.columns)

    if missing:

        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df = df[
        [
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].copy()

    # Normalize every source to timezone-naive UTC. Existing naive XAUUSD
    # timestamps retain their wall-clock values; timezone-aware futures data
    # becomes directly comparable with the frozen OOS boundary.
    df["time"] = (
        pd.to_datetime(df["time"], errors="coerce", utc=True)
        .dt.tz_convert(None)
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df
        .dropna()
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    if df.empty:

        raise ValueError(
            "No valid market data remains."
        )

    validate_market_data(df)

    return df


def validate_market_data(df):
    """Reject OHLCV data that cannot support defensible historical research."""
    if not df["time"].is_monotonic_increasing or df["time"].duplicated().any():
        raise ValueError("Market data must have unique timestamps in chronological order.")
    invalid_ohlc = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        | (df["high"] < df["low"])
    )
    if invalid_ohlc.any():
        raise ValueError(f"Invalid OHLC relationships in {int(invalid_ohlc.sum())} row(s).")
    if (df["volume"] < 0).any():
        raise ValueError("Volume must not be negative.")


def add_frozen_order_flow_score(data):
    """Apply the canonical v0.5 Score 5-7 definition without tuning."""
    scored = data.copy()
    scored["component_delta"] = (
        ((scored["direction"] == "LONG") & (scored["delta_zscore"] >= 1.0))
        | ((scored["direction"] == "SHORT") & (scored["delta_zscore"] <= -1.0))
    ).astype(int)
    scored["component_delta_change"] = (
        ((scored["direction"] == "LONG") & (scored["delta_change"] > 0))
        | ((scored["direction"] == "SHORT") & (scored["delta_change"] < 0))
    ).astype(int)
    scored["component_momentum"] = (
        ((scored["direction"] == "LONG") & (scored["momentum_atr"] > 0))
        | ((scored["direction"] == "SHORT") & (scored["momentum_atr"] < 0))
    ).astype(int)
    scored["component_efficiency"] = (scored["candle_efficiency"] >= 0.60).astype(int)
    scored["component_sweep"] = (
        ((scored["direction"] == "LONG") & (scored["bullish_sweep"] == 1))
        | ((scored["direction"] == "SHORT") & (scored["bearish_sweep"] == 1))
    ).astype(int)
    scored["component_volume"] = (
        (scored["relative_volume"] >= 1.5) & (scored["component_delta"] == 1)
    ).astype(int)
    scored["order_flow_score"] = (
        (scored["component_delta"] * 2)
        + scored["component_delta_change"]
        + scored["component_momentum"]
        + scored["component_efficiency"]
        + scored["component_volume"]
        + scored["component_sweep"]
    )
    return scored


# ============================================================
# ATR
# ============================================================

def calculate_atr(df):

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],

            (
                df["high"]
                - previous_close
            ).abs(),

            (
                df["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return (
        true_range
        .rolling(ATR_LENGTH)
        .mean()
    )


# ============================================================
# FEATURE ENGINE
# ============================================================

def create_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # RANGE
    # --------------------------------------------------------

    df["range"] = (
        df["high"]
        - df["low"]
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    df["atr"] = calculate_atr(df)

    # --------------------------------------------------------
    # BODY
    # --------------------------------------------------------

    df["body"] = (
        df["close"]
        - df["open"]
    ).abs()

    df["body_ratio"] = np.where(
        df["range"] > 0,
        df["body"] / df["range"],
        0,
    )

    # --------------------------------------------------------
    # WICKS
    # --------------------------------------------------------

    df["upper_wick"] = (
        df["high"]
        - df[["open", "close"]].max(axis=1)
    )

    df["lower_wick"] = (
        df[["open", "close"]].min(axis=1)
        - df["low"]
    )

    df["upper_wick_ratio"] = np.where(
        df["range"] > 0,
        df["upper_wick"] / df["range"],
        0,
    )

    df["lower_wick_ratio"] = np.where(
        df["range"] > 0,
        df["lower_wick"] / df["range"],
        0,
    )

    # --------------------------------------------------------
    # RANGE NORMALIZED BY ATR
    # --------------------------------------------------------

    df["range_atr"] = np.where(
        df["atr"] > 0,
        df["range"] / df["atr"],
        0,
    )

    # --------------------------------------------------------
    # CANDLE EFFICIENCY
    # --------------------------------------------------------

    df["candle_efficiency"] = np.where(
        df["range"] > 0,
        df["body"] / df["range"],
        0,
    )

    # --------------------------------------------------------
    # RELATIVE VOLUME
    # --------------------------------------------------------

    volume_mean = (
        df["volume"]
        .rolling(VOLUME_LOOKBACK)
        .mean()
    )

    volume_std = (
        df["volume"]
        .rolling(VOLUME_LOOKBACK)
        .std()
    )

    df["relative_volume"] = np.where(
        volume_mean > 0,
        df["volume"] / volume_mean,
        0,
    )

    # Volume z-score
    df["volume_zscore"] = np.where(
        volume_std > 0,
        (
            df["volume"]
            - volume_mean
        ) / volume_std,
        0,
    )

    # --------------------------------------------------------
    # VOLUME-BASED DELTA PROXY
    # --------------------------------------------------------
    #
    # This is NOT true bid/ask delta.
    #
    # It estimates directional pressure from:
    #
    #   candle close location
    #   multiplied by volume
    #
    # Positive = buying-pressure proxy
    # Negative = selling-pressure proxy
    #
    # --------------------------------------------------------

    close_location = np.where(
        df["range"] > 0,

        (
            (df["close"] - df["low"])
            -
            (df["high"] - df["close"])
        )
        / df["range"],

        0,
    )

    df["delta_proxy"] = (
        close_location
        * df["volume"]
    )

    # --------------------------------------------------------
    # NORMALIZED DELTA
    # --------------------------------------------------------

    delta_mean = (
        df["delta_proxy"]
        .rolling(VOLUME_LOOKBACK)
        .mean()
    )

    delta_std = (
        df["delta_proxy"]
        .rolling(VOLUME_LOOKBACK)
        .std()
    )

    df["delta_zscore"] = np.where(
        delta_std > 0,
        (
            df["delta_proxy"]
            - delta_mean
        ) / delta_std,
        0,
    )

    # --------------------------------------------------------
    # DELTA CHANGE
    # --------------------------------------------------------

    df["delta_change"] = (
        df["delta_proxy"]
        - df["delta_proxy"].shift(1)
    )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    df["return_1"] = (
        df["close"]
        .pct_change(1)
    )

    df["return_5"] = (
        df["close"]
        .pct_change(5)
    )

    df["momentum_atr"] = np.where(
        df["atr"] > 0,

        (
            df["close"]
            - df["close"].shift(5)
        ) / df["atr"],

        0,
    )

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    df["ema_50"] = (
        df["close"]
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    df["ema_200"] = (
        df["close"]
        .ewm(
            span=200,
            adjust=False,
        )
        .mean()
    )

    df["trend"] = np.select(
        [
            df["ema_50"] > df["ema_200"],
            df["ema_50"] < df["ema_200"],
        ],
        [
            1,
            -1,
        ],
        default=0,
    )

    # --------------------------------------------------------
    # LIQUIDITY LEVELS
    # --------------------------------------------------------

    previous_high = (
        df["high"]
        .shift(1)
        .rolling(LIQUIDITY_LOOKBACK)
        .max()
    )

    previous_low = (
        df["low"]
        .shift(1)
        .rolling(LIQUIDITY_LOOKBACK)
        .min()
    )

    df["previous_high"] = previous_high

    df["previous_low"] = previous_low

    # --------------------------------------------------------
    # LIQUIDITY SWEEP
    # --------------------------------------------------------

    bullish_sweep = (
        (df["low"] < previous_low)
        &
        (df["close"] > previous_low)
    )

    bearish_sweep = (
        (df["high"] > previous_high)
        &
        (df["close"] < previous_high)
    )

    df["bullish_sweep"] = (
        bullish_sweep.astype(int)
    )

    df["bearish_sweep"] = (
        bearish_sweep.astype(int)
    )

    df["liquidity_sweep"] = (
        df["bullish_sweep"]
        |
        df["bearish_sweep"]
    ).astype(int)

    # --------------------------------------------------------
    # SWEEP MAGNITUDE
    # --------------------------------------------------------

    df["bullish_sweep_size"] = np.where(
        bullish_sweep & (df["atr"] > 0),

        (
            previous_low
            - df["low"]
        ) / df["atr"],

        0,
    )

    df["bearish_sweep_size"] = np.where(
        bearish_sweep & (df["atr"] > 0),

        (
            df["high"]
            - previous_high
        ) / df["atr"],

        0,
    )

    # --------------------------------------------------------
    # VOLATILITY REGIME
    # --------------------------------------------------------

    df["volatility"] = (
        df["return_1"]
        .rolling(50)
        .std()
    )

    volatility_median = (
        df["volatility"]
        .rolling(200)
        .median()
    )

    df["high_volatility"] = (
        df["volatility"]
        > volatility_median
    ).astype(int)

    return df


# ============================================================
# SETUP DETECTION
# ============================================================

def detect_setup(row):

    long_score = 0

    short_score = 0

    # Higher-timeframe directional proxy
    if row["trend"] == 1:
        long_score += 1

    if row["trend"] == -1:
        short_score += 1

    # Momentum
    if row["momentum_atr"] > 0:
        long_score += 1

    if row["momentum_atr"] < 0:
        short_score += 1

    # Delta proxy
    if row["delta_zscore"] > 1:
        long_score += 1

    if row["delta_zscore"] < -1:
        short_score += 1

    # Relative volume
    if row["relative_volume"] > 1.5:

        if row["delta_proxy"] > 0:
            long_score += 1

        elif row["delta_proxy"] < 0:
            short_score += 1

    # Liquidity sweep
    if row["bullish_sweep"] == 1:
        long_score += 2

    if row["bearish_sweep"] == 1:
        short_score += 2

    # Require actual evidence.
    if long_score >= 4:
        return "LONG"

    if short_score >= 4:
        return "SHORT"

    return "NONE"


# ============================================================
# FORWARD OUTCOME
# ============================================================

def evaluate_forward_path(
    df,
    index,
    direction,
    entry,
    atr,
):

    risk = atr * RISK_ATR

    if direction == "LONG":

        stop = entry - risk

        targets = [
            entry + risk,
            entry + risk * 2,
            entry + risk * 3,
        ]

    else:

        stop = entry + risk

        targets = [
            entry - risk,
            entry - risk * 2,
            entry - risk * 3,
        ]

    future = df.iloc[
        index + 1:
        index + 1 + FORWARD_BARS
    ]

    hit_1r = False
    hit_2r = False
    hit_3r = False

    stopped = False

    mfe = 0.0
    mae = 0.0

    for _, candle in future.iterrows():

        if direction == "LONG":

            favorable = (
                candle["high"] - entry
            ) / risk

            adverse = (
                entry - candle["low"]
            ) / risk

            mfe = max(
                mfe,
                favorable,
            )

            mae = max(
                mae,
                adverse,
            )

            if candle["high"] >= targets[0]:
                hit_1r = True

            if candle["high"] >= targets[1]:
                hit_2r = True

            if candle["high"] >= targets[2]:
                hit_3r = True

            if candle["low"] <= stop:

                stopped = True
                break

        else:

            favorable = (
                entry - candle["low"]
            ) / risk

            adverse = (
                candle["high"] - entry
            ) / risk

            mfe = max(
                mfe,
                favorable,
            )

            mae = max(
                mae,
                adverse,
            )

            if candle["low"] <= targets[0]:
                hit_1r = True

            if candle["low"] <= targets[1]:
                hit_2r = True

            if candle["low"] <= targets[2]:
                hit_3r = True

            if candle["high"] >= stop:

                stopped = True
                break

    # Conservative outcome:
    # if stop was encountered before the final target,
    # do not claim the target as the trade outcome.

    if stopped:

        outcome_r = -1.0

    elif hit_3r:

        outcome_r = 3.0

    elif hit_2r:

        outcome_r = 2.0

    elif hit_1r:

        outcome_r = 1.0

    else:

        outcome_r = 0.0

    return {
        "hit_1r": int(hit_1r),
        "hit_2r": int(hit_2r),
        "hit_3r": int(hit_3r),
        "stopped": int(stopped),
        "mfe": mfe,
        "mae": mae,
        "outcome_r": outcome_r,
    }


# ============================================================
# DATABASE
# ============================================================

def create_tables(connection):

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_observations (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp TEXT NOT NULL,

            symbol TEXT NOT NULL,

            timeframe TEXT NOT NULL,

            direction TEXT NOT NULL,

            close REAL NOT NULL,

            atr REAL,

            volume REAL,

            relative_volume REAL,

            volume_zscore REAL,

            delta_proxy REAL,

            delta_zscore REAL,

            delta_change REAL,

            momentum_atr REAL,

            candle_efficiency REAL,

            body_ratio REAL,

            upper_wick_ratio REAL,

            lower_wick_ratio REAL,

            range_atr REAL,

            trend INTEGER,

            liquidity_sweep INTEGER,

            bullish_sweep INTEGER,

            bearish_sweep INTEGER,

            bullish_sweep_size REAL,

            bearish_sweep_size REAL,

            volatility REAL,

            high_volatility INTEGER,

            hit_1r INTEGER,

            hit_2r INTEGER,

            hit_3r INTEGER,

            stopped INTEGER,

            mfe REAL,

            mae REAL,

            outcome_r REAL

        )
        """
    )

    connection.commit()


# ============================================================
# BUILD OBSERVATIONS
# ============================================================

def build_database(df, connection):

    create_tables(connection)

    # Rebuild historical observations on every research run.
    # This prevents duplicate observations from accumulating in SQLite.
    connection.execute("DELETE FROM feature_observations")
    connection.commit()

    records = []

    minimum_history = 250

    last_index = (
        len(df)
        - FORWARD_BARS
        - 1
    )

    for index in range(
        minimum_history,
        last_index,
    ):

        row = df.iloc[index]

        if pd.isna(row["atr"]):
            continue

        direction = detect_setup(row)

        if direction == "NONE":
            continue

        outcome = evaluate_forward_path(
            df,
            index,
            direction,
            row["close"],
            row["atr"],
        )

        records.append(
            (
                str(row["time"]),
                "XAUUSD",
                TIMEFRAME,
                direction,
                row["close"],
                row["atr"],
                row["volume"],
                row["relative_volume"],
                row["volume_zscore"],
                row["delta_proxy"],
                row["delta_zscore"],
                row["delta_change"],
                row["momentum_atr"],
                row["candle_efficiency"],
                row["body_ratio"],
                row["upper_wick_ratio"],
                row["lower_wick_ratio"],
                row["range_atr"],
                row["trend"],
                row["liquidity_sweep"],
                row["bullish_sweep"],
                row["bearish_sweep"],
                row["bullish_sweep_size"],
                row["bearish_sweep_size"],
                row["volatility"],
                row["high_volatility"],
                outcome["hit_1r"],
                outcome["hit_2r"],
                outcome["hit_3r"],
                outcome["stopped"],
                outcome["mfe"],
                outcome["mae"],
                outcome["outcome_r"],
            )
        )

    if not records:
        return 0

    connection.executemany(
        """
        INSERT INTO feature_observations (

            timestamp,
            symbol,
            timeframe,
            direction,
            close,
            atr,
            volume,
            relative_volume,
            volume_zscore,
            delta_proxy,
            delta_zscore,
            delta_change,
            momentum_atr,
            candle_efficiency,
            body_ratio,
            upper_wick_ratio,
            lower_wick_ratio,
            range_atr,
            trend,
            liquidity_sweep,
            bullish_sweep,
            bearish_sweep,
            bullish_sweep_size,
            bearish_sweep_size,
            volatility,
            high_volatility,
            hit_1r,
            hit_2r,
            hit_3r,
            stopped,
            mfe,
            mae,
            outcome_r

        )

        VALUES (
            ?,?,?,?,?,?,?,?,?,?,
            ?,?,?,?,?,?,?,?,?,?,
            ?,?,?,?,?,?,?,?,?,
            ?,?,?,?
        )
        """,
        records,
    )

    connection.commit()

    return len(records)


# ============================================================
# RESEARCH REPORT
# ============================================================

def generate_report(connection):

    query = """
        SELECT
            direction,
            COUNT(*) AS samples,

            ROUND(
                AVG(hit_1r) * 100,
                2
            ) AS hit_1r_percent,

            ROUND(
                AVG(hit_2r) * 100,
                2
            ) AS hit_2r_percent,

            ROUND(
                AVG(hit_3r) * 100,
                2
            ) AS hit_3r_percent,

            ROUND(
                AVG(stopped) * 100,
                2
            ) AS stopped_percent,

            ROUND(
                AVG(outcome_r),
                3
            ) AS average_R,

            ROUND(
                AVG(mfe),
                3
            ) AS average_MFE,

            ROUND(
                AVG(mae),
                3
            ) AS average_MAE

        FROM feature_observations

        GROUP BY direction
    """

    report = pd.read_sql_query(
        query,
        connection,
    )

    print()
    print("=" * 75)
    print("ITRF ORDER FLOW RESEARCH REPORT")
    print("=" * 75)

    if report.empty:

        print("No qualifying setups found.")

    else:

        print(
            report.to_string(
                index=False
            )
        )

    print("=" * 75)


# ============================================================
# FEATURE-CONDITIONED ANALYSIS
# ============================================================

def generate_feature_analysis(connection):

    query = """
        SELECT
            direction,
            close,
            atr,
            relative_volume,
            volume_zscore,
            delta_proxy,
            delta_zscore,
            delta_change,
            momentum_atr,
            candle_efficiency,
            body_ratio,
            upper_wick_ratio,
            lower_wick_ratio,
            range_atr,
            trend,
            liquidity_sweep,
            bullish_sweep,
            bearish_sweep,
            bullish_sweep_size,
            bearish_sweep_size,
            volatility,
            high_volatility,
            hit_1r,
            hit_2r,
            hit_3r,
            stopped,
            mfe,
            mae,
            outcome_r
        FROM feature_observations
    """

    df = pd.read_sql_query(query, connection)

    print()
    print("=" * 95)
    print("ITRF FEATURE-CONDITIONED ANALYSIS")
    print("=" * 95)

    if df.empty:
        print("No observations available.")
        print("=" * 95)
        return

    conditions = [
        ("High Relative Volume", df["relative_volume"] >= 1.50),
        ("Very High Relative Volume", df["relative_volume"] >= 2.00),
        ("Positive Delta Z", df["delta_zscore"] >= 1.00),
        ("Negative Delta Z", df["delta_zscore"] <= -1.00),
        ("Strong Positive Delta Change", df["delta_change"] > 0),
        ("Strong Negative Delta Change", df["delta_change"] < 0),
        ("Positive Momentum", df["momentum_atr"] > 0),
        ("Negative Momentum", df["momentum_atr"] < 0),
        ("High Candle Efficiency", df["candle_efficiency"] >= 0.60),
        ("Liquidity Sweep", df["liquidity_sweep"] == 1),
        ("Bullish Sweep", df["bullish_sweep"] == 1),
        ("Bearish Sweep", df["bearish_sweep"] == 1),
        ("High Volatility", df["high_volatility"] == 1),
        (
            "Long + Bullish Sweep + Positive Delta",
            (df["direction"] == "LONG")
            & (df["bullish_sweep"] == 1)
            & (df["delta_zscore"] >= 1.00),
        ),
        (
            "Short + Bearish Sweep + Negative Delta",
            (df["direction"] == "SHORT")
            & (df["bearish_sweep"] == 1)
            & (df["delta_zscore"] <= -1.00),
        ),
        (
            "Long + Trend + Positive Delta",
            (df["direction"] == "LONG")
            & (df["trend"] == 1)
            & (df["delta_zscore"] >= 1.00),
        ),
        (
            "Short + Trend + Negative Delta",
            (df["direction"] == "SHORT")
            & (df["trend"] == -1)
            & (df["delta_zscore"] <= -1.00),
        ),
        (
            "Long + Sweep + High Volume",
            (df["direction"] == "LONG")
            & (df["bullish_sweep"] == 1)
            & (df["relative_volume"] >= 1.50),
        ),
        (
            "Short + Sweep + High Volume",
            (df["direction"] == "SHORT")
            & (df["bearish_sweep"] == 1)
            & (df["relative_volume"] >= 1.50),
        ),
    ]

    rows = []
    minimum_sample = 30

    for name, mask in conditions:
        subset = df.loc[mask].copy()
        samples = len(subset)

        if samples < minimum_sample:
            continue

        rows.append(
            {
                "condition": name,
                "samples": samples,
                "1R_%": round(subset["hit_1r"].mean() * 100, 2),
                "2R_%": round(subset["hit_2r"].mean() * 100, 2),
                "3R_%": round(subset["hit_3r"].mean() * 100, 2),
                "stopped_%": round(subset["stopped"].mean() * 100, 2),
                "average_R": round(subset["outcome_r"].mean(), 3),
                "average_MFE": round(subset["mfe"].mean(), 3),
                "average_MAE": round(subset["mae"].mean(), 3),
            }
        )

    report = pd.DataFrame(rows)

    if report.empty:
        print("No conditions met the minimum sample requirement.")
    else:
        report = report.sort_values(
            ["average_R", "samples"],
            ascending=[False, False],
        )
        print(report.to_string(index=False))

    print("=" * 95)
    print("Minimum sample size per condition:", minimum_sample)
    print("Thresholds are fixed research conditions, not optimized parameters.")
    print("Delta remains a volume-based proxy, not true bid/ask flow.")
    print("=" * 95)


# ============================================================
# REGIME-CONDITIONED RESEARCH v0.2.1
# ============================================================

def generate_regime_analysis(connection):

    query = """
        SELECT
            direction,
            trend,
            bullish_sweep,
            bearish_sweep,
            relative_volume,
            delta_zscore,
            hit_1r,
            hit_2r,
            hit_3r,
            stopped,
            mfe,
            mae,
            outcome_r
        FROM feature_observations
    """

    df = pd.read_sql_query(
        query,
        connection,
    )

    print()
    print("=" * 110)
    print("ITRF REGIME-CONDITIONED ANALYSIS v0.2.2")
    print("=" * 110)

    if df.empty:
        print("No observations available.")
        print("=" * 110)
        return

    # --------------------------------------------------------
    # Explicit direction-aware conditions
    # --------------------------------------------------------

    conditions = [

        ("LONG baseline",
         df["direction"] == "LONG"),

        ("LONG + Trend aligned",
         (df["direction"] == "LONG")
         & (df["trend"] == 1)),

        ("LONG + Delta aligned",
         (df["direction"] == "LONG")
         & (df["delta_zscore"] >= 1.00)),

        ("LONG + Sweep aligned",
         (df["direction"] == "LONG")
         & (df["bullish_sweep"] == 1)),

        ("LONG + Trend + Delta",
         (df["direction"] == "LONG")
         & (df["trend"] == 1)
         & (df["delta_zscore"] >= 1.00)),

        ("LONG + Trend + Sweep",
         (df["direction"] == "LONG")
         & (df["trend"] == 1)
         & (df["bullish_sweep"] == 1)),

        ("LONG + Sweep + Delta",
         (df["direction"] == "LONG")
         & (df["bullish_sweep"] == 1)
         & (df["delta_zscore"] >= 1.00)),

        ("LONG + Trend + Sweep + Delta",
         (df["direction"] == "LONG")
         & (df["trend"] == 1)
         & (df["bullish_sweep"] == 1)
         & (df["delta_zscore"] >= 1.00)),

        ("LONG + Trend + Sweep + Delta + High Volume",
         (df["direction"] == "LONG")
         & (df["trend"] == 1)
         & (df["bullish_sweep"] == 1)
         & (df["delta_zscore"] >= 1.00)
         & (df["relative_volume"] >= 1.50)),

        ("LONG + Trend + Sweep + Delta + Very High Volume",
         (df["direction"] == "LONG")
         & (df["trend"] == 1)
         & (df["bullish_sweep"] == 1)
         & (df["delta_zscore"] >= 1.00)
         & (df["relative_volume"] >= 2.00)),

        ("SHORT baseline",
         df["direction"] == "SHORT"),

        ("SHORT + Trend aligned",
         (df["direction"] == "SHORT")
         & (df["trend"] == -1)),

        ("SHORT + Delta aligned",
         (df["direction"] == "SHORT")
         & (df["delta_zscore"] <= -1.00)),

        ("SHORT + Sweep aligned",
         (df["direction"] == "SHORT")
         & (df["bearish_sweep"] == 1)),

        ("SHORT + Trend + Delta",
         (df["direction"] == "SHORT")
         & (df["trend"] == -1)
         & (df["delta_zscore"] <= -1.00)),

        ("SHORT + Trend + Sweep",
         (df["direction"] == "SHORT")
         & (df["trend"] == -1)
         & (df["bearish_sweep"] == 1)),

        ("SHORT + Sweep + Delta",
         (df["direction"] == "SHORT")
         & (df["bearish_sweep"] == 1)
         & (df["delta_zscore"] <= -1.00)),

        ("SHORT + Trend + Sweep + Delta",
         (df["direction"] == "SHORT")
         & (df["trend"] == -1)
         & (df["bearish_sweep"] == 1)
         & (df["delta_zscore"] <= -1.00)),

        ("SHORT + Trend + Sweep + Delta + High Volume",
         (df["direction"] == "SHORT")
         & (df["trend"] == -1)
         & (df["bearish_sweep"] == 1)
         & (df["delta_zscore"] <= -1.00)
         & (df["relative_volume"] >= 1.50)),

        ("SHORT + Trend + Sweep + Delta + Very High Volume",
         (df["direction"] == "SHORT")
         & (df["trend"] == -1)
         & (df["bearish_sweep"] == 1)
         & (df["delta_zscore"] <= -1.00)
         & (df["relative_volume"] >= 2.00)),
    ]

    minimum_sample = 50
    rows = []

    for name, mask in conditions:

        subset = df.loc[mask].copy()
        samples = len(subset)

        if samples < minimum_sample:
            continue

        rows.append(
            {
                "condition": name,
                "samples": samples,
                "1R_%": round(
                    subset["hit_1r"].mean() * 100, 2
                ),
                "2R_%": round(
                    subset["hit_2r"].mean() * 100, 2
                ),
                "3R_%": round(
                    subset["hit_3r"].mean() * 100, 2
                ),
                "stopped_%": round(
                    subset["stopped"].mean() * 100, 2
                ),
                "average_R": round(
                    subset["outcome_r"].mean(), 3
                ),
                "average_MFE": round(
                    subset["mfe"].mean(), 3
                ),
                "average_MAE": round(
                    subset["mae"].mean(), 3
                ),
            }
        )

    report = pd.DataFrame(rows)

    if report.empty:
        print("No conditions met the minimum sample requirement.")
    else:
        report = report.sort_values(
            ["average_R", "samples"],
            ascending=[False, False],
        )
        print(report.to_string(index=False))

    # --------------------------------------------------------
    # Integrity checks
    # --------------------------------------------------------

    long_count = (
        df["direction"] == "LONG"
    ).sum()

    short_count = (
        df["direction"] == "SHORT"
    ).sum()

    print("=" * 110)
    print("Total observations:", len(df))
    print("LONG observations:", long_count)
    print("SHORT observations:", short_count)
    print("LONG + SHORT:", long_count + short_count)

    if long_count + short_count == len(df):
        print("Direction accounting check: PASS")
    else:
        print("Direction accounting check: FAIL")

    print(
        "Minimum sample size per condition:",
        minimum_sample,
    )
    print(
        "All regime conditions are explicitly LONG or SHORT."
    )
    print(
        "No parameter optimization has been performed."
    )
    print(
        "Results are in-sample and require out-of-sample validation."
    )
    print(
        "Delta remains a volume-based proxy, not true bid/ask flow."
    )
    print("=" * 110)

# ============================================================
# OOS REGIME SCORE ANALYSIS v0.7
# ============================================================

def generate_oos_regime_score_analysis(connection, oos_start):

    query = """
        SELECT
            timestamp,
            direction,
            trend,
            relative_volume,
            delta_zscore,
            delta_change,
            momentum_atr,
            candle_efficiency,
            bullish_sweep,
            bearish_sweep,
            hit_1r,
            hit_2r,
            hit_3r,
            stopped,
            mfe,
            mae,
            outcome_r
        FROM feature_observations
        ORDER BY timestamp
    """

    df = pd.read_sql_query(query, connection)

    print()
    print("=" * 110)
    print("ITRF OOS REGIME SCORE ANALYSIS v0.7")
    print("=" * 110)

    if df.empty:
        print("No observations available.")
        print("=" * 110)
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    split_time = resolve_oos_split(df, oos_start, "OOS regime-score analysis")
    if split_time is None:
        print("=" * 110)
        return

    # --------------------------------------------------------
    # Frozen OOS period
    # --------------------------------------------------------

    oos_df = df.loc[
        df["timestamp"] >= split_time
    ].copy()

    if oos_df.empty:
        print("No OOS observations available.")
        print("=" * 110)
        return

    # --------------------------------------------------------
    # Rebuild frozen Score 5-7
    # --------------------------------------------------------

    oos_df["component_delta"] = 0

    oos_df.loc[
        (
            ((oos_df["direction"] == "LONG") &
             (oos_df["delta_zscore"] >= 1.0))
            |
            ((oos_df["direction"] == "SHORT") &
             (oos_df["delta_zscore"] <= -1.0))
        ),
        "component_delta"
    ] = 1

    oos_df["component_delta_change"] = 0

    oos_df.loc[
        (
            ((oos_df["direction"] == "LONG") &
             (oos_df["delta_change"] > 0))
            |
            ((oos_df["direction"] == "SHORT") &
             (oos_df["delta_change"] < 0))
        ),
        "component_delta_change"
    ] = 1

    oos_df["component_momentum"] = 0
    oos_df.loc[
        (
            ((oos_df["direction"] == "LONG") &
             (oos_df["momentum_atr"] > 0))
            |
            ((oos_df["direction"] == "SHORT") &
             (oos_df["momentum_atr"] < 0))
        ),
        "component_momentum"
    ] = 1

    oos_df["component_efficiency"] = 0

    oos_df.loc[
        oos_df["candle_efficiency"] >= 0.60,
        "component_efficiency"
    ] = 1

    oos_df["component_sweep"] = 0

    oos_df.loc[
        (
            ((oos_df["direction"] == "LONG") &
             (oos_df["bullish_sweep"] == 1))
            |
            ((oos_df["direction"] == "SHORT") &
             (oos_df["bearish_sweep"] == 1))
        ),
        "component_sweep"
    ] = 1

    oos_df["component_volume"] = 0

    oos_df.loc[
        (
            (oos_df["relative_volume"] >= 1.5)
            &
            (oos_df["component_delta"] == 1)
        ),
        "component_volume"
    ] = 1

    # --------------------------------------------------------
    # Frozen score
    # --------------------------------------------------------

    oos_df["order_flow_score"] = (
        (oos_df["component_delta"] * 2)
        + oos_df["component_delta_change"]
        + oos_df["component_momentum"]
        + oos_df["component_efficiency"]
        + oos_df["component_volume"]
        + oos_df["component_sweep"]
    )

    # --------------------------------------------------------
    # Keep only Score 5-7
    # --------------------------------------------------------

    score_df = oos_df.loc[
        oos_df["order_flow_score"].between(5, 7)
    ].copy()

    if score_df.empty:
        print("No OOS Score 5-7 observations available.")
        print("=" * 110)
        return

    # --------------------------------------------------------
    # Summary helper
    # --------------------------------------------------------

    def summarize(data):

        if data.empty:
            return {
                "samples": 0,
                "1R_%": 0,
                "2R_%": 0,
                "3R_%": 0,
                "stopped_%": 0,
                "average_R": 0,
                "average_MFE": 0,
                "average_MAE": 0,
            }

        return {
            "samples": len(data),
            "1R_%": round(
                data["hit_1r"].mean() * 100,
                2
            ),
            "2R_%": round(
                data["hit_2r"].mean() * 100,
                2
            ),
            "3R_%": round(
                data["hit_3r"].mean() * 100,
                2
            ),
            "stopped_%": round(
                data["stopped"].mean() * 100,
                2
            ),
            "average_R": round(
                data["outcome_r"].mean(),
                3
            ),
            "average_MFE": round(
                data["mfe"].mean(),
                3
            ),
            "average_MAE": round(
                data["mae"].mean(),
                3
            ),
        }

    # --------------------------------------------------------
    # REGIME CONDITIONS
    # --------------------------------------------------------

    conditions = [

        (
            "Trend aligned",
            (
                ((score_df["direction"] == "LONG") &
                 (score_df["trend"] == 1))
                |
                ((score_df["direction"] == "SHORT") &
                 (score_df["trend"] == -1))
            )
        ),

        (
            "Trend misaligned",
            (
                ((score_df["direction"] == "LONG") &
                 (score_df["trend"] == -1))
                |
                ((score_df["direction"] == "SHORT") &
                 (score_df["trend"] == 1))
            )
        ),

        (
            "Sweep aligned",
            (
                ((score_df["direction"] == "LONG") &
                 (score_df["bullish_sweep"] == 1))
                |
                ((score_df["direction"] == "SHORT") &
                 (score_df["bearish_sweep"] == 1))
            )
        ),

        (
            "No aligned sweep",
            (
                ((score_df["direction"] == "LONG") &
                 (score_df["bullish_sweep"] == 0))
                |
                ((score_df["direction"] == "SHORT") &
                 (score_df["bearish_sweep"] == 0))
            )
        ),

        (
            "High volume",
            score_df["relative_volume"] >= 1.50
        ),

        (
            "Very high volume",
            score_df["relative_volume"] >= 2.00
        ),

        (
            "Trend + Sweep aligned",
            (
                (
                    ((score_df["direction"] == "LONG") &
                     (score_df["trend"] == 1))
                    |
                    ((score_df["direction"] == "SHORT") &
                     (score_df["trend"] == -1))
                )
                &
                (
                    ((score_df["direction"] == "LONG") &
                     (score_df["bullish_sweep"] == 1))
                    |
                    ((score_df["direction"] == "SHORT") &
                     (score_df["bearish_sweep"] == 1))
                )
            )
        ),

        (
            "Trend + High volume",
            (
                (
                    ((score_df["direction"] == "LONG") &
                     (score_df["trend"] == 1))
                    |
                    ((score_df["direction"] == "SHORT") &
                     (score_df["trend"] == -1))
                )
                &
                (score_df["relative_volume"] >= 1.50)
            )
        ),

        (
            "Sweep + High volume",
            (
                (
                    ((score_df["direction"] == "LONG") &
                     (score_df["bullish_sweep"] == 1))
                    |
                    ((score_df["direction"] == "SHORT") &
                     (score_df["bearish_sweep"] == 1))
                )
                &
                (score_df["relative_volume"] >= 1.50)
            )
        ),

        (
            "Trend + Sweep + High volume",
            (
                (
                    ((score_df["direction"] == "LONG") &
                     (score_df["trend"] == 1))
                    |
                    ((score_df["direction"] == "SHORT") &
                     (score_df["trend"] == -1))
                )
                &
                (
                    ((score_df["direction"] == "LONG") &
                     (score_df["bullish_sweep"] == 1))
                    |
                    ((score_df["direction"] == "SHORT") &
                     (score_df["bearish_sweep"] == 1))
                )
                &
                (score_df["relative_volume"] >= 1.50)
            )
        ),
    ]

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    records = []

    for name, mask in conditions:

        subset = score_df.loc[mask]

        if len(subset) < 30:
            continue

        result = summarize(subset)

        records.append(
            {
                "condition": name,
                **result
            }
        )

    report = pd.DataFrame(records)

    print()
    print("-" * 110)
    print("OOS SCORE 5-7 REGIME CONDITIONS")
    print("-" * 110)

    if report.empty:
        print("No conditions met the minimum sample requirement.")
    else:
        print(
            report.to_string(index=False)
        )

    # --------------------------------------------------------
    # DIRECTION + REGIME
    # --------------------------------------------------------

    print()
    print("-" * 110)
    print("OOS SCORE 5-7 DIRECTION + TREND")
    print("-" * 110)

    direction_records = []

    for direction in ["LONG", "SHORT"]:

        for trend_name, trend_value in [
            ("Trend aligned", 1 if direction == "LONG" else -1),
            ("Trend misaligned", -1 if direction == "LONG" else 1),
        ]:

            subset = score_df.loc[
                (score_df["direction"] == direction)
                &
                (score_df["trend"] == trend_value)
            ]

            if len(subset) < 30:
                continue

            result = summarize(subset)

            direction_records.append(
                {
                    "direction": direction,
                    "trend_condition": trend_name,
                    **result
                }
            )

    direction_table = pd.DataFrame(
        direction_records
    )

    if direction_table.empty:
        print("No direction/trend conditions met minimum sample.")
    else:
        print(
            direction_table.to_string(index=False)
        )

    # --------------------------------------------------------
    # RESEARCH INTEGRITY
    # --------------------------------------------------------

    print()
    print("-" * 110)
    print("RESEARCH INTEGRITY")
    print("-" * 110)

    print(
        "OOS start:",
        split_time
    )

    print(
        "Score 5-7 observations:",
        len(score_df)
    )

    print(
        "Minimum sample size per condition: 30"
    )

    print(
        "Score definition remains frozen from v0.5."
    )

    print(
        "No parameter optimization was performed."
    )

    print(
        "This analysis evaluates regime conditioning only."
    )

    print(
        "Results remain research evidence, "
        "not proof of profitability."
    )

    print(
        "Delta remains a volume-based proxy, "
        "not true bid/ask flow."
    )

    print("=" * 110)

# ============================================================
# ORDER FLOW SCORE RESEARCH v0.3
# ============================================================

def generate_order_flow_score_analysis(connection):

    query = """
        SELECT
            direction,
            trend,
            relative_volume,
            delta_zscore,
            delta_change,
            momentum_atr,
            candle_efficiency,
            bullish_sweep,
            bearish_sweep,
            hit_1r,
            hit_2r,
            hit_3r,
            stopped,
            mfe,
            mae,
            outcome_r
        FROM feature_observations
    """

    df = pd.read_sql_query(query, connection)

    print()
    print("=" * 110)
    print("ITRF ORDER FLOW SCORE ANALYSIS v0.3")
    print("=" * 110)

    if df.empty:
        print("No observations available.")
        print("=" * 110)
        return

    # --------------------------------------------------------
    # Direction-aware component scoring
    #
    # IMPORTANT:
    # Score is evaluated relative to the existing LONG/SHORT
    # setup direction.
    #
    # Delta remains a volume-based proxy.
    # It is NOT true bid/ask order flow.
    # --------------------------------------------------------

    df["score_delta"] = 0
    df.loc[
        (df["direction"] == "LONG")
        & (df["delta_zscore"] >= 1.00),
        "score_delta"
    ] = 1

    df.loc[
        (df["direction"] == "LONG")
        & (df["delta_zscore"] <= -1.00),
        "score_delta"
    ] = -1

    df.loc[
        (df["direction"] == "SHORT")
        & (df["delta_zscore"] <= -1.00),
        "score_delta"
    ] = 1

    df.loc[
        (df["direction"] == "SHORT")
        & (df["delta_zscore"] >= 1.00),
        "score_delta"
    ] = -1

    # --------------------------------------------------------
    # Delta change alignment
    # --------------------------------------------------------

    df["score_delta_change"] = 0

    df.loc[
        (df["direction"] == "LONG")
        & (df["delta_change"] > 0),
        "score_delta_change"
    ] = 1

    df.loc[
        (df["direction"] == "LONG")
        & (df["delta_change"] < 0),
        "score_delta_change"
    ] = -1

    df.loc[
        (df["direction"] == "SHORT")
        & (df["delta_change"] < 0),
        "score_delta_change"
    ] = 1

    df.loc[
        (df["direction"] == "SHORT")
        & (df["delta_change"] > 0),
        "score_delta_change"
    ] = -1

    # --------------------------------------------------------
    # Momentum alignment
    # --------------------------------------------------------

    df["score_momentum"] = 0

    df.loc[
        (df["direction"] == "LONG")
        & (df["momentum_atr"] > 0),
        "score_momentum"
    ] = 1

    df.loc[
        (df["direction"] == "LONG")
        & (df["momentum_atr"] < 0),
        "score_momentum"
    ] = -1

    df.loc[
        (df["direction"] == "SHORT")
        & (df["momentum_atr"] < 0),
        "score_momentum"
    ] = 1

    df.loc[
        (df["direction"] == "SHORT")
        & (df["momentum_atr"] > 0),
        "score_momentum"
    ] = -1

    # --------------------------------------------------------
    # Liquidity sweep alignment
    #
    # Sweep receives 2 points because it represents a specific
    # liquidity event rather than general participation.
    # --------------------------------------------------------

    df["score_sweep"] = 0

    df.loc[
        (df["direction"] == "LONG")
        & (df["bullish_sweep"] == 1),
        "score_sweep"
    ] = 2

    df.loc[
        (df["direction"] == "LONG")
        & (df["bearish_sweep"] == 1),
        "score_sweep"
    ] = -2

    df.loc[
        (df["direction"] == "SHORT")
        & (df["bearish_sweep"] == 1),
        "score_sweep"
    ] = 2

    df.loc[
        (df["direction"] == "SHORT")
        & (df["bullish_sweep"] == 1),
        "score_sweep"
    ] = -2

    # --------------------------------------------------------
    # High-volume confirmation
    #
    # Volume itself is NOT directional.
    # It only receives a point when elevated volume agrees
    # with the directional delta proxy.
    # --------------------------------------------------------

    df["score_volume"] = 0

    df.loc[
        (df["relative_volume"] >= 1.50)
        & (df["score_delta"] == 1),
        "score_volume"
    ] = 1

    df.loc[
        (df["relative_volume"] >= 1.50)
        & (df["score_delta"] == -1),
        "score_volume"
    ] = -1

    # --------------------------------------------------------
    # Candle efficiency
    #
    # Efficiency is non-directional, therefore it receives
    # positive confirmation only.
    # --------------------------------------------------------

    df["score_efficiency"] = 0

    df.loc[
        df["candle_efficiency"] >= 0.60,
        "score_efficiency"
    ] = 1

    # --------------------------------------------------------
    # TOTAL ORDER FLOW SCORE
    # --------------------------------------------------------

    df["order_flow_score"] = (
        df["score_delta"]
        + df["score_delta_change"]
        + df["score_momentum"]
        + df["score_sweep"]
        + df["score_volume"]
        + df["score_efficiency"]
    )

    # --------------------------------------------------------
    # Score-conditioned research
    # --------------------------------------------------------

    score_conditions = [
        ("Score <= -2", df["order_flow_score"] <= -2),
        ("Score -1 to 0",
         (df["order_flow_score"] >= -1)
         & (df["order_flow_score"] <= 0)),
        ("Score 1 to 2",
         (df["order_flow_score"] >= 1)
         & (df["order_flow_score"] <= 2)),
        ("Score 3 to 4",
         (df["order_flow_score"] >= 3)
         & (df["order_flow_score"] <= 4)),
        ("Score 5 to 7",
         (df["order_flow_score"] >= 5)
         & (df["order_flow_score"] <= 7)),
    ]

    minimum_sample = 50
    rows = []

    for name, mask in score_conditions:

        subset = df.loc[mask].copy()

        samples = len(subset)

        if samples < minimum_sample:
            continue

        rows.append(
            {
                "score_condition": name,
                "samples": samples,
                "1R_%": round(
                    subset["hit_1r"].mean() * 100,
                    2,
                ),
                "2R_%": round(
                    subset["hit_2r"].mean() * 100,
                    2,
                ),
                "3R_%": round(
                    subset["hit_3r"].mean() * 100,
                    2,
                ),
                "stopped_%": round(
                    subset["stopped"].mean() * 100,
                    2,
                ),
                "average_R": round(
                    subset["outcome_r"].mean(),
                    3,
                ),
                "average_MFE": round(
                    subset["mfe"].mean(),
                    3,
                ),
                "average_MAE": round(
                    subset["mae"].mean(),
                    3,
                ),
            }
        )

    report = pd.DataFrame(rows)

    if report.empty:

        print("No score conditions met the minimum sample requirement.")

    else:

        print(report.to_string(index=False))

    # --------------------------------------------------------
    # Direction-specific score analysis
    # --------------------------------------------------------

    print()
    print("-" * 110)
    print("DIRECTION-SPECIFIC ORDER FLOW SCORE")
    print("-" * 110)

    direction_rows = []

    for direction in ["LONG", "SHORT"]:

        direction_df = df.loc[
            df["direction"] == direction
        ]

        for name, mask in score_conditions:

            subset = direction_df.loc[mask].copy()

            samples = len(subset)

            if samples < minimum_sample:
                continue

            direction_rows.append(
                {
                    "direction": direction,
                    "score_condition": name,
                    "samples": samples,
                    "1R_%": round(
                        subset["hit_1r"].mean() * 100,
                        2,
                    ),
                    "2R_%": round(
                        subset["hit_2r"].mean() * 100,
                        2,
                    ),
                    "3R_%": round(
                        subset["hit_3r"].mean() * 100,
                        2,
                    ),
                    "stopped_%": round(
                        subset["stopped"].mean() * 100,
                        2,
                    ),
                    "average_R": round(
                        subset["outcome_r"].mean(),
                        3,
                    ),
                }
            )

    direction_report = pd.DataFrame(direction_rows)

    if direction_report.empty:

        print(
            "No direction-specific score conditions "
            "met the minimum sample requirement."
        )

    else:

        print(
            direction_report.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Score distribution
    # --------------------------------------------------------

    print()
    print("-" * 110)
    print("ORDER FLOW SCORE DISTRIBUTION")
    print("-" * 110)

    distribution = (
        df["order_flow_score"]
        .value_counts()
        .sort_index()
    )

    print(distribution.to_string())

    print("=" * 110)
    print(
        "Minimum sample size per score condition:",
        minimum_sample,
    )
    print(
        "Score weights are fixed research rules."
    )
    print(
        "No parameter optimization has been performed."
    )
    print(
        "Score is evaluated relative to LONG/SHORT setup direction."
    )
    print(
        "Results are in-sample and require out-of-sample validation."
    )
    print(
        "Delta remains a volume-based proxy, not true bid/ask flow."
    )
    print("=" * 110)


# ============================================================
# ORDER FLOW COMPONENT ATTRIBUTION v0.4
# ============================================================

def generate_component_attribution_analysis(connection):

    query = """
        SELECT
            direction,
            relative_volume,
            delta_zscore,
            delta_change,
            momentum_atr,
            candle_efficiency,
            bullish_sweep,
            bearish_sweep,
            hit_1r,
            hit_2r,
            hit_3r,
            stopped,
            mfe,
            mae,
            outcome_r
        FROM feature_observations
    """

    df = pd.read_sql_query(query, connection)

    print()
    print("=" * 110)
    print("ITRF ORDER FLOW COMPONENT ATTRIBUTION v0.4")
    print("=" * 110)

    if df.empty:
        print("No observations available.")
        print("=" * 110)
        return

    # --------------------------------------------------------
    # Direction-aware component definitions
    #
    # Each component is evaluated independently.
    #
    # IMPORTANT:
    # This is attribution research, NOT parameter optimization.
    # Delta remains a volume-based proxy.
    # --------------------------------------------------------

    df["component_delta"] = 0
    df.loc[
        (
            ((df["direction"] == "LONG") & (df["delta_zscore"] >= 1.0))
            |
            ((df["direction"] == "SHORT") & (df["delta_zscore"] <= -1.0))
        ),
        "component_delta"
    ] = 1

    df["component_delta_change"] = 0
    df.loc[
        (
            ((df["direction"] == "LONG") & (df["delta_change"] > 0))
            |
            ((df["direction"] == "SHORT") & (df["delta_change"] < 0))
        ),
        "component_delta_change"
    ] = 1

    df["component_momentum"] = 0
    df.loc[
        (
            ((df["direction"] == "LONG") & (df["momentum_atr"] > 0))
            |
            ((df["direction"] == "SHORT") & (df["momentum_atr"] < 0))
        ),
        "component_momentum"
    ] = 1

    df["component_sweep"] = 0
    df.loc[
        (
            ((df["direction"] == "LONG") & (df["bullish_sweep"] == 1))
            |
            ((df["direction"] == "SHORT") & (df["bearish_sweep"] == 1))
        ),
        "component_sweep"
    ] = 1

    df["component_volume"] = 0
    df.loc[
        (
            (df["relative_volume"] >= 1.5)
            & (df["component_delta"] == 1)
        ),
        "component_volume"
    ] = 1

    df["component_efficiency"] = 0
    df.loc[
        df["candle_efficiency"] >= 0.60,
        "component_efficiency"
    ] = 1

    components = [
        ("Delta", "component_delta"),
        ("Delta Change", "component_delta_change"),
        ("Momentum", "component_momentum"),
        ("Liquidity Sweep", "component_sweep"),
        ("Relative Volume", "component_volume"),
        ("Candle Efficiency", "component_efficiency"),
    ]

    # --------------------------------------------------------
    # Individual component analysis
    # --------------------------------------------------------

    rows = []

    for name, column in components:

        subset = df.loc[df[column] == 1].copy()

        if len(subset) < 50:
            continue

        rows.append(
            {
                "component": name,
                "samples": len(subset),
                "1R_%": round(subset["hit_1r"].mean() * 100, 2),
                "2R_%": round(subset["hit_2r"].mean() * 100, 2),
                "3R_%": round(subset["hit_3r"].mean() * 100, 2),
                "stopped_%": round(subset["stopped"].mean() * 100, 2),
                "average_R": round(subset["outcome_r"].mean(), 3),
                "average_MFE": round(subset["mfe"].mean(), 3),
                "average_MAE": round(subset["mae"].mean(), 3),
            }
        )

    report = pd.DataFrame(rows)

    print()
    print("-" * 110)
    print("INDIVIDUAL COMPONENT ATTRIBUTION")
    print("-" * 110)

    if report.empty:
        print("No components met the minimum sample requirement.")
    else:
        print(report.to_string(index=False))

    # --------------------------------------------------------
    # LONG / SHORT attribution
    # --------------------------------------------------------

    print()
    print("-" * 110)
    print("DIRECTION-SPECIFIC COMPONENT ATTRIBUTION")
    print("-" * 110)

    direction_rows = []

    for direction in ["LONG", "SHORT"]:

        direction_df = df.loc[
            df["direction"] == direction
        ]

        for name, column in components:

            subset = direction_df.loc[
                direction_df[column] == 1
            ].copy()

            if len(subset) < 50:
                continue

            direction_rows.append(
                {
                    "direction": direction,
                    "component": name,
                    "samples": len(subset),
                    "1R_%": round(subset["hit_1r"].mean() * 100, 2),
                    "2R_%": round(subset["hit_2r"].mean() * 100, 2),
                    "3R_%": round(subset["hit_3r"].mean() * 100, 2),
                    "stopped_%": round(subset["stopped"].mean() * 100, 2),
                    "average_R": round(subset["outcome_r"].mean(), 3),
                }
            )

    direction_report = pd.DataFrame(direction_rows)

    if direction_report.empty:
        print("No direction-specific component conditions met minimum sample.")
    else:
        print(direction_report.to_string(index=False))

    # --------------------------------------------------------
    # Two-component interaction analysis
    # --------------------------------------------------------

    print()
    print("-" * 110)
    print("TWO-COMPONENT INTERACTION ANALYSIS")
    print("-" * 110)

    interaction_rows = []

    for i in range(len(components)):

        name_a, column_a = components[i]

        for j in range(i + 1, len(components)):

            name_b, column_b = components[j]

            subset = df.loc[
                (df[column_a] == 1)
                & (df[column_b] == 1)
            ].copy()

            if len(subset) < 50:
                continue

            interaction_rows.append(
                {
                    "components": f"{name_a} + {name_b}",
                    "samples": len(subset),
                    "1R_%": round(subset["hit_1r"].mean() * 100, 2),
                    "2R_%": round(subset["hit_2r"].mean() * 100, 2),
                    "3R_%": round(subset["hit_3r"].mean() * 100, 2),
                    "stopped_%": round(subset["stopped"].mean() * 100, 2),
                    "average_R": round(subset["outcome_r"].mean(), 3),
                }
            )

    interaction_report = pd.DataFrame(interaction_rows)

    if interaction_report.empty:
        print("No interactions met minimum sample requirement.")
    else:

        interaction_report = interaction_report.sort_values(
            "average_R",
            ascending=False,
        )

        print(
            interaction_report.head(15).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Baseline comparison
    # --------------------------------------------------------

    print()
    print("-" * 110)
    print("BASELINE")
    print("-" * 110)

    print(
        "All observations:",
        len(df),
    )

    print(
        "Average R:",
        round(df["outcome_r"].mean(), 3),
    )

    print(
        "1R:",
        round(df["hit_1r"].mean() * 100, 2),
        "%",
    )

    print(
        "2R:",
        round(df["hit_2r"].mean() * 100, 2),
        "%",
    )

    print(
        "3R:",
        round(df["hit_3r"].mean() * 100, 2),
        "%",
    )

    print("=" * 110)
    print(
        "Minimum sample size:",
        50,
    )
    print(
        "No parameter optimization has been performed."
    )
    print(
        "Interaction ranking is descriptive research only."
    )
    print(
        "Results are in-sample and require out-of-sample validation."
    )
    print(
        "Delta remains a volume-based proxy, not true bid/ask flow."
    )
    print("=" * 110)
# ============================================================
# OUT-OF-SAMPLE VALIDATION
# ============================================================

def generate_out_of_sample_validation(connection):

    query = """
        SELECT
            timestamp,
            direction,
            relative_volume,
            delta_zscore,
            delta_change,
            momentum_atr,
            candle_efficiency,
            bullish_sweep,
            bearish_sweep,
            hit_1r,
            hit_2r,
            hit_3r,
            stopped,
            mfe,
            mae,
            outcome_r
        FROM feature_observations
        ORDER BY timestamp
    """

    df = pd.read_sql_query(query, connection)

    print()
    print("=" * 110)
    print("ITRF OUT-OF-SAMPLE VALIDATION v0.1")
    print("=" * 110)

    if df.empty:
        print("No observations available.")
        print("=" * 110)
        return

    # --------------------------------------------------------
    # Chronological 70 / 30 split
    #
    # IMPORTANT:
    # The validation period is completely unseen.
    # No parameter optimization is performed here.
    # --------------------------------------------------------

    split_timestamp = pd.Timestamp("2025-07-28 17:00:00")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    development_df = df.loc[
        df["timestamp"] < split_timestamp
    ].copy()

    validation_df = df.loc[
        df["timestamp"] >= split_timestamp
    ].copy()

    print()
    print("DEVELOPMENT PERIOD")
    print(
        development_df["timestamp"].min(),
        "->",
        development_df["timestamp"].max()
    )
    print(
        "Observations:",
        len(development_df)
    )

    print()
    print("VALIDATION PERIOD")
    print(
        validation_df["timestamp"].min(),
        "->",
        validation_df["timestamp"].max()
    )
    print(
        "Observations:",
        len(validation_df)
    )

    if validation_df.empty:
        print("Validation dataset is empty.")
        print("=" * 110)
        return

    # --------------------------------------------------------
    # Recreate the FIXED v0.4 components
    #
    # These thresholds are NOT optimized on validation data.
    # --------------------------------------------------------

    validation_df["component_delta"] = 0

    validation_df.loc[
        (
            (
                (validation_df["direction"] == "LONG")
                & (validation_df["delta_zscore"] >= 1.0)
            )
            |
            (
                (validation_df["direction"] == "SHORT")
                & (validation_df["delta_zscore"] <= -1.0)
            )
        ),
        "component_delta"
    ] = 1

    validation_df["component_delta_change"] = 0

    validation_df.loc[
        (
            (
                (validation_df["direction"] == "LONG")
                & (validation_df["delta_change"] > 0)
            )
            |
            (
                (validation_df["direction"] == "SHORT")
                & (validation_df["delta_change"] < 0)
            )
        ),
        "component_delta_change"
    ] = 1

    validation_df["component_momentum"] = 0

    validation_df.loc[
        (
            (
                (validation_df["direction"] == "LONG")
                & (validation_df["momentum_atr"] > 0)
            )
            |
            (
                (validation_df["direction"] == "SHORT")
                & (validation_df["momentum_atr"] < 0)
            )
        ),
        "component_momentum"
    ] = 1

    validation_df["component_sweep"] = 0

    validation_df.loc[
        (
            (
                (validation_df["direction"] == "LONG")
                & (validation_df["bullish_sweep"] == 1)
            )
            |
            (
                (validation_df["direction"] == "SHORT")
                & (validation_df["bearish_sweep"] == 1)
            )
        ),
        "component_sweep"
    ] = 1

    validation_df["component_volume"] = 0

    validation_df.loc[
        (
            (validation_df["relative_volume"] >= 1.5)
            & (validation_df["component_delta"] == 1)
        ),
        "component_volume"
    ] = 1

    validation_df["component_efficiency"] = 0

    validation_df.loc[
        validation_df["candle_efficiency"] >= 0.60,
        "component_efficiency"
    ] = 1

    components = [
        ("Delta", "component_delta"),
        ("Delta Change", "component_delta_change"),
        ("Momentum", "component_momentum"),
        ("Relative Volume", "component_volume"),
        ("Candle Efficiency", "component_efficiency"),
    ]

    # --------------------------------------------------------
    # Validation component analysis
    # --------------------------------------------------------

    print()
    print("-" * 110)
    print("VALIDATION COMPONENT PERFORMANCE")
    print("-" * 110)

    rows = []

    for name, column in components:

        subset = validation_df.loc[
            validation_df[column] == 1
        ].copy()

        if len(subset) < 30:
            continue

        rows.append(
            {
                "component": name,
                "samples": len(subset),
                "1R_%": round(
                    subset["hit_1r"].mean() * 100,
                    2
                ),
                "2R_%": round(
                    subset["hit_2r"].mean() * 100,
                    2
                ),
                "3R_%": round(
                    subset["hit_3r"].mean() * 100,
                    2
                ),
                "stopped_%": round(
                    subset["stopped"].mean() * 100,
                    2
                ),
                "average_R": round(
                    subset["outcome_r"].mean(),
                    3
                ),
                "average_MFE": round(
                    subset["mfe"].mean(),
                    3
                ),
                "average_MAE": round(
                    subset["mae"].mean(),
                    3
                ),
            }
        )

    report = pd.DataFrame(rows)

    if report.empty:
        print(
            "No validation components met "
            "the minimum sample requirement."
        )
    else:
        print(
            report.to_string(index=False)
        )

    # --------------------------------------------------------
    # Validation direction analysis
    # --------------------------------------------------------

    print()
    print("-" * 110)
    print("VALIDATION LONG / SHORT PERFORMANCE")
    print("-" * 110)

    direction_rows = []

    for direction in ["LONG", "SHORT"]:

        subset = validation_df.loc[
            validation_df["direction"] == direction
        ].copy()

        if len(subset) < 30:
            continue

        direction_rows.append(
            {
                "direction": direction,
                "samples": len(subset),
                "1R_%": round(
                    subset["hit_1r"].mean() * 100,
                    2
                ),
                "2R_%": round(
                    subset["hit_2r"].mean() * 100,
                    2
                ),
                "3R_%": round(
                    subset["hit_3r"].mean() * 100,
                    2
                ),
                "stopped_%": round(
                    subset["stopped"].mean() * 100,
                    2
                ),
                "average_R": round(
                    subset["outcome_r"].mean(),
                    3
                ),
            }
        )

    direction_report = pd.DataFrame(
        direction_rows
    )

    if direction_report.empty:
        print(
            "No validation direction "
            "conditions met minimum sample."
        )
    else:
        print(
            direction_report.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Validation baseline
    # --------------------------------------------------------

    print()
    print("-" * 110)
    print("VALIDATION BASELINE")
    print("-" * 110)

    print(
        "Observations:",
        len(validation_df)
    )

    print(
        "Average R:",
        round(
            validation_df["outcome_r"].mean(),
            3
        )
    )

    print(
        "1R:",
        round(
            validation_df["hit_1r"].mean() * 100,
            2
        ),
        "%"
    )

    print(
        "2R:",
        round(
            validation_df["hit_2r"].mean() * 100,
            2
        ),
        "%"
    )

    print(
        "3R:",
        round(
            validation_df["hit_3r"].mean() * 100,
            2
        ),
        "%"
    )

    print(
        "Stopped:",
        round(
            validation_df["stopped"].mean() * 100,
            2
        ),
        "%"
    )

    print("=" * 110)

    print(
        "Validation thresholds are fixed from "
        "the existing v0.4 research."
    )

    print(
        "No validation-period parameter optimization "
        "has been performed."
    )

    print(
        "This is a chronological out-of-sample test."
    )

    print(
        "Delta remains a volume-based proxy, "
        "not true bid/ask flow."
    )

    print("=" * 110)
# ============================================================
# OUT-OF-SAMPLE VALIDATION v0.5
# ============================================================

def generate_oos_validation_analysis(connection, oos_start):

    query = """
        SELECT
            timestamp,
            direction,
            relative_volume,
            delta_zscore,
            delta_change,
            momentum_atr,
            candle_efficiency,
            bullish_sweep,
            bearish_sweep,
            hit_1r,
            hit_2r,
            hit_3r,
            stopped,
            mfe,
            mae,
            outcome_r
        FROM feature_observations
        ORDER BY timestamp
    """

    df = pd.read_sql_query(query, connection)

    print()
    print("=" * 110)
    print("ITRF OUT-OF-SAMPLE VALIDATION v0.5")
    print("=" * 110)

    if df.empty:
        print("No observations available.")
        print("=" * 110)
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    split_time = resolve_oos_split(df, oos_start, "OOS validation analysis")
    if split_time is None:
        print("=" * 110)
        return

    train_df = df.loc[
        df["timestamp"] < split_time
    ].copy()

    oos_df = df.loc[
        df["timestamp"] >= split_time
    ].copy()

    print()
    print("CHRONOLOGICAL SPLIT")
    print("-" * 110)

    print(
        "Training observations:",
        len(train_df)
    )

    print(
        "OOS observations:",
        len(oos_df)
    )

    print(
        "Training period:",
        train_df["timestamp"].min(),
        "to",
        train_df["timestamp"].max()
    )

    print(
        "OOS period:",
        oos_df["timestamp"].min(),
        "to",
        oos_df["timestamp"].max()
    )

    # --------------------------------------------------------
    # Fixed research conditions
    #
    # IMPORTANT:
    # These rules are inherited from previous research.
    # They are NOT optimized using OOS data.
    # --------------------------------------------------------

    def add_conditions(data):

        data = data.copy()

        data["component_delta"] = 0

        data.loc[
            (
                ((data["direction"] == "LONG") &
                 (data["delta_zscore"] >= 1.0))
                |
                ((data["direction"] == "SHORT") &
                 (data["delta_zscore"] <= -1.0))
            ),
            "component_delta"
        ] = 1

        data["component_delta_change"] = 0

        data.loc[
            (
                ((data["direction"] == "LONG") &
                 (data["delta_change"] > 0))
                |
                ((data["direction"] == "SHORT") &
                 (data["delta_change"] < 0))
            ),
            "component_delta_change"
        ] = 1

        data["component_momentum"] = 0

        data.loc[
            (
                ((data["direction"] == "LONG") &
                 (data["momentum_atr"] > 0))
                |
                ((data["direction"] == "SHORT") &
                 (data["momentum_atr"] < 0))
            ),
            "component_momentum"
        ] = 1

        data["component_efficiency"] = 0

        data.loc[
            data["candle_efficiency"] >= 0.60,
            "component_efficiency"
        ] = 1

        data["component_sweep"] = 0

        data.loc[
            (
                (
                    (data["direction"] == "LONG")
                    & (data["bullish_sweep"] == 1)
                )
                |
                (
                    (data["direction"] == "SHORT")
                    & (data["bearish_sweep"] == 1)
                )
            ),
            "component_sweep"
        ] = 1

        data["component_volume"] = 0


        data.loc[
            (
                (data["relative_volume"] >= 1.5)
                &
                (data["component_delta"] == 1)
            ),
            "component_volume"
        ] = 1

        # Fixed order-flow score.
        #
        # Score is calculated relative to setup direction.
        #
        # Delta aligned      = +2
        # Delta change       = +1
        # Momentum aligned   = +1
        # Candle efficiency  = +1
        # Relative volume    = +1
        # Liquidity sweep    = +1
        #
        # Existing research score is preserved.
        data["order_flow_score"] = 0

        data.loc[
            data["component_delta"] == 1,
            "order_flow_score"
        ] += 2

        data.loc[
            data["component_delta_change"] == 1,
            "order_flow_score"
        ] += 1

        data.loc[
            data["component_momentum"] == 1,
            "order_flow_score"
        ] += 1
        data.loc[
            data["component_sweep"] == 1,
            "order_flow_score"
        ] += 1

        data.loc[
            data["component_efficiency"] == 1,
            "order_flow_score"
        ] += 1

        data.loc[
            data["component_volume"] == 1,
            "order_flow_score"
        ] += 1

        return data

    train_df = add_conditions(train_df)
    oos_df = add_conditions(oos_df)

    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------

    def summarize(data):

        if data.empty:
            return {
                "samples": 0,
                "1R_%": 0,
                "2R_%": 0,
                "3R_%": 0,
                "stopped_%": 0,
                "average_R": 0,
                "average_MFE": 0,
                "average_MAE": 0,
            }

        return {
            "samples": len(data),
            "1R_%": round(
                data["hit_1r"].mean() * 100,
                2
            ),
            "2R_%": round(
                data["hit_2r"].mean() * 100,
                2
            ),
            "3R_%": round(
                data["hit_3r"].mean() * 100,
                2
            ),
            "stopped_%": round(
                data["stopped"].mean() * 100,
                2
            ),
            "average_R": round(
                data["outcome_r"].mean(),
                3
            ),
            "average_MFE": round(
                data["mfe"].mean(),
                3
            ),
            "average_MAE": round(
                data["mae"].mean(),
                3
            ),
        }

    # --------------------------------------------------------
    # Baseline comparison
    # --------------------------------------------------------

    print()
    print("-" * 110)
    print("TRAIN vs OOS BASELINE")
    print("-" * 110)

    train_summary = summarize(train_df)
    oos_summary = summarize(oos_df)

    baseline_report = pd.DataFrame(
        [
            {
                "period": "TRAIN",
                **train_summary,
            },
            {
                "period": "OOS",
                **oos_summary,
            },
        ]
    )

    print(
        baseline_report.to_string(index=False)
    )

    # --------------------------------------------------------
    # Direction comparison
    # --------------------------------------------------------

    print()
    print("-" * 110)
    print("OOS DIRECTION ANALYSIS")
    print("-" * 110)

    direction_rows = []

    for direction in ["LONG", "SHORT"]:

        subset = oos_df.loc[
            oos_df["direction"] == direction
        ]

        if len(subset) < 50:
            continue

        direction_rows.append(
            {
                "direction": direction,
                **summarize(subset),
            }
        )

    direction_report = pd.DataFrame(
        direction_rows
    )

    if direction_report.empty:
        print(
            "No direction met minimum sample requirement."
        )
    else:
        print(
            direction_report.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # OOS score analysis
    # --------------------------------------------------------

    print()
    print("-" * 110)
    print("OOS ORDER FLOW SCORE ANALYSIS")
    print("-" * 110)

    score_conditions = [
        ("Score -1 to 0", -1, 0),
        ("Score 1 to 2", 1, 2),
        ("Score 3 to 4", 3, 4),
        ("Score 5 to 7", 5, 7),
    ]

    score_rows = []

    for name, low, high in score_conditions:

        subset = oos_df.loc[
            (oos_df["order_flow_score"] >= low)
            &
            (oos_df["order_flow_score"] <= high)
        ]

        if len(subset) < 50:
            continue

        score_rows.append(
            {
                "score_condition": name,
                **summarize(subset),
            }
        )

    score_report = pd.DataFrame(
        score_rows
    )

    if score_report.empty:
        print(
            "No score condition met minimum sample requirement."
        )
    else:
        print(
            score_report.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # OOS direction + score
    # --------------------------------------------------------

    print()
    print("-" * 110)
    print("OOS DIRECTION-SPECIFIC ORDER FLOW SCORE")
    print("-" * 110)

    direction_score_rows = []

    for direction in ["LONG", "SHORT"]:

        direction_df = oos_df.loc[
            oos_df["direction"] == direction
        ]

        for name, low, high in score_conditions:

            subset = direction_df.loc[
                (direction_df["order_flow_score"] >= low)
                &
                (direction_df["order_flow_score"] <= high)
            ]

            if len(subset) < 50:
                continue

            direction_score_rows.append(
                {
                    "direction": direction,
                    "score_condition": name,
                    **summarize(subset),
                }
            )

    direction_score_report = pd.DataFrame(
        direction_score_rows
    )

    if direction_score_report.empty:
        print(
            "No direction-score condition met minimum sample."
        )
    else:
        print(
            direction_score_report.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # OOS component analysis
    # --------------------------------------------------------

    print()
    print("-" * 110)
    print("OOS COMPONENT VALIDATION")
    print("-" * 110)

    components = [
        ("Delta", "component_delta"),
        ("Delta Change", "component_delta_change"),
        ("Momentum", "component_momentum"),
        ("Relative Volume", "component_volume"),
        ("Candle Efficiency", "component_efficiency"),
    ]

    component_rows = []

    for name, column in components:

        subset = oos_df.loc[
            oos_df[column] == 1
        ]

        if len(subset) < 50:
            continue

        component_rows.append(
            {
                "component": name,
                **summarize(subset),
            }
        )

    component_report = pd.DataFrame(
        component_rows
    )

    if component_report.empty:
        print(
            "No component met minimum sample requirement."
        )
    else:
        print(
            component_report.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Train vs OOS performance decay
    # --------------------------------------------------------

    print()
    print("-" * 110)
    print("TRAIN → OOS PERFORMANCE CHANGE")
    print("-" * 110)

    train_average_r = train_df["outcome_r"].mean()
    oos_average_r = oos_df["outcome_r"].mean()

    print(
        "TRAIN average R:",
        round(train_average_r, 3)
    )

    print(
        "OOS average R:",
        round(oos_average_r, 3)
    )

    print(
        "OOS minus TRAIN:",
        round(
            oos_average_r - train_average_r,
            3
        )
    )

    print()
    print(
        "OOS rules were frozen before validation."
    )

    print(
        "No OOS parameter optimization was performed."
    )

    print(
        "Results remain research evidence, not proof of profitability."
    )

    print(
        "Delta remains a volume-based proxy, not true bid/ask flow."
    )

    print("=" * 110)

# ============================================================
# OOS ROBUSTNESS ANALYSIS v0.8
# ============================================================

def generate_oos_robustness_analysis(connection, oos_start):

    BOOTSTRAP_ITERATIONS = 2000
    RANDOM_SEED = 42

    query = """
        SELECT
            timestamp,
            direction,
            relative_volume,
            delta_zscore,
            delta_change,
            momentum_atr,
            candle_efficiency,
            bullish_sweep,
            bearish_sweep, 
            outcome_r
        FROM feature_observations
        ORDER BY timestamp
    """

    df = pd.read_sql_query(
        query,
        connection
    )

    print()
    print("=" * 110)
    print("ITRF OOS ROBUSTNESS ANALYSIS v0.8")
    print("=" * 110)

    if df.empty:
        print("No observations available.")
        print("=" * 110)
        return

    # Frozen score.# --------------------------------------------------------
    # Prepare timestamps
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    split_time = resolve_oos_split(df, oos_start, "OOS robustness analysis")
    if split_time is None:
        print("=" * 110)
        return

    # --------------------------------------------------------
    # Frozen chronological OOS period
    # --------------------------------------------------------

    oos_df = df[
        df["timestamp"] >= split_time
    ].copy()

    if oos_df.empty:
        print("No OOS observations available.")
        print("=" * 110)
        return
    # Use the same canonical definition as the frozen v0.5 report. Earlier
    # v0.8 code accidentally loosened two thresholds and omitted the sweep.
    oos_df = add_frozen_order_flow_score(oos_df)

    # --------------------------------------------------------
    # Frozen Score 5-7 condition
    # --------------------------------------------------------

    score_5_7 = oos_df[
        oos_df["order_flow_score"].between(
            5,
            7
        )
    ].copy()

    if score_5_7.empty:
        print("No OOS Score 5-7 observations available.")
        print("=" * 110)
        return

    # --------------------------------------------------------
    # Helper function
    # --------------------------------------------------------

    def calculate_statistics(
        series,
        bootstrap_iterations,
        random_seed,
    ):

        values = (
            pd.to_numeric(
                series,
                errors="coerce"
            )
            .dropna()
            .to_numpy()
        )

        if len(values) == 0:
            return {
                "samples": 0,
                "mean_r": np.nan,
                "median_r": np.nan,
                "std_r": np.nan,
                "positive_r_percent": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
            }

        mean_r = float(
            np.mean(values)
        )

        median_r = float(
            np.median(values)
        )

        std_r = float(
            np.std(
                values,
                ddof=1
            )
        ) if len(values) > 1 else 0.0

        positive_r_percent = float(
            np.mean(values > 0) * 100
        )

        # ----------------------------------------------------
        # Bootstrap confidence interval
        # ----------------------------------------------------

        rng = np.random.default_rng(
            random_seed
        )

        bootstrap_means = np.empty(
            bootstrap_iterations
        )

        for i in range(
            bootstrap_iterations
        ):

            sample = rng.choice(
                values,
                size=len(values),
                replace=True,
            )

            bootstrap_means[i] = np.mean(
                sample
            )

        ci_lower = float(
            np.percentile(
                bootstrap_means,
                2.5
            )
        )

        ci_upper = float(
            np.percentile(
                bootstrap_means,
                97.5
            )
        )

        return {
            "samples": len(values),
            "mean_r": mean_r,
            "median_r": median_r,
            "std_r": std_r,
            "positive_r_percent": positive_r_percent,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    # ========================================================
    # OOS BASELINE
    # ========================================================

    baseline_stats = calculate_statistics(
        oos_df["outcome_r"],
        BOOTSTRAP_ITERATIONS,
        RANDOM_SEED,
    )

    # ========================================================
    # OOS SCORE 5-7
    # ========================================================

    score_stats = calculate_statistics(
        score_5_7["outcome_r"],
        BOOTSTRAP_ITERATIONS,
        RANDOM_SEED,
    )

    print()
    print("-" * 110)
    print("OOS BASELINE vs SCORE 5-7")
    print("-" * 110)

    print(
        f"Baseline samples: "
        f"{baseline_stats['samples']:,}"
    )

    print(
        f"Baseline average R: "
        f"{baseline_stats['mean_r']:.3f}"
    )

    print(
        f"Score 5-7 samples: "
        f"{score_stats['samples']:,}"
    )

    print(
        f"Score 5-7 average R: "
        f"{score_stats['mean_r']:.3f}"
    )

    print(
        f"Score 5-7 median R: "
        f"{score_stats['median_r']:.3f}"
    )

    print(
        f"Score 5-7 standard deviation: "
        f"{score_stats['std_r']:.3f}"
    )

    print(
        f"Score 5-7 positive outcome R: "
        f"{score_stats['positive_r_percent']:.2f}%"
    )

    print(
        f"Score 5-7 bootstrap 95% CI: "
        f"{score_stats['ci_lower']:.3f} "
        f"to "
        f"{score_stats['ci_upper']:.3f}"
    )

    print(
        f"Score 5-7 minus OOS baseline: "
        f"{score_stats['mean_r'] - baseline_stats['mean_r']:.3f}"
    )

    # ========================================================
    # DIRECTION ROBUSTNESS
    # ========================================================

    print()
    print("-" * 110)
    print("SCORE 5-7 DIRECTION ROBUSTNESS")
    print("-" * 110)

    for direction in [
        "LONG",
        "SHORT",
    ]:

        direction_df = score_5_7[
            score_5_7["direction"] == direction
        ].copy()

        if len(direction_df) == 0:
            print()
            print(
                f"{direction}: "
                "No observations."
            )
            continue

        direction_stats = calculate_statistics(
            direction_df["outcome_r"],
            BOOTSTRAP_ITERATIONS,
            RANDOM_SEED,
        )

        print()
        print(
            f"{direction}"
        )

        print(
            f"  Samples: "
            f"{direction_stats['samples']:,}"
        )

        print(
            f"  Average R: "
            f"{direction_stats['mean_r']:.3f}"
        )

        print(
            f"  Median R: "
            f"{direction_stats['median_r']:.3f}"
        )

        print(
            f"  Positive R: "
            f"{direction_stats['positive_r_percent']:.2f}%"
        )

        print(
            f"  Bootstrap 95% CI: "
            f"{direction_stats['ci_lower']:.3f} "
            f"to "
            f"{direction_stats['ci_upper']:.3f}"
        )

    # ========================================================
    # OOS PERIOD ROBUSTNESS
    # ========================================================

    print()
    print("-" * 110)
    print("SCORE 5-7 OOS PERIOD ROBUSTNESS")
    print("-" * 110)

    period_size = len(
        score_5_7
    ) // 4

    if period_size < 30:

        print(
            "Insufficient observations for "
            "four-period robustness analysis."
        )

    else:

        score_5_7 = score_5_7.sort_values(
            "timestamp"
        ).reset_index(
            drop=True
        )

        for period_number in range(
            1,
            5
        ):

            start_index = (
                (period_number - 1)
                * period_size
            )

            if period_number == 4:

                end_index = len(
                    score_5_7
                )

            else:

                end_index = (
                    period_number
                    * period_size
                )

            period_df = score_5_7.iloc[
                start_index:end_index
            ]

            period_stats = calculate_statistics(
                period_df["outcome_r"],
                BOOTSTRAP_ITERATIONS,
                RANDOM_SEED + period_number,
            )

            print()
            print(
                f"OOS Period {period_number}"
            )

            print(
                f"  Samples: "
                f"{period_stats['samples']:,}"
            )

            print(
                f"  Average R: "
                f"{period_stats['mean_r']:.3f}"
            )

            print(
                f"  Median R: "
                f"{period_stats['median_r']:.3f}"
            )

            print(
                f"  Positive R: "
                f"{period_stats['positive_r_percent']:.2f}%"
            )

            print(
                f"  Bootstrap 95% CI: "
                f"{period_stats['ci_lower']:.3f} "
                f"to "
                f"{period_stats['ci_upper']:.3f}"
            )

    # ========================================================
    # RESEARCH INTERPRETATION
    # ========================================================

    print()
    print("-" * 110)
    print("ROBUSTNESS CHECK")
    print("-" * 110)

    print(
        "Score definition remains frozen from v0.5."
    )

    print(
        "OOS start remains fixed at:",
        split_time
    )

    print(
        "No parameter optimization was performed."
    )

    print(
        "Bootstrap seed:",
        RANDOM_SEED
    )

    print(
        "Bootstrap iterations:",
        BOOTSTRAP_ITERATIONS
    )

    print(
        "This analysis measures statistical uncertainty "
        "around observed OOS performance."
    )

    print(
        "A confidence interval crossing zero means "
        "positive average R is not statistically robust "
        "at the 95% bootstrap level."
    )

    print(
        "Results remain research evidence, "
        "not proof of profitability."
    )

    print(
        "Delta remains a volume-based proxy, "
        "not true bid/ask flow."
    )

    print("=" * 110)

# ============================================================
# MAIN
# ============================================================

def _parse_arguments():
    """Keep v0.8 as the default while exposing explicit v0.9 research runs."""
    parser = argparse.ArgumentParser(
        description="ITRF-XAUUSD research engine; v0.8 baseline plus optional v0.9 studies."
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=DATA_FILE,
        help="OHLCV CSV input. Defaults to the preserved XAUUSD baseline file.",
    )
    parser.add_argument(
        "--database-file",
        type=Path,
        default=DATABASE_FILE,
        help="SQLite output for the v0.8 baseline observations.",
    )
    parser.add_argument(
        "--v09-context-database-file",
        type=Path,
        default=PROJECT_ROOT / "database" / "itrf_v09_research.db",
        help="Separate SQLite output for the v0.9 context study.",
    )
    parser.add_argument(
        "--v09-trade-management-database-file",
        type=Path,
        default=PROJECT_ROOT / "database" / "itrf_v09_trade_management.db",
        help="Separate SQLite output for the v0.9 exit-model study.",
    )
    parser.add_argument(
        "--v091-database-file",
        type=Path,
        default=PROJECT_ROOT / "database" / "itrf_v091_research.db",
        help="Separate SQLite output for the v0.9.1 causal context study.",
    )
    parser.add_argument(
        "--v092-database-file",
        type=Path,
        default=PROJECT_ROOT / "database" / "itrf_v092_orderflow.db",
        help="Separate SQLite output for the v0.9.2 order-flow study.",
    )
    parser.add_argument(
        "--v092-orderflow-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "databento" / "GC_front_orderflow_15m.csv",
        help="Historical CME Gold order-flow validation features.",
    )
    parser.add_argument(
        "--v09-context",
        action="store_true",
        help="Run the isolated v0.9 market-context study after the v0.8 baseline.",
    )
    parser.add_argument(
        "--v09-trade-management",
        action="store_true",
        help="Run the fixed, non-optimized v0.9 exit-model comparison after v0.8.",
    )
    parser.add_argument(
        "--v09-all",
        action="store_true",
        help="Run both v0.9 studies after the v0.8 baseline.",
    )
    parser.add_argument(
        "--v091-context",
        action="store_true",
        help="Run the pre-registered v0.9.1 causal context diagnostic after v0.8.",
    )
    parser.add_argument(
        "--v092-orderflow",
        action="store_true",
        help="Run the pre-registered historical v0.9.2 order-flow diagnostic after v0.8.",
    )
    parser.add_argument(
        "--oos-start",
        default=DEFAULT_OOS_START,
        help=(
            "Frozen chronological OOS start (YYYY-MM-DD or timestamp). "
            "The engine safely skips OOS reports when it is outside the data range."
        ),
    )
    return parser.parse_args()


def main():

    arguments = _parse_arguments()

    print()
    print("=" * 75)
    print("ITRF-XAUUSD RESEARCH ENGINE v0.2")
    print("=" * 75)

    print(f"Loading market data from {arguments.data_file}...")

    df = load_market_data(arguments.data_file)

    print(
        f"Loaded {len(df):,} candles."
    )

    print("Calculating research features...")

    df = create_features(df)

    print("Building historical observations...")

    arguments.database_file.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(
        arguments.database_file
    ) as connection:

        count = build_database(
            df,
            connection,
        )

        print(
            f"Historical setups created: {count:,}"
        )

        generate_report(
            connection
        )

        generate_feature_analysis(
            connection
        )

        generate_order_flow_score_analysis(
            connection
        )
        generate_component_attribution_analysis(
            connection
        )

        generate_oos_validation_analysis(
            connection,
            arguments.oos_start,
        )
        generate_oos_stability_analysis(
            connection,
            arguments.oos_start,
        )
        generate_oos_regime_score_analysis(
            connection,
            arguments.oos_start,
        )

        generate_oos_robustness_analysis(
            connection,
            arguments.oos_start,
        )
        generate_regime_analysis(
            connection
        )

    print()
    print(
        "Research engine completed."
    )

    print(
        f"Database: {arguments.database_file}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Delta values are volume-based proxies."
    )

    print(
        "They are NOT true bid/ask buyer-seller counts."
    )

    if arguments.v09_context or arguments.v09_all:
        from run_v09_research import main as run_v09_context_study

        print("\nRunning v0.9 market-context study in its separate database...")
        run_v09_context_study(
            data_file=arguments.data_file,
            database_file=arguments.v09_context_database_file,
            oos_start=arguments.oos_start,
        )

    if arguments.v09_trade_management or arguments.v09_all:
        from run_v09_trade_management import main as run_v09_trade_management_study

        print("\nRunning v0.9 fixed trade-management comparison in its separate database...")
        run_v09_trade_management_study(
            data_file=arguments.data_file,
            database_file=arguments.v09_trade_management_database_file,
            oos_start=arguments.oos_start,
        )

    if arguments.v091_context:
        from run_v091_research import main as run_v091_context_study

        print("\nRunning v0.9.1 causal context diagnostic in its separate database...")
        run_v091_context_study(
            data_file=arguments.data_file,
            database_file=arguments.v091_database_file,
        )

    if arguments.v092_orderflow:
        from run_v092_orderflow_research import main as run_v092_orderflow_study

        print("\nRunning v0.9.2 historical order-flow diagnostic in its separate database...")
        run_v092_orderflow_study(
            data_file=arguments.data_file,
            orderflow_file=arguments.v092_orderflow_file,
            database_file=arguments.v092_database_file,
        )

# ============================================================
# OOS STABILITY ANALYSIS v0.6
# ============================================================

def generate_oos_stability_analysis(connection, oos_start):

    query = """
        SELECT
            timestamp,
            direction,
            relative_volume,
            delta_zscore,
            delta_change,
            momentum_atr,
            candle_efficiency,
            bullish_sweep,
            bearish_sweep,
            hit_1r,
            hit_2r,
            hit_3r,
            stopped,
            mfe,
            mae,
            outcome_r
        FROM feature_observations
        ORDER BY timestamp
    """

    df = pd.read_sql_query(query, connection)

    print()
    print("=" * 110)
    print("ITRF OOS STABILITY ANALYSIS v0.6")
    print("=" * 110)

    if df.empty:
        print("No observations available.")
        print("=" * 110)
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # --------------------------------------------------------
    # Keep only the frozen OOS period.
    # --------------------------------------------------------

    split_time = resolve_oos_split(df, oos_start, "OOS stability analysis")
    if split_time is None:
        print("=" * 110)
        return

    oos_df = df.loc[
        df["timestamp"] >= split_time
    ].copy()

    if oos_df.empty:
        print("No OOS observations available.")
        print("=" * 110)
        return

    # --------------------------------------------------------
    # Rebuild the frozen research score.
    # --------------------------------------------------------

    oos_df["component_delta"] = 0

    oos_df.loc[
        (
            ((oos_df["direction"] == "LONG") &
             (oos_df["delta_zscore"] >= 1.0))
            |
            ((oos_df["direction"] == "SHORT") &
             (oos_df["delta_zscore"] <= -1.0))
        ),
        "component_delta"
    ] = 1

    oos_df["component_delta_change"] = 0

    oos_df.loc[
        (
            ((oos_df["direction"] == "LONG") &
             (oos_df["delta_change"] > 0))
            |
            ((oos_df["direction"] == "SHORT") &
             (oos_df["delta_change"] < 0))
        ),
        "component_delta_change"
    ] = 1

    oos_df["component_momentum"] = 0

    oos_df.loc[
        (
            ((oos_df["direction"] == "LONG") &
             (oos_df["momentum_atr"] > 0))
            |
            ((oos_df["direction"] == "SHORT") &
             (oos_df["momentum_atr"] < 0))
        ),
        "component_momentum"
    ] = 1

    oos_df["component_efficiency"] = 0

    oos_df.loc[
        oos_df["candle_efficiency"] >= 0.60,
        "component_efficiency"
    ] = 1

    oos_df["component_sweep"] = 0

    oos_df.loc[
        (
            (
                (oos_df["direction"] == "LONG")
                & (oos_df["bullish_sweep"] == 1)
            )
            |
            (
                (oos_df["direction"] == "SHORT")
                & (oos_df["bearish_sweep"] == 1)
            )
        ),
        "component_sweep"
    ] = 1

    oos_df["component_volume"] = 0

    oos_df.loc[
        (
            (oos_df["relative_volume"] >= 1.5)
            &
            (oos_df["component_delta"] == 1)
        ),
        "component_volume"
    ] = 1

    # --------------------------------------------------------
    # Frozen score.
    # --------------------------------------------------------

    oos_df["order_flow_score"] = (
        (oos_df["component_delta"] * 2)
        + oos_df["component_delta_change"]
        + oos_df["component_momentum"]
        + oos_df["component_efficiency"]
        + oos_df["component_volume"]
        + oos_df["component_sweep"]
    )

    # --------------------------------------------------------
    # Divide OOS chronologically into four equal periods.
    # --------------------------------------------------------

    oos_df = oos_df.sort_values("timestamp").reset_index(drop=True)

    oos_df["oos_period"] = pd.qcut(
        oos_df.index,
        q=4,
        labels=[
            "OOS Period 1",
            "OOS Period 2",
            "OOS Period 3",
            "OOS Period 4"
        ]
    )

    # --------------------------------------------------------
    # Summary helper.
    # --------------------------------------------------------

    def summarize(data):

        if data.empty:
            return {
                "samples": 0,
                "1R_%": 0,
                "2R_%": 0,
                "3R_%": 0,
                "stopped_%": 0,
                "average_R": 0,
                "average_MFE": 0,
                "average_MAE": 0,
            }

        return {
            "samples": len(data),
            "1R_%": round(
                data["hit_1r"].mean() * 100,
                2
            ),
            "2R_%": round(
                data["hit_2r"].mean() * 100,
                2
            ),
            "3R_%": round(
                data["hit_3r"].mean() * 100,
                2
            ),
            "stopped_%": round(
                data["stopped"].mean() * 100,
                2
            ),
            "average_R": round(
                data["outcome_r"].mean(),
                3
            ),
            "average_MFE": round(
                data["mfe"].mean(),
                3
            ),
            "average_MAE": round(
                data["mae"].mean(),
                3
            ),
        }

    # --------------------------------------------------------
    # PERIOD STABILITY
    # --------------------------------------------------------

    print()
    print("-" * 110)
    print("OOS PERIOD STABILITY")
    print("-" * 110)

    period_records = []

    for period, period_df in oos_df.groupby(
        "oos_period",
        observed=True
    ):

        result = summarize(period_df)

        period_records.append(
            {
                "period": period,
                **result
            }
        )

    period_table = pd.DataFrame(period_records)

    print(
        period_table.to_string(index=False)
    )

    # --------------------------------------------------------
    # SCORE 3-4 VS SCORE 5-7 BY PERIOD
    # --------------------------------------------------------

    print()
    print("-" * 110)
    print("SCORE STABILITY BY OOS PERIOD")
    print("-" * 110)

    score_records = []

    for period, period_df in oos_df.groupby(
        "oos_period",
        observed=True
    ):

        for score_name, score_condition in [
            (
                "Score 3 to 4",
                period_df["order_flow_score"].between(3, 4)
            ),
            (
                "Score 5 to 7",
                period_df["order_flow_score"].between(5, 7)
            ),
        ]:

            subset = period_df.loc[
                score_condition
            ]

            result = summarize(subset)

            score_records.append(
                {
                    "period": period,
                    "score_condition": score_name,
                    **result
                }
            )

    score_table = pd.DataFrame(score_records)

    print(
        score_table.to_string(index=False)
    )

    # --------------------------------------------------------
    # DIRECTIONAL SCORE 5-7 STABILITY
    # --------------------------------------------------------

    print()
    print("-" * 110)
    print("LONG / SHORT SCORE 5-7 STABILITY")
    print("-" * 110)

    direction_records = []

    for period, period_df in oos_df.groupby(
        "oos_period",
        observed=True
    ):

        for direction in ["LONG", "SHORT"]:

            subset = period_df.loc[
                (period_df["direction"] == direction)
                &
                (period_df["order_flow_score"].between(5, 7))
            ]

            result = summarize(subset)

            direction_records.append(
                {
                    "period": period,
                    "direction": direction,
                    **result
                }
            )

    direction_table = pd.DataFrame(
        direction_records
    )

    print(
        direction_table.to_string(index=False)
    )

    # --------------------------------------------------------
    # STABILITY CHECK
    # --------------------------------------------------------

    score_57 = oos_df.loc[
        oos_df["order_flow_score"].between(5, 7)
    ]

    period_average_r = []

    for period, period_df in score_57.groupby(
        "oos_period",
        observed=True
    ):

        if not period_df.empty:
            period_average_r.append(
                period_df["outcome_r"].mean()
            )

    positive_periods = sum(
        value > 0
        for value in period_average_r
    )

    print()
    print("-" * 110)
    print("STABILITY CHECK")
    print("-" * 110)

    print(
        "Score 5-7 positive periods:",
        positive_periods,
        "/",
        len(period_average_r)
    )

    if period_average_r:
        print(
            "Score 5-7 period average R range:",
            round(min(period_average_r), 3),
            "to",
            round(max(period_average_r), 3)
        )

    print()
    print(
        "Rules remain frozen from v0.5."
    )

    print(
        "No parameter optimization was performed."
    )

    print(
        "This analysis tests temporal stability only."
    )

    print(
        "Results remain research evidence, "
        "not proof of profitability."
    )

    print(
        "Delta remains a volume-based proxy, "
        "not true bid/ask flow."
    )

    print("=" * 110)

if __name__ == "__main__":
    main()
