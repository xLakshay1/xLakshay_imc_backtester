from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import itertools
import io
import json
import math
import re
import sys
import time
import types
from contextlib import redirect_stdout
from types import FunctionType

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


ROUND1_DATA_DIR = Path("/Users/lakshaykumar/Downloads/ROUND1")
ROUND2_DATA_DIR = Path("/Users/lakshaykumar/Downloads/ROUND2")
DEFAULT_DATA_DIR = ROUND2_DATA_DIR if ROUND2_DATA_DIR.exists() else ROUND1_DATA_DIR
DATASET_DIRS = {
    "Round 1": (ROUND1_DATA_DIR,),
    "Round 2": (ROUND2_DATA_DIR,),
    "Rounds 1 + 2": (ROUND1_DATA_DIR, ROUND2_DATA_DIR),
}
DEFAULT_STRATEGY_ROOT = Path("/Users/lakshaykumar/Documents/Playground/imc-prosperity-4-fresh")
BID_COLOR = "#2364AA"
ASK_COLOR = "#D62828"
BUY_TRADE_COLOR = "#2A9D8F"
SELL_TRADE_COLOR = "#8A1C7C"
UNKNOWN_TRADE_COLOR = "#4A4A4A"

DEFAULT_CUSTOM_STRATEGY = """\
def strategy(row, state):
    product = row["product"]
    position = state["positions"].get(product, 0)
    params = state["params"]

    # Return a list of orders. price=None means cross visible book like a market order.
    # Examples:
    # return [{"side": "buy", "quantity": 10, "price": row["ask_price_1"]}]
    # return [{"side": "sell", "quantity": position, "price": row["bid_price_1"]}]
    return []
"""


@dataclass(frozen=True)
class DataFiles:
    price_paths: tuple[str, ...]
    trade_paths: tuple[str, ...]


def file_day(path: str) -> int | None:
    match = re.search(r"_day_(-?\d+)\.csv$", Path(path).name)
    return int(match.group(1)) if match else None


def file_round(path: str) -> int | None:
    match = re.search(r"_round_(\d+)_day_", Path(path).name)
    return int(match.group(1)) if match else None


@st.cache_data(show_spinner=False)
def discover_files(folder: str) -> DataFiles:
    root = Path(folder).expanduser()
    return DataFiles(
        price_paths=tuple(str(path) for path in sorted(root.glob("prices_round_*_day_*.csv"))),
        trade_paths=tuple(str(path) for path in sorted(root.glob("trades_round_*_day_*.csv"))),
    )


@st.cache_data(show_spinner=False)
def discover_dataset_files(folders: tuple[str, ...]) -> DataFiles:
    price_paths: list[str] = []
    trade_paths: list[str] = []
    for folder in folders:
        files = discover_files(folder)
        price_paths.extend(files.price_paths)
        trade_paths.extend(files.trade_paths)
    return DataFiles(tuple(sorted(price_paths)), tuple(sorted(trade_paths)))


def apply_session_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["source_round"] = pd.to_numeric(output["source_round"], errors="coerce")
    output["source_day"] = pd.to_numeric(output["source_day"], errors="coerce")
    session_keys = (
        output[["source_round", "source_day"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["source_round", "source_day"])
    )
    has_multiple_rounds = int(session_keys["source_round"].nunique()) > 1 if not session_keys.empty else False
    if has_multiple_rounds:
        mapping = {
            (int(row.source_round), int(row.source_day)): index
            for index, row in enumerate(session_keys.itertuples(index=False))
        }
        output["day"] = [
            mapping.get((int(round_id), int(day_id)), int(day_id))
            for round_id, day_id in zip(output["source_round"], output["source_day"])
        ]
        output["session_label"] = [
            f"R{int(round_id)} {day_label(int(day_id))}"
            for round_id, day_id in zip(output["source_round"], output["source_day"])
        ]
    else:
        output["day"] = pd.to_numeric(output["day"], errors="coerce")
        output["session_label"] = output["day"].map(lambda value: day_label(int(value)) if not pd.isna(value) else "")
    return output


@st.cache_data(show_spinner=False)
def load_prices(paths: tuple[str, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, sep=";")
        frame["source_round"] = file_round(path)
        frame["source_day"] = file_day(path)
        frame["source_file"] = Path(path).name
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, ignore_index=True)
    numeric_columns = [
        "day",
        "timestamp",
        "mid_price",
        "profit_and_loss",
        "source_round",
        "source_day",
        "bid_price_1",
        "bid_volume_1",
        "bid_price_2",
        "bid_volume_2",
        "bid_price_3",
        "bid_volume_3",
        "ask_price_1",
        "ask_volume_1",
        "ask_price_2",
        "ask_volume_2",
        "ask_price_3",
        "ask_volume_3",
    ]
    for column in numeric_columns:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    prices = apply_session_columns(prices)

    has_two_sided_top = prices["bid_price_1"].notna() & prices["ask_price_1"].notna()
    prices["csv_mid_price"] = prices["mid_price"]
    prices["mid_price"] = float("nan")
    prices.loc[has_two_sided_top, "mid_price"] = (
        prices.loc[has_two_sided_top, "bid_price_1"]
        + prices.loc[has_two_sided_top, "ask_price_1"]
    ) / 2.0
    prices["mid_price"] = pd.to_numeric(prices["mid_price"], errors="coerce")

    total_top_volume = prices["bid_volume_1"].abs() + prices["ask_volume_1"].abs()
    prices["spread"] = prices["ask_price_1"] - prices["bid_price_1"]
    prices["microprice"] = (
        prices["ask_price_1"] * prices["bid_volume_1"].abs()
        + prices["bid_price_1"] * prices["ask_volume_1"].abs()
    ) / total_top_volume.replace(0, pd.NA)
    prices["top_imbalance"] = (
        prices["bid_volume_1"].abs() - prices["ask_volume_1"].abs()
    ) / total_top_volume.replace(0, pd.NA)

    weighted_bid = pd.Series(0.0, index=prices.index)
    weighted_ask = pd.Series(0.0, index=prices.index)
    for level in range(1, 4):
        weighted_bid += (
            prices[f"bid_price_{level}"].fillna(0) * prices[f"bid_volume_{level}"].abs().fillna(0)
        )
        weighted_ask += (
            prices[f"ask_price_{level}"].fillna(0) * prices[f"ask_volume_{level}"].abs().fillna(0)
        )
    all_depth = sum(
        prices[f"bid_volume_{level}"].abs().fillna(0) + prices[f"ask_volume_{level}"].abs().fillna(0)
        for level in range(1, 4)
    )
    prices["depth_vwap"] = (weighted_bid + weighted_ask) / all_depth.replace(0, pd.NA)

    def smooth_depth_vwap(series: pd.Series) -> pd.Series:
        cleaned = pd.to_numeric(series, errors="coerce").replace(0, pd.NA)
        cleaned = cleaned.interpolate(limit_direction="both")
        median_line = cleaned.rolling(90, min_periods=15, center=True).median()
        trend = median_line.ewm(span=80, adjust=False, min_periods=10).mean()
        return trend.fillna(median_line).fillna(cleaned)

    def live_smooth_depth_vwap(series: pd.Series) -> pd.Series:
        cleaned = pd.to_numeric(series, errors="coerce").replace(0, pd.NA)
        cleaned = cleaned.ffill()
        median_line = cleaned.rolling(90, min_periods=15).median()
        trend = median_line.ewm(span=80, adjust=False, min_periods=10).mean()
        return trend.fillna(median_line).fillna(cleaned)

    prices["depth_vwap_trend"] = prices.groupby(["product", "day"], sort=False)[
        "depth_vwap"
    ].transform(smooth_depth_vwap)
    prices["depth_vwap_live_trend"] = prices.groupby(["product", "day"], sort=False)[
        "depth_vwap"
    ].transform(live_smooth_depth_vwap)

    prices["pepper_linear_trend"] = float("nan")
    pepper_mask = prices["product"].eq("INTARIAN_PEPPER_ROOT")
    for (_, day), group in prices[pepper_mask].groupby(["product", "day"], sort=False):
        valid = group.dropna(subset=["timestamp", "mid_price"]).sort_values("timestamp")
        if valid.empty:
            continue
        warmup = valid.head(25)
        raw_intercept = (warmup["mid_price"] - 0.001 * warmup["timestamp"]).median()
        detected_intercept = round(float(raw_intercept) / 1000.0) * 1000.0
        prices.loc[group.index, "pepper_linear_trend"] = (
            detected_intercept + 0.001 * prices.loc[group.index, "timestamp"]
        )

    def live_density_fair(series: pd.Series) -> pd.Series:
        cleaned = pd.to_numeric(series, errors="coerce").ffill()
        rounded = (cleaned * 2).round() / 2

        def densest_level(values) -> float:
            values = pd.Series(values).dropna()
            if values.empty:
                return float("nan")
            counts = values.value_counts()
            top_count = counts.iloc[0]
            candidates = pd.Series(counts[counts == top_count].index, dtype="float64")
            return float(candidates.median())

        fair = rounded.rolling(240, min_periods=30).apply(densest_level, raw=True)
        return fair.fillna(cleaned)

    def smooth_wall_mid(series: pd.Series) -> pd.Series:
        cleaned = pd.to_numeric(series, errors="coerce").ffill()
        median_line = cleaned.rolling(301, min_periods=35, center=True).median()
        smooth_line = median_line.rolling(81, min_periods=20, center=True).mean()
        return smooth_line.fillna(median_line).fillna(cleaned)

    prices["osmium_density_fair"] = float("nan")
    osmium_mask = prices["product"].eq("ASH_COATED_OSMIUM")
    prices.loc[osmium_mask, "osmium_density_fair"] = prices[osmium_mask].groupby(
        ["product", "day"], sort=False
    )["mid_price"].transform(live_density_fair)
    prices["osmium_wall_mid_smooth"] = float("nan")
    prices.loc[osmium_mask, "osmium_wall_mid_smooth"] = prices[osmium_mask].groupby(
        ["product", "day"], sort=False
    )["mid_price"].transform(smooth_wall_mid)

    grouped = prices.groupby(["product", "day"], sort=False)["mid_price"]
    prices["day_mid_mean"] = grouped.transform("mean")
    prices["day_mid_std"] = grouped.transform("std").replace(0, pd.NA)
    prices["rolling_mid_mean"] = grouped.transform(
        lambda series: series.rolling(120, min_periods=20).mean()
    )
    prices["rolling_mid_std"] = grouped.transform(
        lambda series: series.rolling(120, min_periods=20).std()
    ).replace(0, pd.NA)
    prices["rolling_mid_mean"] = prices["rolling_mid_mean"].fillna(prices["day_mid_mean"])
    prices["rolling_mid_std"] = prices["rolling_mid_std"].fillna(prices["day_mid_std"])
    return prices.sort_values(["day", "product", "timestamp"])


@st.cache_data(show_spinner=False)
def load_trades(paths: tuple[str, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, sep=";")
        frame["source_round"] = file_round(path)
        frame["source_day"] = file_day(path)
        frame["day"] = file_day(path)
        frame["source_file"] = Path(path).name
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    trades = pd.concat(frames, ignore_index=True)
    for column in ("day", "timestamp", "price", "quantity", "source_round", "source_day"):
        trades[column] = pd.to_numeric(trades[column], errors="coerce")
    trades = apply_session_columns(trades)
    for column in ("buyer", "seller", "symbol", "currency"):
        trades[column] = trades[column].fillna("").astype(str)
    return trades.sort_values(["day", "symbol", "timestamp"])


def nearest_book_for_trades(product_prices: pd.DataFrame, product_trades: pd.DataFrame) -> pd.DataFrame:
    if product_trades.empty:
        return product_trades

    book_columns = [
        "timestamp",
        "bid_price_1",
        "ask_price_1",
        "mid_price",
        "microprice",
        "depth_vwap",
        "depth_vwap_trend",
        "depth_vwap_live_trend",
        "day_mid_mean",
        "day_mid_std",
        "rolling_mid_mean",
        "rolling_mid_std",
    ]
    book = product_prices[book_columns].sort_values("timestamp")
    trades = product_trades.sort_values("timestamp")
    merged = pd.merge_asof(trades, book, on="timestamp", direction="nearest")

    merged["aggressor"] = "unknown"
    merged.loc[merged["price"] >= merged["ask_price_1"], "aggressor"] = "buyer taker"
    merged.loc[merged["price"] <= merged["bid_price_1"], "aggressor"] = "seller taker"
    return merged


def normalize_label(option: str) -> str:
    labels = {
        "None": "price",
        "mid_price": "price - mid_price",
        "microprice": "price - microprice",
        "depth_vwap": "price - depth_vwap",
        "depth_vwap_trend": "price - smoothed depth_vwap",
        "depth_vwap_live_trend": "price - live-safe depth_vwap trend",
        "pepper_linear_trend": "price - Pepper linear trend",
        "osmium_density_fair": "price - Osmium density fair",
        "osmium_wall_mid_smooth": "price - smooth Osmium wall mid",
        "Day volatility z-score": "(price - day mean) / day volatility",
        "Rolling volatility z-score": "(price - rolling mean) / rolling volatility",
    }
    return labels.get(option, "price")


def normalization_parts(frame: pd.DataFrame, option: str) -> tuple[pd.Series | None, pd.Series | None]:
    if option == "None":
        return None, None
    if option in {
        "mid_price",
        "microprice",
        "depth_vwap",
        "depth_vwap_trend",
        "depth_vwap_live_trend",
        "pepper_linear_trend",
        "osmium_density_fair",
        "osmium_wall_mid_smooth",
    }:
        return frame[option], None
    if option == "Day volatility z-score":
        return frame["day_mid_mean"], frame["day_mid_std"]
    if option == "Rolling volatility z-score":
        return frame["rolling_mid_mean"], frame["rolling_mid_std"]
    return None, None


def normalized(series: pd.Series, base: pd.Series | None, scale: pd.Series | None = None) -> pd.Series:
    if base is None:
        return series
    values = series - base
    if scale is None:
        return values
    return values / scale.replace(0, pd.NA)


def install_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding: 1.35rem 1rem 1.75rem;
            max-width: none;
        }
        h1, h2, h3 {
            margin: 0.25rem 0 0.55rem;
        }
        div[data-testid="stVerticalBlock"] {
            gap: 1.05rem;
        }
        div[data-testid="stHorizontalBlock"] {
            gap: 0.95rem;
        }
        div[data-testid="stElementContainer"] {
            margin-bottom: 0.5rem;
        }
        div[data-testid="stPlotlyChart"] {
            padding-top: 0.9rem;
            padding-bottom: 0.75rem;
        }
        div[data-testid="stExpander"] {
            margin-top: 0.55rem;
            margin-bottom: 0.65rem;
        }
        div[data-testid="stTabs"] {
            margin-top: 1rem;
        }
        div[data-testid="stTabs"] [role="tabpanel"] {
            padding-top: 1.45rem;
        }
        div[data-testid="stMetric"] {
            background: #f7f7f7;
            border: 1px solid #d2d2d2;
            padding: 0.45rem 0.55rem;
        }
        .panel-title {
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0;
            margin: 0.15rem 0 0.45rem;
            text-transform: uppercase;
        }
        .right-panel {
            border: 2px solid #111;
            padding: 0.55rem 0.65rem;
            background: #fbfbfb;
            min-height: 3.1rem;
            position: relative;
        }
        .number-badge {
            align-items: center;
            background: #050505;
            border-radius: 50%;
            color: #fff;
            display: inline-flex;
            font-size: 1rem;
            font-weight: 900;
            height: 2.2rem;
            justify-content: center;
            line-height: 1;
            margin-right: 0.35rem;
            width: 2.2rem;
        }
        .tiny-note {
            color: #444;
            font-size: 0.72rem;
            line-height: 1.42;
        }
        .trade-key {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.24rem;
            margin-top: 0.45rem;
        }
        .trade-chip {
            color: #111;
            font-size: 0.72rem;
            font-weight: 900;
            padding: 0.22rem 0.28rem;
            text-align: center;
        }
        .main-label {
            align-items: center;
            background: #050505;
            border: 2px solid #111;
            color: white;
            display: inline-flex;
            font-size: 1.05rem;
            font-weight: 900;
            gap: 0.35rem;
            margin-bottom: 0.45rem;
            padding: 0.42rem 0.65rem;
        }
        .mc-title {
            align-items: center;
            background: #111418;
            border: 1px solid #2a2f37;
            color: #d9dce2;
            display: flex;
            gap: 0.55rem;
            font-size: 1.05rem;
            font-weight: 800;
            margin: 1.4rem 0 0.85rem;
            padding: 0.75rem 0.9rem;
        }
        .mc-panel {
            background: #15181d;
            border: 1px solid #303640;
            color: #d9dce2;
            margin-bottom: 0.9rem;
            padding: 0.9rem 1rem;
        }
        .mc-heading {
            color: #c6c9cf;
            font-size: 1.25rem;
            font-weight: 800;
            letter-spacing: 0;
            margin-bottom: 0.15rem;
        }
        .mc-subtle {
            color: #8f96a3;
            font-size: 0.82rem;
        }
        .mc-chip {
            background: #19314d;
            border-radius: 8px;
            color: #8dbbf4;
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 800;
            margin-left: 0.45rem;
            padding: 0.22rem 0.5rem;
            text-transform: uppercase;
        }
        .mc-card {
            background: #171b21;
            border: 1px solid #303640;
            color: #d9dce2;
            margin-bottom: 0.85rem;
            min-height: 5.6rem;
            padding: 0.9rem 1rem;
        }
        .mc-card-label {
            color: #b8bcc4;
            font-size: 0.88rem;
            font-weight: 800;
            margin-bottom: 0.8rem;
        }
        .mc-card-value {
            color: #e4e7ed;
            font-size: 1.65rem;
            font-weight: 900;
            line-height: 1;
        }
        .mc-card-caption {
            color: #8f96a3;
            font-size: 0.78rem;
            margin-top: 0.35rem;
        }
        .mc-section-title {
            color: #c6c9cf;
            font-size: 0.98rem;
            font-weight: 800;
            margin: 0.1rem 0 0.65rem;
        }
        .mc-table {
            border-collapse: collapse;
            color: #d3d6dc;
            font-size: 0.82rem;
            width: 100%;
        }
        .mc-table th {
            background: #1b2027;
            border: 1px solid #303640;
            color: #c9cdd5;
            font-weight: 800;
            padding: 0.48rem 0.55rem;
            text-align: left;
        }
        .mc-table td {
            background: #15181d;
            border: 1px solid #303640;
            padding: 0.45rem 0.55rem;
        }
        .mc-table tr:nth-child(even) td {
            background: #20242b;
        }
        .mc-note {
            color: #8f96a3;
            font-size: 0.78rem;
            line-height: 1.45;
            margin-top: 0.5rem;
        }
        .mc-terminal {
            background: #050607;
            border: 1px solid #303640;
            color: #d6d6d6;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 0.9rem;
            line-height: 1.32;
            margin: 0;
            overflow-x: auto;
            padding: 0.9rem 1rem;
            white-space: pre;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge(number: int, title: str) -> None:
    st.markdown(
        f'<div class="panel-title"><span class="number-badge">{number}</span>{title}</div>',
        unsafe_allow_html=True,
    )


def make_book_points(
    prices: pd.DataFrame,
    base: pd.Series | None,
    scale: pd.Series | None,
    min_qty: int,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for side, color in (("bid", BID_COLOR), ("ask", ASK_COLOR)):
        for level in range(1, 4):
            price_col = f"{side}_price_{level}"
            volume_col = f"{side}_volume_{level}"
            level_frame = prices[["timestamp", price_col, volume_col]].copy()
            level_frame["side"] = side
            level_frame["level"] = level
            level_frame["color"] = color
            level_frame["price"] = level_frame[price_col]
            level_frame["volume"] = level_frame[volume_col].abs()
            level_frame["plot_price"] = normalized(level_frame["price"], base, scale)
            level_frame = level_frame.dropna(subset=["price", "volume", "plot_price"])
            level_frame = level_frame[level_frame["volume"] >= min_qty]
            pieces.append(level_frame[["timestamp", "side", "level", "price", "plot_price", "volume", "color"]])
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def downsample_by_timestamp(frame: pd.DataFrame, max_timestamps: int) -> pd.DataFrame:
    timestamps = frame["timestamp"].dropna().sort_values().unique()
    if len(timestamps) <= max_timestamps:
        return frame
    stride = max(1, len(timestamps) // max_timestamps)
    keep = set(timestamps[::stride])
    return frame[frame["timestamp"].isin(keep)]


def trade_price_path(visible_trades: pd.DataFrame) -> pd.DataFrame:
    if visible_trades.empty:
        return pd.DataFrame()

    frame = visible_trades.copy()
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce").fillna(0).abs()
    frame = frame[frame["quantity"] > 0]
    if frame.empty:
        return pd.DataFrame()

    frame["price_x_qty"] = frame["price"] * frame["quantity"]
    frame["plot_x_qty"] = frame["plot_price"] * frame["quantity"]
    grouped = frame.groupby("timestamp", as_index=False).agg(
        trade_price=("price_x_qty", "sum"),
        plot_price=("plot_x_qty", "sum"),
        quantity=("quantity", "sum"),
        trades=("price", "count"),
    )
    grouped["trade_price"] = grouped["trade_price"] / grouped["quantity"].replace(0, pd.NA)
    grouped["plot_price"] = grouped["plot_price"] / grouped["quantity"].replace(0, pd.NA)
    grouped = grouped.dropna(subset=["trade_price", "plot_price"]).sort_values("timestamp")
    grouped["trade_live_trend"] = (
        grouped["plot_price"]
        .rolling(20, min_periods=3)
        .median()
        .ewm(span=18, adjust=False, min_periods=3)
        .mean()
    )
    grouped["trade_live_trend"] = grouped["trade_live_trend"].fillna(grouped["plot_price"])
    return grouped


def main_order_book_chart(
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    overlays: list[str],
    normalize_by: str,
    show_bids: bool,
    show_asks: bool,
    show_public_trades: bool,
    show_maker_mirror: bool,
    show_trade_path: bool,
    show_trade_trend: bool,
    min_book_qty: int,
    trade_qty_range: tuple[int, int],
    max_timestamps: int,
    focus_timestamp: int,
) -> go.Figure:
    base, scale = normalization_parts(prices, normalize_by)
    book_points = make_book_points(prices, base, scale, min_book_qty)
    book_points = downsample_by_timestamp(book_points, max_timestamps)

    fig = go.Figure()
    side_options = []
    if show_bids:
        side_options.append("bid")
    if show_asks:
        side_options.append("ask")

    for side in side_options:
        side_frame = book_points[book_points["side"] == side]
        for level in range(1, 4):
            level_frame = side_frame[side_frame["level"] == level]
            if level_frame.empty:
                continue
            fig.add_trace(
                go.Scattergl(
                    x=level_frame["timestamp"],
                    y=level_frame["plot_price"],
                    mode="markers",
                    name=f"{side} L{level}",
                    marker={
                        "color": ASK_COLOR if side == "ask" else BID_COLOR,
                        "size": (level_frame["volume"].clip(1, 60) ** 0.5) * 2.2,
                        "opacity": 0.46 if level > 1 else 0.78,
                        "symbol": "circle",
                    },
                    customdata=level_frame[["price", "volume"]],
                    hovertemplate=(
                        "t=%{x}<br>"
                        f"{side} L{level}<br>"
                        "price=%{customdata[0]}<br>"
                        "qty=%{customdata[1]}<extra></extra>"
                    ),
                )
            )

    for overlay in overlays:
            if overlay in prices:
                y_values = normalized(prices[overlay], base, scale)
                line_style = {"width": 1.4}
                if overlay == "depth_vwap_trend":
                    line_style = {"width": 2.8, "color": "#ffffff"}
                elif overlay == "depth_vwap_live_trend":
                    line_style = {"width": 2.8, "color": "#ffe45c", "dash": "dash"}
                elif overlay == "pepper_linear_trend":
                    line_style = {"width": 4.2, "color": "#8B0000", "dash": "longdash"}
                elif overlay == "osmium_density_fair":
                    line_style = {"width": 2.0, "color": "#8B0000", "dash": "dot"}
                elif overlay == "osmium_wall_mid_smooth":
                    line_style = {"width": 4.8, "color": "#5A0000"}
                fig.add_trace(
                    go.Scattergl(
                        x=prices["timestamp"],
                        y=y_values,
                    mode="lines",
                    line=line_style,
                    connectgaps=False,
                    name=overlay,
                    hovertemplate=f"t=%{{x}}<br>{overlay}=%{{y}}<extra></extra>",
                )
            )

    if (show_public_trades or show_trade_path or show_trade_trend) and not trades.empty:
        low, high = trade_qty_range
        visible_trades = trades[(trades["quantity"] >= low) & (trades["quantity"] <= high)].copy()
        if not visible_trades.empty:
            trade_base, trade_scale = normalization_parts(visible_trades, normalize_by)
            visible_trades["plot_price"] = normalized(visible_trades["price"], trade_base, trade_scale)
            path = trade_price_path(visible_trades)

            if show_trade_path and not path.empty:
                fig.add_trace(
                    go.Scattergl(
                        x=path["timestamp"],
                        y=path["plot_price"],
                        mode="lines+markers",
                        name="actual trade price path",
                        line={"color": "#00d9ff", "width": 2.0},
                        marker={"color": "#00d9ff", "size": 4.5, "opacity": 0.82},
                        customdata=path[["trade_price", "quantity", "trades"]],
                        hovertemplate=(
                            "t=%{x}<br>"
                            "trade VWAP=%{customdata[0]:.2f}<br>"
                            "total qty=%{customdata[1]}<br>"
                            "trades=%{customdata[2]}<extra></extra>"
                        ),
                    )
                )

            if show_trade_trend and not path.empty:
                fig.add_trace(
                    go.Scattergl(
                        x=path["timestamp"],
                        y=path["trade_live_trend"],
                        mode="lines",
                        name="actual trade live trend",
                        line={"color": "#ffe45c", "width": 2.7, "dash": "dot"},
                        customdata=path[["trade_price", "quantity", "trades"]],
                        hovertemplate=(
                            "t=%{x}<br>"
                            "trade trend=%{y:.2f}<br>"
                            "trade VWAP=%{customdata[0]:.2f}<br>"
                            "total qty=%{customdata[1]}<br>"
                            "trades=%{customdata[2]}<extra></extra>"
                        ),
                    )
                )

            if show_public_trades:
                trade_groups = [
                    ("buyer taker", "triangle-up", BUY_TRADE_COLOR, "public buy taker"),
                    ("seller taker", "triangle-down", SELL_TRADE_COLOR, "public sell taker"),
                    ("unknown", "diamond", UNKNOWN_TRADE_COLOR, "public unknown"),
                ]
                for aggressor, symbol, color, label in trade_groups:
                    group = visible_trades[visible_trades["aggressor"] == aggressor]
                    if group.empty:
                        continue
                    fig.add_trace(
                        go.Scattergl(
                            x=group["timestamp"],
                            y=group["plot_price"],
                            mode="markers",
                            name=label,
                            marker={
                                "symbol": symbol,
                                "color": color,
                                "size": (group["quantity"].clip(1, 80) ** 0.5) * 4.2,
                                "line": {"width": 1, "color": "#FFFFFF"},
                            },
                            customdata=group[["price", "quantity", "buyer", "seller", "aggressor"]],
                            hovertemplate=(
                                "t=%{x}<br>"
                                "trade price=%{customdata[0]}<br>"
                                "qty=%{customdata[1]}<br>"
                                "buyer=%{customdata[2]}<br>"
                                "seller=%{customdata[3]}<br>"
                                "%{customdata[4]}<extra></extra>"
                            ),
                        )
                    )

            if show_public_trades and show_maker_mirror:
                maker = visible_trades[visible_trades["aggressor"] != "unknown"].copy()
                if not maker.empty:
                    maker["maker_side"] = maker["aggressor"].map(
                        {"buyer taker": "sell maker", "seller taker": "buy maker"}
                    )
                    fig.add_trace(
                        go.Scattergl(
                            x=maker["timestamp"],
                            y=maker["plot_price"],
                            mode="markers",
                            name="inferred maker side",
                            marker={
                                "symbol": "square-open",
                                "color": "#111111",
                                "size": (maker["quantity"].clip(1, 80) ** 0.5) * 4.8,
                                "line": {"width": 1.5},
                            },
                            customdata=maker[["price", "quantity", "maker_side"]],
                            hovertemplate=(
                                "t=%{x}<br>"
                                "trade price=%{customdata[0]}<br>"
                                "qty=%{customdata[1]}<br>"
                                "%{customdata[2]}<extra></extra>"
                            ),
                        )
                    )

    fig.update_layout(
        height=625,
        margin={"l": 22, "r": 18, "t": 78, "b": 42},
        xaxis_title="timestamp",
        yaxis_title=normalize_label(normalize_by),
        hovermode="closest",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0},
        template="plotly_white",
        font={"size": 10},
        xaxis={
            "showgrid": True,
            "gridcolor": "#e6e6e6",
            "minor": {"showgrid": True, "gridcolor": "#f2f2f2"},
        },
        yaxis={
            "showgrid": True,
            "gridcolor": "#e2e2e2",
            "minor": {"showgrid": True, "gridcolor": "#f1f1f1"},
        },
        shapes=[
            {
                "type": "line",
                "xref": "x",
                "yref": "paper",
                "x0": focus_timestamp,
                "x1": focus_timestamp,
                "y0": 0,
                "y1": 1,
                "line": {"color": "#111", "width": 2},
            }
        ],
        annotations=[
            {
                "text": "1",
                "x": 0.50,
                "y": 0.44,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"color": "white", "size": 16},
                "bgcolor": "#000",
                "bordercolor": "#000",
                "borderpad": 8,
                "opacity": 0.92,
            }
        ],
    )
    return fig


def small_line_chart(frame: pd.DataFrame, columns: list[str], title: str, focus_timestamp: int) -> go.Figure:
    fig = go.Figure()
    for column in columns:
        fig.add_trace(
            go.Scattergl(
                x=frame["timestamp"],
                y=frame[column],
                mode="lines",
                name=column,
                line={"color": "#111", "width": 2} if len(columns) == 1 else None,
            )
        )
    fig.update_layout(
        title=title,
        height=145,
        margin={"l": 22, "r": 18, "t": 62, "b": 30},
        template="plotly_white",
        legend={"orientation": "h", "font": {"size": 9}},
        font={"size": 9},
        xaxis={"showgrid": True, "gridcolor": "#eeeeee", "title": None},
        yaxis={"showgrid": True, "gridcolor": "#e5e5e5", "title": None},
        shapes=[
            {
                "type": "line",
                "xref": "x",
                "yref": "paper",
                "x0": focus_timestamp,
                "x1": focus_timestamp,
                "y0": 0,
                "y1": 1,
                "line": {"color": "#111", "width": 1.5},
            }
        ],
    )
    return fig


def position_placeholder(frame: pd.DataFrame, focus_timestamp: int) -> go.Figure:
    position = pd.Series(0, index=frame.index)
    plot_frame = frame.assign(position=position)
    return small_line_chart(plot_frame, ["position"], "Position Panel", focus_timestamp)


def focused_snapshot(prices: pd.DataFrame, trades: pd.DataFrame, timestamp: int) -> tuple[pd.Series, pd.DataFrame]:
    if prices.empty:
        return pd.Series(dtype=object), trades.head(0)
    nearest_index = (prices["timestamp"] - timestamp).abs().idxmin()
    snapshot = prices.loc[nearest_index]
    nearby_trades = trades[(trades["timestamp"] - timestamp).abs() <= 200]
    return snapshot, nearby_trades


def book_levels(row: pd.Series, side: str) -> list[tuple[int, int]]:
    levels: list[tuple[int, int]] = []
    for level in range(1, 4):
        price = row.get(f"{side}_price_{level}")
        volume = row.get(f"{side}_volume_{level}")
        if pd.isna(price) or pd.isna(volume) or int(abs(volume)) <= 0:
            continue
        levels.append((int(price), int(abs(volume))))

    if side == "ask":
        return sorted(levels)
    return sorted(levels, reverse=True)


def take_from_asks(row: pd.Series, desired_quantity: int, max_price: float | None = None) -> tuple[int, float]:
    filled = 0
    cash_spent = 0.0
    for price, available in book_levels(row, "ask"):
        if max_price is not None and price > max_price:
            break
        quantity = min(max(0, desired_quantity - filled), available)
        if quantity <= 0:
            break
        filled += quantity
        cash_spent += quantity * price
    return filled, cash_spent


def hit_bids(row: pd.Series, desired_quantity: int, min_price: float | None = None) -> tuple[int, float]:
    filled = 0
    cash_received = 0.0
    for price, available in book_levels(row, "bid"):
        if min_price is not None and price < min_price:
            break
        quantity = min(max(0, desired_quantity - filled), available)
        if quantity <= 0:
            break
        filled += quantity
        cash_received += quantity * price
    return filled, cash_received


def empty_backtest_state(products: list[str]) -> dict[str, dict[str, float]]:
    return {
        product: {
            "position": 0.0,
            "cash": 0.0,
            "ema": float("nan"),
            "buy_qty": 0.0,
            "sell_qty": 0.0,
            "buy_value": 0.0,
            "sell_value": 0.0,
            "fills": 0.0,
        }
        for product in products
    }


def record_fill(state: dict[str, float], side: str, quantity: int, value: float) -> None:
    if quantity <= 0:
        return
    state["fills"] += 1
    if side == "buy":
        state["position"] += quantity
        state["cash"] -= value
        state["buy_qty"] += quantity
        state["buy_value"] += value
    else:
        state["position"] -= quantity
        state["cash"] += value
        state["sell_qty"] += quantity
        state["sell_value"] += value


def compile_custom_strategy(code: str) -> FunctionType:
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "pow": pow,
        "range": range,
        "round": round,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    namespace: dict[str, object] = {
        "__builtins__": safe_builtins,
        "math": __import__("math"),
    }
    exec(code, namespace)
    strategy = namespace.get("strategy")
    if not callable(strategy):
        raise ValueError('Custom strategy must define `def strategy(row, state): ...`')
    return strategy


def prosperity_datamodel():
    backtester_root = DEFAULT_STRATEGY_ROOT / "backtester"
    if str(backtester_root) not in sys.path:
        sys.path.insert(0, str(backtester_root))
    if str(DEFAULT_STRATEGY_ROOT) not in sys.path:
        sys.path.insert(0, str(DEFAULT_STRATEGY_ROOT))

    try:
        from prosperity4mcbt import datamodel as datamodel_module
    except ModuleNotFoundError:
        jsonpickle_stub = types.ModuleType("jsonpickle")
        jsonpickle_stub.encode = lambda value: json.dumps(
            value,
            default=lambda inner: getattr(inner, "__dict__", str(inner)),
        )
        sys.modules.setdefault("jsonpickle", jsonpickle_stub)
        from prosperity4mcbt import datamodel as datamodel_module

    sys.modules.setdefault("datamodel", datamodel_module)
    sys.modules.setdefault("prosperity3bt.datamodel", datamodel_module)
    sys.modules.setdefault("prosperity4mcbt.datamodel", datamodel_module)
    return datamodel_module


def compile_trader_code(code: str):
    datamodel_module = prosperity_datamodel()
    namespace: dict[str, object] = {
        "__name__": "dashboard_submitted_strategy",
        "math": math,
        "json": json,
    }
    exec(code, namespace)
    trader_cls = namespace.get("Trader")
    if trader_cls is None:
        raise ValueError("Submitted code must define a Trader class.")
    return trader_cls(), datamodel_module


def custom_state_view(
    states: dict[str, dict[str, float]],
    memory: dict[str, object],
    last_mid: dict[str, float],
    params: dict[str, float | int | str],
) -> dict[str, object]:
    return {
        "positions": {product: int(state["position"]) for product, state in states.items()},
        "cash": {product: float(state["cash"]) for product, state in states.items()},
        "last_mid": dict(last_mid),
        "memory": memory,
        "params": params,
    }


def apply_custom_orders(
    row: pd.Series,
    state: dict[str, float],
    orders: object,
    max_position: int,
    fill_model: str,
) -> None:
    if orders is None:
        return
    if isinstance(orders, dict):
        orders = [orders]
    if not isinstance(orders, list):
        raise ValueError("Custom strategy must return a list of order dicts.")

    for order in orders:
        if not isinstance(order, dict):
            raise ValueError("Each custom order must be a dict.")
        side = str(order.get("side", "")).lower()
        quantity = int(order.get("quantity", 0))
        price = order.get("price")
        price_limit = None if price is None else float(price)

        if quantity <= 0:
            continue
        if side == "buy":
            quantity = min(quantity, max_position - int(state["position"]))
            if fill_model == "Pessimistic maker-only" and price_limit is not None and price_limit < float(row.get("ask_price_1", float("inf"))):
                filled, spent = 0, 0.0
            else:
                filled, spent = take_from_asks(row, quantity, max_price=price_limit)
            record_fill(state, "buy", filled, spent)
        elif side == "sell":
            quantity = min(quantity, max_position + int(state["position"]))
            if fill_model == "Pessimistic maker-only" and price_limit is not None and price_limit > float(row.get("bid_price_1", float("-inf"))):
                filled, received = 0, 0.0
            else:
                filled, received = hit_bids(row, quantity, min_price=price_limit)
            record_fill(state, "sell", filled, received)
        else:
            raise ValueError("Custom order side must be `buy` or `sell`.")


def row_order_depth(row: pd.Series, datamodel_module):
    depth = datamodel_module.OrderDepth()
    for price, volume in book_levels(row, "bid"):
        depth.buy_orders[int(price)] = int(volume)
    for price, volume in book_levels(row, "ask"):
        depth.sell_orders[int(price)] = -int(volume)
    return depth


def build_trading_state(
    timestamp: int,
    stamp_frame: pd.DataFrame,
    products: list[str],
    states: dict[str, dict[str, float]],
    trader_data: str,
    datamodel_module,
):
    listings = {
        product: datamodel_module.Listing(product, product, "XIRECS")
        for product in products
    }
    order_depths = {}
    for product in products:
        product_rows = stamp_frame[stamp_frame["product"] == product]
        if product_rows.empty:
            continue
        order_depths[product] = row_order_depth(product_rows.iloc[0], datamodel_module)
    own_trades = {product: [] for product in products}
    market_trades = {product: [] for product in products}
    position = {product: int(states[product]["position"]) for product in products}
    observations = datamodel_module.Observation({}, {})
    return datamodel_module.TradingState(
        traderData=trader_data,
        timestamp=int(timestamp),
        listings=listings,
        order_depths=order_depths,
        own_trades=own_trades,
        market_trades=market_trades,
        position=position,
        observations=observations,
    )


def apply_trader_orders(
    stamp_frame: pd.DataFrame,
    states: dict[str, dict[str, float]],
    orders_by_product: object,
    max_position: int,
    fill_model: str,
) -> None:
    if not isinstance(orders_by_product, dict):
        return
    rows_by_product = {
        str(row["product"]): row
        for _, row in stamp_frame.iterrows()
    }
    for product, orders in orders_by_product.items():
        if product not in states or product not in rows_by_product:
            continue
        if orders is None:
            continue
        for order in orders:
            quantity = int(getattr(order, "quantity", 0))
            price = int(getattr(order, "price", 0))
            if quantity > 0:
                apply_custom_orders(
                    rows_by_product[product],
                    states[product],
                    [{"side": "buy", "quantity": quantity, "price": price}],
                    max_position,
                    fill_model,
                )
            elif quantity < 0:
                apply_custom_orders(
                    rows_by_product[product],
                    states[product],
                    [{"side": "sell", "quantity": abs(quantity), "price": price}],
                    max_position,
                    fill_model,
                )


def portfolio_rows(
    states: dict[str, dict[str, float]],
    last_mid: dict[str, float],
    day: int,
    timestamp: int,
    global_timestamp: int,
    completed_pnl_by_product: dict[str, float],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    current_total = 0.0
    for product, state in states.items():
        mid = last_mid.get(product)
        marked_pnl = float(state["cash"])
        if mid is not None and not pd.isna(mid):
            marked_pnl += float(state["position"]) * float(mid)
        marked_pnl += float(completed_pnl_by_product.get(product, 0.0))
        current_total += marked_pnl
        rows.append(
            {
                "day": day,
                "timestamp": timestamp,
                "global_timestamp": global_timestamp,
                "product": product,
                "position": int(state["position"]),
                "cash": float(state["cash"]),
                "mid_price": float(mid) if mid is not None and not pd.isna(mid) else float("nan"),
                "pnl": marked_pnl,
            }
        )

    rows.append(
        {
            "day": day,
            "timestamp": timestamp,
            "global_timestamp": global_timestamp,
            "product": "TOTAL",
            "position": 0,
            "cash": 0.0,
            "mid_price": float("nan"),
            "pnl": current_total,
        }
    )
    return rows


@st.cache_data(show_spinner=False)
def run_combined_backtest(
    prices: pd.DataFrame,
    selected_products: tuple[str, ...],
    selected_days: tuple[int, ...],
    start_day: int,
    start_timestamp: int,
    end_day: int,
    end_timestamp: int,
    reset_each_day: bool,
    max_position: int,
    strategy_source: str,
    custom_strategy_code: str,
    trader_code: str,
    custom_params_json: str,
    fill_model: str,
    pepper_mode: str,
    pepper_exit_timestamp: int,
    osmium_mode: str,
    osmium_fair: float,
    osmium_edge: float,
    osmium_ema_alpha: float,
    liquidate_end: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    products = list(selected_products)
    start_key = (int(start_day), int(start_timestamp))
    end_key = (int(end_day), int(end_timestamp))
    if start_key > end_key:
        start_key, end_key = end_key, start_key

    frame = prices[
        prices["product"].isin(products) & prices["day"].isin(selected_days)
    ].sort_values(["day", "timestamp", "product"])
    frame = frame[
        ((frame["day"] > start_key[0]) | ((frame["day"] == start_key[0]) & (frame["timestamp"] >= start_key[1])))
        & ((frame["day"] < end_key[0]) | ((frame["day"] == end_key[0]) & (frame["timestamp"] <= end_key[1])))
    ]
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    states = empty_backtest_state(products)
    last_mid: dict[str, float] = {}
    custom_memory: dict[str, object] = {}
    custom_params = json.loads(custom_params_json) if custom_params_json.strip() else {}
    custom_strategy = (
        compile_custom_strategy(custom_strategy_code)
        if strategy_source == "Custom Python"
        else None
    )
    trader = None
    datamodel_module = None
    trader_data = ""
    if strategy_source == "Trader class":
        trader, datamodel_module = compile_trader_code(trader_code)
    trace: list[dict[str, float | int | str]] = []
    summaries: list[dict[str, float | int | str]] = []
    completed_pnl_by_product = {product: 0.0 for product in products}
    day_span = int(frame["timestamp"].max() - frame["timestamp"].min() + 100)

    for day_index, day in enumerate(sorted(selected_days)):
        day_frame = frame[frame["day"] == day]
        if day_frame.empty:
            continue
        if reset_each_day:
            states = empty_backtest_state(products)
            last_mid = {}
            custom_memory = {}
            if strategy_source == "Trader class":
                trader, datamodel_module = compile_trader_code(trader_code)
                trader_data = ""

        for timestamp, stamp_frame in day_frame.groupby("timestamp", sort=True):
            for _, row in stamp_frame.iterrows():
                product = str(row["product"])
                if not pd.isna(row.get("mid_price")):
                    last_mid[product] = float(row["mid_price"])

            if trader is not None and datamodel_module is not None:
                trading_state = build_trading_state(
                    int(timestamp),
                    stamp_frame,
                    products,
                    states,
                    trader_data,
                    datamodel_module,
                )
                with redirect_stdout(io.StringIO()):
                    result = trader.run(trading_state)
                if isinstance(result, tuple) and len(result) >= 3:
                    orders_by_product, _conversions, trader_data = result[:3]
                else:
                    orders_by_product = result
                apply_trader_orders(
                    stamp_frame,
                    states,
                    orders_by_product,
                    max_position,
                    fill_model,
                )
            else:
                for _, row in stamp_frame.iterrows():
                    product = str(row["product"])
                    state = states[product]

                    if custom_strategy is not None:
                        orders = custom_strategy(
                            row.to_dict(),
                            custom_state_view(states, custom_memory, last_mid, custom_params),
                        )
                        apply_custom_orders(row, state, orders, max_position, fill_model)
                    elif product == "INTARIAN_PEPPER_ROOT":
                        if pepper_mode in {"Buy and hold", "Buy then sell"} and (
                            pepper_mode == "Buy and hold" or int(timestamp) < pepper_exit_timestamp
                        ):
                            desired = max_position - int(state["position"])
                            filled, spent = take_from_asks(row, desired)
                            record_fill(state, "buy", filled, spent)
                        elif pepper_mode == "Buy then sell" and int(state["position"]) > 0:
                            filled, received = hit_bids(row, int(state["position"]))
                            record_fill(state, "sell", filled, received)

                    elif product == "ASH_COATED_OSMIUM" and osmium_mode != "Off":
                        if osmium_mode == "Fixed fair taker":
                            fair = osmium_fair
                        else:
                            mid = float(row["mid_price"])
                            if pd.isna(state["ema"]):
                                state["ema"] = mid
                            else:
                                state["ema"] = osmium_ema_alpha * mid + (1 - osmium_ema_alpha) * state["ema"]
                            fair = float(state["ema"])

                        buy_room = max_position - int(state["position"])
                        if buy_room > 0:
                            filled, spent = take_from_asks(row, buy_room, max_price=fair - osmium_edge)
                            record_fill(state, "buy", filled, spent)

                        sell_room = max_position + int(state["position"])
                        if sell_room > 0:
                            filled, received = hit_bids(row, sell_room, min_price=fair + osmium_edge)
                            record_fill(state, "sell", filled, received)

            global_timestamp = day_index * day_span + int(timestamp)
            trace.extend(
                portfolio_rows(
                    states,
                    last_mid,
                    int(day),
                    int(timestamp),
                    global_timestamp,
                    completed_pnl_by_product,
                )
            )

        if liquidate_end:
            final_rows = day_frame.sort_values("timestamp").groupby("product", sort=False).tail(1)
            for _, row in final_rows.iterrows():
                product = str(row["product"])
                state = states[product]
                if state["position"] > 0:
                    filled, received = hit_bids(row, int(state["position"]))
                    record_fill(state, "sell", filled, received)
                elif state["position"] < 0:
                    filled, spent = take_from_asks(row, int(abs(state["position"])))
                    record_fill(state, "buy", filled, spent)
            final_timestamp = int(day_frame["timestamp"].max()) + 1
            trace.extend(
                portfolio_rows(
                    states,
                    last_mid,
                    int(day),
                    final_timestamp,
                    day_index * day_span + final_timestamp,
                    completed_pnl_by_product,
                )
            )

        day_pnl_by_product: dict[str, float] = {}
        for product, state in states.items():
            mid = last_mid.get(product, float("nan"))
            pnl = float(state["cash"])
            if not pd.isna(mid):
                pnl += float(state["position"]) * float(mid)
            day_pnl_by_product[product] = pnl
            summaries.append(
                {
                    "day": int(day),
                    "product": product,
                    "final_position": int(state["position"]),
                    "cash": round(float(state["cash"]), 2),
                    "mark_mid": round(float(mid), 2) if not pd.isna(mid) else float("nan"),
                    "pnl": round(pnl, 2),
                    "fills": int(state["fills"]),
                    "buy_qty": int(state["buy_qty"]),
                    "sell_qty": int(state["sell_qty"]),
                    "gross_qty": int(state["buy_qty"] + state["sell_qty"]),
                    "buy_value": round(float(state["buy_value"]), 2),
                    "sell_value": round(float(state["sell_value"]), 2),
                    "gross_value": round(float(state["buy_value"] + state["sell_value"]), 2),
                    "avg_buy": round(float(state["buy_value"] / state["buy_qty"]), 2)
                    if state["buy_qty"] > 0
                    else float("nan"),
                    "avg_sell": round(float(state["sell_value"] / state["sell_qty"]), 2)
                    if state["sell_qty"] > 0
                    else float("nan"),
                }
            )

        if reset_each_day:
            for product, pnl in day_pnl_by_product.items():
                completed_pnl_by_product[product] = completed_pnl_by_product.get(product, 0.0) + float(pnl)

    trace_frame = pd.DataFrame(trace)
    summary_frame = pd.DataFrame(summaries)
    if not summary_frame.empty:
        if reset_each_day:
            total = summary_frame.groupby("product", as_index=False).agg(
                {
                    "pnl": "sum",
                    "fills": "sum",
                    "buy_qty": "sum",
                    "sell_qty": "sum",
                    "gross_qty": "sum",
                    "buy_value": "sum",
                    "sell_value": "sum",
                    "gross_value": "sum",
                }
            )
            total["avg_buy"] = total["buy_value"] / total["buy_qty"].replace(0, pd.NA)
            total["avg_sell"] = total["sell_value"] / total["sell_qty"].replace(0, pd.NA)
            total["final_position"] = float("nan")
            total["cash"] = float("nan")
            total["mark_mid"] = float("nan")
        else:
            total = summary_frame.sort_values("day").groupby("product", as_index=False).tail(1)
        total["day"] = "ALL"
        total["pnl"] = pd.to_numeric(total["pnl"], errors="coerce").round(2)
        for column in ("buy_value", "sell_value", "gross_value", "avg_buy", "avg_sell"):
            total[column] = pd.to_numeric(total[column], errors="coerce").round(2)
        summary_frame = pd.concat([summary_frame, total[summary_frame.columns]], ignore_index=True)
        summary_frame["day"] = summary_frame["day"].astype(str)
    return trace_frame, summary_frame


def combined_backtest_chart(trace: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if trace.empty:
        return fig

    total = trace[trace["product"] == "TOTAL"]
    fig.add_trace(
        go.Scatter(
            x=total["global_timestamp"],
            y=total["pnl"],
            mode="lines",
            name="Total marked PnL",
            line={"color": "#111", "width": 2.2},
        )
    )
    for product, product_frame in trace[trace["product"] != "TOTAL"].groupby("product", sort=True):
        fig.add_trace(
            go.Scatter(
                x=product_frame["global_timestamp"],
                y=product_frame["pnl"],
                mode="lines",
                name=f"{product} PnL",
                line={"width": 1.3},
            )
        )
    for day, day_frame in trace[trace["product"] == "TOTAL"].groupby("day", sort=True):
        first_x = day_frame["global_timestamp"].min()
        fig.add_vline(x=first_x, line={"color": "#aaaaaa", "width": 1, "dash": "dot"})
        fig.add_annotation(x=first_x, y=1, yref="paper", text=f"day {day}", showarrow=False, yanchor="bottom")
    fig.update_layout(
        height=345,
        margin={"l": 48, "r": 30, "t": 84, "b": 44},
        template="plotly_white",
        title="Combined Backtest Marked PnL",
        legend={"orientation": "h", "font": {"size": 9}},
        font={"size": 9},
        xaxis={"title": "combined replay time", "showgrid": True, "gridcolor": "#eeeeee"},
        yaxis={"title": "PnL", "showgrid": True, "gridcolor": "#e5e5e5"},
    )
    return fig


def combined_position_chart(trace: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if trace.empty:
        return fig

    for product, product_frame in trace[trace["product"] != "TOTAL"].groupby("product", sort=True):
        fig.add_trace(
            go.Scatter(
                x=product_frame["global_timestamp"],
                y=product_frame["position"],
                mode="lines",
                name=f"{product} position",
                line={"width": 1.6},
            )
        )
    fig.update_layout(
        height=270,
        margin={"l": 48, "r": 30, "t": 82, "b": 42},
        template="plotly_white",
        title="Position Through Replay",
        legend={"orientation": "h", "font": {"size": 9}},
        font={"size": 9},
        xaxis={"title": "combined replay time", "showgrid": True, "gridcolor": "#eeeeee"},
        yaxis={"title": "position", "showgrid": True, "gridcolor": "#e5e5e5"},
    )
    return fig


def individual_product_backtest_chart(trace: pd.DataFrame, product: str) -> go.Figure:
    product_frame = trace[trace["product"] == product].sort_values("global_timestamp")
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.115,
        subplot_titles=("Marked PnL and Cash", "Position", "Mid Price"),
    )
    if product_frame.empty:
        return fig

    fig.add_trace(
        go.Scatter(
            x=product_frame["global_timestamp"],
            y=product_frame["pnl"],
            mode="lines",
            name="PnL",
            line={"color": "#111", "width": 2},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=product_frame["global_timestamp"],
            y=product_frame["cash"],
            mode="lines",
            name="Cash",
            line={"color": "#777", "width": 1.2, "dash": "dot"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=product_frame["global_timestamp"],
            y=product_frame["position"],
            mode="lines",
            name="Position",
            line={"color": BID_COLOR, "width": 1.8},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=product_frame["global_timestamp"],
            y=product_frame["mid_price"],
            mode="lines",
            name="Mid",
            line={"color": ASK_COLOR, "width": 1.4},
        ),
        row=3,
        col=1,
    )
    for day, day_frame in product_frame.groupby("day", sort=True):
        first_x = day_frame["global_timestamp"].min()
        fig.add_vline(x=first_x, line={"color": "#bbbbbb", "width": 1, "dash": "dot"})
        fig.add_annotation(
            x=first_x,
            y=1,
            yref="paper",
            text=f"day {day}",
            showarrow=False,
            yanchor="bottom",
            font={"size": 9},
        )
    fig.update_layout(
        height=620,
        margin={"l": 52, "r": 32, "t": 130, "b": 50},
        template="plotly_white",
        title={"text": f"{product} Individual Backtest", "y": 0.985, "yanchor": "top"},
        legend={"orientation": "h", "font": {"size": 9}},
        font={"size": 9},
        xaxis3={"title": "combined replay time"},
    )
    fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5")
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    return fig


def max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    drawdown = series - series.cummax()
    return float(drawdown.min())


def backtest_risk_table(trace: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if trace.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    for product, product_frame in trace.groupby("product", sort=True):
        product_frame = product_frame.sort_values("global_timestamp")
        pnl = product_frame["pnl"].astype(float)
        step = pnl.diff().dropna()
        step_sd = float(step.std()) if len(step) > 1 else 0.0
        step_mean = float(step.mean()) if len(step) > 0 else 0.0
        sharpe_like = (step_mean / step_sd * (len(step) ** 0.5)) if step_sd > 0 else 0.0

        product_summary = summary[summary["product"] == product]
        summary_total = product_summary[product_summary["day"] == "ALL"]
        final_pnl = (
            float(pd.to_numeric(summary_total["pnl"], errors="coerce").sum())
            if not summary_total.empty
            else float(pnl.iloc[-1])
            if not pnl.empty
            else 0.0
        )
        execution_source = summary_total if not summary_total.empty else product_summary
        gross_qty = pd.to_numeric(execution_source.get("gross_qty", pd.Series(dtype=float)), errors="coerce").sum()
        gross_value = pd.to_numeric(execution_source.get("gross_value", pd.Series(dtype=float)), errors="coerce").sum()
        fills = pd.to_numeric(execution_source.get("fills", pd.Series(dtype=float)), errors="coerce").sum()

        rows.append(
            {
                "product": product,
                "final_pnl": round(final_pnl, 2),
                "pnl_step_sd": round(step_sd, 4),
                "mean_step_pnl": round(step_mean, 4),
                "best_step": round(float(step.max()) if len(step) else 0.0, 2),
                "worst_step": round(float(step.min()) if len(step) else 0.0, 2),
                "max_drawdown": round(max_drawdown(pnl), 2),
                "sharpe_like": round(float(sharpe_like), 3),
                "min_pnl": round(float(pnl.min()), 2),
                "max_pnl": round(float(pnl.max()), 2),
                "fills": int(fills) if product != "TOTAL" else pd.NA,
                "gross_qty": int(gross_qty) if product != "TOTAL" else pd.NA,
                "gross_value": round(float(gross_value), 2) if product != "TOTAL" else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def parse_sweep_values(raw: str, default: list[float]) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values or default


def backtest_score_row(trace: pd.DataFrame, summary: pd.DataFrame) -> dict[str, float]:
    risk = backtest_risk_table(trace, summary)
    total = risk[risk["product"] == "TOTAL"]
    if total.empty:
        return {
            "pnl": 0.0,
            "pnl_step_sd": 0.0,
            "max_drawdown": 0.0,
            "sharpe_like": 0.0,
            "gross_qty": 0.0,
            "fills": 0.0,
        }
    row = total.iloc[0]
    fills = pd.to_numeric(summary[summary["day"] == "ALL"].get("fills", pd.Series(dtype=float)), errors="coerce").sum()
    gross_qty = pd.to_numeric(summary[summary["day"] == "ALL"].get("gross_qty", pd.Series(dtype=float)), errors="coerce").sum()
    return {
        "pnl": float(row["final_pnl"]),
        "pnl_step_sd": float(row["pnl_step_sd"]),
        "max_drawdown": float(row["max_drawdown"]),
        "sharpe_like": float(row["sharpe_like"]),
        "gross_qty": float(gross_qty),
        "fills": float(fills),
    }


def strategy_diagnostics(trace: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if trace.empty:
        return pd.DataFrame()
    rows: list[dict[str, float | str]] = []
    for product, frame in trace[trace["product"] != "TOTAL"].groupby("product", sort=True):
        frame = frame.sort_values("global_timestamp").copy()
        frame["pnl_step"] = frame["pnl"].diff()
        frame["abs_position"] = frame["position"].abs()
        nonzero = frame[frame["position"] != 0]
        rows.append(
            {
                "product": product,
                "ticks": int(len(frame)),
                "active_ticks": int(len(nonzero)),
                "active_pct": round(100 * len(nonzero) / max(1, len(frame)), 2),
                "avg_abs_position": round(float(frame["abs_position"].mean()), 2),
                "max_abs_position": int(frame["abs_position"].max()),
                "positive_pnl_steps": int((frame["pnl_step"] > 0).sum()),
                "negative_pnl_steps": int((frame["pnl_step"] < 0).sum()),
                "end_pnl": round(float(frame["pnl"].iloc[-1]), 2),
            }
        )
    return pd.DataFrame(rows)


def product_research_report(prices: pd.DataFrame, products: tuple[str, ...], days: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    frame = prices[prices["product"].isin(products) & prices["day"].isin(days)].copy()
    for (product, day), group in frame.groupby(["product", "day"], sort=True):
        group = group.sort_values("timestamp")
        mid = group["mid_price"].dropna()
        if len(mid) < 3:
            continue
        x = pd.Series(range(len(mid)), dtype="float64")
        slope = float(((x - x.mean()) * (mid.reset_index(drop=True) - mid.mean())).sum() / ((x - x.mean()) ** 2).sum())
        returns = mid.diff().dropna()
        rows.append(
            {
                "product": product,
                "day": int(day),
                "start_mid": round(float(mid.iloc[0]), 2),
                "end_mid": round(float(mid.iloc[-1]), 2),
                "net_move": round(float(mid.iloc[-1] - mid.iloc[0]), 2),
                "trend_slope_per_tick": round(slope, 6),
                "return_sd": round(float(returns.std()), 4),
                "realized_vol": round(float((returns ** 2).sum() ** 0.5), 4),
                "avg_spread": round(float(group["spread"].mean()), 4),
                "avg_top_imbalance": round(float(group["top_imbalance"].mean()), 4),
                "large_move_pct": round(float((returns.abs() >= 5).mean() * 100), 3),
            }
        )
    return pd.DataFrame(rows)


def day_label(day: int) -> str:
    if day < 0:
        return f"D{day}"
    return f"day-{day}"


def day_tick_span(day_prices: pd.DataFrame) -> int:
    timestamps = pd.to_numeric(day_prices["timestamp"], errors="coerce").dropna().sort_values().unique()
    if len(timestamps) == 0:
        return 0
    if len(timestamps) == 1:
        return 1
    diffs = pd.Series(timestamps).diff().dropna()
    step = int(diffs[diffs > 0].min()) if not diffs.empty else 1
    return int(timestamps[-1] - timestamps[0] + step)


@st.cache_data(show_spinner="Running deterministic backtest...")
def run_uploaded_submission_report(
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    trader_code: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from round_backtester_core import run_submission_backtest

    result = run_submission_backtest(
        prices=prices,
        trades=trades,
        trader_code=trader_code,
        strategy_root=DEFAULT_STRATEGY_ROOT,
    )
    return result.trace, result.summary, result.daily, result.product_pnl, result.stats


def submission_daily_pnl_chart(daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if daily.empty:
        return fig
    fig.add_trace(
        go.Bar(
            x=daily["SET"],
            y=daily["FINAL_PNL"],
            marker={"color": "#111111"},
            text=daily["FINAL_PNL"].map(lambda value: f"{value:,.0f}"),
            textposition="outside",
            name="Final PnL",
        )
    )
    fig.update_layout(
        height=360,
        margin={"l": 52, "r": 32, "t": 80, "b": 48},
        template="plotly_white",
        title="Final PnL by Day",
        font={"size": 11},
        yaxis={"title": "PnL", "showgrid": True, "gridcolor": "#e5e5e5"},
        xaxis={"title": None},
        showlegend=False,
    )
    return fig


def submission_product_pnl_chart(product_pnl: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if product_pnl.empty:
        return fig
    day_columns = [column for column in product_pnl.columns if column != "PRODUCT"]
    for day_column in day_columns:
        fig.add_trace(
            go.Bar(
                x=product_pnl["PRODUCT"],
                y=product_pnl[day_column],
                name=day_column,
                text=product_pnl[day_column].map(lambda value: f"{value:,.0f}"),
                textposition="outside",
            )
        )
    fig.update_layout(
        barmode="group",
        height=390,
        margin={"l": 52, "r": 32, "t": 82, "b": 92},
        template="plotly_white",
        title="Product PnL by Day",
        legend={"orientation": "h", "font": {"size": 10}},
        font={"size": 11},
        yaxis={"title": "PnL", "showgrid": True, "gridcolor": "#e5e5e5"},
        xaxis={"title": None, "tickangle": -12},
    )
    return fig


def safe_sd(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 2:
        return 0.0
    value = float(numeric.std(ddof=1))
    return 0.0 if math.isnan(value) else value


def linear_r2(values: pd.Series) -> float:
    y = pd.to_numeric(values, errors="coerce").dropna().reset_index(drop=True)
    if len(y) < 3:
        return 0.0
    x = pd.Series(range(len(y)), dtype="float64")
    y_mean = float(y.mean())
    ss_total = float(((y - y_mean) ** 2).sum())
    if ss_total <= 0:
        return 1.0
    slope = float(((x - x.mean()) * (y - y_mean)).sum() / ((x - x.mean()) ** 2).sum())
    intercept = y_mean - slope * float(x.mean())
    fitted = intercept + slope * x
    ss_residual = float(((y - fitted) ** 2).sum())
    return max(0.0, min(1.0, 1.0 - ss_residual / ss_total))


def fmt_number(value: object, decimals: int = 2) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "-"
    return f"{numeric:,.{decimals}f}"


def fmt_money(value: object, decimals: int = 0) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "-"
    return f"{numeric:,.{decimals}f}"


def mc_table(frame: pd.DataFrame) -> None:
    display = frame.copy()
    st.markdown(
        display.to_html(index=False, classes="mc-table", border=0, escape=False),
        unsafe_allow_html=True,
    )


def mc_card(label: str, value: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="mc-card">
          <div class="mc-card-label">{label}</div>
          <div class="mc-card-value">{value}</div>
          <div class="mc-card-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mc_big_result(label: str, value: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="mc-card" style="border:1px solid #5b7cfa; background:#202634; padding:1rem 1.15rem;">
          <div class="mc-card-label">{label}</div>
          <div class="mc-card-value" style="font-size:2.25rem; color:#ffffff;">{value}</div>
          <div class="mc-card-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mc_terminal(text: str) -> None:
    st.markdown(
        f'<pre class="mc-terminal">{html.escape(text)}</pre>',
        unsafe_allow_html=True,
    )


def product_day_pnl_map(product_pnl: pd.DataFrame) -> dict[str, pd.Series]:
    output: dict[str, pd.Series] = {}
    if product_pnl.empty:
        return output
    day_columns = [column for column in product_pnl.columns if column != "PRODUCT"]
    for _, row in product_pnl.iterrows():
        output[str(row["PRODUCT"])] = pd.to_numeric(row[day_columns], errors="coerce")
    return output


def normalize_daily_report(daily: pd.DataFrame, target_ticks: int) -> pd.DataFrame:
    output = daily.copy()
    if output.empty:
        return output
    ticks = pd.to_numeric(output["TICKS"], errors="coerce").replace(0, pd.NA)
    output["FINAL_PNL"] = (
        pd.to_numeric(output["FINAL_PNL"], errors="coerce") * float(target_ticks) / ticks
    ).round(2)
    output.attrs["pnl_basis_ticks"] = int(target_ticks)
    return output


def normalize_product_report(
    product_pnl: pd.DataFrame,
    daily: pd.DataFrame,
    target_ticks: int,
) -> pd.DataFrame:
    output = product_pnl.copy()
    if output.empty:
        return output
    ticks_by_set = daily.set_index("SET")["TICKS"].to_dict() if not daily.empty else {}
    for column in output.columns:
        if column == "PRODUCT":
            continue
        ticks = float(ticks_by_set.get(column, 0) or 0)
        if ticks > 0:
            output[column] = (pd.to_numeric(output[column], errors="coerce") * float(target_ticks) / ticks).round(2)
    return output


def normalize_summary_pnl(
    summary: pd.DataFrame,
    daily: pd.DataFrame,
    target_ticks: int,
) -> pd.DataFrame:
    output = summary.copy()
    if output.empty:
        return output
    ticks_by_day = daily.set_index("DAY")["TICKS"].to_dict() if not daily.empty else {}
    for index, row in output.iterrows():
        if str(row.get("day")) == "ALL":
            continue
        try:
            day = int(row["day"])
        except (TypeError, ValueError):
            continue
        ticks = float(ticks_by_day.get(day, 0) or 0)
        if ticks > 0:
            output.at[index, "pnl"] = round(float(row["pnl"]) * float(target_ticks) / ticks, 2)
    for product in output["product"].dropna().unique():
        all_mask = (output["product"] == product) & (output["day"].astype(str) == "ALL")
        if not all_mask.any():
            continue
        day_mask = (output["product"] == product) & (output["day"].astype(str) != "ALL")
        output.loc[all_mask, "pnl"] = round(float(pd.to_numeric(output.loc[day_mask, "pnl"], errors="coerce").sum()), 2)
    return output


def normalize_trace_pnl(
    trace: pd.DataFrame,
    daily: pd.DataFrame,
    target_ticks: int,
    product_pnl: pd.DataFrame | None = None,
) -> pd.DataFrame:
    output = trace.copy()
    if output.empty:
        return output
    ticks_by_day = daily.set_index("DAY")["TICKS"].to_dict() if not daily.empty else {}
    days = list(daily["DAY"].astype(int)) if not daily.empty else sorted(output["day"].astype(int).unique())

    raw_day_pnl: dict[str, dict[int, float]] = {"TOTAL": {}}
    for _, row in daily.iterrows():
        raw_day_pnl["TOTAL"][int(row["DAY"])] = float(row["FINAL_PNL"])
    day_by_set = {
        str(row["SET"]): int(row["DAY"])
        for _, row in daily.iterrows()
    }
    if product_pnl is not None and not product_pnl.empty:
        for product, values in product_day_pnl_map(product_pnl).items():
            raw_day_pnl[product] = {}
            for label, value in values.items():
                mapped_day = day_by_set.get(str(label))
                if mapped_day is None:
                    continue
                raw_day_pnl[product][mapped_day] = float(value)

    raw_offsets: dict[tuple[str, int], float] = {}
    normalized_offsets: dict[tuple[str, int], float] = {}
    scales_by_day = {
        int(day): float(target_ticks) / float(ticks_by_day.get(int(day), 1) or 1)
        for day in days
    }
    for product, pnl_by_day in raw_day_pnl.items():
        raw_running = 0.0
        normalized_running = 0.0
        for day in days:
            raw_offsets[(product, int(day))] = raw_running
            normalized_offsets[(product, int(day))] = normalized_running
            day_pnl = float(pnl_by_day.get(int(day), 0.0))
            raw_running += day_pnl
            normalized_running += day_pnl * scales_by_day.get(int(day), 1.0)

    normalized_pnl = []
    for _, row in output.iterrows():
        product = str(row["product"])
        day = int(row["day"])
        raw_offset = raw_offsets.get((product, day), 0.0)
        normalized_offset = normalized_offsets.get((product, day), 0.0)
        scale = scales_by_day.get(day, 1.0)
        local_pnl = float(row["pnl"]) - raw_offset
        normalized_pnl.append(round(normalized_offset + local_pnl * scale, 2))
    output["pnl"] = normalized_pnl
    return output


def stats_from_daily_report(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    pnl = pd.to_numeric(daily["FINAL_PNL"], errors="coerce")
    return pd.DataFrame(
        [
            {"METRIC": "daily_pnl_mean", "VALUE": round(float(pnl.mean()), 2)},
            {"METRIC": "daily_pnl_sd", "VALUE": round(safe_sd(pnl), 2)},
            {"METRIC": "daily_pnl_min", "VALUE": round(float(pnl.min()), 2)},
            {"METRIC": "daily_pnl_max", "VALUE": round(float(pnl.max()), 2)},
            {"METRIC": "sum_final_pnl", "VALUE": round(float(pnl.sum()), 2)},
            {"METRIC": "own_trades_sum", "VALUE": int(daily["OWN_TRADES"].sum())},
        ]
    )


def submission_artifact_text(daily: pd.DataFrame, product_pnl: pd.DataFrame) -> str:
    daily_display = daily.copy()
    if "FINAL_PNL" in daily_display:
        daily_display["FINAL_PNL"] = daily_display["FINAL_PNL"].map(lambda value: f"{float(value):.1f}")
    for column in ("DAY", "TICKS", "SNAPSHOTS", "OWN_TRADES"):
        if column in daily_display:
            daily_display[column] = daily_display[column].map(lambda value: f"{int(value)}")

    product_display = product_pnl.copy()
    for column in product_display.columns:
        if column != "PRODUCT":
            product_display[column] = product_display[column].map(lambda value: f"{float(value):.2f}")

    return (
        "artifacts: log-only\n"
        f"{daily_display.to_string(index=False)}\n\n"
        f"{product_display.to_string(index=False)}"
    )


def pnl_denominator_by_set(daily: pd.DataFrame) -> dict[str, float]:
    basis_ticks = daily.attrs.get("pnl_basis_ticks")
    if basis_ticks:
        return {str(row["SET"]): float(basis_ticks) for _, row in daily.iterrows()}
    return daily.set_index("SET")["TICKS"].astype(float).to_dict()


def submission_product_detail_table(
    summary: pd.DataFrame,
    daily: pd.DataFrame,
    product: str,
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = summary[(summary["product"] == product) & (summary["day"] != "ALL")].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["DAY"] = rows["day"].astype(int)
    set_map = daily.set_index("DAY")["SET"].to_dict() if not daily.empty and "SET" in daily else {}
    rows["SET"] = rows["DAY"].map(lambda day: set_map.get(int(day), day_label(int(day))))
    tick_map = daily.set_index("DAY")["TICKS"].to_dict() if not daily.empty else {}
    snapshot_map = (
        daily.set_index("DAY")["SNAPSHOTS"].to_dict()
        if not daily.empty and "SNAPSHOTS" in daily
        else {}
    )
    rows["TICKS"] = rows["DAY"].map(tick_map).fillna(0).astype(int)
    rows["SNAPSHOTS"] = rows["DAY"].map(snapshot_map).fillna(0).astype(int)
    output = rows[
        [
            "SET",
            "DAY",
            "TICKS",
            "SNAPSHOTS",
            "fills",
            "pnl",
            "final_position",
            "cash",
            "mark_mid",
            "buy_qty",
            "sell_qty",
            "avg_buy",
            "avg_sell",
        ]
    ].rename(
        columns={
            "fills": "OWN_TRADES",
            "pnl": "FINAL_PNL",
            "final_position": "FINAL_POSITION",
            "cash": "CASH",
            "mark_mid": "MARK_MID",
            "buy_qty": "BUY_QTY",
            "sell_qty": "SELL_QTY",
            "avg_buy": "AVG_BUY",
            "avg_sell": "AVG_SELL",
        }
    )
    for column in ("FINAL_PNL", "CASH", "MARK_MID", "AVG_BUY", "AVG_SELL"):
        output[column] = output[column].map(lambda value: fmt_money(value, 2))
    for column in ("DAY", "TICKS", "SNAPSHOTS", "OWN_TRADES", "FINAL_POSITION", "BUY_QTY", "SELL_QTY"):
        output[column] = output[column].map(lambda value: f"{int(value):,}" if not pd.isna(value) else "-")
    return output


def submission_r2_by_session(trace: pd.DataFrame) -> pd.DataFrame:
    if trace.empty:
        return pd.DataFrame()
    rows: list[dict[str, float | int | str]] = []
    for (day, product), frame in trace.groupby(["day", "product"], sort=True):
        rows.append(
            {
                "DAY": int(day),
                "PRODUCT": product,
                "R2": linear_r2(frame.sort_values("global_timestamp")["pnl"]),
            }
        )
    return pd.DataFrame(rows)


def submission_profitability_table(
    daily: pd.DataFrame,
    product_pnl: pd.DataFrame,
    trace: pd.DataFrame,
) -> pd.DataFrame:
    product_values = product_day_pnl_map(product_pnl)
    ticks_by_set = pnl_denominator_by_set(daily)
    total_per_step = pd.Series(
        [
            float(row["FINAL_PNL"]) / ticks_by_set.get(str(row["SET"]), float(row["TICKS"]) or 1.0)
            for _, row in daily.iterrows()
        ]
    )

    r2 = submission_r2_by_session(trace)
    metric_rows: list[dict[str, str]] = []
    products = list(product_values.keys())

    def product_per_step(product: str) -> pd.Series:
        values = product_values.get(product, pd.Series(dtype=float))
        return pd.Series(
            [
                float(value) / ticks_by_set.get(str(day_label), 1.0)
                for day_label, value in values.items()
                if ticks_by_set.get(str(day_label), 0.0) > 0
            ]
        )

    def product_r2(product: str) -> pd.Series:
        if r2.empty:
            return pd.Series(dtype=float)
        return pd.to_numeric(r2[r2["PRODUCT"] == product]["R2"], errors="coerce")

    rowspec = [
        (
            "Profitability",
            "Mean final PnL per replay step.",
            total_per_step.mean(),
            lambda product: product_per_step(product).mean(),
            4,
        ),
        (
            "Stability",
            "Mean line-fit R². Higher means smoother PnL paths.",
            pd.to_numeric(r2[r2["PRODUCT"] == "TOTAL"]["R2"], errors="coerce").mean() if not r2.empty else 0.0,
            lambda product: product_r2(product).mean(),
            3,
        ),
        (
            "Profitability 1σ",
            "Cross-day spread of PnL per replay step.",
            safe_sd(total_per_step),
            lambda product: safe_sd(product_per_step(product)),
            4,
        ),
        (
            "Stability 1σ",
            "Cross-day spread of smoothness.",
            safe_sd(pd.to_numeric(r2[r2["PRODUCT"] == "TOTAL"]["R2"], errors="coerce")) if not r2.empty else 0.0,
            lambda product: safe_sd(product_r2(product)),
            3,
        ),
    ]
    for metric, meaning, total_value, product_func, decimals in rowspec:
        row = {
            "Metric": metric,
            "Meaning": meaning,
            "Total": fmt_number(total_value, decimals),
        }
        for product in products:
            row[product] = fmt_number(product_func(product), decimals)
        metric_rows.append(row)
    return pd.DataFrame(metric_rows)


def submission_summary_table(label: str, values: pd.Series) -> pd.DataFrame:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        numeric = pd.Series([0.0])
    rows = [
        ("Mean", float(numeric.mean())),
        ("1σ", safe_sd(numeric)),
        ("P05", float(numeric.quantile(0.05))),
        ("Median", float(numeric.median())),
        ("P95", float(numeric.quantile(0.95))),
    ]
    return pd.DataFrame({label: [name for name, _ in rows], "Value": [fmt_money(value, 2) for _, value in rows]})


def submission_session_table(
    daily: pd.DataFrame,
    product_pnl: pd.DataFrame,
    trace: pd.DataFrame,
    ascending: bool,
) -> pd.DataFrame:
    table = daily[["SET", "FINAL_PNL"]].rename(columns={"SET": "Session", "FINAL_PNL": "Total"}).copy()
    products = product_day_pnl_map(product_pnl)
    for product, values in products.items():
        mapped = {str(day_label): float(value) for day_label, value in values.items()}
        table[product] = table["Session"].map(mapped).fillna(0.0)
    table["Total $/step"] = table["Total"] / daily["TICKS"].astype(float).replace(0, pd.NA).to_numpy()
    pnl_denominators = pnl_denominator_by_set(daily)
    table["Total $/step"] = [
        float(total) / pnl_denominators.get(str(session), 1.0)
        for session, total in zip(table["Session"], table["Total"])
    ]
    r2 = submission_r2_by_session(trace)
    total_r2_by_day = (
        r2[r2["PRODUCT"] == "TOTAL"].set_index("DAY")["R2"].to_dict()
        if not r2.empty
        else {}
    )
    table["Total R²"] = daily["DAY"].map(total_r2_by_day).fillna(0.0)
    table = table.sort_values("Total", ascending=ascending).head(8)
    for column in table.columns:
        if column == "Session":
            continue
        decimals = 3 if column in {"Total $/step", "Total R²"} else 2
        table[column] = table[column].map(lambda value: fmt_number(value, decimals))
    return table


def apply_mc_chart_layout(fig: go.Figure, title: str, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 58, "r": 28, "t": 74, "b": 56},
        template="plotly_dark",
        title={"text": title, "font": {"size": 16}},
        paper_bgcolor="#15181d",
        plot_bgcolor="#15181d",
        font={"color": "#c6c9cf", "size": 11},
        legend={"orientation": "h", "y": -0.18, "x": 0.5, "xanchor": "center"},
    )
    fig.update_xaxes(gridcolor="#3a404a", zerolinecolor="#3a404a")
    fig.update_yaxes(gridcolor="#3a404a", zerolinecolor="#3a404a")
    return fig


def normal_curve(values: pd.Series, bins: int = 16) -> tuple[list[float], list[float]]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 2:
        return [], []
    mu = float(numeric.mean())
    sd = safe_sd(numeric)
    if sd <= 0:
        return [], []
    low = float(numeric.min() - sd)
    high = float(numeric.max() + sd)
    xs = [low + (high - low) * i / 80 for i in range(81)]
    bin_width = max((float(numeric.max()) - float(numeric.min())) / max(1, bins), 1.0)
    scale = len(numeric) * bin_width
    ys = [
        scale * (1 / (sd * (2 * math.pi) ** 0.5)) * math.exp(-0.5 * ((x - mu) / sd) ** 2)
        for x in xs
    ]
    return xs, ys


def pnl_distribution_chart(values: pd.Series, title: str, color: str) -> go.Figure:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=numeric,
            nbinsx=max(3, min(16, len(numeric) * 3)),
            name="PnL",
            marker={"color": color, "line": {"color": "#aab7ff", "width": 1}},
            opacity=0.82,
        )
    )
    xs, ys = normal_curve(numeric)
    if xs:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name="Normal fit",
                line={"color": "#e15759", "width": 2},
            )
        )
    subtitle = f"Normal fit μ {numeric.mean():,.0f} · σ {safe_sd(numeric):,.0f}" if not numeric.empty else ""
    fig.add_annotation(
        text=subtitle,
        x=0.5,
        y=1.04,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 11, "color": "#aeb4bd"},
    )
    fig.update_xaxes(title="Final PnL")
    fig.update_yaxes(title="Session count")
    return apply_mc_chart_layout(fig, title)


def cross_product_scatter(product_pnl: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    products = product_day_pnl_map(product_pnl)
    product_names = list(products.keys())
    if len(product_names) >= 2:
        x_product, y_product = product_names[:2]
        x_values = pd.to_numeric(products[x_product], errors="coerce")
        y_values = pd.to_numeric(products[y_product], errors="coerce")
        common = pd.DataFrame({"x": x_values, "y": y_values}).dropna()
        fig.add_trace(
            go.Scatter(
                x=common["x"],
                y=common["y"],
                mode="markers",
                name="Sessions",
                marker={"color": "#5b7cfa", "size": 9},
            )
        )
        if len(common) >= 2:
            slope = float(common["x"].cov(common["y"]) / common["x"].var()) if common["x"].var() else 0.0
            intercept = float(common["y"].mean() - slope * common["x"].mean())
            xs = [float(common["x"].min()), float(common["x"].max())]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=[intercept + slope * x for x in xs],
                    mode="lines",
                    name="Linear fit",
                    line={"color": "#e15759", "width": 1.6},
                )
            )
            if len(common) > 1 and float(common["x"].var()) > 0 and float(common["y"].var()) > 0:
                corr = float(common["x"].corr(common["y"]))
            else:
                corr = 0.0
            r2 = corr * corr if not math.isnan(corr) else 0.0
            subtitle = f"corr {corr:.2f} · R² {r2:.2f}"
        else:
            subtitle = "not enough sessions"
        fig.add_annotation(
            text=subtitle,
            x=0.5,
            y=1.04,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 11, "color": "#aeb4bd"},
        )
        fig.update_xaxes(title=f"{x_product} PnL")
        fig.update_yaxes(title=f"{y_product} PnL")
    return apply_mc_chart_layout(fig, "Cross Product Scatter")


def profitability_distribution_chart(daily: pd.DataFrame, product_pnl: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    ticks = daily.set_index("SET")["TICKS"].astype(float).to_dict()
    total = daily.set_index("SET")["FINAL_PNL"].astype(float)
    fig.add_trace(
        go.Scatter(
            x=[float(value) / ticks.get(label, 1.0) for label, value in total.items()],
            y=[1] * len(total),
            mode="lines+markers",
            name="Total",
            line={"color": "#5b7cfa", "width": 1.6},
            marker={"size": 7},
        )
    )
    colors = ["#6ccf9c", "#f28e2b", "#e15759", "#76b7b2"]
    for color, (product, values) in zip(colors, product_day_pnl_map(product_pnl).items()):
        fig.add_trace(
            go.Scatter(
                x=[
                    float(value) / ticks.get(str(label), 1.0)
                    for label, value in values.items()
                    if ticks.get(str(label), 0.0) > 0
                ],
                y=[1 + 0.08 * (len(fig.data))] * len(values),
                mode="lines+markers",
                name=product,
                line={"color": color, "width": 1.5},
                marker={"size": 7},
            )
        )
    fig.update_xaxes(title="$ / step")
    fig.update_yaxes(title="Density proxy", showticklabels=False)
    return apply_mc_chart_layout(fig, "Profitability Distribution")


def stability_distribution_chart(trace: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    r2 = submission_r2_by_session(trace)
    colors = {"TOTAL": "#5b7cfa", "ASH_COATED_OSMIUM": "#6ccf9c", "INTARIAN_PEPPER_ROOT": "#f28e2b"}
    for product, frame in r2.groupby("PRODUCT", sort=True):
        fig.add_trace(
            go.Scatter(
                x=frame["R2"],
                y=[1 + 0.12 * len(fig.data)] * len(frame),
                mode="lines+markers",
                name=str(product).replace("_", " "),
                line={"color": colors.get(product, "#b8bcc4"), "width": 1.5},
                marker={"size": 7},
            )
        )
    fig.update_xaxes(title="R²", range=[0, 1.02])
    fig.update_yaxes(title="Density proxy", showticklabels=False)
    return apply_mc_chart_layout(fig, "Stability Distribution")


def trace_product_chart(trace: pd.DataFrame, product: str) -> go.Figure:
    fig = go.Figure()
    frame = trace[trace["product"] == product].sort_values("global_timestamp")
    fig.add_trace(
        go.Scatter(
            x=frame["global_timestamp"],
            y=frame["pnl"],
            mode="lines",
            name="Marked PnL",
            line={"color": "#5b7cfa", "width": 1.7},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["global_timestamp"],
            y=frame["position"],
            mode="lines",
            name="Position",
            line={"color": "#f28e2b", "width": 1.1, "dash": "dot"},
            yaxis="y2",
        )
    )
    fig.update_layout(
        yaxis2={
            "title": "Position",
            "overlaying": "y",
            "side": "right",
            "gridcolor": "rgba(0,0,0,0)",
        }
    )
    fig.update_xaxes(title="Replay time")
    fig.update_yaxes(title="PnL")
    return apply_mc_chart_layout(fig, f"{product} PnL Path", height=330)


def research_value(percent: float) -> float:
    return 200_000.0 * math.log1p(max(0.0, percent)) / math.log1p(100.0)


def scale_value(percent: float) -> float:
    return 7.0 * max(0.0, percent) / 100.0


def speed_multiplier(rank: int, field_size: int) -> float:
    if field_size <= 1:
        return 0.9
    safe_rank = min(max(1, int(rank)), int(field_size))
    return 0.9 - 0.8 * (safe_rank - 1) / (field_size - 1)


def normal_cdf(value: float, mean: float, std: float) -> float:
    if std <= 0:
        return 1.0 if value >= mean else 0.0
    z = (value - mean) / std
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def investment_budget_used(research_pct: float, scale_pct: float, speed_pct: float) -> float:
    return 50_000.0 * (research_pct + scale_pct + speed_pct) / 100.0


def investment_pnl(
    research_pct: float,
    scale_pct: float,
    speed_pct: float,
    hit_rate: float,
) -> float:
    gross = research_value(research_pct) * scale_value(scale_pct) * hit_rate
    return gross - investment_budget_used(research_pct, scale_pct, speed_pct)


def investment_pillar_chart() -> go.Figure:
    xs = list(range(0, 101))
    research_values = [research_value(x) for x in xs]
    scale_values = [scale_value(x) for x in xs]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=research_values,
            mode="lines",
            name="Research(x)",
            line={"color": "#5b7cfa", "width": 2.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=[value * 28_571.428571 for value in scale_values],
            mode="lines",
            name="Scale(x), rescaled",
            line={"color": "#6ccf9c", "width": 2.2, "dash": "dot"},
        )
    )
    fig.update_xaxes(title="Investment percentage")
    fig.update_yaxes(title="Outcome value")
    return apply_mc_chart_layout(fig, "Research And Scale Curves", height=330)


def investment_speed_chart(field_size: int, selected_rank: int) -> go.Figure:
    ranks = list(range(1, max(1, int(field_size)) + 1))
    multipliers = [speed_multiplier(rank, field_size) for rank in ranks]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ranks,
            y=multipliers,
            mode="lines+markers",
            name="Speed multiplier",
            line={"color": "#f28e2b", "width": 2.4},
            marker={"size": 5},
        )
    )
    chosen = speed_multiplier(selected_rank, field_size)
    fig.add_trace(
        go.Scatter(
            x=[selected_rank],
            y=[chosen],
            mode="markers",
            name="Your assumed rank",
            marker={"color": "#e15759", "size": 12, "symbol": "diamond"},
        )
    )
    fig.update_xaxes(title="Speed rank, 1 is best", autorange="reversed")
    fig.update_yaxes(title="Hit-rate multiplier", range=[0.05, 0.95])
    return apply_mc_chart_layout(fig, "Speed Rank Curve", height=330)


def investment_pnl_heatmap(speed_pct: float, hit_rate: float) -> go.Figure:
    values = list(range(0, 101, 2))
    z: list[list[float | None]] = []
    for scale_pct in values:
        row: list[float | None] = []
        for research_pct in values:
            if research_pct + scale_pct + speed_pct > 100:
                row.append(None)
            else:
                row.append(investment_pnl(research_pct, scale_pct, speed_pct, hit_rate))
        z.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            x=values,
            y=values,
            z=z,
            colorscale="Viridis",
            colorbar={"title": "PnL"},
            hovertemplate=(
                "Research=%{x}%<br>"
                "Scale=%{y}%<br>"
                "PnL=%{z:,.0f}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="Research %")
    fig.update_yaxes(title="Scale %")
    return apply_mc_chart_layout(fig, "PnL Heatmap With Speed Fixed", height=410)


def optimal_research_scale_split(total_pct: float) -> tuple[float, float, float]:
    total_pct = min(100.0, max(0.0, float(total_pct)))
    if total_pct <= 0:
        return 0.0, 0.0, 0.0

    best_research = 0.0
    best_scale = total_pct
    best_value = -1.0
    steps = max(1, int(round(total_pct * 10)))
    for step in range(steps + 1):
        research_pct = total_pct * step / steps
        scale_pct = total_pct - research_pct
        value = research_value(research_pct) * scale_value(scale_pct)
        if value > best_value:
            best_value = value
            best_research = research_pct
            best_scale = scale_pct
    return best_research, best_scale, best_value


def optimal_research_scale_curve() -> pd.DataFrame:
    rows = []
    for total_pct in range(1, 101):
        research_pct, scale_pct, edge_value = optimal_research_scale_split(total_pct)
        rows.append(
            {
                "total_pct": total_pct,
                "research_pct": research_pct,
                "scale_pct": scale_pct,
                "research_share": research_pct / total_pct if total_pct > 0 else 0.0,
                "scale_share": scale_pct / total_pct if total_pct > 0 else 0.0,
                "edge_value": edge_value,
            }
        )
    return pd.DataFrame(rows)


def optimal_research_scale_chart(selected_total_pct: float) -> go.Figure:
    curve = optimal_research_scale_curve()
    selected_research, selected_scale, _ = optimal_research_scale_split(selected_total_pct)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve["total_pct"],
            y=curve["research_pct"],
            mode="lines",
            name="Ideal Research %",
            line={"color": "#5b7cfa", "width": 2.6},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curve["total_pct"],
            y=curve["scale_pct"],
            mode="lines",
            name="Ideal Scale %",
            line={"color": "#6ccf9c", "width": 2.6},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[selected_total_pct, selected_total_pct],
            y=[selected_research, selected_scale],
            mode="markers",
            name="Selected n",
            marker={"color": "#e15759", "size": 11, "symbol": "diamond"},
            customdata=[[selected_research, selected_scale], [selected_research, selected_scale]],
            hovertemplate=(
                "n=%{x:.0f}%<br>"
                "ideal research=%{customdata[0]:.1f}%<br>"
                "ideal scale=%{customdata[1]:.1f}%<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="n = Research% + Scale%")
    fig.update_yaxes(title="Optimal allocation %")
    return apply_mc_chart_layout(fig, "Best Scale / Research Split For Each n", height=360)


def optimal_research_scale_share_chart(selected_total_pct: float) -> go.Figure:
    curve = optimal_research_scale_curve()
    selected_research, selected_scale, _ = optimal_research_scale_split(selected_total_pct)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve["total_pct"],
            y=100 * curve["research_share"],
            mode="lines",
            name="Research share of n",
            line={"color": "#5b7cfa", "width": 2.4},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curve["total_pct"],
            y=100 * curve["scale_share"],
            mode="lines",
            name="Scale share of n",
            line={"color": "#6ccf9c", "width": 2.4},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[selected_total_pct],
            y=[100 * selected_scale / selected_total_pct if selected_total_pct > 0 else 0],
            mode="markers",
            name="Selected scale share",
            marker={"color": "#e15759", "size": 11, "symbol": "diamond"},
        )
    )
    fig.update_xaxes(title="n = Research% + Scale%")
    fig.update_yaxes(title="Share of n (%)", range=[0, 100])
    return apply_mc_chart_layout(fig, "Ideal Scale / Research Ratio As n Changes", height=360)


def speed_mu_pnl(speed_pct: float, mu: float, std: float) -> dict[str, float]:
    std = max(1e-9, std)
    percentile = normal_cdf(speed_pct, mu, std)
    multiplier = 0.1 + 0.8 * percentile
    remaining = max(0.0, 100.0 - speed_pct)
    research_pct, scale_pct, edge_value = optimal_research_scale_split(remaining)
    gross = edge_value * multiplier
    final = gross - investment_budget_used(research_pct, scale_pct, speed_pct)
    return {
        "speed_pct": speed_pct,
        "mu": mu,
        "std": std,
        "percentile": percentile,
        "multiplier": multiplier,
        "research_pct": research_pct,
        "scale_pct": scale_pct,
        "gross": gross,
        "pnl": final,
    }


def speed_mu_pnl_chart(selected_speed_pct: float, mu: float, std: float) -> go.Figure:
    speeds = list(range(0, 101))
    rows = [speed_mu_pnl(speed, mu, std) for speed in speeds]
    selected = speed_mu_pnl(selected_speed_pct, mu, std)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=speeds,
            y=[row["pnl"] for row in rows],
            mode="lines",
            name="PnL",
            line={"color": "#5b7cfa", "width": 2.6},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[selected_speed_pct],
            y=[selected["pnl"]],
            mode="markers",
            name="Selected speed",
            marker={"color": "#e15759", "size": 12, "symbol": "diamond"},
            customdata=[
                [
                    selected["percentile"] * 100,
                    selected["multiplier"],
                    selected["research_pct"],
                    selected["scale_pct"],
                ]
            ],
            hovertemplate=(
                "Speed=%{x:.0f}%<br>"
                "PnL=%{y:,.0f}<br>"
                "percentile=%{customdata[0]:.1f}%<br>"
                "multiplier=%{customdata[1]:.3f}<br>"
                "Research=%{customdata[2]:.1f}%<br>"
                "Scale=%{customdata[3]:.1f}%<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="Your Speed %")
    fig.update_yaxes(title="PnL")
    title = f"PnL As A Function Of Speed And mu · selected {selected['pnl']:,.0f}"
    fig.add_annotation(
        x=selected_speed_pct,
        y=selected["pnl"],
        text=f"PnL {selected['pnl']:,.0f}",
        showarrow=True,
        arrowhead=2,
        ax=35,
        ay=-35,
        bgcolor="#202634",
        bordercolor="#5b7cfa",
        font={"color": "#ffffff", "size": 12},
    )
    return apply_mc_chart_layout(fig, title, height=370)


def rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def main() -> None:
    st.set_page_config(page_title="Prosperity Order Book Visualizer", layout="wide")
    install_style()

    available_datasets = [
        label
        for label, folder_paths in DATASET_DIRS.items()
        if discover_dataset_files(tuple(str(path) for path in folder_paths)).price_paths
    ]
    if not available_datasets:
        st.error("No price CSVs found in the configured Round 1 or Round 2 folders.")
        return

    default_dataset = "Round 2" if "Round 2" in available_datasets else available_datasets[-1]
    selected_dataset_label = st.sidebar.selectbox(
        "Dataset",
        available_datasets,
        index=available_datasets.index(default_dataset),
    )
    folder_paths = tuple(str(path) for path in DATASET_DIRS[selected_dataset_label])
    folder = ", ".join(folder_paths)
    files = discover_dataset_files(folder_paths)
    if not files.price_paths:
        st.error(f"No price CSVs found for {selected_dataset_label}.")
        return

    prices = load_prices(files.price_paths)
    trades = load_trades(files.trade_paths)
    if prices.empty:
        st.warning("No price rows found for the selected day set.")
        return

    selected_days = sorted(int(day) for day in prices["day"].dropna().unique())
    day_label_map = (
        prices.dropna(subset=["day"])
        .drop_duplicates("day")
        .set_index("day")["session_label"]
        .to_dict()
    )
    products = sorted(prices["product"].dropna().unique())
    controls, charts = st.columns([1.22, 4.95], gap="small")

    with controls:
        with st.container(border=True):
            badge(4, "Snapshot")
            product = st.selectbox("Product", products, label_visibility="collapsed")
            plot_day = st.selectbox(
                "Day",
                selected_days,
                index=len(selected_days) - 1,
                label_visibility="collapsed",
                format_func=lambda value: day_label_map.get(int(value), day_label(int(value))),
            )

            product_prices = prices[(prices["product"] == product) & (prices["day"] == plot_day)].copy()
            product_trades = trades[(trades["symbol"] == product) & (trades["day"] == plot_day)].copy()
            product_trades = nearest_book_for_trades(product_prices, product_trades)

            min_ts = int(product_prices["timestamp"].min())
            max_ts = int(product_prices["timestamp"].max())
            manual_cursor_key = f"manual_cursor_{product}_{plot_day}"
            playback_cursor_key = f"playback_cursor_{product}_{plot_day}"
            playback_active_key = f"playback_active_{product}_{plot_day}"

            if manual_cursor_key not in st.session_state:
                st.session_state[manual_cursor_key] = min_ts
            if playback_cursor_key not in st.session_state:
                st.session_state[playback_cursor_key] = min_ts
            st.session_state[manual_cursor_key] = int(
                min(max(st.session_state[manual_cursor_key], min_ts), max_ts)
            )
            st.session_state[playback_cursor_key] = int(
                min(max(st.session_state[playback_cursor_key], min_ts), max_ts)
            )

            st.markdown('<div class="panel-title">Playback</div>', unsafe_allow_html=True)
            playback_start, playback_end = st.slider(
                "Range",
                min_ts,
                max_ts,
                (min_ts, max_ts),
                step=100,
                key=f"playback_range_{product}_{plot_day}",
            )
            playback_step = st.number_input(
                "Step",
                min_value=100,
                max_value=10000,
                value=100,
                step=100,
                key=f"playback_step_{product}_{plot_day}",
            )
            playback_delay = st.slider(
                "Delay",
                0.03,
                1.0,
                0.15,
                step=0.03,
                key=f"playback_delay_{product}_{plot_day}",
            )
            play_cols = st.columns(4)
            if play_cols[0].button("Play", key=f"play_{product}_{plot_day}"):
                st.session_state[playback_active_key] = True
                st.session_state[playback_cursor_key] = int(
                    min(max(st.session_state[manual_cursor_key], playback_start), playback_end)
                )
                rerun()
            if play_cols[1].button("Pause", key=f"pause_{product}_{plot_day}"):
                st.session_state[playback_active_key] = False
            if play_cols[2].button("Start", key=f"start_{product}_{plot_day}"):
                st.session_state[playback_active_key] = False
                st.session_state[playback_cursor_key] = int(playback_start)
                st.session_state[manual_cursor_key] = int(playback_start)
                rerun()
            if play_cols[3].button("End", key=f"end_{product}_{plot_day}"):
                st.session_state[playback_active_key] = False
                st.session_state[playback_cursor_key] = int(playback_end)
                st.session_state[manual_cursor_key] = int(playback_end)
                rerun()

            manual_timestamp = st.slider(
                "Manual timestamp",
                min_ts,
                max_ts,
                value=st.session_state[manual_cursor_key],
                step=100,
                key=manual_cursor_key,
            )

            if st.session_state.get(playback_active_key, False):
                focus_timestamp = int(st.session_state[playback_cursor_key])
                st.caption(f"Playing t={focus_timestamp:,}")
            else:
                focus_timestamp = int(manual_timestamp)
                st.session_state[playback_cursor_key] = focus_timestamp

            snapshot, nearby_trades = focused_snapshot(product_prices, product_trades, focus_timestamp)
            st.markdown(
                f"""
                <div class="tiny-note">
                <b>General:</b><br>
                TIMESTAMP={int(snapshot.get("timestamp", focus_timestamp))}<br><br>
                <b>NFC:</b><br>
                {product}<br><br>
                <b>Book:</b><br>
                BID {snapshot.get("bid_volume_1", ""):.0f}@{snapshot.get("bid_price_1", float("nan")):.1f}<br>
                ASK {snapshot.get("ask_volume_1", ""):.0f}@{snapshot.get("ask_price_1", float("nan")):.1f}<br>
                MID {snapshot.get("mid_price", float("nan")):.2f}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            badge(5, "Selection")
            log_file = st.file_uploader("Import log", type=["log", "txt", "csv"])
            overlays = st.multiselect(
                "Overlays",
                [
                    "mid_price",
                    "microprice",
                    "depth_vwap",
                    "depth_vwap_trend",
                    "depth_vwap_live_trend",
                    "pepper_linear_trend",
                    "osmium_density_fair",
                    "osmium_wall_mid_smooth",
                ],
                default=(
                    ["mid_price", "pepper_linear_trend"]
                    if product == "INTARIAN_PEPPER_ROOT"
                    else ["mid_price"]
                ),
            )
            plot_overlays = list(overlays)
            if "depth_vwap" in plot_overlays and "depth_vwap_trend" not in plot_overlays:
                plot_overlays.append("depth_vwap_trend")
            normalize_by = st.selectbox(
                "Normalize by",
                [
                    "None",
                    "mid_price",
                    "microprice",
                    "depth_vwap",
                    "depth_vwap_trend",
                    "depth_vwap_live_trend",
                    "pepper_linear_trend",
                    "osmium_density_fair",
                    "osmium_wall_mid_smooth",
                    "Day volatility z-score",
                    "Rolling volatility z-score",
                ],
            )
            if normalize_by == "Day volatility z-score":
                st.caption("Shows each price as standard deviations from that product/day mean.")
            elif normalize_by == "Rolling volatility z-score":
                st.caption("Uses a 120-tick rolling mean and rolling price volatility.")

        with st.container(border=True):
            badge(6, "Trades and Book")
            show_bids = st.checkbox("OB bid levels", value=True)
            show_asks = st.checkbox("OB ask levels", value=True)
            show_public_trades = st.checkbox("All taker trades", value=True)
            show_maker_mirror = st.checkbox("M maker squares", value=True)
            show_trade_path = st.checkbox("Actual trade price path", value=True)
            show_trade_trend = st.checkbox("Actual trade live trend", value=True)
            show_small_takers = st.checkbox("S small takers", value=True)
            show_big_takers = st.checkbox("B big takers", value=True)
            _show_own_trades = st.checkbox("F own trade crosses", value=False, disabled=True)
            max_quantity = int(max(1, trades["quantity"].max())) if not trades.empty else 1
            trade_qty_range = st.slider("Trade quantity", 1, max_quantity, (1, max_quantity))
            st.markdown(
                """
                <div class="trade-key">
                  <div class="trade-chip" style="background:#f07f24">M1</div>
                  <div class="trade-chip" style="background:#d96b2b">M2</div>
                  <div class="trade-chip" style="background:#f2c23e">M3</div>
                  <div class="trade-chip" style="background:#91d8ff">S</div>
                  <div class="trade-chip" style="background:#c9ff4f">S1</div>
                  <div class="trade-chip" style="background:#83df67">S2</div>
                  <div class="trade-chip" style="background:#cd79ff">B</div>
                  <div class="trade-chip" style="background:#6b49d8">I</div>
                  <div class="trade-chip" style="background:#ffe45c">F</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            badge(7, "Performance")
            min_book_qty = st.slider("Book qty threshold", 0, 100, 0)
            max_timestamps = st.slider("Max timestamps", 500, 10000, 5000, step=500)
            st.caption(f"{len(product_prices):,} snapshots / {len(product_trades):,} trades")

        with st.container(border=True):
            badge(8, "Submission Backtester")
            st.caption(f"Using {selected_dataset_label} data from `{folder}`")
            submission_upload = st.file_uploader(
                "Upload .py submission",
                type=["py"],
                key="submission_backtest_upload",
            )
            submission_key = None
            submission_code = (
                submission_upload.getvalue().decode("utf-8", errors="replace")
                if submission_upload is not None
                else ""
            )
            if submission_upload is None:
                st.caption("Upload one Prosperity file containing `class Trader`.")
            else:
                submission_key = f"{submission_upload.name}:{submission_upload.size}"
                st.caption(f"Ready: `{submission_upload.name}`")
                pnl_basis_label = st.selectbox(
                    "Normalize PnL to",
                    ["100,000 ticks", "1,000,000 ticks"],
                    index=1,
                    key="submission_pnl_basis",
                )
                if st.button(
                    "Submit / Run Backtest",
                    type="primary",
                    use_container_width=True,
                    key="submission_submit_button",
                ):
                    st.session_state["submitted_submission_key"] = submission_key
                    st.session_state["submitted_submission_name"] = submission_upload.name
                    st.session_state["submitted_submission_code"] = submission_code
                    st.session_state["submitted_pnl_basis_label"] = pnl_basis_label
                if st.session_state.get("submitted_submission_key") != submission_key:
                    st.caption("Click Submit / Run Backtest when you want to run this file.")

    low_qty, high_qty = trade_qty_range
    if not show_small_takers:
        low_qty = max(low_qty, 8)
    if not show_big_takers:
        high_qty = min(high_qty, 14)
    if low_qty > high_qty:
        show_public_trades = False
        show_trade_path = False
        show_trade_trend = False
        low_qty, high_qty = trade_qty_range

    with charts:
        st.markdown(
            f'<div class="main-label">{product} &nbsp; {day_label_map.get(int(plot_day), day_label(int(plot_day)))} &nbsp; t={focus_timestamp}</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
                main_order_book_chart(
                    product_prices,
                    product_trades,
                    plot_overlays,
                    normalize_by,
                    show_bids,
                show_asks,
                show_public_trades,
                show_maker_mirror,
                show_trade_path,
                show_trade_trend,
                min_book_qty,
                (low_qty, high_qty),
                max_timestamps,
                focus_timestamp,
            ),
            use_container_width=True,
            config={"scrollZoom": True, "displaylogo": False},
        )

        st.plotly_chart(
            small_line_chart(product_prices, ["profit_and_loss"], "2  PnL Panel", focus_timestamp),
            use_container_width=True,
            config={"displayModeBar": False, "displaylogo": False},
        )
        st.plotly_chart(
            position_placeholder(product_prices, focus_timestamp),
            use_container_width=True,
            config={"displayModeBar": False, "displaylogo": False},
        )

        log_left, log_right = st.columns([1, 1], gap="small")
        with log_left:
            with st.container(border=True):
                st.markdown('<div class="panel-title">Hover Tooltip / Nearby Trades</div>', unsafe_allow_html=True)
                if nearby_trades.empty:
                    st.caption("No public trades within +/- 200 timestamp units.")
                else:
                    st.dataframe(
                        nearby_trades[["timestamp", "price", "quantity", "aggressor", "buyer", "seller"]],
                        width="stretch",
                        height=155,
                        hide_index=True,
                    )
        with log_right:
            with st.container(border=True):
                st.markdown('<div class="panel-title">Log Viewer</div>', unsafe_allow_html=True)
                if log_file is None:
                    st.caption("No own log loaded. Upload a logger output in panel 5.")
                else:
                    text = log_file.getvalue().decode("utf-8", errors="replace")
                    lines = [line for line in text.splitlines() if str(focus_timestamp) in line]
                    st.code("\n".join(lines[:18]) or "No matching log lines.")

        submitted_key = st.session_state.get("submitted_submission_key")
        if submission_upload is not None and submitted_key == submission_key:
            submitted_name = st.session_state.get("submitted_submission_name", submission_upload.name)
            submitted_code = st.session_state.get("submitted_submission_code", submission_code)
            st.markdown(
                f'<div class="mc-title">{selected_dataset_label} Backtester <span class="mc-chip">upload-only</span></div>',
                unsafe_allow_html=True,
            )
            try:
                submission_trace, submission_summary, daily_report, product_report, stats_report = (
                    run_uploaded_submission_report(prices, trades, submitted_code)
                )
            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
                st.stop()

            if daily_report.empty:
                st.warning("No backtest rows were produced.")
            else:
                maf_bid = daily_report.attrs.get("maf_bid")
                backtest_warnings = daily_report.attrs.get("warnings", ())
                basis_label = st.session_state.get(
                    "submission_pnl_basis",
                    st.session_state.get("submitted_pnl_basis_label", "100,000 ticks"),
                )
                normalization_ticks = 1_000_000 if "1,000,000" in str(basis_label) else 100_000
                normalized_daily_report = normalize_daily_report(daily_report, normalization_ticks)
                normalized_product_report = normalize_product_report(
                    product_report,
                    daily_report,
                    normalization_ticks,
                )
                normalized_summary = normalize_summary_pnl(
                    submission_summary,
                    daily_report,
                    normalization_ticks,
                )
                normalized_trace = normalize_trace_pnl(
                    submission_trace,
                    daily_report,
                    normalization_ticks,
                    product_report,
                )
                normalized_stats_report = stats_from_daily_report(normalized_daily_report)

                total_values = pd.to_numeric(normalized_daily_report["FINAL_PNL"], errors="coerce")
                total_sum = float(total_values.sum())
                total_mean = float(total_values.mean())
                total_sd = safe_sd(total_values)
                total_low = float(total_values.quantile(0.05))
                total_high = float(total_values.quantile(0.95))
                own_trades = int(pd.to_numeric(daily_report["OWN_TRADES"], errors="coerce").sum())
                ticks = int(pd.to_numeric(daily_report["TICKS"], errors="coerce").sum())
                snapshots = int(pd.to_numeric(daily_report.get("SNAPSHOTS", pd.Series(dtype=float)), errors="coerce").sum())
                maf_chip = (
                    f'<span class="mc-chip">MAF bid {int(maf_bid):,}</span>'
                    if maf_bid is not None
                    else '<span class="mc-chip">no MAF bid</span>'
                )
                day_list_caption = ", ".join(str(value) for value in normalized_daily_report["SET"].tolist())

                st.markdown(
                    f"""
                    <div class="mc-panel">
                      <div class="mc-heading">{selected_dataset_label} Backtest Results</div>
                      <div class="mc-subtle">{submitted_name}</div>
                      <div style="margin-top:0.65rem;">
                        <span class="mc-chip">{len(daily_report):,} days</span>
                        <span class="mc-chip">{ticks:,} ticks</span>
                        <span class="mc-chip">{snapshots:,} snapshots</span>
                        <span class="mc-chip">PnL / {normalization_ticks:,} ticks</span>
                        <span class="mc-chip">{own_trades:,} own trades</span>
                        {maf_chip}
                        <span class="mc-chip">simulated</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                for warning in backtest_warnings:
                    st.warning(str(warning))

                metric_left, metric_mid, metric_right = st.columns([1, 1, 1], gap="small")
                with metric_left:
                    mc_card(
                        f"Mean PnL Across {len(daily_report):,} Days / {normalization_ticks:,} ticks",
                        fmt_money(total_mean, 0),
                        f"95% mean CI {fmt_money(total_low, 0)} to {fmt_money(total_high, 0)}",
                    )
                with metric_mid:
                    mc_card(
                        f"Total PnL Across {len(daily_report):,} Days",
                        fmt_money(total_sum, 0),
                        f"Sum of {day_list_caption} after normalization.",
                    )
                with metric_right:
                    mc_card(
                        f"Total PnL 1σ / {normalization_ticks:,} ticks",
                        fmt_money(total_sd, 0),
                        f"P05 {fmt_money(total_low, 0)} · P95 {fmt_money(total_high, 0)}",
                    )

                top_left, top_right = st.columns([1.35, 1], gap="medium")
                with top_left:
                    st.markdown('<div class="mc-panel"><div class="mc-section-title">Profitability And Statistics</div>', unsafe_allow_html=True)
                    mc_table(submission_profitability_table(normalized_daily_report, normalized_product_report, normalized_trace))
                    st.markdown("</div>", unsafe_allow_html=True)
                with top_right:
                    st.markdown(
                        f"""
                        <div class="mc-panel">
                          <div class="mc-section-title">Fair Value Models</div>
                          <table class="mc-table">
                            <tr><td>ASH_COATED_OSMIUM</td><td>Uploaded Trader class decides fair value and quoting. Replay marks inventory to mid.</td></tr>
                            <tr><td>INTARIAN_PEPPER_ROOT</td><td>Uploaded Trader class decides entries/exits. Replay uses visible book crossing fills.</td></tr>
                          </table>
                          <div class="mc-note">This is a deterministic {selected_dataset_label} backtest over the provided CSV days. The fresh engine consumes visible book volume once per timestamp, passes previous own trades into TradingState, includes public market trades, and marks inventory to mid. Displayed PnL is normalized to {normalization_ticks:,} ticks while TICKS remains the actual replay length.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                summaries = [("Total PnL Summary", total_values)]
                for product, values in product_day_pnl_map(normalized_product_report).items():
                    summaries.append((f"{product} PnL Summary", values))
                summary_cols = st.columns(len(summaries), gap="medium")
                for col, (title, values) in zip(summary_cols, summaries):
                    with col:
                        st.markdown(f'<div class="mc-panel"><div class="mc-section-title">{title}</div>', unsafe_allow_html=True)
                        mc_table(submission_summary_table("Metric", values))
                        st.markdown("</div>", unsafe_allow_html=True)

                dist_left, scatter_right = st.columns([1, 1], gap="medium")
                with dist_left:
                    st.plotly_chart(
                        pnl_distribution_chart(total_values, "Total PnL Distribution", "#5b7cfa"),
                        use_container_width=True,
                        config={"displaylogo": False},
                    )
                with scatter_right:
                    st.plotly_chart(
                        cross_product_scatter(normalized_product_report),
                        use_container_width=True,
                        config={"displaylogo": False},
                    )

                product_items = list(product_day_pnl_map(normalized_product_report).items())
                product_chart_cols = st.columns(max(1, len(product_items)), gap="medium")
                product_colors = ["#6ccf9c", "#f28e2b", "#e15759"]
                for col, (color, (product_name, values)) in zip(product_chart_cols, zip(product_colors, product_items)):
                    with col:
                        st.plotly_chart(
                            pnl_distribution_chart(values, f"{product_name} PnL Distribution", color),
                            use_container_width=True,
                            config={"displaylogo": False},
                        )

                spread_left, spread_right = st.columns([1, 1], gap="medium")
                with spread_left:
                    st.plotly_chart(
                        profitability_distribution_chart(normalized_daily_report, normalized_product_report),
                        use_container_width=True,
                        config={"displaylogo": False},
                    )
                with spread_right:
                    st.plotly_chart(
                        stability_distribution_chart(normalized_trace),
                        use_container_width=True,
                        config={"displaylogo": False},
                    )

                best_left, worst_right = st.columns([1, 1], gap="medium")
                with best_left:
                    st.markdown('<div class="mc-panel"><div class="mc-section-title">Best Sessions</div>', unsafe_allow_html=True)
                    mc_table(submission_session_table(normalized_daily_report, normalized_product_report, normalized_trace, ascending=False))
                    st.markdown("</div>", unsafe_allow_html=True)
                with worst_right:
                    st.markdown('<div class="mc-panel"><div class="mc-section-title">Worst Sessions</div>', unsafe_allow_html=True)
                    mc_table(submission_session_table(normalized_daily_report, normalized_product_report, normalized_trace, ascending=True))
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(f'<div class="mc-panel"><div class="mc-section-title">Artifacts: log-only, PnL / {normalization_ticks:,} ticks</div>', unsafe_allow_html=True)
                mc_terminal(submission_artifact_text(normalized_daily_report, normalized_product_report))
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="mc-section-title">Individual Commodity Data</div>', unsafe_allow_html=True)
                product_order = ["INTARIAN_PEPPER_ROOT", "ASH_COATED_OSMIUM"]
                product_order.extend(
                    product_name
                    for product_name in product_day_pnl_map(normalized_product_report).keys()
                    if product_name not in product_order
                )
                product_order = [
                    product_name
                    for product_name in product_order
                    if product_name in set(normalized_product_report["PRODUCT"])
                ]
                detail_cols = st.columns(max(1, len(product_order)), gap="medium")
                for col, product_name in zip(detail_cols, product_order):
                    with col:
                        detail_table = submission_product_detail_table(
                            normalized_summary,
                            daily_report,
                            product_name,
                        )
                        st.markdown(
                            f'<div class="mc-panel"><div class="mc-section-title">{product_name}</div>',
                            unsafe_allow_html=True,
                        )
                        if detail_table.empty:
                            st.caption("No executions for this product.")
                        else:
                            mc_table(detail_table)
                        st.markdown("</div>", unsafe_allow_html=True)

                for product_name in product_day_pnl_map(normalized_product_report).keys():
                    st.plotly_chart(
                        trace_product_chart(normalized_trace, product_name),
                        use_container_width=True,
                        config={"displaylogo": False},
                    )

                with st.expander("Detailed execution tables and downloads", expanded=False):
                    detail_left, detail_right = st.columns([1, 1], gap="medium")
                    with detail_left:
                        st.markdown("**Stats**")
                        st.dataframe(normalized_stats_report, width="stretch", hide_index=True)
                        st.markdown("**Product/day execution**")
                        st.dataframe(normalized_summary, width="stretch", height=280, hide_index=True)
                    with detail_right:
                        st.markdown("**Trace preview**")
                        st.dataframe(
                            normalized_trace.sort_values("global_timestamp").tail(40),
                            width="stretch",
                            height=395,
                            hide_index=True,
                        )
                    download_cols = st.columns(4)
                    download_cols[0].download_button(
                        "Download days",
                        normalized_daily_report.to_csv(index=False),
                        "submission_day_summary.csv",
                        "text/csv",
                    )
                    download_cols[1].download_button(
                        "Download products",
                        normalized_product_report.to_csv(index=False),
                        "submission_product_pnl.csv",
                        "text/csv",
                    )
                    download_cols[2].download_button(
                        "Download stats",
                        normalized_stats_report.to_csv(index=False),
                        "submission_stats.csv",
                        "text/csv",
                    )
                    download_cols[3].download_button(
                        "Download trace",
                        normalized_trace.to_csv(index=False),
                        "submission_trace.csv",
                        "text/csv",
                    )

        st.markdown(
            '<div class="mc-title">Invest & Expand <span class="mc-chip">manual challenge</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="mc-panel">
              <div class="mc-section-title">Equations</div>
              <table class="mc-table">
                <tr><td>Budget used</td><td>50,000 × (Research% + Scale% + Speed%) / 100</td></tr>
                <tr><td>Research outcome</td><td>200,000 × ln(1 + Research%) / ln(101)</td></tr>
                <tr><td>Scale outcome</td><td>7 × Scale% / 100</td></tr>
                <tr><td>Speed outcome</td><td>0.9 − 0.8 × (rank − 1) / (field size − 1)</td></tr>
                <tr><td>Final PnL</td><td>Research outcome × Scale outcome × Speed outcome − Budget used</td></tr>
              </table>
              <div class="mc-note">Speed percentage only matters through your rank against other players. Since their allocations are unknown, this panel lets you choose an assumed rank and field size.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        invest_controls, invest_charts = st.columns([1.0, 2.1], gap="medium")
        with invest_controls:
            st.markdown('<div class="mc-panel"><div class="mc-section-title">Allocation</div>', unsafe_allow_html=True)
            research_pct = st.slider("Research %", 0.0, 100.0, 45.0, step=1.0)
            scale_pct = st.slider("Scale %", 0.0, 100.0, 35.0, step=1.0)
            speed_pct = st.slider("Speed %", 0.0, 100.0, 20.0, step=1.0)
            total_pct = research_pct + scale_pct + speed_pct
            field_size = st.number_input("Assumed field size", 2, 10000, 100, step=1)
            assumed_rank = st.number_input(
                "Assumed speed rank",
                1,
                int(field_size),
                max(1, int(field_size * 0.25)),
                step=1,
            )
            hit_rate = speed_multiplier(int(assumed_rank), int(field_size))
            used_budget = investment_budget_used(research_pct, scale_pct, speed_pct)
            gross_pnl = research_value(research_pct) * scale_value(scale_pct) * hit_rate
            final_pnl = gross_pnl - used_budget
            if total_pct > 100:
                st.error(f"Total allocation is {total_pct:.0f}%. It must be <= 100%.")
            else:
                st.success(f"Total allocation: {total_pct:.0f}%")
            mc_card("Research outcome", fmt_money(research_value(research_pct), 0), "Logarithmic edge curve.")
            mc_card("Scale outcome", f"{scale_value(scale_pct):.3f}", "Linear deployment breadth.")
            mc_card("Speed multiplier", f"{hit_rate:.3f}", f"Rank {int(assumed_rank)} of {int(field_size)}.")
            mc_card("Final PnL", fmt_money(final_pnl, 0), f"Gross {fmt_money(gross_pnl, 0)} minus budget {fmt_money(used_budget, 0)}.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="mc-panel"><div class="mc-section-title">Best split for Research + Scale = n</div>', unsafe_allow_html=True)
            rs_total_pct = st.slider("n = Research% + Scale%", 1.0, 100.0, 80.0, step=1.0)
            ideal_research, ideal_scale, ideal_edge = optimal_research_scale_split(rs_total_pct)
            scale_to_research = ideal_scale / ideal_research if ideal_research > 0 else float("inf")
            ideal_gross = ideal_edge * hit_rate
            ideal_final = ideal_gross - investment_budget_used(ideal_research, ideal_scale, speed_pct)
            mc_card(
                "Ideal Scale / Research",
                f"{ideal_scale:.1f}% / {ideal_research:.1f}%",
                f"Ratio S:R = {scale_to_research:.2f}:1",
            )
            mc_card(
                "Ideal PnL at this n",
                fmt_money(ideal_final, 0),
                f"Using current speed rank and Speed% = {speed_pct:.0f}.",
            )
            st.caption("This optimizes only the split between Research and Scale for a fixed n. Speed rank is handled separately.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="mc-panel"><div class="mc-section-title">PnL from Speed and Percentile</div>', unsafe_allow_html=True)
            chosen_speed_pct = st.slider("Speed chosen %", 0.0, 100.0, 33.0, step=1.0)
            speed_percentile = st.slider("Speed percentile", 0.0, 100.0, 75.0, step=1.0)
            reduced_total = max(0.0, 100.0 - chosen_speed_pct)
            reduced_research, reduced_scale, reduced_edge = optimal_research_scale_split(reduced_total)
            percentile_multiplier = 0.1 + 0.8 * speed_percentile / 100.0
            reduced_gross = reduced_edge * percentile_multiplier
            reduced_final = reduced_gross - investment_budget_used(
                reduced_research,
                reduced_scale,
                chosen_speed_pct,
            )
            reduced_ratio = reduced_scale / reduced_research if reduced_research > 0 else float("inf")
            mc_card(
                "Optimal R/S after Speed",
                f"R {reduced_research:.1f}% · S {reduced_scale:.1f}%",
                f"Scale:Research = {reduced_ratio:.2f}:1",
            )
            mc_card(
                "Percentile multiplier",
                f"{percentile_multiplier:.3f}",
                f"{speed_percentile:.0f}th percentile maps from 0.1 to 0.9.",
            )
            mc_card(
                "Reduced-form PnL",
                fmt_money(reduced_final, 0),
                f"Speed {chosen_speed_pct:.0f}%, n = {reduced_total:.0f}%.",
            )
            st.caption("Here percentile means the fraction of players whose Speed allocation you beat. This ignores exact rank ties.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="mc-panel"><div class="mc-section-title">Normal speed field</div>', unsafe_allow_html=True)
            normal_speed_pct = st.slider(
                "Your Speed % under normal model",
                0.0,
                100.0,
                33.0,
                step=1.0,
                key="manual_normal_speed_pct",
            )
            speed_mu = st.slider(
                "mu: average opponent Speed %",
                1.0,
                100.0,
                25.0,
                step=1.0,
                key="manual_speed_mu",
            )
            speed_sd = st.slider(
                "sd: opponent Speed standard deviation",
                1.0,
                50.0,
                17.0,
                step=1.0,
                key="manual_speed_sd",
            )
            normal_result = speed_mu_pnl(normal_speed_pct, speed_mu, speed_sd)
            normal_ratio = (
                normal_result["scale_pct"] / normal_result["research_pct"]
                if normal_result["research_pct"] > 0
                else float("inf")
            )
            mc_big_result(
                "PnL from normal speed model",
                fmt_money(normal_result["pnl"], 0),
                f"Your speed {normal_speed_pct:.0f}%, mu {speed_mu:.0f}, sd {speed_sd:.0f}, percentile {100 * normal_result['percentile']:.1f}%.",
            )
            mc_card(
                "Opponent speed model",
                f"N({speed_mu:.0f}, {speed_sd:.0f})",
                "Mean and standard deviation are controlled separately.",
            )
            mc_card(
                "Your implied percentile",
                f"{100 * normal_result['percentile']:.1f}%",
                f"Speed multiplier {normal_result['multiplier']:.3f}.",
            )
            mc_card(
                "Optimal R/S and PnL",
                f"R {normal_result['research_pct']:.1f}% · S {normal_result['scale_pct']:.1f}%",
                f"S:R {normal_ratio:.2f}:1 · PnL {fmt_money(normal_result['pnl'], 0)}",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with invest_charts:
            top_curve_left, top_curve_right = st.columns([1, 1], gap="medium")
            with top_curve_left:
                st.plotly_chart(
                    investment_pillar_chart(),
                    use_container_width=True,
                    config={"displaylogo": False},
                )
            with top_curve_right:
                st.plotly_chart(
                    investment_speed_chart(int(field_size), int(assumed_rank)),
                    use_container_width=True,
                    config={"displaylogo": False},
                )
            st.plotly_chart(
                investment_pnl_heatmap(speed_pct, hit_rate),
                use_container_width=True,
                config={"displaylogo": False},
            )
            split_left, split_right = st.columns([1, 1], gap="medium")
            with split_left:
                st.plotly_chart(
                    optimal_research_scale_chart(rs_total_pct),
                    use_container_width=True,
                    config={"displaylogo": False},
                )
            with split_right:
                st.plotly_chart(
                    optimal_research_scale_share_chart(rs_total_pct),
                    use_container_width=True,
                    config={"displaylogo": False},
                )
            st.plotly_chart(
                    speed_mu_pnl_chart(normal_speed_pct, speed_mu, speed_sd),
                use_container_width=True,
                config={"displaylogo": False},
            )

    if st.session_state.get(playback_active_key, False):
        current_timestamp = int(st.session_state[playback_cursor_key])
        next_timestamp = current_timestamp + int(playback_step)
        if next_timestamp > int(playback_end):
            st.session_state[playback_cursor_key] = int(playback_end)
            st.session_state[playback_active_key] = False
        else:
            st.session_state[playback_cursor_key] = int(next_timestamp)
        time.sleep(float(playback_delay))
        rerun()


if __name__ == "__main__":
    main()
