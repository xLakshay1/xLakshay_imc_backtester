from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import hashlib
import itertools
import io
import json
import math
import os
import re
import sys
import time
import types
from contextlib import redirect_stdout
from types import FunctionType

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import streamlit as st
from statsmodels.tsa.stattools import adfuller, coint


REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("IMC_DATA_ROOT", str(REPO_ROOT / "data")))


def _has_round_price_csvs(path: Path) -> bool:
    return path.exists() and any(path.glob("prices_round_*_day_*.csv"))


def _resolve_round_dir(env_var: str, default_dir: Path, fallback_dirs: tuple[Path, ...]) -> Path:
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value)
    if _has_round_price_csvs(default_dir):
        return default_dir
    for candidate in fallback_dirs:
        if _has_round_price_csvs(candidate):
            return candidate
    return default_dir


_P4_BUNDLED_ROUND_FALLBACKS = {
    "ROUND1": (
        REPO_ROOT.parent / "imc-prosperity-4" / "backtester" / "prosperity3bt" / "resources" / "round1",
        REPO_ROOT.parent / "imc-prosperity-4-fresh" / "backtester" / "prosperity3bt" / "resources" / "round1",
    ),
    "ROUND2": (
        REPO_ROOT.parent / "imc-prosperity-4" / "backtester" / "prosperity3bt" / "resources" / "round2",
        REPO_ROOT.parent / "imc-prosperity-4-fresh" / "backtester" / "prosperity3bt" / "resources" / "round2",
    ),
    "ROUND3": (
        REPO_ROOT.parent / "imc-prosperity-4" / "backtester" / "prosperity3bt" / "resources" / "round3",
        REPO_ROOT.parent / "imc-prosperity-4-fresh" / "backtester" / "prosperity3bt" / "resources" / "round3",
    ),
}

ROUND1_DATA_DIR = _resolve_round_dir(
    "IMC_ROUND1_DATA_DIR",
    DATA_ROOT / "ROUND1",
    _P4_BUNDLED_ROUND_FALLBACKS["ROUND1"],
)
ROUND2_DATA_DIR = _resolve_round_dir(
    "IMC_ROUND2_DATA_DIR",
    DATA_ROOT / "ROUND2",
    _P4_BUNDLED_ROUND_FALLBACKS["ROUND2"],
)
ROUND3_DATA_DIR = _resolve_round_dir(
    "IMC_ROUND3_DATA_DIR",
    DATA_ROOT / "ROUND3",
    _P4_BUNDLED_ROUND_FALLBACKS["ROUND3"],
)
DEFAULT_DATA_DIR = (
    ROUND3_DATA_DIR
    if ROUND3_DATA_DIR.exists()
    else ROUND2_DATA_DIR
    if ROUND2_DATA_DIR.exists()
    else ROUND1_DATA_DIR
)
DATASET_DIRS = {
    "Round 1": (ROUND1_DATA_DIR,),
    "Round 2": (ROUND2_DATA_DIR,),
    "Round 3": (ROUND3_DATA_DIR,),
    "Rounds 1 + 2": (ROUND1_DATA_DIR, ROUND2_DATA_DIR),
    "Rounds 1 + 2 + 3": (ROUND1_DATA_DIR, ROUND2_DATA_DIR, ROUND3_DATA_DIR),
}
DEFAULT_STRATEGY_ROOT = Path(os.environ.get("IMC_STRATEGY_ROOT", str(REPO_ROOT)))
SUBMISSION_HISTORY_DIR = DEFAULT_STRATEGY_ROOT / "dashboard_submission_history"
SUBMISSION_HISTORY_FILE = SUBMISSION_HISTORY_DIR / "history.json"
SUBMISSION_HISTORY_CODE_DIR = SUBMISSION_HISTORY_DIR / "code_snapshots"
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

ACTUAL_R2_SPEED_COUNTS = [
    453, 125, 84, 52, 27, 129, 27, 31, 30, 12, 249, 53, 29, 17, 5, 137, 35, 22, 23, 13,
    281, 78, 44, 33, 22, 172, 63, 54, 22, 14, 242, 57, 42, 68, 99, 155, 184, 118, 75, 37,
    227, 139, 119, 89, 40, 100, 69, 42, 20, 14, 86, 62, 56, 40, 22, 33, 21, 17, 17, 4,
    30, 15, 3, 7, 8, 9, 5, 4, 3, 3, 8, 9, 2, 0, 1, 1, 1, 2, 1, 1,
    4, 2, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2,
]

P3_RESOURCES_DIR = Path(
    os.environ.get(
        "IMC_P3_RESOURCES_DIR",
        str(REPO_ROOT / "backtester" / "prosperity3bt" / "resources"),
    )
)
P3_ROUND3_DATA_DIR = P3_RESOURCES_DIR / "round3"
P3_R3_UNDERLYING = "VOLCANIC_ROCK"
P3_R3_HUMAN_DISTRIBUTION_IMAGE = Path(
    os.environ.get(
        "IMC_P3_R3_HUMAN_DISTRIBUTION_IMAGE",
        str(REPO_ROOT / "assets" / "p3_r3_human_distribution.png"),
    )
)
P3_R3_OPTION_STRIKES = (9500, 9750, 10000, 10250, 10500)
P3_R3_OPTION_PRODUCTS = tuple(
    f"VOLCANIC_ROCK_VOUCHER_{strike}" for strike in P3_R3_OPTION_STRIKES
)
P3_R3_FRANKFURT_SMILE_COEFFS = np.array([0.27362531, 0.01007566, 0.14876677], dtype=float)
P3_R3_THR_OPEN = 0.5
P3_R3_THR_CLOSE = 0.0
P3_R3_LOW_VEGA_THR_ADJ = 0.5
P3_R3_THEO_NORM_WINDOW = 20
P3_R3_IV_SCALPING_WINDOW = 100
P3_R3_IV_SCALPING_THR = 0.7
P3_R3_UNDERLYING_MR_WINDOW = 10
P3_R3_UNDERLYING_MR_THR = 15.0
P3_R3_OPTIONS_MR_WINDOW = 30
P3_R3_OPTIONS_MR_THR = 5.0
P4_R3_UNDERLYING = "VELVETFRUIT_EXTRACT"
P4_R3_HYDROGEL = "HYDROGEL_PACK"
P4_R3_VOUCHER_PREFIX = "VEV_"
P4_R3_VOUCHER_STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
P4_R3_HISTORICAL_START_TTE_DAYS = 8.0
P4_R3_MANUAL_RESALE_PRICE = 920
P4_R3_MANUAL_RESERVE_VALUES = np.arange(670, 925, 5, dtype=float)
P4_R3_MANUAL_ESTIMATED_MU = 862.0


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


def p3_r3_standard_normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def p3_r3_standard_normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def p3_r3_bs_call_metrics(spot: float, strike: float, tau: float, sigma: float) -> tuple[float, float, float, float]:
    if any(math.isnan(value) for value in (spot, strike, tau, sigma)) or spot <= 0 or strike <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    if tau <= 0 or sigma <= 0:
        intrinsic = max(spot - strike, 0.0)
        delta = 1.0 if spot > strike else 0.0
        return intrinsic, delta, 0.0, 0.0

    root_tau = math.sqrt(tau)
    d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * tau) / (sigma * root_tau)
    d2 = d1 - sigma * root_tau
    call_value = spot * p3_r3_standard_normal_cdf(d1) - strike * p3_r3_standard_normal_cdf(d2)
    delta = p3_r3_standard_normal_cdf(d1)
    gamma = p3_r3_standard_normal_pdf(d1) / max(spot * sigma * root_tau, 1e-12)
    vega = spot * p3_r3_standard_normal_pdf(d1) * root_tau
    return call_value, delta, gamma, vega


def p3_r3_implied_volatility(spot: float, strike: float, tau: float, price: float) -> float:
    if any(pd.isna(value) for value in (spot, strike, tau, price)):
        return float("nan")
    intrinsic = max(float(spot) - float(strike), 0.0)
    if price <= intrinsic + 1e-9 or tau <= 0 or spot <= 0 or strike <= 0:
        return float("nan")

    low, high = 1e-4, 5.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        mid_price, _, _, _ = p3_r3_bs_call_metrics(float(spot), float(strike), float(tau), float(mid))
        if pd.isna(mid_price):
            return float("nan")
        if mid_price > price:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


@st.cache_data(show_spinner=False)
def load_p4_r3_option_market() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not ROUND3_DATA_DIR.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    files = discover_files(str(ROUND3_DATA_DIR))
    prices = load_prices(files.price_paths)
    trades = load_trades(files.trade_paths)
    if prices.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    products = {P4_R3_UNDERLYING, P4_R3_HYDROGEL, *[f"{P4_R3_VOUCHER_PREFIX}{strike}" for strike in P4_R3_VOUCHER_STRIKES]}
    round3_prices = prices[prices["product"].isin(products)].copy()
    round3_trades = trades[trades["symbol"].isin(products)].copy()

    if round3_prices.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    options = round3_prices[round3_prices["product"].str.startswith(P4_R3_VOUCHER_PREFIX, na=False)].copy()
    options["strike"] = (
        options["product"].str.extract(r"(\d+)$").iloc[:, 0].astype(float)
    )
    options["option_wall_mid"] = options["mid_price"]
    ask_only = options["option_wall_mid"].isna() & options["ask_price_1"].notna()
    bid_only = options["option_wall_mid"].isna() & options["bid_price_1"].notna()
    options.loc[ask_only, "option_wall_mid"] = options.loc[ask_only, "ask_price_1"] - 0.5
    options.loc[bid_only, "option_wall_mid"] = options.loc[bid_only, "bid_price_1"] + 0.5

    underlying = round3_prices[round3_prices["product"] == P4_R3_UNDERLYING].copy()
    underlying["underlying_wall_mid"] = underlying["mid_price"]
    ask_only = underlying["underlying_wall_mid"].isna() & underlying["ask_price_1"].notna()
    bid_only = underlying["underlying_wall_mid"].isna() & underlying["bid_price_1"].notna()
    underlying.loc[ask_only, "underlying_wall_mid"] = underlying.loc[ask_only, "ask_price_1"] - 0.5
    underlying.loc[bid_only, "underlying_wall_mid"] = underlying.loc[bid_only, "bid_price_1"] + 0.5

    merge_columns = [
        "day",
        "timestamp",
        "underlying_wall_mid",
        "bid_price_1",
        "ask_price_1",
    ]
    options = options.merge(
        underlying[merge_columns].rename(
            columns={
                "bid_price_1": "underlying_bid_1",
                "ask_price_1": "underlying_ask_1",
            }
        ),
        on=["day", "timestamp"],
        how="left",
    )
    return options, underlying, round3_trades


@st.cache_data(show_spinner=False)
def build_p4_r3_option_analysis(selected_day: int, sample_points_per_strike: int = 350) -> dict[str, object]:
    options_raw, underlying_raw, trades_raw = load_p4_r3_option_market()
    if options_raw.empty:
        return {
            "options": pd.DataFrame(),
            "underlying": pd.DataFrame(),
            "trades": pd.DataFrame(),
            "scatter": pd.DataFrame(),
            "coeffs": np.array([float("nan")] * 3, dtype=float),
            "strike_table": pd.DataFrame(),
        }

    options = options_raw[options_raw["day"] == int(selected_day)].copy()
    underlying = underlying_raw[underlying_raw["day"] == int(selected_day)].copy()
    trades = trades_raw[trades_raw["day"] == int(selected_day)].copy()
    if options.empty or underlying.empty:
        return {
            "options": pd.DataFrame(),
            "underlying": underlying,
            "trades": trades,
            "scatter": pd.DataFrame(),
            "coeffs": np.array([float("nan")] * 3, dtype=float),
            "strike_table": pd.DataFrame(),
        }

    options["tte_days"] = (
        P4_R3_HISTORICAL_START_TTE_DAYS - float(selected_day) - options["timestamp"] / 1_000_000.0
    ).clip(lower=0.05)
    options["tau_years"] = options["tte_days"] / 365.0
    options["underlying_price"] = pd.to_numeric(options["underlying_wall_mid"], errors="coerce")
    options["option_price"] = pd.to_numeric(options["option_wall_mid"], errors="coerce")
    options["voucher"] = options["product"]
    options["intrinsic"] = (options["underlying_price"] - options["strike"]).clip(lower=0.0)
    options["extrinsic"] = options["option_price"] - options["intrinsic"]
    options["log_moneyness"] = np.log(
        options["strike"] / options["underlying_price"].replace(0, pd.NA)
    )

    iv_inputs = options[["underlying_price", "strike", "tau_years", "option_price"]].itertuples(index=False, name=None)
    options["market_iv"] = [
        p3_r3_implied_volatility(spot, strike, tau, price)
        for spot, strike, tau, price in iv_inputs
    ]

    fit_frame = options.dropna(subset=["market_iv", "log_moneyness"]).copy()
    fit_frame = fit_frame[fit_frame["market_iv"].between(0.01, 3.0)]
    if len(fit_frame) >= 12:
        coeffs = np.polyfit(fit_frame["log_moneyness"], fit_frame["market_iv"], 2)
    else:
        coeffs = np.array([0.0, 0.0, float(fit_frame["market_iv"].median()) if not fit_frame.empty else 0.25], dtype=float)

    options["smile_iv"] = np.polyval(coeffs, options["log_moneyness"])
    options["iv_residual"] = options["market_iv"] - options["smile_iv"]

    scatter = (
        fit_frame.groupby("strike", group_keys=False)
        .apply(lambda frame: frame.sample(min(len(frame), int(sample_points_per_strike)), random_state=42))
        .reset_index(drop=True)
        if not fit_frame.empty
        else fit_frame
    )

    strike_table = (
        options.groupby(["product", "strike"], as_index=False)
        .agg(
            mean_iv=("market_iv", "mean"),
            mean_smile_iv=("smile_iv", "mean"),
            mean_residual=("iv_residual", "mean"),
            residual_std=("iv_residual", "std"),
            mean_price=("option_price", "mean"),
        )
        .sort_values("strike")
    )

    return {
        "options": options.sort_values(["timestamp", "strike"]),
        "underlying": underlying.sort_values("timestamp"),
        "trades": trades.sort_values(["timestamp", "symbol"]),
        "scatter": scatter.sort_values(["timestamp", "strike"]) if not scatter.empty else scatter,
        "coeffs": np.asarray(coeffs, dtype=float),
        "strike_table": strike_table,
    }


def p4_r3_smile_fit_chart(
    scatter: pd.DataFrame,
    options: pd.DataFrame,
    coeffs: np.ndarray,
    focus_timestamp: int,
) -> go.Figure:
    fig = go.Figure()
    if scatter.empty or options.empty:
        return apply_mc_chart_layout(fig, "Round 3 Option Smile Fit", height=380)

    for strike, sub in scatter.groupby("strike", sort=True):
        fig.add_trace(
            go.Scatter(
                x=sub["log_moneyness"],
                y=sub["market_iv"],
                mode="markers",
                name=f"VEV_{int(strike)}",
                marker={"size": 4, "opacity": 0.28},
                hovertemplate="m=%{x:.4f}<br>IV=%{y:.4f}<extra></extra>",
            )
        )

    grid_x = np.linspace(float(scatter["log_moneyness"].min()), float(scatter["log_moneyness"].max()), 200)
    fig.add_trace(
        go.Scatter(
            x=grid_x,
            y=np.polyval(coeffs, grid_x),
            mode="lines",
            name="Quadratic smile fit",
            line={"color": "#111111", "width": 4},
            hovertemplate="m=%{x:.4f}<br>fit=%{y:.4f}<extra></extra>",
        )
    )

    snap = options.loc[options["timestamp"] == int(focus_timestamp)].dropna(subset=["market_iv", "log_moneyness"])
    if not snap.empty:
        fig.add_trace(
            go.Scatter(
                x=snap["log_moneyness"],
                y=snap["market_iv"],
                mode="markers+text",
                name=f"Current snapshot t={focus_timestamp}",
                text=[f"{int(k)}" for k in snap["strike"]],
                textposition="top center",
                marker={"size": 9, "color": "#ff5f5f", "line": {"color": "#ffffff", "width": 1}},
                hovertemplate="strike=%{text}<br>m=%{x:.4f}<br>IV=%{y:.4f}<extra></extra>",
            )
        )

    fig.update_xaxes(title="log-moneyness log(K / S)")
    fig.update_yaxes(title="observed implied volatility")
    return apply_mc_chart_layout(fig, "Round 3 Option Smile Fit (Parabola Across Strikes)", height=420)


def p4_r3_iv_residual_chart(options: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if options.empty:
        return apply_mc_chart_layout(fig, "Round 3 Detrended IV Residuals", height=380)

    for strike, sub in options.dropna(subset=["iv_residual"]).groupby("strike", sort=True):
        fig.add_trace(
            go.Scatter(
                x=sub["timestamp"],
                y=sub["iv_residual"],
                mode="lines",
                name=f"VEV_{int(strike)}",
                line={"width": 1.8},
                hovertemplate="t=%{x}<br>resid=%{y:.4f}<extra></extra>",
            )
        )
    fig.add_hline(y=0.0, line={"color": "#8792a2", "width": 1, "dash": "dot"})
    fig.update_xaxes(title="timestamp")
    fig.update_yaxes(title="market IV - quadratic smile IV")
    return apply_mc_chart_layout(fig, "Round 3 Detrended IV Residuals", height=420)


def p4_r3_snapshot_residual_chart(options: pd.DataFrame, focus_timestamp: int) -> go.Figure:
    fig = go.Figure()
    if options.empty:
        return apply_mc_chart_layout(fig, "Snapshot Residuals by Strike", height=360)

    snapshot = (
        options.assign(ts_distance=(options["timestamp"] - int(focus_timestamp)).abs())
        .sort_values(["strike", "ts_distance", "timestamp"])
        .groupby("strike", as_index=False)
        .first()
    )
    snapshot = snapshot.dropna(subset=["iv_residual", "market_iv", "smile_iv"])
    if snapshot.empty:
        return apply_mc_chart_layout(fig, "Snapshot Residuals by Strike", height=360)

    fig.add_trace(
        go.Bar(
            x=[f"VEV_{int(strike)}" for strike in snapshot["strike"]],
            y=snapshot["iv_residual"],
            marker_color=["#59c17a" if value < 0 else "#ff7b72" for value in snapshot["iv_residual"]],
            text=[f"{value:+.4f}" for value in snapshot["iv_residual"]],
            textposition="outside",
            name="Detrended IV residual",
            hovertemplate="%{x}<br>residual=%{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(y=0.0, line={"color": "#8792a2", "width": 1, "dash": "dot"})
    fig.update_xaxes(title="voucher strike")
    fig.update_yaxes(title="IV residual")
    return apply_mc_chart_layout(fig, "Snapshot Residuals by Strike", height=360)


@st.cache_data(show_spinner=False)
def build_p4_r3_delta1_diagnostics(
    selected_day: int,
    rolling_window: int = 120,
    beta_window: int = 250,
    max_lag: int = 20,
    random_baselines: int = 40,
) -> dict[str, object]:
    if not ROUND3_DATA_DIR.exists():
        return {}

    files = discover_files(str(ROUND3_DATA_DIR))
    prices = load_prices(files.price_paths)
    if prices.empty:
        return {}

    subset = prices[
        (prices["day"] == int(selected_day))
        & (prices["product"].isin([P4_R3_HYDROGEL, P4_R3_UNDERLYING]))
    ].copy()
    if subset.empty:
        return {}

    subset["wall_mid"] = subset["mid_price"]
    ask_only = subset["wall_mid"].isna() & subset["ask_price_1"].notna()
    bid_only = subset["wall_mid"].isna() & subset["bid_price_1"].notna()
    subset.loc[ask_only, "wall_mid"] = subset.loc[ask_only, "ask_price_1"] - 0.5
    subset.loc[bid_only, "wall_mid"] = subset.loc[bid_only, "bid_price_1"] + 0.5
    bid_cols = [f"bid_volume_{level}" for level in (1, 2, 3)]
    ask_cols = [f"ask_volume_{level}" for level in (1, 2, 3)]
    subset["bid_total"] = subset[bid_cols].fillna(0).sum(axis=1)
    subset["ask_total"] = subset[ask_cols].fillna(0).sum(axis=1)
    subset["depth_total"] = subset["bid_total"] + subset["ask_total"]
    subset["imbalance"] = np.where(
        subset["depth_total"] > 0,
        (subset["bid_total"] - subset["ask_total"]) / subset["depth_total"],
        np.nan,
    )
    subset["l1_total"] = subset["bid_volume_1"].fillna(0) + subset["ask_volume_1"].fillna(0)
    subset["l1_imbalance"] = np.where(
        subset["l1_total"] > 0,
        (subset["bid_volume_1"].fillna(0) - subset["ask_volume_1"].fillna(0)) / subset["l1_total"],
        np.nan,
    )

    wide = (
        subset.pivot_table(index="timestamp", columns="product", values="wall_mid", aggfunc="last")
        .sort_index()
        .rename_axis(columns=None)
    )
    if P4_R3_HYDROGEL not in wide.columns or P4_R3_UNDERLYING not in wide.columns:
        return {}
    wide = wide[[P4_R3_HYDROGEL, P4_R3_UNDERLYING]].dropna().copy()
    if len(wide) < max(rolling_window + 5, beta_window + 5, 100):
        return {}

    wide["hydrogel_log"] = np.log(wide[P4_R3_HYDROGEL].replace(0, np.nan))
    wide["velvet_log"] = np.log(wide[P4_R3_UNDERLYING].replace(0, np.nan))
    wide["hydrogel_ret"] = wide["hydrogel_log"].diff()
    wide["velvet_ret"] = wide["velvet_log"].diff()
    wide = wide.dropna().copy()
    if len(wide) < max(rolling_window + 5, beta_window + 5, 80):
        return {}

    rolling_autocorr_rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(42 + int(selected_day))
    for product_name, ret_col in [("HYDROGEL_PACK", "hydrogel_ret"), ("VELVETFRUIT_EXTRACT", "velvet_ret")]:
        series = wide[ret_col].astype(float)
        real = series.rolling(rolling_window).corr(series.shift(1))
        for timestamp, value in real.dropna().items():
            rolling_autocorr_rows.append(
                {"timestamp": int(timestamp), "value": float(value), "series": product_name}
            )

        centered_std = float(series.std(ddof=1))
        if not np.isfinite(centered_std) or centered_std <= 0:
            centered_std = 1e-6
        for baseline_id in range(int(random_baselines)):
            random_series = pd.Series(
                rng.normal(0.0, centered_std, size=len(series)),
                index=series.index,
            )
            rand_ac = random_series.rolling(rolling_window).corr(random_series.shift(1))
            for timestamp, value in rand_ac.dropna().items():
                rolling_autocorr_rows.append(
                    {
                        "timestamp": int(timestamp),
                        "value": float(value),
                        "series": product_name,
                        "baseline": baseline_id + 1,
                        "kind": "random",
                    }
                )

    autocorr_frame = pd.DataFrame(rolling_autocorr_rows)
    autocorr_frame["kind"] = autocorr_frame.get("kind", "real").fillna("real")

    x = wide["hydrogel_log"].astype(float)
    y = wide["velvet_log"].astype(float)
    coint_t, coint_p, crit = coint(y, x)
    ols_fit = sm.OLS(y, sm.add_constant(x)).fit()
    beta_static = float(ols_fit.params.iloc[1])
    alpha_static = float(ols_fit.params.iloc[0])
    wide["spread_static"] = y - (alpha_static + beta_static * x)

    rolling_cov = wide["velvet_log"].rolling(beta_window).cov(wide["hydrogel_log"])
    rolling_var = wide["hydrogel_log"].rolling(beta_window).var()
    wide["rolling_beta"] = rolling_cov / rolling_var.replace(0, np.nan)
    wide["rolling_beta"] = wide["rolling_beta"].replace([np.inf, -np.inf], np.nan).ffill()
    wide["rolling_alpha"] = (
        wide["velvet_log"].rolling(beta_window).mean()
        - wide["rolling_beta"] * wide["hydrogel_log"].rolling(beta_window).mean()
    )
    wide["rolling_spread"] = wide["velvet_log"] - (
        wide["rolling_alpha"] + wide["rolling_beta"] * wide["hydrogel_log"]
    )
    spread_mean = wide["rolling_spread"].rolling(beta_window).mean()
    spread_std = wide["rolling_spread"].rolling(beta_window).std().replace(0, np.nan)
    wide["spread_z"] = (wide["rolling_spread"] - spread_mean) / spread_std

    lag_rows: list[dict[str, float | int]] = []
    hydro_ret = wide["hydrogel_ret"]
    velvet_ret = wide["velvet_ret"]
    for lag in range(-int(max_lag), int(max_lag) + 1):
        corr_h_to_v = hydro_ret.corr(velvet_ret.shift(-lag))
        corr_v_to_h = velvet_ret.corr(hydro_ret.shift(-lag))
        lag_rows.append({"lag": lag, "pair": "Hydrogel leads Velvet", "corr": float(corr_h_to_v)})
        lag_rows.append({"lag": lag, "pair": "Velvet leads Hydrogel", "corr": float(corr_v_to_h)})
    lead_lag_frame = pd.DataFrame(lag_rows)

    adf_frame = wide["spread_static"].dropna()
    adf_p_value = float("nan")
    if len(adf_frame) > 40:
        try:
            adf_p_value = float(adfuller(adf_frame, maxlag=1, regression="c", autolag=None)[1])
        except Exception:
            adf_p_value = float("nan")

    half_life = float("nan")
    spread_reg = pd.DataFrame(
        {"delta": wide["spread_static"].diff(), "lagged": wide["spread_static"].shift(1)}
    ).dropna()
    if len(spread_reg) > 25:
        try:
            hl_fit = sm.OLS(spread_reg["delta"], sm.add_constant(spread_reg["lagged"])).fit()
            decay = float(hl_fit.params.iloc[1])
            if np.isfinite(decay) and decay < 0:
                half_life = float(np.log(2.0) / (-decay))
        except Exception:
            half_life = float("nan")

    signal_rows: list[dict[str, object]] = []
    regime_rows: list[dict[str, object]] = []
    for product_name in [P4_R3_HYDROGEL, P4_R3_UNDERLYING]:
        prod = subset[subset["product"] == product_name].sort_values("timestamp").copy()
        prod["log_price"] = np.log(prod["wall_mid"].replace(0, np.nan))
        prod["ret_1"] = prod["log_price"].diff()
        for horizon in (1, 5, 10):
            prod[f"future_ret_{horizon}"] = prod["log_price"].shift(-horizon) - prod["log_price"]

        for feature_name, label in [("imbalance", "Total imbalance"), ("l1_imbalance", "L1 imbalance")]:
            for horizon in (1, 5, 10):
                frame = prod[[feature_name, f"future_ret_{horizon}"]].dropna()
                if frame.empty:
                    corr = float("nan")
                    beta_bp = float("nan")
                else:
                    corr = float(frame[feature_name].corr(frame[f"future_ret_{horizon}"]))
                    beta_bp = float(frame[f"future_ret_{horizon}"].cov(frame[feature_name]) / frame[feature_name].var()) * 1e4
                signal_rows.append(
                    {
                        "product": product_name,
                        "feature": label,
                        "horizon": horizon,
                        "corr": corr,
                        "beta_bp": beta_bp,
                    }
                )

        depth_rank = prod["depth_total"].rank(method="first")
        if depth_rank.notna().sum() >= 9:
            prod["depth_regime"] = pd.qcut(depth_rank, 3, labels=["thin", "mid", "thick"])
            for regime, frame in prod.groupby("depth_regime", observed=False):
                ret_frame = frame["ret_1"].dropna()
                if len(ret_frame) < 10:
                    continue
                regime_rows.append(
                    {
                        "product": product_name,
                        "regime": str(regime),
                        "lag1_corr": float(ret_frame.autocorr(lag=1)),
                        "flip_rate": float((np.sign(ret_frame) != np.sign(ret_frame.shift(1))).dropna().mean()),
                    }
                )

    signal_frame = pd.DataFrame(signal_rows)
    regime_frame = pd.DataFrame(regime_rows)

    summary = pd.DataFrame(
        [
            {
                "Metric": "Static cointegration beta",
                "Value": f"{beta_static:.4f}",
                "Meaning": "OLS slope in log-price space for Velvet on Hydrogel.",
            },
            {
                "Metric": "Engle-Granger t-stat",
                "Value": f"{coint_t:.4f}",
                "Meaning": "More negative means stronger evidence of a stationary spread.",
            },
            {
                "Metric": "Engle-Granger p-value",
                "Value": f"{coint_p:.6f}",
                "Meaning": "Small p-value supports cointegration rather than just visual co-movement.",
            },
            {
                "Metric": "Spread ADF p-value",
                "Value": f"{adf_p_value:.6f}" if np.isfinite(adf_p_value) else "n/a",
                "Meaning": "Tests whether the static Hydrogel/Velvet spread itself looks stationary on this day.",
            },
            {
                "Metric": "Spread half-life",
                "Value": f"{half_life:.2f} ticks" if np.isfinite(half_life) else "n/a",
                "Meaning": "Crude OU-style estimate of how quickly spread shocks decay back toward equilibrium.",
            },
            {
                "Metric": "Hydrogel mean rolling AR(1)",
                "Value": f"{autocorr_frame[(autocorr_frame['series']=='HYDROGEL_PACK') & (autocorr_frame['kind']=='real')]['value'].mean():.4f}",
                "Meaning": "Negative values indicate short-horizon mean reversion in returns.",
            },
            {
                "Metric": "Velvet mean rolling AR(1)",
                "Value": f"{autocorr_frame[(autocorr_frame['series']=='VELVETFRUIT_EXTRACT') & (autocorr_frame['kind']=='real')]['value'].mean():.4f}",
                "Meaning": "More negative than Hydrogel means cleaner short-horizon snap-back.",
            },
            {
                "Metric": "Best lead-lag corr",
                "Value": f"{lead_lag_frame.loc[lead_lag_frame['corr'].abs().idxmax(), 'corr']:.4f}",
                "Meaning": f"{lead_lag_frame.loc[lead_lag_frame['corr'].abs().idxmax(), 'pair']} at lag {int(lead_lag_frame.loc[lead_lag_frame['corr'].abs().idxmax(), 'lag'])}.",
            },
            {
                "Metric": "Velvet total imbalance -> h1 return",
                "Value": f"{signal_frame[(signal_frame['product']==P4_R3_UNDERLYING) & (signal_frame['feature']=='Total imbalance') & (signal_frame['horizon']==1)]['corr'].iloc[0]:.4f}",
                "Meaning": "Negative means broad-book bid pressure tends to reverse rather than continue.",
            },
            {
                "Metric": "Velvet thick-book AR(1)",
                "Value": f"{regime_frame[(regime_frame['product']==P4_R3_UNDERLYING) & (regime_frame['regime']=='thick')]['lag1_corr'].iloc[0]:.4f}" if not regime_frame[(regime_frame['product']==P4_R3_UNDERLYING) & (regime_frame['regime']=='thick')].empty else "n/a",
                "Meaning": "More negative in thick books means snap-back is strongest when the visible book is deepest.",
            },
        ]
    )

    return {
        "wide": wide.reset_index(),
        "autocorr_frame": autocorr_frame,
        "lead_lag_frame": lead_lag_frame,
        "signal_frame": signal_frame,
        "regime_frame": regime_frame,
        "summary": summary,
        "cointegration": {
            "beta": beta_static,
            "alpha": alpha_static,
            "t_stat": float(coint_t),
            "p_value": float(coint_p),
            "crit_1": float(crit[0]),
            "crit_5": float(crit[1]),
            "crit_10": float(crit[2]),
            "adf_p_value": adf_p_value,
            "half_life": half_life,
        },
    }


def p4_r3_rolling_autocorr_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return apply_mc_chart_layout(fig, "Rolling Return Autocorrelation vs Random", height=420)

    for product in ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]:
        random_sub = frame[(frame["series"] == product) & (frame["kind"] == "random")]
        for _, sub in random_sub.groupby("baseline", sort=False):
            fig.add_trace(
                go.Scatter(
                    x=sub["timestamp"],
                    y=sub["value"],
                    mode="lines",
                    line={"color": "rgba(180,180,180,0.10)", "width": 1},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        real = frame[(frame["series"] == product) & (frame["kind"] == "real")]
        fig.add_trace(
            go.Scatter(
                x=real["timestamp"],
                y=real["value"],
                mode="lines",
                name=product,
                line={"width": 3},
                hovertemplate="t=%{x}<br>rolling AR(1)=%{y:.4f}<extra></extra>",
            )
        )

    fig.add_hline(y=0.0, line={"color": "#8792a2", "width": 1, "dash": "dot"})
    fig.update_xaxes(title="timestamp")
    fig.update_yaxes(title="rolling lag-1 return autocorrelation")
    return apply_mc_chart_layout(fig, "Rolling Return Autocorrelation vs Random", height=420)


def p4_r3_rolling_beta_spread_chart(frame: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("Rolling beta (Velvet on Hydrogel)", "Rolling spread z-score"),
    )
    if frame.empty:
        return apply_mc_chart_layout(fig, "Rolling Beta and Spread", height=520)

    fig.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["rolling_beta"],
            mode="lines",
            name="rolling beta",
            line={"color": "#5dade2", "width": 2.5},
            hovertemplate="t=%{x}<br>beta=%{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["spread_z"],
            mode="lines",
            name="spread z-score",
            line={"color": "#f4a261", "width": 2.5},
            hovertemplate="t=%{x}<br>z=%{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0.0, line={"color": "#8792a2", "width": 1, "dash": "dot"}, row=2, col=1)
    fig.add_hline(y=2.0, line={"color": "#8792a2", "width": 1, "dash": "dash"}, row=2, col=1)
    fig.add_hline(y=-2.0, line={"color": "#8792a2", "width": 1, "dash": "dash"}, row=2, col=1)
    fig.update_xaxes(title="timestamp", row=2, col=1)
    fig.update_yaxes(title="beta", row=1, col=1)
    fig.update_yaxes(title="spread z", row=2, col=1)
    return apply_mc_chart_layout(fig, "Rolling Beta and Spread", height=560)


def p4_r3_lead_lag_heatmap(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return apply_mc_chart_layout(fig, "Lead-Lag Correlation Heatmap", height=320)
    pivot = frame.pivot(index="pair", columns="lag", values="corr").loc[
        ["Hydrogel leads Velvet", "Velvet leads Hydrogel"]
    ]
    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.astype(int),
            y=pivot.index,
            colorscale="RdBu",
            zmid=0.0,
            colorbar={"title": "corr"},
            hovertemplate="%{y}<br>lag=%{x}<br>corr=%{z:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="lag")
    fig.update_yaxes(title="")
    return apply_mc_chart_layout(fig, "Lead-Lag Correlation Heatmap", height=320)


def p4_r3_delta1_signal_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return apply_mc_chart_layout(fig, "Book Imbalance vs Future Returns", height=360)

    chart = frame[frame["horizon"].isin([1, 5])].copy()
    chart["label"] = chart["product"].map(
        {
            P4_R3_HYDROGEL: "Hydrogel",
            P4_R3_UNDERLYING: "Velvet",
        }
    ) + " · " + chart["feature"]
    for horizon, sub in chart.groupby("horizon", sort=True):
        fig.add_trace(
            go.Bar(
                x=sub["label"],
                y=sub["corr"],
                name=f"h={int(horizon)}",
                text=[f"{value:+.3f}" for value in sub["corr"]],
                textposition="outside",
                hovertemplate="%{x}<br>corr=%{y:.4f}<extra></extra>",
            )
        )

    fig.add_hline(y=0.0, line={"color": "#8792a2", "width": 1, "dash": "dot"})
    fig.update_xaxes(title="")
    fig.update_yaxes(title="corr(signal, future return)")
    return apply_mc_chart_layout(fig, "Book Imbalance vs Future Returns", height=380)


def p4_r3_depth_regime_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return apply_mc_chart_layout(fig, "Depth Regime and Mean Reversion", height=360)

    order = ["thin", "mid", "thick"]
    colors = {
        P4_R3_HYDROGEL: "#5dade2",
        P4_R3_UNDERLYING: "#f4a261",
    }
    for product_name in [P4_R3_HYDROGEL, P4_R3_UNDERLYING]:
        sub = frame[frame["product"] == product_name].copy()
        sub["regime"] = pd.Categorical(sub["regime"], categories=order, ordered=True)
        sub = sub.sort_values("regime")
        fig.add_trace(
            go.Bar(
                x=sub["regime"],
                y=sub["lag1_corr"],
                name="Hydrogel" if product_name == P4_R3_HYDROGEL else "Velvet",
                marker_color=colors[product_name],
                text=[f"{value:+.3f}" for value in sub["lag1_corr"]],
                textposition="outside",
                hovertemplate="%{x}<br>lag1 corr=%{y:.4f}<extra></extra>",
            )
        )
    fig.add_hline(y=0.0, line={"color": "#8792a2", "width": 1, "dash": "dot"})
    fig.update_xaxes(title="depth regime")
    fig.update_yaxes(title="lag-1 return autocorr")
    return apply_mc_chart_layout(fig, "Depth Regime and Mean Reversion", height=360)


@st.cache_data(show_spinner=False)
def build_p4_r3_option_microstructure_diagnostics(selected_day: int, max_lag: int = 12) -> dict[str, object]:
    analysis = build_p4_r3_option_analysis(int(selected_day))
    options = analysis.get("options", pd.DataFrame()).copy()
    underlying = analysis.get("underlying", pd.DataFrame()).copy()
    if options.empty or underlying.empty:
        return {}

    options = options.sort_values(["strike", "timestamp"]).copy()
    options["option_ret_1"] = options.groupby("strike", sort=False)["option_price"].diff()
    for horizon in (1, 3, 5):
        options[f"future_ret_{horizon}"] = (
            options.groupby("strike", sort=False)["option_price"].shift(-horizon) - options["option_price"]
        )

    signal_rows: list[dict[str, object]] = []
    for strike, sub in options.groupby("strike", sort=True):
        for horizon in (1, 3, 5):
            frame = sub[["iv_residual", f"future_ret_{horizon}"]].dropna()
            corr = float(frame["iv_residual"].corr(frame[f"future_ret_{horizon}"])) if not frame.empty else float("nan")
            hit_rate = float((np.sign(frame["iv_residual"]) != np.sign(frame[f"future_ret_{horizon}"])).mean()) if not frame.empty else float("nan")
            signal_rows.append(
                {
                    "strike": int(strike),
                    "voucher": f"VEV_{int(strike)}",
                    "horizon": horizon,
                    "corr": corr,
                    "mean_reversion_hit": hit_rate,
                }
            )
    residual_signal = pd.DataFrame(signal_rows)

    under = underlying.sort_values("timestamp").copy()
    under["under_ret"] = np.log(under["underlying_wall_mid"].replace(0, np.nan)).diff()
    lead_lag_rows: list[dict[str, object]] = []
    for strike, sub in options.groupby("strike", sort=True):
        merged = pd.merge(
            under[["timestamp", "under_ret"]],
            sub[["timestamp", "option_ret_1"]],
            on="timestamp",
            how="inner",
        ).dropna()
        for lag in range(-int(max_lag), int(max_lag) + 1):
            corr = float(merged["under_ret"].corr(merged["option_ret_1"].shift(-lag)))
            lead_lag_rows.append(
                {
                    "strike": int(strike),
                    "voucher": f"VEV_{int(strike)}",
                    "lag": lag,
                    "corr": corr,
                }
            )
    under_option_lag = pd.DataFrame(lead_lag_rows)

    options["spread"] = options["ask_price_1"] - options["bid_price_1"]
    options["bid_total"] = options[[f"bid_volume_{level}" for level in (1, 2, 3)]].fillna(0).sum(axis=1)
    options["ask_total"] = options[[f"ask_volume_{level}" for level in (1, 2, 3)]].fillna(0).sum(axis=1)
    options["depth_total"] = options["bid_total"] + options["ask_total"]
    options["total_imbalance"] = np.where(
        options["depth_total"] > 0,
        (options["bid_total"] - options["ask_total"]) / options["depth_total"],
        np.nan,
    )
    options["l1_size_pair"] = (
        options["bid_volume_1"].fillna(0).astype(int).astype(str)
        + " x "
        + options["ask_volume_1"].fillna(0).astype(int).astype(str)
    )
    options["quote_signature"] = (
        options["bid_price_1"].fillna(-1).astype(int).astype(str)
        + "|"
        + options["ask_price_1"].fillna(-1).astype(int).astype(str)
        + "|"
        + options["l1_size_pair"]
    )

    imbalance_wide = (
        options.pivot_table(index="timestamp", columns="voucher", values="total_imbalance", aggfunc="last")
        .sort_index()
    )
    imbalance_corr = imbalance_wide.corr()

    sync_rows: list[dict[str, object]] = []
    prev_quotes: pd.Series | None = None
    for timestamp, frame in options.groupby("timestamp", sort=True):
        active = frame["voucher"].nunique()
        if active == 0:
            continue
        spread_counts = frame["spread"].dropna().value_counts()
        size_counts = frame["l1_size_pair"].dropna().value_counts()
        spread_mode_share = float(spread_counts.max() / active) if not spread_counts.empty else float("nan")
        size_mode_share = float(size_counts.max() / active) if not size_counts.empty else float("nan")
        current_quotes = frame.set_index("voucher")["quote_signature"].sort_index()
        quote_change_share = float("nan")
        if prev_quotes is not None:
            aligned = pd.concat([prev_quotes.rename("prev"), current_quotes.rename("curr")], axis=1).dropna()
            if not aligned.empty:
                quote_change_share = float((aligned["prev"] != aligned["curr"]).mean())
        prev_quotes = current_quotes
        sync_rows.append(
            {
                "timestamp": int(timestamp),
                "active_strikes": active,
                "spread_mode_share": spread_mode_share,
                "size_mode_share": size_mode_share,
                "quote_change_share": quote_change_share,
            }
        )
    sync_frame = pd.DataFrame(sync_rows)

    template_table = (
        options["l1_size_pair"]
        .value_counts(dropna=False)
        .rename_axis("L1 bid x ask")
        .reset_index(name="Snapshots")
        .head(8)
    )
    template_table["Share"] = template_table["Snapshots"] / max(len(options), 1)

    adjacent_corrs: list[float] = []
    ordered_vouchers = [f"VEV_{strike}" for strike in P4_R3_VOUCHER_STRIKES]
    for left, right in zip(ordered_vouchers[:-1], ordered_vouchers[1:]):
        if left in imbalance_corr.index and right in imbalance_corr.columns:
            adjacent_corrs.append(float(imbalance_corr.loc[left, right]))

    bot_summary = pd.DataFrame(
        [
            {
                "Metric": "Mean same-spread share",
                "Value": f"{sync_frame['spread_mode_share'].mean():.3f}" if not sync_frame.empty else "n/a",
                "Meaning": "Share of active strikes quoting the modal bid-ask spread at the same timestamp.",
            },
            {
                "Metric": "Mean same-size share",
                "Value": f"{sync_frame['size_mode_share'].mean():.3f}" if not sync_frame.empty else "n/a",
                "Meaning": "Share of active strikes reusing the same L1 size template at the same timestamp.",
            },
            {
                "Metric": "Mean quote-change share",
                "Value": f"{sync_frame['quote_change_share'].dropna().mean():.3f}" if not sync_frame['quote_change_share'].dropna().empty else "n/a",
                "Meaning": "Fraction of strikes whose top quote changed together from one timestamp to the next.",
            },
            {
                "Metric": "Adj. imbalance corr",
                "Value": f"{np.nanmean(adjacent_corrs):.4f}" if adjacent_corrs else "n/a",
                "Meaning": "Average adjacent-strike correlation in total imbalance; high values imply synchronized quoting templates.",
            },
            {
                "Metric": "Residual mean-reversion sweet spot",
                "Value": residual_signal.loc[residual_signal["corr"].abs().idxmax(), "voucher"] if not residual_signal.empty else "n/a",
                "Meaning": "Strike with the strongest residual-to-future-return relationship on this day.",
            },
        ]
    )

    return {
        "residual_signal": residual_signal,
        "under_option_lag": under_option_lag,
        "imbalance_corr": imbalance_corr,
        "sync_frame": sync_frame,
        "template_table": template_table,
        "bot_summary": bot_summary,
    }


def p4_r3_option_residual_signal_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return apply_mc_chart_layout(fig, "Residual Mispricing vs Future Option Move", height=360)

    pivot = frame.pivot(index="voucher", columns="horizon", values="corr").sort_index()
    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=[f"h={int(column)}" for column in pivot.columns],
            y=pivot.index,
            colorscale="RdBu",
            zmid=0.0,
            colorbar={"title": "corr"},
            hovertemplate="%{y}<br>%{x}<br>corr=%{z:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="future return horizon")
    fig.update_yaxes(title="")
    return apply_mc_chart_layout(fig, "Residual Mispricing vs Future Option Move", height=380)


def p4_r3_underlying_option_lag_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return apply_mc_chart_layout(fig, "Underlying vs Voucher Lead-Lag", height=380)

    pivot = frame.pivot(index="voucher", columns="lag", values="corr").sort_index()
    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.astype(int),
            y=pivot.index,
            colorscale="RdBu",
            zmid=0.0,
            colorbar={"title": "corr"},
            hovertemplate="%{y}<br>lag=%{x}<br>corr=%{z:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="lag (underlying return vs option return)")
    fig.update_yaxes(title="")
    return apply_mc_chart_layout(fig, "Underlying vs Voucher Lead-Lag", height=400)


def p4_r3_voucher_sync_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return apply_mc_chart_layout(fig, "Voucher Quote Synchronization", height=360)

    for column, label, color in [
        ("spread_mode_share", "Same spread share", "#5dade2"),
        ("size_mode_share", "Same L1 size share", "#f4a261"),
        ("quote_change_share", "Synchronized quote changes", "#59c17a"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=frame["timestamp"],
                y=frame[column],
                mode="lines",
                name=label,
                line={"width": 2.2, "color": color},
                hovertemplate="t=%{x}<br>share=%{y:.3f}<extra></extra>",
            )
        )
    fig.update_xaxes(title="timestamp")
    fig.update_yaxes(title="share of active voucher strikes")
    return apply_mc_chart_layout(fig, "Voucher Quote Synchronization", height=380)


def p4_r3_voucher_corr_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if corr.empty:
        return apply_mc_chart_layout(fig, "Cross-Strike Imbalance Correlation", height=360)

    ordered = [label for label in [f"VEV_{strike}" for strike in P4_R3_VOUCHER_STRIKES] if label in corr.index]
    corr = corr.loc[ordered, ordered]
    fig.add_trace(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="Viridis",
            zmin=-1.0,
            zmax=1.0,
            colorbar={"title": "corr"},
            hovertemplate="%{y} vs %{x}<br>corr=%{z:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="")
    fig.update_yaxes(title="")
    return apply_mc_chart_layout(fig, "Cross-Strike Imbalance Correlation", height=400)


def p4_r3_manual_penalty_multiplier(second_bid: float, estimated_mu: float) -> float:
    if second_bid > estimated_mu:
        return 1.0
    denominator = P4_R3_MANUAL_RESALE_PRICE - second_bid
    numerator = P4_R3_MANUAL_RESALE_PRICE - estimated_mu
    if denominator <= 0:
        return 1.0
    return float((numerator / denominator) ** 3)


def p4_r3_manual_metrics(first_bid: float, second_bid: float, estimated_mu: float) -> dict[str, float]:
    reserves = P4_R3_MANUAL_RESERVE_VALUES
    penalty = p4_r3_manual_penalty_multiplier(second_bid, estimated_mu)
    first_mask = first_bid > reserves
    second_mask = (~first_mask) & (second_bid > reserves)
    first_fill_count = int(first_mask.sum())
    second_fill_count = int(second_mask.sum())
    first_profit = first_fill_count * (P4_R3_MANUAL_RESALE_PRICE - first_bid)
    second_profit = second_fill_count * (P4_R3_MANUAL_RESALE_PRICE - second_bid) * penalty
    total_count = len(reserves)
    return {
        "penalty": penalty,
        "first_fill_count": first_fill_count,
        "second_fill_count": second_fill_count,
        "first_fill_share": first_fill_count / total_count,
        "second_fill_share": second_fill_count / total_count,
        "combined_fill_share": (first_fill_count + second_fill_count) / total_count,
        "expected_pnl_per_counterparty": (first_profit + second_profit) / total_count,
        "first_profit": first_profit / total_count,
        "second_profit": second_profit / total_count,
    }


def p4_r3_manual_second_bid_curve(first_bid: float, estimated_mu: float) -> go.Figure:
    second_bids = np.arange(670, 921, 5, dtype=float)
    expected_pnls = []
    penalties = []
    for second_bid in second_bids:
        metrics = p4_r3_manual_metrics(first_bid, float(second_bid), estimated_mu)
        expected_pnls.append(metrics["expected_pnl_per_counterparty"])
        penalties.append(metrics["penalty"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=second_bids,
            y=expected_pnls,
            mode="lines",
            name="Expected PnL",
            line={"color": "#6ccf9c", "width": 3},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=second_bids,
            y=penalties,
            mode="lines",
            name="Penalty multiplier",
            line={"color": "#f4a261", "width": 2, "dash": "dot"},
        ),
        secondary_y=True,
    )
    fig.add_vline(
        x=float(estimated_mu),
        line={"color": "#b8c0ff", "dash": "dash", "width": 1.5},
        annotation_text=f"mu ≈ {estimated_mu:.0f}",
        annotation_position="top right",
    )
    fig.update_xaxes(title="Second bid")
    fig.update_yaxes(title="Expected PnL per counterparty", secondary_y=False)
    fig.update_yaxes(title="Penalty multiplier", secondary_y=True)
    return apply_mc_chart_layout(fig, "Manual Round 3: Second-Bid Curve", height=390)


def render_p4_r3_manual_bid_panel() -> None:
    st.markdown(
        """
        <div class="mc-panel">
          <div class="mc-section-title">Manual Challenge: Celestial Gardeners’ Guild</div>
          <div class="mc-note">
            This panel treats reserve prices as <b>uniform on 670, 675, ..., 920</b>. The first bid fills everyone below it at <b>b1</b>. The second bid fills the remaining reserves below <b>b2</b>,
            but if <b>b2 ≤ μ</b> the per-trade PnL is penalized by <b>((920 − μ) / (920 − b2))³</b>. The output below is the expected profit
            <b>per counterparty</b>, so multiply by your believed number of guild members if you want a rough total.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    controls, charts = st.columns([1.0, 1.35], gap="medium")
    with controls:
        st.markdown('<div class="mc-panel"><div class="mc-section-title">Assumptions</div>', unsafe_allow_html=True)
        first_bid = st.number_input(
            "First bid (b1)",
            min_value=670.0,
            max_value=920.0,
            value=float(st.session_state.get("p4_r3_manual_b1_input", 780.0)),
            step=1.0,
            key="p4_r3_manual_b1_input",
        )
        second_seed = max(float(first_bid), float(st.session_state.get("p4_r3_manual_b2_input", 862.0)))
        if second_seed != float(st.session_state.get("p4_r3_manual_b2_input", second_seed)):
            st.session_state["p4_r3_manual_b2_input"] = second_seed
        second_bid = st.number_input(
            "Second bid (b2)",
            min_value=float(first_bid),
            max_value=920.0,
            value=float(st.session_state.get("p4_r3_manual_b2_input", second_seed)),
            step=1.0,
            key="p4_r3_manual_b2_input",
        )
        estimated_mu = st.slider(
            "Estimated crowd mean of second bids (μ)",
            836,
            900,
            int(P4_R3_MANUAL_ESTIMATED_MU),
            step=1,
            key="p4_r3_manual_mu",
        )
        metrics = p4_r3_manual_metrics(float(first_bid), float(second_bid), float(estimated_mu))
        mc_card("Estimated μ", f"{estimated_mu:.0f}", "Current crowd-average second-bid assumption.")
        mc_card("Penalty multiplier", f"{metrics['penalty']:.3f}", "Equals 1.0 whenever b2 is above μ.")
        mc_card(
            "Expected PnL / counterparty",
            fmt_money(metrics["expected_pnl_per_counterparty"], 2),
            f"First leg {fmt_money(metrics['first_profit'], 2)} · second leg {fmt_money(metrics['second_profit'], 2)}",
        )
        mc_card(
            "Combined fill share",
            f"{metrics['combined_fill_share']:.1%}",
            f"First {metrics['first_fill_share']:.1%} · second {metrics['second_fill_share']:.1%}",
        )
        mc_card(
            "Expected PnL for 50 counterparties",
            fmt_money(50.0 * metrics["expected_pnl_per_counterparty"], 2),
            "Scale the per-counterparty edge by your own guess for the guild size.",
        )
        st.caption("You can type any bid value. Reserve prices are still modeled on the true 5-point support grid.")
        st.markdown("</div>", unsafe_allow_html=True)

    with charts:
        st.plotly_chart(
            p4_r3_manual_second_bid_curve(float(first_bid), float(estimated_mu)),
            use_container_width=True,
            config={"displaylogo": False},
        )
        reserve_frame = pd.DataFrame({"Reserve price": P4_R3_MANUAL_RESERVE_VALUES})
        first_mask = first_bid > reserve_frame["Reserve price"]
        second_mask = (~first_mask) & (second_bid > reserve_frame["Reserve price"])
        reserve_frame["Route"] = np.where(first_mask, "First bid", np.where(second_mask, "Second bid", "No trade"))
        reserve_frame["Per-counterparty PnL"] = np.where(
            first_mask,
            P4_R3_MANUAL_RESALE_PRICE - first_bid,
            np.where(
                second_mask,
                (P4_R3_MANUAL_RESALE_PRICE - second_bid)
                * p4_r3_manual_penalty_multiplier(float(second_bid), float(estimated_mu)),
                0.0,
            ),
        )
        st.markdown('<div class="mc-panel"><div class="mc-section-title">Reserve-by-Reserve Breakdown</div>', unsafe_allow_html=True)
        mc_table(reserve_frame)
        st.markdown("</div>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_p3_r3_option_market() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not P3_ROUND3_DATA_DIR.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    files = discover_files(str(P3_ROUND3_DATA_DIR))
    prices = load_prices(files.price_paths)
    trades = load_trades(files.trade_paths)

    volcanic_prices = prices[prices["product"].isin((P3_R3_UNDERLYING, *P3_R3_OPTION_PRODUCTS))].copy()
    if volcanic_prices.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    options = volcanic_prices[volcanic_prices["product"].isin(P3_R3_OPTION_PRODUCTS)].copy()
    options["strike"] = (
        options["product"].str.extract(r"(\d+)$").iloc[:, 0].astype(float)
    )
    options["option_wall_mid"] = options["mid_price"]
    ask_only = options["option_wall_mid"].isna() & options["ask_price_1"].notna()
    bid_only = options["option_wall_mid"].isna() & options["bid_price_1"].notna()
    options.loc[ask_only, "option_wall_mid"] = options.loc[ask_only, "ask_price_1"] - 0.5
    options.loc[bid_only, "option_wall_mid"] = options.loc[bid_only, "bid_price_1"] + 0.5

    underlying = volcanic_prices[volcanic_prices["product"] == P3_R3_UNDERLYING].copy()
    underlying["underlying_wall_mid"] = underlying["mid_price"]
    ask_only = underlying["underlying_wall_mid"].isna() & underlying["ask_price_1"].notna()
    bid_only = underlying["underlying_wall_mid"].isna() & underlying["bid_price_1"].notna()
    underlying.loc[ask_only, "underlying_wall_mid"] = underlying.loc[ask_only, "ask_price_1"] - 0.5
    underlying.loc[bid_only, "underlying_wall_mid"] = underlying.loc[bid_only, "bid_price_1"] + 0.5

    columns = [
        "day",
        "timestamp",
        "bid_price_1",
        "ask_price_1",
        "underlying_wall_mid",
        "source_file",
        "session_label",
    ]
    options = options.merge(
        underlying[columns].rename(
            columns={
                "bid_price_1": "underlying_bid_1",
                "ask_price_1": "underlying_ask_1",
            }
        ),
        on=["day", "timestamp"],
        how="left",
    )

    volcanic_trades = trades[trades["symbol"].isin((P3_R3_UNDERLYING, *P3_R3_OPTION_PRODUCTS))].copy()
    volcanic_trades["strike"] = (
        volcanic_trades["symbol"].str.extract(r"(\d+)$").iloc[:, 0].astype(float)
    )
    return options, underlying, volcanic_trades


@st.cache_data(show_spinner=False)
def build_p3_r3_option_analysis(
    selected_day: int,
    days_to_expiry_open: float,
    extrinsic_floor: float,
    fit_mode: str,
    scatter_points_per_strike: int,
    random_sample_count: int,
    max_lag: int,
) -> dict[str, object]:
    options_raw, underlying_raw, trades_raw = load_p3_r3_option_market()
    if options_raw.empty or underlying_raw.empty:
        return {
            "options": pd.DataFrame(),
            "underlying": pd.DataFrame(),
            "focus": pd.DataFrame(),
            "focus_trades": pd.DataFrame(),
            "scatter": pd.DataFrame(),
            "insight_table": pd.DataFrame(),
            "published_coeffs": P3_R3_FRANKFURT_SMILE_COEFFS,
            "dynamic_coeffs": P3_R3_FRANKFURT_SMILE_COEFFS,
            "acf_frame": pd.DataFrame(),
        }

    options = options_raw[options_raw["day"] == selected_day].copy().sort_values(["product", "timestamp"])
    underlying = underlying_raw[underlying_raw["day"] == selected_day].copy().sort_values("timestamp")
    trades = trades_raw[trades_raw["day"] == selected_day].copy().sort_values(["symbol", "timestamp"])

    options["tau"] = (
        (float(days_to_expiry_open) - options["day"] - options["timestamp"] / 1_000_000.0)
        .clip(lower=1e-4)
        / 365.0
    )
    options["intrinsic"] = np.maximum(options["underlying_wall_mid"] - options["strike"], 0.0)
    options["extrinsic"] = options["option_wall_mid"] - options["intrinsic"]
    options["moneyness"] = np.log(options["strike"] / options["underlying_wall_mid"]) / np.sqrt(options["tau"])
    options["market_iv"] = [
        p3_r3_implied_volatility(spot, strike, tau, price)
        for spot, strike, tau, price in zip(
            options["underlying_wall_mid"],
            options["strike"],
            options["tau"],
            options["option_wall_mid"],
        )
    ]

    fit_mask = (
        options["market_iv"].notna()
        & options["moneyness"].notna()
        & options["extrinsic"].ge(float(extrinsic_floor))
        & options["market_iv"].between(0.05, 1.00)
    )
    if int(fit_mask.sum()) >= 15:
        dynamic_coeffs = np.polyfit(
            options.loc[fit_mask, "moneyness"],
            options.loc[fit_mask, "market_iv"],
            2,
        )
    else:
        dynamic_coeffs = P3_R3_FRANKFURT_SMILE_COEFFS.copy()

    published_poly = np.poly1d(P3_R3_FRANKFURT_SMILE_COEFFS)
    dynamic_poly = np.poly1d(dynamic_coeffs)
    options["fair_iv_published"] = published_poly(options["moneyness"])
    options["fair_iv_dynamic"] = dynamic_poly(options["moneyness"])
    if fit_mode == "Published Frankfurt fit":
        options["fair_iv"] = options["fair_iv_published"]
    else:
        options["fair_iv"] = options["fair_iv_dynamic"]
    options["fair_iv"] = pd.to_numeric(options["fair_iv"], errors="coerce").clip(lower=0.05, upper=1.20)

    metrics = [
        p3_r3_bs_call_metrics(float(spot), float(strike), float(tau), float(sigma))
        if pd.notna(spot) and pd.notna(strike) and pd.notna(tau) and pd.notna(sigma)
        else (float("nan"), float("nan"), float("nan"), float("nan"))
        for spot, strike, tau, sigma in zip(
            options["underlying_wall_mid"],
            options["strike"],
            options["tau"],
            options["fair_iv"],
        )
    ]
    options["fair_price"] = [item[0] for item in metrics]
    options["delta"] = [item[1] for item in metrics]
    options["gamma"] = [item[2] for item in metrics]
    options["vega"] = [item[3] for item in metrics]
    options["iv_deviation"] = options["market_iv"] - options["fair_iv"]
    options["price_deviation"] = options["option_wall_mid"] - options["fair_price"]

    grouped = options.groupby("product", sort=False)["price_deviation"]
    options["mean_theo_diff"] = grouped.transform(
        lambda series: series.ewm(span=P3_R3_THEO_NORM_WINDOW, adjust=False, min_periods=1).mean()
    )
    options["switch_mean"] = options.groupby("product", sort=False).apply(
        lambda frame: (
            (frame["price_deviation"] - frame["mean_theo_diff"]).abs()
            .ewm(span=P3_R3_IV_SCALPING_WINDOW, adjust=False, min_periods=1)
            .mean()
        )
    ).reset_index(level=0, drop=True)
    options["scalper_on"] = options["switch_mean"] >= P3_R3_IV_SCALPING_THR
    options["low_vega_adj"] = np.where(options["vega"] <= 1.0, P3_R3_LOW_VEGA_THR_ADJ, 0.0)
    options["bid_signal"] = options["bid_price_1"] - options["fair_price"] - options["mean_theo_diff"]
    options["ask_signal"] = options["ask_price_1"] - options["fair_price"] - options["mean_theo_diff"]
    options["would_sell"] = options["scalper_on"] & (options["bid_signal"] >= (P3_R3_THR_OPEN + options["low_vega_adj"]))
    options["would_buy"] = options["scalper_on"] & (options["ask_signal"] <= -(P3_R3_THR_OPEN + options["low_vega_adj"]))
    options["normalized_deviation_pct"] = 100.0 * options["price_deviation"] / options["fair_price"].replace(0, pd.NA)

    scatter_parts: list[pd.DataFrame] = []
    for product, frame in options.groupby("product", sort=False):
        valid = frame[fit_mask.loc[frame.index]].copy()
        if valid.empty:
            continue
        sample_n = min(int(scatter_points_per_strike), len(valid))
        if sample_n <= 0:
            continue
        scatter_parts.append(valid.sample(sample_n, random_state=7))
    scatter = pd.concat(scatter_parts, ignore_index=True) if scatter_parts else pd.DataFrame()

    insight_table = (
        options.groupby("product", sort=False)
        .agg(
            Strike=("strike", "first"),
            Mean_IV_Dev=("iv_deviation", "mean"),
            Price_Dev_SD=("price_deviation", "std"),
            Mean_Abs_Price_Dev=("price_deviation", lambda series: series.abs().mean()),
            Scalp_On_Pct=("scalper_on", "mean"),
            Avg_Vega=("vega", "mean"),
        )
        .reset_index()
        .rename(columns={"product": "Voucher"})
    )
    if not insight_table.empty:
        insight_table["Mean_IV_Dev"] = insight_table["Mean_IV_Dev"].map(lambda value: f"{float(value):+.4f}")
        insight_table["Price_Dev_SD"] = insight_table["Price_Dev_SD"].map(lambda value: fmt_number(value, 3))
        insight_table["Mean_Abs_Price_Dev"] = insight_table["Mean_Abs_Price_Dev"].map(lambda value: fmt_number(value, 3))
        insight_table["Scalp_On_Pct"] = insight_table["Scalp_On_Pct"].map(lambda value: f"{100.0 * float(value):.1f}%")
        insight_table["Avg_Vega"] = insight_table["Avg_Vega"].map(lambda value: fmt_number(value, 3))
        insight_table["Strike"] = insight_table["Strike"].map(lambda value: f"{int(value)}")

    returns = pd.to_numeric(underlying["underlying_wall_mid"], errors="coerce").diff().dropna().to_numpy()
    acf_rows: list[dict[str, float | int | str]] = []
    if len(returns) > max_lag + 5:
        centered = returns - returns.mean()
        denom = float(np.dot(centered, centered))
        lags = range(1, int(max_lag) + 1)
        for lag in lags:
            if lag >= len(centered):
                break
            numer = float(np.dot(centered[:-lag], centered[lag:]))
            acf_rows.append({"lag": lag, "value": numer / denom if denom else 0.0, "series": "VOLCANIC_ROCK"})
        rng = np.random.default_rng(11)
        scale = float(np.std(returns)) if float(np.std(returns)) > 0 else 1.0
        for sample_id in range(int(random_sample_count)):
            sample = rng.normal(0.0, scale, len(returns))
            sample = sample - sample.mean()
            sample_denom = float(np.dot(sample, sample))
            for lag in lags:
                if lag >= len(sample):
                    break
                numer = float(np.dot(sample[:-lag], sample[lag:]))
                acf_rows.append(
                    {
                        "lag": lag,
                        "value": numer / sample_denom if sample_denom else 0.0,
                        "series": f"random_{sample_id + 1}",
                    }
                )
    acf_frame = pd.DataFrame(acf_rows)

    return {
        "options": options,
        "underlying": underlying,
        "trades": trades,
        "scatter": scatter,
        "insight_table": insight_table,
        "published_coeffs": P3_R3_FRANKFURT_SMILE_COEFFS,
        "dynamic_coeffs": dynamic_coeffs,
        "acf_frame": acf_frame,
    }


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
            table-layout: fixed;
            width: 100%;
        }
        .mc-table th {
            background: #1b2027;
            border: 1px solid #303640;
            color: #c9cdd5;
            font-weight: 800;
            overflow-wrap: anywhere;
            padding: 0.48rem 0.55rem;
            text-align: left;
            vertical-align: top;
        }
        .mc-table td {
            background: #15181d;
            border: 1px solid #303640;
            overflow-wrap: anywhere;
            padding: 0.45rem 0.55rem;
            vertical-align: top;
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


def submission_code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:12]


def submission_safe_name(name: str) -> str:
    stem = Path(name).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return cleaned or "submission"


def submission_family_name(name: str) -> str:
    stem = submission_safe_name(name)
    parts = [part for part in stem.split("_") if part]
    if not parts:
        return "submission"
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return "_".join(parts)
    return "_".join(parts[:3])


def load_submission_history() -> list[dict[str, object]]:
    if not SUBMISSION_HISTORY_FILE.exists():
        return []
    try:
        raw = json.loads(SUBMISSION_HISTORY_FILE.read_text())
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    return [entry for entry in raw if isinstance(entry, dict)]


def save_submission_history(entries: list[dict[str, object]]) -> None:
    SUBMISSION_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_HISTORY_CODE_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_HISTORY_FILE.write_text(json.dumps(entries, indent=2))


def record_submission_history(
    *,
    event_id: str,
    submitted_name: str,
    submitted_code: str,
    dataset_label: str,
    normalization_ticks: int,
    total_pnl: float,
    mean_pnl: float,
    sd_pnl: float,
    own_trades: int,
    day_count: int,
) -> None:
    if not event_id:
        return
    entries = load_submission_history()
    if any(str(entry.get("event_id", "")) == str(event_id) for entry in entries):
        return

    SUBMISSION_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_HISTORY_CODE_DIR.mkdir(parents=True, exist_ok=True)

    digest = submission_code_hash(submitted_code)
    safe_name = submission_safe_name(submitted_name)
    timestamp_ns = time.time_ns()
    snapshot_name = f"{timestamp_ns}_{safe_name}_{digest}.py"
    snapshot_path = SUBMISSION_HISTORY_CODE_DIR / snapshot_name
    snapshot_path.write_text(submitted_code)

    entries.append(
        {
            "event_id": str(event_id),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "submitted_name": submitted_name,
            "dataset_label": dataset_label,
            "normalization_ticks": int(normalization_ticks),
            "total_pnl": float(total_pnl),
            "mean_pnl": float(mean_pnl),
            "sd_pnl": float(sd_pnl),
            "own_trades": int(own_trades),
            "day_count": int(day_count),
            "code_hash": digest,
            "code_path": str(snapshot_path),
        }
    )
    entries.sort(key=lambda entry: str(entry.get("created_at", "")), reverse=True)
    save_submission_history(entries)


def submission_history_dataframe(
    normalization_ticks: int,
    dataset_label: str | None = None,
) -> pd.DataFrame:
    rows = []
    for entry in load_submission_history():
        if int(entry.get("normalization_ticks", -1)) != int(normalization_ticks):
            continue
        if dataset_label is not None and str(entry.get("dataset_label", "")) != dataset_label:
            continue
        rows.append(
            {
                "When": str(entry.get("created_at", "")),
                "File": str(entry.get("submitted_name", "")),
                "Dataset": str(entry.get("dataset_label", "")),
                "Total PnL": float(entry.get("total_pnl", 0.0)),
                "Mean PnL": float(entry.get("mean_pnl", 0.0)),
                "1σ": float(entry.get("sd_pnl", 0.0)),
                "Trades": int(entry.get("own_trades", 0)),
                "Days": int(entry.get("day_count", 0)),
                "Hash": str(entry.get("code_hash", "")),
                "Code path": str(entry.get("code_path", "")),
                "Family": submission_family_name(str(entry.get("submitted_name", ""))),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["Total PnL", "When"], ascending=[False, False]).reset_index(drop=True)


def submission_family_leaderboard(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    grouped = (
        history.groupby("Family", dropna=False)
        .agg(
            Best_PnL=("Total PnL", "max"),
            Mean_PnL=("Total PnL", "mean"),
            Runs=("Total PnL", "size"),
            Best_File=("File", "first"),
            Latest_When=("When", "max"),
        )
        .reset_index()
        .sort_values(["Best_PnL", "Runs"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return grouped


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


def empirical_speed_stats(speed_pct: float, counts: list[int] | tuple[int, ...] = ACTUAL_R2_SPEED_COUNTS) -> dict[str, float]:
    if not counts:
        return {
            "speed_pct": speed_pct,
            "speed_int": 0,
            "count": 0.0,
            "field_size": 0.0,
            "pct_exact": 0.0,
            "pct_below": 0.0,
            "pct_at_or_below": 0.0,
            "pct_above": 0.0,
            "pct_at_or_above": 0.0,
            "multiplier_below": 0.1,
            "multiplier_at_or_below": 0.1,
        }
    speed_int = min(max(int(round(speed_pct)), 0), len(counts) - 1)
    field_size = float(sum(counts))
    below = float(sum(counts[:speed_int]))
    at = float(counts[speed_int])
    at_or_below = below + at
    pct_exact = at / field_size if field_size > 0 else 0.0
    pct_below = below / field_size if field_size > 0 else 0.0
    pct_at_or_below = at_or_below / field_size if field_size > 0 else 0.0
    pct_above = 1.0 - pct_at_or_below
    pct_at_or_above = 1.0 - pct_below
    return {
        "speed_pct": speed_pct,
        "speed_int": speed_int,
        "count": at,
        "field_size": field_size,
        "pct_exact": pct_exact,
        "pct_below": pct_below,
        "pct_at_or_below": pct_at_or_below,
        "pct_above": pct_above,
        "pct_at_or_above": pct_at_or_above,
        "multiplier_below": 0.1 + 0.8 * pct_below,
        "multiplier_at_or_below": 0.1 + 0.8 * pct_at_or_below,
    }


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


def empirical_speed_percentile_chart(selected_speed_pct: float) -> go.Figure:
    speeds = list(range(len(ACTUAL_R2_SPEED_COUNTS)))
    total = max(1, sum(ACTUAL_R2_SPEED_COUNTS))
    shares = [100.0 * count / total for count in ACTUAL_R2_SPEED_COUNTS]
    selected_speed = min(max(int(round(selected_speed_pct)), 0), len(ACTUAL_R2_SPEED_COUNTS) - 1)
    selected_stats = empirical_speed_stats(selected_speed)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=speeds,
            y=shares,
            name="Actual crowd share",
            marker={"color": "#5b7cfa", "line": {"color": "#aab7ff", "width": 0.6}},
            hovertemplate="Speed %{x}<br>Share %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_vline(
        x=selected_speed,
        line={"color": "#e15759", "width": 2.6, "dash": "dash"},
    )
    fig.add_trace(
        go.Scatter(
            x=[selected_speed],
            y=[shares[selected_speed]],
            mode="markers",
            name="Selected speed",
            marker={"color": "#e15759", "size": 11, "symbol": "diamond"},
            customdata=[[
                selected_stats["pct_below"] * 100.0,
                selected_stats["pct_at_or_below"] * 100.0,
                int(selected_stats["count"]),
            ]],
            hovertemplate=(
                "Speed %{x}<br>"
                "Share %{y:.2f}%<br>"
                "Below percentile %{customdata[0]:.1f}%<br>"
                "At or below %{customdata[1]:.1f}%<br>"
                "Exact count %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.add_annotation(
        x=selected_speed,
        y=max(shares) * 1.02 if shares else 0.0,
        text=f"Speed {selected_speed}",
        showarrow=False,
        font={"size": 12, "color": "#e15759"},
        bgcolor="rgba(21,24,29,0.85)",
    )
    fig.update_xaxes(title="Speed %", range=[-0.5, len(speeds) - 0.5])
    fig.update_yaxes(title="Actual crowd share (%)")
    return apply_mc_chart_layout(fig, "Actual 2026 Speed Distribution", height=360)


def empirical_speed_pnl_chart(selected_speed_pct: float) -> go.Figure:
    speeds = list(range(0, 101))
    pnls: list[float] = []
    for speed in speeds:
        stats = empirical_speed_stats(speed)
        remaining = max(0.0, 100.0 - float(speed))
        research_pct, scale_pct, edge_value = optimal_research_scale_split(remaining)
        gross = edge_value * stats["multiplier_below"]
        pnl = gross - investment_budget_used(research_pct, scale_pct, float(speed))
        pnls.append(float(pnl))

    selected_speed = min(max(int(round(selected_speed_pct)), 0), 100)
    selected_pnl = pnls[selected_speed]
    selected_stats = empirical_speed_stats(selected_speed)
    selected_remaining = max(0.0, 100.0 - float(selected_speed))
    selected_research, selected_scale, _ = optimal_research_scale_split(selected_remaining)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=speeds,
            y=pnls,
            mode="lines",
            name="Empirical PnL",
            line={"color": "#6ccf9c", "width": 2.8},
            hovertemplate="Speed %{x}<br>PnL %{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[selected_speed],
            y=[selected_pnl],
            mode="markers",
            name="Selected speed",
            marker={"color": "#e15759", "size": 12, "symbol": "diamond"},
            customdata=[[
                selected_stats["pct_below"] * 100.0,
                selected_research,
                selected_scale,
            ]],
            hovertemplate=(
                "Speed %{x}<br>"
                "PnL %{y:,.0f}<br>"
                "Empirical percentile %{customdata[0]:.1f}%<br>"
                "R %{customdata[1]:.1f}% · S %{customdata[2]:.1f}%<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="Speed %", range=[0, 100])
    fig.update_yaxes(title="PnL")
    return apply_mc_chart_layout(fig, "Actual 2026 Speed vs PnL", height=360)


def p3_container_base_frame() -> pd.DataFrame:
    rows = [
        {"multiplier": 10, "inhabitants": 1, "nash_density": 1.84, "actual_density": 1.0},
        {"multiplier": 17, "inhabitants": 1, "nash_density": 3.83, "actual_density": 7.5},
        {"multiplier": 20, "inhabitants": 2, "nash_density": 3.68, "actual_density": 1.7},
        {"multiplier": 31, "inhabitants": 2, "nash_density": 6.80, "actual_density": 7.0},
        {"multiplier": 37, "inhabitants": 3, "nash_density": 7.50, "actual_density": 5.0},
        {"multiplier": 50, "inhabitants": 4, "nash_density": 10.18, "actual_density": 8.5},
        {"multiplier": 73, "inhabitants": 4, "nash_density": 16.71, "actual_density": 24.0},
        {"multiplier": 80, "inhabitants": 6, "nash_density": 16.69, "actual_density": 18.0},
        {"multiplier": 89, "inhabitants": 8, "nash_density": 17.24, "actual_density": 15.0},
        {"multiplier": 90, "inhabitants": 10, "nash_density": 15.53, "actual_density": 12.0},
    ]
    frame = pd.DataFrame(rows)
    frame["actual_density"] = 100.0 * frame["actual_density"] / frame["actual_density"].sum()
    return frame


def p3_container_payoff(multiplier: pd.Series | float, density_percent: pd.Series | float, inhabitants: pd.Series | float) -> pd.Series | float:
    return pd.to_numeric(multiplier, errors="coerce") * 10_000.0 / (
        pd.to_numeric(density_percent, errors="coerce") + pd.to_numeric(inhabitants, errors="coerce")
    )


def p3_strict_rank(
    values: pd.Series | list[float],
    tie_break: pd.Series | list[float],
    *,
    value_ascending: bool,
    tie_break_desc: bool = True,
) -> pd.Series:
    rank_frame = pd.DataFrame(
        {
            "value": pd.to_numeric(values, errors="coerce"),
            "tie_break": pd.to_numeric(tie_break, errors="coerce"),
        }
    ).reset_index()
    rank_frame = rank_frame.sort_values(
        ["value", "tie_break"],
        ascending=[value_ascending, not tie_break_desc],
    )
    rank_frame["rank"] = np.arange(1, len(rank_frame) + 1, dtype=int)
    ranked = rank_frame.set_index("index")["rank"].sort_index()
    return pd.Series(ranked, dtype=int)


def p3_first_order_payoff_frame() -> pd.DataFrame:
    frame = p3_container_base_frame().copy()
    multipliers = pd.to_numeric(frame["multiplier"], errors="coerce").astype(float)
    inhabitants = pd.to_numeric(frame["inhabitants"], errors="coerce").astype(float)
    total_multiplier = max(1.0, float(multipliers.sum()))
    frame["island"] = range(len(frame))
    frame["first_order_pf_percent"] = 100.0 * multipliers / total_multiplier
    frame["first_order_denominator"] = frame["first_order_pf_percent"] + inhabitants
    frame["first_order_payout_proxy"] = p3_container_payoff(
        frame["multiplier"],
        frame["first_order_pf_percent"],
        frame["inhabitants"],
    )
    frame["first_order_rank_score"] = p3_strict_rank(
        frame["first_order_payout_proxy"],
        frame["multiplier"],
        value_ascending=True,
        tie_break_desc=True,
    )
    frame["first_order_rank"] = frame["first_order_rank_score"]
    return frame


def p3_first_order_rank_tuple() -> tuple[int, ...]:
    frame = p3_first_order_payoff_frame()
    return tuple(int(value) for value in frame["first_order_rank_score"].tolist())


def p3_actual_payoff_rank_tuple() -> tuple[int, ...]:
    frame = p3_container_base_frame().copy()
    frame["actual_payout_proxy"] = p3_container_payoff(
        frame["multiplier"],
        frame["actual_density"],
        frame["inhabitants"],
    )
    ranks = p3_strict_rank(
        frame["actual_payout_proxy"],
        frame["multiplier"],
        value_ascending=False,
        tie_break_desc=True,
    )
    return tuple(int(value) for value in ranks.tolist())


def p3_known_ranking_inferred_frame(known_final_rank_tuple: tuple[int, ...] | None = None) -> pd.DataFrame:
    frame = p3_first_order_payoff_frame().copy()
    multipliers = pd.to_numeric(frame["multiplier"], errors="coerce").astype(float)
    inhabitants = pd.to_numeric(frame["inhabitants"], errors="coerce").astype(float)
    if known_final_rank_tuple is not None and len(known_final_rank_tuple) == len(frame):
        frame["known_final_payoff_rank"] = pd.Series(
            [int(value) for value in known_final_rank_tuple],
            index=frame.index,
            dtype=int,
        )
    else:
        frame["known_final_payoff_rank"] = pd.to_numeric(
            frame["first_order_rank_score"],
            errors="coerce",
        ).astype(int)
    max_rank = int(frame["known_final_payoff_rank"].max())
    frame["known_final_payoff_score"] = (max_rank - frame["known_final_payoff_rank"] + 1).astype(float)
    scale = 10_000.0 * float((multipliers / frame["known_final_payoff_score"]).sum()) / (
        100.0 + float(inhabitants.sum())
    )
    frame["rank_inferred_pf_percent"] = (
        10_000.0 * multipliers / (scale * frame["known_final_payoff_score"])
    ) - inhabitants
    frame["rank_inferred_denominator"] = frame["rank_inferred_pf_percent"] + inhabitants
    frame["rank_inferred_payout_proxy"] = p3_container_payoff(
        frame["multiplier"],
        frame["rank_inferred_pf_percent"],
        frame["inhabitants"],
    )
    frame["rank_inferred_payoff_rank"] = pd.to_numeric(
        frame["rank_inferred_payout_proxy"],
        errors="coerce",
    ).rank(ascending=False, method="dense").astype(int)
    return frame


def normalize_component(values: pd.Series | list[float]) -> pd.Series:
    series = pd.Series(values, dtype=float)
    series = series.clip(lower=0.0)
    total = float(series.sum())
    if total <= 0:
        return pd.Series([100.0 / len(series)] * len(series))
    return 100.0 * series / total


def minmax_component(values: pd.Series | list[float]) -> pd.Series:
    series = pd.Series(values, dtype=float)
    spread = float(series.max() - series.min())
    if spread <= 0:
        return pd.Series([0.0] * len(series), dtype=float)
    return (series - float(series.min())) / spread


def centered_component(values: pd.Series | list[float]) -> pd.Series:
    series = pd.Series(values, dtype=float)
    spread = float(series.max() - series.min())
    if spread <= 0:
        return pd.Series([0.0] * len(series), dtype=float)
    scaled = (series - float(series.min())) / spread
    return scaled - float(scaled.mean())


def softmax_density(logits: pd.Series | list[float]) -> pd.Series:
    series = pd.Series(logits, dtype=float).clip(lower=-8.0, upper=8.0)
    shifted = series - float(series.max())
    scores = np.exp(shifted.to_numpy(dtype=float))
    total = float(scores.sum())
    if total <= 0:
        return pd.Series([100.0 / len(series)] * len(series), dtype=float)
    return pd.Series(100.0 * scores / total, index=series.index, dtype=float)


def p3_digit_bias_scores(multipliers: pd.Series) -> list[float]:
    digit_scores = []
    for multiplier in multipliers:
        if int(multiplier) == 37:
            digit_scores.append(0.15)
            continue
        text = str(int(multiplier))
        score = 0.25
        if "3" in text:
            score += 1.0
        if "7" in text:
            score += 1.0
        if int(multiplier) == 73:
            score += 1.75
        digit_scores.append(score)
    return digit_scores


def p3_container_components(
    high_number_exponent: float = 3.0,
    known_final_rank_tuple: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    frame = p3_container_base_frame()
    first_order_frame = p3_first_order_payoff_frame()
    known_rank_frame = p3_known_ranking_inferred_frame(known_final_rank_tuple)
    multiplier_text = frame["multiplier"].astype(str)
    multipliers = pd.to_numeric(frame["multiplier"], errors="coerce").astype(float)
    inhabitants = pd.to_numeric(frame["inhabitants"], errors="coerce").astype(float)
    digit_scores = p3_digit_bias_scores(frame["multiplier"])
    factor_counts = multipliers.map(lambda value: divisor_count(int(value))).astype(float)
    nash_payoff_proxy = p3_container_payoff(multipliers, frame["nash_density"], inhabitants)
    first_order_payoff = pd.to_numeric(first_order_frame["first_order_payout_proxy"], errors="coerce").astype(float)
    value_scores = multipliers / inhabitants.clip(lower=1.0)
    famous_values = {10, 20, 37, 50, 90}
    avoid_famous_scores = [
        0.08 if int(multiplier) in famous_values else 1.0
        for multiplier in frame["multiplier"]
    ]
    high_number_scores = np.exp(max(0.0, float(high_number_exponent)) * minmax_component(multipliers))

    frame["value_feature"] = centered_component(multipliers / inhabitants.clip(lower=1.0))
    frame["digit_feature"] = centered_component(digit_scores)
    frame["special_73_feature"] = centered_component((multipliers == 73).astype(float))
    frame["nash_payoff_feature"] = centered_component(nash_payoff_proxy)
    frame["nash_payout_proxy"] = nash_payoff_proxy
    frame["first_order_pf_percent"] = pd.to_numeric(first_order_frame["first_order_pf_percent"], errors="coerce").astype(float)
    frame["first_order_denominator"] = pd.to_numeric(first_order_frame["first_order_denominator"], errors="coerce").astype(float)
    frame["first_order_payout_proxy"] = first_order_payoff
    frame["known_final_payoff_rank"] = pd.to_numeric(known_rank_frame["known_final_payoff_rank"], errors="coerce").astype(float)
    frame["first_order_rank_score"] = pd.to_numeric(known_rank_frame["known_final_payoff_score"], errors="coerce").astype(float)
    frame["rank_inferred_pf_percent"] = pd.to_numeric(known_rank_frame["rank_inferred_pf_percent"], errors="coerce").astype(float)
    frame["rank_inferred_denominator"] = pd.to_numeric(known_rank_frame["rank_inferred_denominator"], errors="coerce").astype(float)
    frame["rank_inferred_payout_proxy"] = pd.to_numeric(known_rank_frame["rank_inferred_payout_proxy"], errors="coerce").astype(float)
    frame["rank_inferred_payoff_rank"] = pd.to_numeric(known_rank_frame["rank_inferred_payoff_rank"], errors="coerce").astype(float)
    frame["nash_component"] = normalize_component(frame["nash_density"])
    frame["high_number_component"] = normalize_component(high_number_scores)
    frame["avoid_famous_component"] = normalize_component(avoid_famous_scores)
    frame["random_component"] = normalize_component([1.0] * len(frame))
    frame["nice_numbers_component"] = normalize_component(digit_scores)
    frame["value_hunters_component"] = normalize_component(value_scores ** 1.2)
    frame["first_order_pnl_component"] = normalize_component(frame["rank_inferred_pf_percent"])
    frame["digit_component"] = normalize_component(digit_scores)
    frame["factor_component"] = normalize_component(factor_counts)
    frame["prime_component"] = normalize_component(
        multipliers.map(lambda value: 1.0 if is_prime_number(int(value)) else 0.0)
    )
    frame["container"] = multiplier_text
    return frame


def p3_container_modeled_frame(
    nash_weight: float,
    high_number_weight: float,
    avoid_famous_weight: float,
    random_weight: float,
    nice_numbers_weight: float,
    value_hunters_weight: float,
    first_order_pnl_weight: float,
    high_number_exponent: float,
    known_final_rank_tuple: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    frame = p3_container_components(high_number_exponent, known_final_rank_tuple)
    raw_weights = {
        "nash": max(0.0, float(nash_weight)),
        "high_number": max(0.0, float(high_number_weight)),
        "avoid_famous": max(0.0, float(avoid_famous_weight)),
        "random": max(0.0, float(random_weight)),
        "nice_numbers": max(0.0, float(nice_numbers_weight)),
        "value_hunters": max(0.0, float(value_hunters_weight)),
        "first_order_pnl": max(0.0, float(first_order_pnl_weight)),
    }
    total_weight = float(sum(raw_weights.values()))
    if total_weight <= 0:
        raw_weights["random"] = 1.0
        total_weight = 1.0

    modeled_density = pd.Series([0.0] * len(frame), dtype=float)
    for strategy, raw_weight in raw_weights.items():
        normalized_weight = raw_weight / total_weight
        frame[f"{strategy}_strategy_weight"] = normalized_weight * 100.0
        frame[f"{strategy}_contribution"] = normalized_weight * frame[f"{strategy}_component"]
        modeled_density += frame[f"{strategy}_contribution"]

    frame["modeled_density"] = modeled_density
    frame["model_score"] = frame["modeled_density"]
    frame["model_error"] = frame["modeled_density"] - frame["actual_density"]
    frame["absolute_error"] = frame["model_error"].abs()
    frame["actual_payout_proxy"] = p3_container_payoff(frame["multiplier"], frame["actual_density"], frame["inhabitants"])
    frame["modeled_payout_proxy"] = p3_container_payoff(frame["multiplier"], frame["modeled_density"], frame["inhabitants"])
    frame["payoff_error"] = frame["modeled_payout_proxy"] - frame["actual_payout_proxy"]
    return frame


@st.cache_data(show_spinner=False)
def p3_auto_fit_container_model(
    enabled_strategies: tuple[str, ...] | None = None,
    known_final_rank_tuple: tuple[int, ...] | None = None,
    objective: str = "density",
) -> dict[str, float]:
    base = p3_container_base_frame()
    target = pd.to_numeric(base["actual_density"], errors="coerce").to_numpy(dtype=float)
    target = np.clip(target, 0.0001, None)
    target = 100.0 * target / target.sum()
    strategy_specs = [
        ("nash", "nash_component", 15.0),
        ("high_number", "high_number_component", 20.0),
        ("avoid_famous", "avoid_famous_component", 10.0),
        ("random", "random_component", 15.0),
        ("nice_numbers", "nice_numbers_component", 20.0),
        ("value_hunters", "value_hunters_component", 20.0),
        ("first_order_pnl", "first_order_pnl_component", 15.0),
    ]
    if enabled_strategies is None:
        enabled_names = set(name for name, _, _ in strategy_specs)
    else:
        enabled_names = set(enabled_strategies)
    active_specs = [spec for spec in strategy_specs if spec[0] in enabled_names]
    if not active_specs:
        return {
            "nash_weight": 0.0,
            "high_number_weight": 0.0,
            "avoid_famous_weight": 0.0,
            "random_weight": 0.0,
            "nice_numbers_weight": 0.0,
            "value_hunters_weight": 0.0,
            "first_order_pnl_weight": 0.0,
            "high_number_exponent": 3.0,
            "mae": 0.0,
            "rmse": 0.0,
            "score": 0.0,
        }
    prior = np.array([prior_value for _, _, prior_value in active_specs], dtype=float)
    prior = prior / prior.sum()

    def empty_weight_map() -> dict[str, float]:
        return {
            "nash_weight": 0.0,
            "high_number_weight": 0.0,
            "avoid_famous_weight": 0.0,
            "random_weight": 0.0,
            "nice_numbers_weight": 0.0,
            "value_hunters_weight": 0.0,
            "first_order_pnl_weight": 0.0,
        }

    def score_frame(frame: pd.DataFrame, weights: np.ndarray, objective_name: str) -> tuple[float, float, float]:
        density_error = pd.to_numeric(frame["modeled_density"], errors="coerce") - pd.to_numeric(frame["actual_density"], errors="coerce")
        density_mae = float(np.mean(np.abs(density_error)))
        density_rmse = float(np.sqrt(np.mean(density_error * density_error)))
        payoff_error = pd.to_numeric(frame["modeled_payout_proxy"], errors="coerce") - pd.to_numeric(frame["actual_payout_proxy"], errors="coerce")
        payoff_mae = float(np.mean(np.abs(payoff_error))) / 1000.0
        if objective_name == "profit":
            objective_score = payoff_mae + 0.08 * density_mae + 0.015 * float(np.mean((weights - prior) ** 2))
        elif objective_name == "rank_top5":
            top5 = frame.sort_values("known_final_payoff_rank").head(5)
            top5_rank_mae = float((top5["modeled_payoff_rank"] - top5["known_final_payoff_rank"]).abs().mean())
            all_rank_mae = float((frame["modeled_payoff_rank"] - frame["known_final_payoff_rank"]).abs().mean())
            objective_score = top5_rank_mae + 0.25 * all_rank_mae + 0.05 * density_mae + 0.015 * float(np.mean((weights - prior) ** 2))
        else:
            objective_score = density_rmse + 0.15 * density_mae + 0.015 * float(np.mean((weights - prior) ** 2))
        return objective_score, density_mae, density_rmse

    best: dict[str, float] | None = None
    rng = np.random.default_rng(7)
    for exponent in np.arange(0.5, 8.01, 0.5):
        component_frame = p3_container_components(float(exponent), known_final_rank_tuple)
        matrix = np.column_stack([
            pd.to_numeric(component_frame[column], errors="coerce").to_numpy(dtype=float)
            for _, column, _ in active_specs
        ])
        ridge = 1.0
        prior_strength = 0.18
        lhs = matrix.T @ matrix + ridge * np.eye(matrix.shape[1])
        rhs = matrix.T @ target + ridge * prior_strength * prior * 100.0
        density_fit_weights = np.linalg.solve(lhs, rhs)
        density_fit_weights = np.clip(density_fit_weights, 0.0, None)
        if float(density_fit_weights.sum()) <= 0:
            density_fit_weights = prior.copy()
        density_fit_weights = density_fit_weights / density_fit_weights.sum()

        candidate_weights: list[np.ndarray] = [prior.copy(), density_fit_weights.copy()]
        for idx in range(len(active_specs)):
            one_hot = np.zeros(len(active_specs), dtype=float)
            one_hot[idx] = 1.0
            candidate_weights.append(one_hot)
        for _ in range(40):
            candidate_weights.append(rng.dirichlet(np.ones(len(active_specs), dtype=float)))
        for _ in range(40):
            alpha = 1.0 + 24.0 * density_fit_weights
            candidate_weights.append(rng.dirichlet(alpha))

        for weights in candidate_weights:
            weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
            if float(weights.sum()) <= 0:
                continue
            weights = weights / weights.sum()
            modeled_density = matrix @ weights
            frame = component_frame.copy()
            for idx, (strategy_name, _, _) in enumerate(active_specs):
                frame[f"{strategy_name}_strategy_weight"] = float(weights[idx] * 100.0)
                frame[f"{strategy_name}_contribution"] = weights[idx] * frame[f"{strategy_name}_component"]
            for strategy_name, _, _ in strategy_specs:
                if strategy_name not in {name for name, _, _ in active_specs}:
                    frame[f"{strategy_name}_strategy_weight"] = 0.0
                    frame[f"{strategy_name}_contribution"] = 0.0
            frame["modeled_density"] = modeled_density
            frame["model_score"] = frame["modeled_density"]
            frame["model_error"] = frame["modeled_density"] - frame["actual_density"]
            frame["absolute_error"] = frame["model_error"].abs()
            frame["actual_payout_proxy"] = p3_container_payoff(frame["multiplier"], frame["actual_density"], frame["inhabitants"])
            frame["modeled_payout_proxy"] = p3_container_payoff(frame["multiplier"], frame["modeled_density"], frame["inhabitants"])
            frame["payoff_error"] = frame["modeled_payout_proxy"] - frame["actual_payout_proxy"]
            ranked = p3_payoff_ranked_frame(frame)
            objective_score, mae, rmse = score_frame(ranked, weights, objective)
            if best is None or objective_score < best["score"]:
                fitted_weights = empty_weight_map()
                for idx, (strategy_name, _, _) in enumerate(active_specs):
                    fitted_weights[f"{strategy_name}_weight"] = float(weights[idx] * 100.0)
                best = {
                    **fitted_weights,
                    "high_number_exponent": float(exponent),
                    "mae": mae,
                    "rmse": rmse,
                    "score": float(objective_score),
                }
    return best or {
        "nash_weight": 20.0,
        "high_number_weight": 20.0,
        "avoid_famous_weight": 10.0,
        "random_weight": 15.0,
        "nice_numbers_weight": 20.0,
        "value_hunters_weight": 20.0,
        "first_order_pnl_weight": 15.0,
        "high_number_exponent": 3.0,
        "mae": 0.0,
        "rmse": 0.0,
        "score": 0.0,
    }


def p3_container_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["container"],
            y=frame["nash_density"],
            name="Nash",
            marker_color="#4e79a7",
            hovertemplate="x%{x}<br>Nash %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=frame["container"],
            y=frame["actual_density"],
            name="Actual human",
            marker_color="#f2b447",
            hovertemplate="x%{x}<br>Actual %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=frame["container"],
            y=frame["modeled_density"],
            name="Model predicted",
            marker_color="#e15759",
            hovertemplate="x%{x}<br>Model %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Container multiplier")
    fig.update_yaxes(title="Choice density (%)")
    return apply_mc_chart_layout(fig, "Prosperity 3 Round 2 · Nash vs Actual vs Model", height=420)


def p3_container_error_chart(frame: pd.DataFrame) -> go.Figure:
    colors = ["#6ccf9c" if value >= 0 else "#e15759" for value in frame["model_error"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["container"],
            y=frame["model_error"],
            marker_color=colors,
            name="Model - actual",
            hovertemplate="x%{x}<br>Error %{y:+.2f}pp<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"color": "#6f7682", "width": 1})
    fig.update_xaxes(title="Container multiplier")
    fig.update_yaxes(title="Percentage-point error")
    return apply_mc_chart_layout(fig, "Where The Explanation Misses", height=320)


def p3_human_vs_nash_difference_chart(frame: pd.DataFrame) -> go.Figure:
    difference = frame["actual_density"] - frame["nash_density"]
    colors = ["#f2b447" if value >= 0 else "#4e79a7" for value in difference]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["container"],
            y=difference,
            marker_color=colors,
            name="Human - Nash",
            customdata=list(zip(frame["actual_density"], frame["nash_density"])),
            hovertemplate=(
                "x%{x}<br>"
                "Human - Nash %{y:+.2f}pp<br>"
                "Human %{customdata[0]:.2f}%<br>"
                "Nash %{customdata[1]:.2f}%<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line={"color": "#f5f7fb", "width": 1.2})
    fig.add_annotation(
        x=0.02,
        y=1.06,
        xref="paper",
        yref="paper",
        text="Positive = humans over-picked vs Nash · Negative = humans under-picked",
        showarrow=False,
        font={"size": 13, "color": "#aeb4bd"},
        align="left",
    )
    max_abs = max(1.0, float(difference.abs().max()))
    fig.update_xaxes(title="Container multiplier")
    fig.update_yaxes(title="Human choice density minus Nash density (percentage points)", range=[-1.25 * max_abs, 1.25 * max_abs])
    return apply_mc_chart_layout(fig, "Big Human vs Nash Difference", height=520)


def p3_component_contribution_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    components = [
        ("Nash", "nash_contribution", "#4e79a7"),
        ("High-number exponential", "high_number_contribution", "#59a14f"),
        ("Famous avoiders", "avoid_famous_contribution", "#b07aa1"),
        ("Random", "random_contribution", "#8a8f98"),
        ("Nice numbers", "nice_numbers_contribution", "#f2b447"),
        ("Value hunters", "value_hunters_contribution", "#e15759"),
        ("p_f from higher-PnL worse crowd rank", "first_order_pnl_contribution", "#76b7b2"),
    ]
    for name, column, color in components:
        fig.add_trace(
            go.Bar(
                x=frame["container"],
                y=frame[column],
                name=name,
                marker_color=color,
                hovertemplate=f"{name}<br>x%{{x}}<br>%{{y:.2f}}pp<extra></extra>",
            )
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Container multiplier")
    fig.update_yaxes(title="Weighted density contribution (%)")
    return apply_mc_chart_layout(fig, "What Builds The Model Prediction", height=420)


def p3_payoff_ranked_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["actual_payoff_rank"] = ranked["actual_payout_proxy"].rank(
        ascending=False,
        method="min",
    ).astype(int)
    ranked["modeled_payoff_rank"] = ranked["modeled_payout_proxy"].rank(
        ascending=False,
        method="min",
    ).astype(int)
    ranked["nash_payoff_rank"] = ranked["nash_payout_proxy"].rank(
        ascending=False,
        method="min",
    ).astype(int)
    ranked["payoff_rank_delta"] = ranked["modeled_payoff_rank"] - ranked["actual_payoff_rank"]
    return ranked.sort_values(
        ["actual_payoff_rank", "multiplier"],
        ascending=[True, True],
    )


def p3_payout_proxy_chart(frame: pd.DataFrame) -> go.Figure:
    ranked = p3_payoff_ranked_frame(frame)
    category_order = ranked["container"].tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=ranked["container"],
            y=ranked["nash_payout_proxy"],
            name="Nash payoff",
            marker_color="#4e79a7",
            hovertemplate="Nash payoff<br>x%{x}<br>%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=ranked["container"],
            y=ranked["actual_payout_proxy"],
            name="Actual payoff",
            marker_color="#f2b447",
            text=[f"A#{rank}" for rank in ranked["actual_payoff_rank"]],
            textposition="outside",
            cliponaxis=False,
            customdata=list(zip(ranked["actual_payoff_rank"], ranked["actual_density"])),
            hovertemplate=(
                "Actual payoff<br>x%{x}<br>%{y:,.0f}<br>"
                "Actual rank #%{customdata[0]}<br>"
                "Actual density %{customdata[1]:.2f}%<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=ranked["container"],
            y=ranked["modeled_payout_proxy"],
            name="Predicted payoff",
            marker_color="#e15759",
            text=[f"P#{rank}" for rank in ranked["modeled_payoff_rank"]],
            textposition="outside",
            cliponaxis=False,
            customdata=list(zip(ranked["modeled_payoff_rank"], ranked["modeled_density"], ranked["payoff_rank_delta"])),
            hovertemplate=(
                "Predicted payoff<br>x%{x}<br>%{y:,.0f}<br>"
                "Predicted rank #%{customdata[0]}<br>"
                "Predicted density %{customdata[1]:.2f}%<br>"
                "Rank delta %{customdata[2]:+d}<extra></extra>"
            ),
        )
    )
    max_payoff = max(1.0, float(ranked[["nash_payout_proxy", "actual_payout_proxy", "modeled_payout_proxy"]].max().max()))
    fig.update_layout(barmode="group")
    fig.update_xaxes(
        title="Container multiplier, sorted by actual payoff high to low",
        categoryorder="array",
        categoryarray=category_order,
    )
    fig.update_yaxes(title="Payoff", range=[0, max_payoff * 1.18])
    return apply_mc_chart_layout(fig, "Payoff Function Under Each Crowd", height=420)


def p3_payoff_difference_chart(frame: pd.DataFrame) -> go.Figure:
    ranked = p3_payoff_ranked_frame(frame)
    colors = ["#6ccf9c" if value >= 0 else "#e15759" for value in ranked["payoff_error"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=ranked["container"],
            y=ranked["payoff_error"],
            marker_color=colors,
            name="Predicted - actual payoff",
            customdata=list(zip(ranked["modeled_payout_proxy"], ranked["actual_payout_proxy"], ranked["modeled_payoff_rank"], ranked["actual_payoff_rank"])),
            hovertemplate=(
                "x%{x}<br>"
                "Predicted - actual %{y:+,.0f}<br>"
                "Predicted %{customdata[0]:,.0f}<br>"
                "Actual %{customdata[1]:,.0f}<br>"
                "Pred rank #%{customdata[2]} · Actual rank #%{customdata[3]}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line={"color": "#f5f7fb", "width": 1.2})
    fig.update_xaxes(
        title="Container multiplier, sorted by actual payoff high to low",
        categoryorder="array",
        categoryarray=ranked["container"].tolist(),
    )
    fig.update_yaxes(title="Predicted payoff minus actual payoff")
    return apply_mc_chart_layout(fig, "Predicted Payoff Error", height=340)


def divisor_count(value: int) -> int:
    if value <= 0:
        return 0
    count = 0
    root = int(math.sqrt(value))
    for factor in range(1, root + 1):
        if value % factor == 0:
            count += 2
    if root * root == value:
        count -= 1
    return count


def is_prime_number(value: int) -> bool:
    return value > 1 and divisor_count(value) == 2


def p3_behavior_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    multipliers = pd.to_numeric(output["multiplier"], errors="coerce").astype(float)
    inhabitants = pd.to_numeric(output["inhabitants"], errors="coerce").astype(float)
    midpoint = float(multipliers.median())
    half_range = max(1.0, float(multipliers.max() - multipliers.min()) / 2.0)
    text_values = output["multiplier"].astype(int).astype(str)

    digit_bias = []
    for multiplier in output["multiplier"]:
        value = int(multiplier)
        text = str(value)
        score = 0.25
        if value == 37:
            score = 0.15
        else:
            if "3" in text:
                score += 1.0
            if "7" in text:
                score += 1.0
            if value == 73:
                score += 1.75
        digit_bias.append(score)

    output["human_minus_nash"] = output["actual_density"] - output["nash_density"]
    output["multiplier_level"] = multipliers
    output["inhabitants_level"] = inhabitants
    output["nash_crowding"] = output["nash_density"]
    output["nash_payoff_proxy"] = output["nash_payout_proxy"]
    output["crowd_avoidance"] = 1.0 / (output["nash_density"] + 0.75)
    output["multiplier_per_inhabitant"] = multipliers / inhabitants.clip(lower=1.0)
    output["risk_averse_middle"] = 1.0 - ((multipliers - midpoint).abs() / half_range).clip(upper=1.0)
    output["extreme_pick"] = ((multipliers - midpoint).abs() / half_range).clip(upper=1.0)
    output["digit_3_or_7_bias"] = digit_bias
    output["has_3"] = text_values.str.contains("3").astype(float)
    output["has_7"] = text_values.str.contains("7").astype(float)
    output["special_73"] = (multipliers == 73).astype(float)
    output["famous_37"] = (multipliers == 37).astype(float)
    output["round_number"] = (multipliers % 10 == 0).astype(float)
    output["factor_count"] = multipliers.map(lambda value: divisor_count(int(value))).astype(float)
    output["proper_factor_count"] = (output["factor_count"] - 2.0).clip(lower=0.0)
    output["prime_multiplier"] = multipliers.map(lambda value: 1.0 if is_prime_number(int(value)) else 0.0)
    output["composite_multiplier"] = 1.0 - output["prime_multiplier"]
    output["special_73_feature_raw"] = (multipliers == 73).astype(float)
    output["nash_pull"] = output["nash_component"]
    output["digit_pull"] = output["digit_component"]
    output["model_predicted"] = output["modeled_density"]
    output["model_minus_actual"] = output["model_error"]
    return output


def safe_corr(left: pd.Series, right: pd.Series, method: str = "pearson") -> float:
    paired = pd.DataFrame({"left": left, "right": right}).replace([math.inf, -math.inf], pd.NA).dropna()
    if len(paired) < 3:
        return 0.0
    if float(paired["left"].std()) == 0.0 or float(paired["right"].std()) == 0.0:
        return 0.0
    if method == "spearman":
        return float(paired["left"].rank().corr(paired["right"].rank()))
    return float(paired["left"].corr(paired["right"]))


def p3_correlation_table(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    features = p3_behavior_feature_frame(frame)
    feature_labels = {
        "nash_payoff_proxy": "Nash payoff proxy",
        "multiplier_per_inhabitant": "Multiplier / inhabitant",
        "digit_3_or_7_bias": "3/7 bias score",
        "special_73": "Special 73",
    }
    rows = []
    target = pd.to_numeric(features[target_column], errors="coerce")
    for column, label in feature_labels.items():
        series = pd.to_numeric(features[column], errors="coerce")
        pearson = safe_corr(series, target, "pearson")
        spearman = safe_corr(series, target, "spearman")
        significant = abs(pearson) >= 0.632
        rows.append(
            {
                "Feature": label,
                "Pearson": pearson,
                "Spearman": spearman,
                "Abs Pearson": abs(pearson),
                "Significant": significant,
            }
        )
    return pd.DataFrame(rows).sort_values("Abs Pearson", ascending=False)


def p3_correlation_chart(correlations: pd.DataFrame, target_label: str) -> go.Figure:
    top = correlations.head(16).iloc[::-1].copy()
    colors = ["#f2b447" if value >= 0 else "#4e79a7" for value in top["Pearson"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top["Pearson"],
            y=top["Feature"],
            orientation="h",
            marker_color=colors,
            customdata=top["Spearman"],
            hovertemplate=(
                "%{y}<br>"
                "Pearson %{x:+.3f}<br>"
                "Spearman %{customdata:+.3f}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line={"color": "#f5f7fb", "width": 1.2})
    fig.update_xaxes(title=f"Correlation with {target_label}", range=[-1.05, 1.05])
    fig.update_yaxes(title="")
    return apply_mc_chart_layout(fig, "Behavior Feature Correlations", height=620)


def p3_correlation_strength(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude >= 0.8:
        return "very strong"
    if magnitude >= 0.6:
        return "strong"
    if magnitude >= 0.4:
        return "moderate"
    if magnitude >= 0.2:
        return "weak"
    return "tiny"


def p3_target_effect_phrase(target_label: str, sign: float) -> str:
    direction = "higher" if sign >= 0 else "lower"
    if target_label == "Human - Nash difference":
        return (
            f"Containers high on this feature were {direction} relative to Nash; "
            "positive means humans over-picked them, negative means humans under-picked them."
        )
    if target_label == "Actual human density":
        return f"Containers high on this feature got {direction} actual human pick density."
    if target_label == "Model predicted density":
        return f"Containers high on this feature receive {direction} model-predicted crowd density."
    if target_label == "Model - actual error":
        if sign >= 0:
            return "The model tends to over-predict human density where this feature is high."
        return "The model tends to under-predict human density where this feature is high."
    return f"Containers high on this feature move the selected target {direction}."


def p3_feature_explanations() -> dict[str, str]:
    return {
        "Nash payoff proxy": "The estimated payoff if the crowd followed the Nash table.",
        "Multiplier / inhabitant": "Reward divided by fixed crowd. This captures the simple high-upside, low-crowd instinct.",
        "3/7 bias score": "Psychological digit score for numbers containing 3 or 7, with extra weight on 73 and no hidden boost for famous 37.",
        "Special 73": "Whether the container is exactly 73, the strongest 3/7-bias candidate.",
    }


def p3_correlation_display_table(correlations: pd.DataFrame) -> pd.DataFrame:
    output = correlations[["Feature", "Pearson", "Spearman", "Significant"]].copy()
    output["Pearson"] = output["Pearson"].map(lambda value: f"{float(value):+.3f}")
    output["Spearman"] = output["Spearman"].map(lambda value: f"{float(value):+.3f}")
    output["Significant"] = output["Significant"].map(lambda value: "yes, p<0.05" if value else "no")
    return output


def p3_correlation_interpretation_table(correlations: pd.DataFrame, target_label: str) -> pd.DataFrame:
    explanations = p3_feature_explanations()
    output = correlations[["Feature", "Pearson", "Spearman", "Significant"]].copy()
    output["Strength"] = output["Pearson"].map(p3_correlation_strength)
    output["What this factor means"] = output["Feature"].map(explanations).fillna("")
    output["Effect on selected graph"] = output["Pearson"].map(
        lambda value: p3_target_effect_phrase(target_label, float(value))
    )
    output["Pearson"] = output["Pearson"].map(lambda value: f"{float(value):+.3f}")
    output["Spearman"] = output["Spearman"].map(lambda value: f"{float(value):+.3f}")
    output["Significant"] = output["Significant"].map(lambda value: "yes" if value else "no")
    return output


def p3_container_table(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[
        [
            "multiplier",
            "inhabitants",
            "nash_density",
            "actual_density",
            "modeled_density",
            "model_error",
            "modeled_payout_proxy",
        ]
    ].copy()
    output.columns = [
        "mult",
        "inhab",
        "Nash %",
        "Actual %",
        "Model %",
        "Error pp",
        "Pred payoff",
    ]
    for column in ["Nash %", "Actual %", "Model %", "Error pp"]:
        output[column] = output[column].map(lambda value: f"{float(value):.2f}")
    output["Pred payoff"] = output["Pred payoff"].map(lambda value: f"{float(value):,.0f}")
    return output


def p3_payoff_comparison_table(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = p3_payoff_ranked_frame(frame)
    output = ranked[
        [
            "multiplier",
            "inhabitants",
            "actual_density",
            "modeled_density",
            "actual_payoff_rank",
            "modeled_payoff_rank",
            "known_final_payoff_rank",
            "nash_payout_proxy",
            "actual_payout_proxy",
            "modeled_payout_proxy",
            "payoff_rank_delta",
            "payoff_error",
        ]
    ].copy()
    output.columns = [
        "mult",
        "inhab",
        "Actual %",
        "Pred %",
        "Actual rank",
        "Pred rank",
        "x rank",
        "Nash payoff",
        "Actual payoff",
        "Pred payoff",
        "Rank delta",
        "Pred - actual",
    ]
    for column in ["Actual %", "Pred %"]:
        output[column] = output[column].map(lambda value: f"{float(value):.2f}")
    output["x rank"] = output["x rank"].map(lambda value: f"{int(value)}")
    for column in ["Nash payoff", "Actual payoff", "Pred payoff", "Pred - actual"]:
        output[column] = output[column].map(lambda value: f"{float(value):+,.0f}" if column == "Pred - actual" else f"{float(value):,.0f}")
    output["Rank delta"] = output["Rank delta"].map(lambda value: f"{int(value):+d}")
    return output


def p3_first_order_payoff_table(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[
        [
            "multiplier",
            "inhabitants",
            "first_order_pf_percent",
            "first_order_denominator",
            "first_order_payout_proxy",
            "known_final_payoff_rank",
        ]
    ].copy()
    output.insert(0, "Island", range(len(output)))
    output.columns = [
        "Island",
        "M_f",
        "I_f",
        "p_f (%)",
        "Denominator (p_f + I_f)",
        "PnL",
        "Known final payoff rank",
    ]
    output["p_f (%)"] = output["p_f (%)"].map(lambda value: f"{float(value):.1f}")
    output["Denominator (p_f + I_f)"] = output["Denominator (p_f + I_f)"].map(lambda value: f"{float(value):.1f}")
    output["PnL"] = output["PnL"].map(lambda value: f"{float(value):,.0f}")
    output["Known final payoff rank"] = output["Known final payoff rank"].map(lambda value: f"{int(value)}")
    return output


def p3_rank_implied_crowd_table(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[
        [
            "multiplier",
            "inhabitants",
            "known_final_payoff_rank",
            "first_order_rank_score",
            "rank_inferred_pf_percent",
            "rank_inferred_denominator",
            "rank_inferred_payout_proxy",
            "rank_inferred_payoff_rank",
        ]
    ].copy()
    output.columns = [
        "M_f",
        "I_f",
        "Known final payoff rank",
        "Payoff tier score",
        "Estimated p_f (%)",
        "Denominator (p_f + I_f)",
        "Implied final PnL",
        "Implied final payoff rank",
    ]
    output["Known final payoff rank"] = output["Known final payoff rank"].map(lambda value: f"{int(value)}")
    output["Payoff tier score"] = output["Payoff tier score"].map(lambda value: f"{int(value)}")
    output["Estimated p_f (%)"] = output["Estimated p_f (%)"].map(lambda value: f"{float(value):.2f}")
    output["Denominator (p_f + I_f)"] = output["Denominator (p_f + I_f)"].map(lambda value: f"{float(value):.2f}")
    output["Implied final PnL"] = output["Implied final PnL"].map(lambda value: f"{float(value):,.0f}")
    output["Implied final payoff rank"] = output["Implied final payoff rank"].map(lambda value: f"{int(value)}")
    return output


def p3_strategy_prior_table(weights: dict[str, float]) -> pd.DataFrame:
    rows = [
        {
            "Strategy": "Nash",
            "Weight %": weights["nash_weight"],
            "Logic": "Players copy the equilibrium allocation.",
        },
        {
            "Strategy": "High-number exponential",
            "Weight %": weights["high_number_weight"],
            "Logic": "Players increasingly favor higher multipliers; exponent controls sharpness.",
        },
        {
            "Strategy": "Famous avoiders",
            "Weight %": weights["avoid_famous_weight"],
            "Logic": "Players avoid obvious/famous numbers: 10, 20, 37, 50, 90.",
        },
        {
            "Strategy": "Random",
            "Weight %": weights["random_weight"],
            "Logic": "Players pick without reasoning; uniform across containers.",
        },
        {
            "Strategy": "Nice numbers",
            "Weight %": weights["nice_numbers_weight"],
            "Logic": "Players gravitate to memorable 3/7 numbers, especially 73.",
        },
        {
            "Strategy": "Value hunters",
            "Weight %": weights["value_hunters_weight"],
            "Logic": "Players favor high multiplier per fixed inhabitant.",
        },
        {
            "Strategy": "Infer p_f from known final payoff ranking",
            "Weight %": weights["first_order_pnl_weight"],
            "Logic": "Players assume the final payoff ranking is known. Using only that order, they estimate crowd shares by fitting a simple monotone payoff ladder and inverting the payoff formula to solve for p_f.",
        },
    ]
    output = pd.DataFrame(rows)
    output["Weight %"] = output["Weight %"].map(lambda value: f"{float(value):.1f}")
    return output


def p3_strategy_multiplier_table(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[
        [
            "multiplier",
            "nash_component",
            "high_number_component",
            "avoid_famous_component",
            "random_component",
            "nice_numbers_component",
            "value_hunters_component",
            "first_order_pnl_component",
            "modeled_density",
            "actual_density",
        ]
    ].copy()
    output.columns = [
        "mult",
        "Nash %",
        "High # %",
        "Avoid famous %",
        "Random %",
        "Nice %",
        "Value %",
        "Inferred p_f from rank %",
        "Model %",
        "Actual %",
    ]
    for column in output.columns:
        if column != "mult":
            output[column] = output[column].map(lambda value: f"{float(value):.2f}")
    return output


def p3_set_synced_param(key: str, value: float) -> None:
    st.session_state[f"{key}_slider"] = float(value)
    st.session_state[f"{key}_number"] = float(value)


def p3_slider_with_number(
    label: str,
    min_value: float,
    max_value: float,
    default_value: float,
    step: float,
    key: str,
    help_text: str = "",
    disabled: bool = False,
) -> float:
    slider_key = f"{key}_slider"
    number_key = f"{key}_number"
    st.session_state.setdefault(slider_key, float(default_value))
    st.session_state.setdefault(number_key, float(default_value))

    def sync_from_slider() -> None:
        st.session_state[number_key] = float(st.session_state[slider_key])

    def sync_from_number() -> None:
        value = float(st.session_state[number_key])
        value = min(float(max_value), max(float(min_value), value))
        st.session_state[number_key] = value
        st.session_state[slider_key] = value

    slider_col, input_col = st.columns([2.35, 0.85], gap="small")
    with slider_col:
        st.slider(
            label,
            min_value=float(min_value),
            max_value=float(max_value),
            step=float(step),
            key=slider_key,
            on_change=sync_from_slider,
            help=help_text,
            disabled=disabled,
        )
    with input_col:
        st.number_input(
            f"{label} exact",
            min_value=float(min_value),
            max_value=float(max_value),
            step=float(step),
            key=number_key,
            on_change=sync_from_number,
            label_visibility="collapsed",
            disabled=disabled,
        )
    return float(st.session_state[number_key])


def p3_r3_reserve_fill_fraction(
    bid: pd.Series | float,
    reserve_low: float,
    reserve_high: float,
) -> pd.Series | float:
    bid_numeric = pd.to_numeric(bid, errors="coerce")
    fraction = (bid_numeric - float(reserve_low)) / max(float(reserve_high) - float(reserve_low), 1e-9)
    return np.clip(fraction, 0.0, 1.0)


def p3_r3_average_penalty(
    bid: pd.Series | float,
    average_bid: float,
    resale_price: float,
    exponent: float,
) -> pd.Series | float:
    bid_numeric = pd.to_numeric(bid, errors="coerce")
    denominator = np.maximum(float(resale_price) - bid_numeric, 1e-9)
    numerator = max(float(resale_price) - float(average_bid), 0.0)
    raw_penalty = (numerator / denominator) ** float(exponent)
    return np.minimum(raw_penalty, 1.0)


def p3_r3_profit(
    bid: pd.Series | float,
    seller_count: float,
    reserve_low: float,
    reserve_high: float,
    resale_price: float,
    average_bid: float | None = None,
    exponent: float = 3.0,
) -> pd.Series | float:
    bid_numeric = pd.to_numeric(bid, errors="coerce")
    fill_fraction = p3_r3_reserve_fill_fraction(bid_numeric, reserve_low, reserve_high)
    unit_edge = np.maximum(float(resale_price) - bid_numeric, 0.0)
    base = float(seller_count) * fill_fraction * unit_edge
    if average_bid is None:
        return base
    return base * p3_r3_average_penalty(bid_numeric, average_bid, resale_price, exponent)


def p3_r3_payoff_frame(
    seller_count: float,
    average_bid: float,
    exponent: float,
    reserve_low: float = 250.0,
    reserve_high: float = 320.0,
    resale_price: float = 320.0,
) -> pd.DataFrame:
    bids = np.arange(int(math.floor(reserve_low)), int(math.floor(resale_price)) + 1)
    frame = pd.DataFrame({"Bid": bids})
    frame["Reserve fill %"] = 100.0 * p3_r3_reserve_fill_fraction(frame["Bid"], reserve_low, reserve_high)
    frame["Penalty factor"] = p3_r3_average_penalty(frame["Bid"], average_bid, resale_price, exponent)
    frame["Game payoff"] = p3_r3_profit(
        frame["Bid"],
        seller_count,
        reserve_low,
        reserve_high,
        resale_price,
        average_bid,
        exponent,
    )
    frame["No-penalty payoff"] = p3_r3_profit(
        frame["Bid"],
        seller_count,
        reserve_low,
        reserve_high,
        resale_price,
        None,
        exponent,
    )
    return frame


def p3_r3_payoff_chart(frame: pd.DataFrame, selected_bid: float) -> go.Figure:
    selected = frame.iloc[(frame["Bid"] - selected_bid).abs().argsort()[:1]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["Bid"],
            y=frame["No-penalty payoff"],
            mode="lines",
            name="Reserve-only payoff",
            line={"color": "#7aa2f7", "width": 2.4, "dash": "dot"},
            hovertemplate="Bid %{x}<br>Payoff %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["Bid"],
            y=frame["Game payoff"],
            mode="lines",
            name="With average-bid penalty",
            line={"color": "#f6c85f", "width": 3.0},
            hovertemplate="Bid %{x}<br>Payoff %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=selected["Bid"],
            y=selected["Game payoff"],
            mode="markers",
            name="Selected bid",
            marker={"color": "#e15759", "size": 13, "symbol": "diamond"},
            hovertemplate="Selected bid %{x}<br>Payoff %{y:,.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Your bid / reserve price")
    fig.update_yaxes(title="Expected profit")
    return apply_mc_chart_layout(fig, "Round 3 Reserve Price Payoff Curve", height=430)


def p3_r3_penalty_chart(frame: pd.DataFrame, selected_bid: float) -> go.Figure:
    selected = frame.iloc[(frame["Bid"] - selected_bid).abs().argsort()[:1]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["Bid"],
            y=frame["Reserve fill %"],
            mode="lines",
            name="Seller fill %",
            line={"color": "#6ccf9c", "width": 2.5},
            hovertemplate="Bid %{x}<br>Seller fill %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["Bid"],
            y=100.0 * frame["Penalty factor"],
            mode="lines",
            name="Average penalty %",
            line={"color": "#e15759", "width": 2.5},
            hovertemplate="Bid %{x}<br>Penalty %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=selected["Bid"],
            y=100.0 * selected["Penalty factor"],
            mode="markers",
            name="Selected penalty",
            marker={"color": "#f6c85f", "size": 11},
            hovertemplate="Selected bid %{x}<br>Penalty %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_xaxes(title="Your bid")
    fig.update_yaxes(title="Percent")
    return apply_mc_chart_layout(fig, "Fill Rate vs Average-Bid Penalty", height=360)


def p3_r3_top_bid_table(frame: pd.DataFrame, selected_bid: float, limit: int = 12) -> pd.DataFrame:
    table = frame.sort_values("Game payoff", ascending=False).head(limit).copy()
    selected_row = frame.iloc[(frame["Bid"] - selected_bid).abs().argsort()[:1]].copy()
    selected_row["Rank label"] = "selected"
    table["Rank label"] = [f"#{idx}" for idx in range(1, len(table) + 1)]
    if int(selected_row["Bid"].iloc[0]) not in set(table["Bid"].astype(int)):
        table = pd.concat([table, selected_row], ignore_index=True)
    table = table[["Rank label", "Bid", "Reserve fill %", "Penalty factor", "No-penalty payoff", "Game payoff"]]
    for column in ["Reserve fill %"]:
        table[column] = table[column].map(lambda value: f"{float(value):.1f}%")
    for column in ["Penalty factor"]:
        table[column] = table[column].map(lambda value: f"{float(value):.3f}")
    for column in ["No-penalty payoff", "Game payoff"]:
        table[column] = table[column].map(lambda value: fmt_number(value, 2))
    return table


def render_p3_reserve_manual_page() -> None:
    st.markdown(
        '<div class="mc-title">Prosperity 3 Round 3 <span class="mc-chip">reserve price game</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="mc-panel">
          <div class="mc-section-title">Problem Statement</div>
          <div class="mc-note">
            Round 3 was a reserve-price optimization game. Sellers had private reserve prices. A seller trades with you only if your bid is at least their reserve price. Goods bought in the challenge could be resold for <b>320</b>, so bidding higher wins more sellers but earns less profit per unit.
            <br><br>
            The game-theory twist: when your bid is below the crowd average, your payoff is scaled down. This creates a tension between the pure mathematical optimum and the strategic guess of where other teams will bid.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    controls, main_panel = st.columns([1.0, 2.15], gap="medium")
    with controls:
        st.markdown('<div class="mc-panel"><div class="mc-section-title">Sliders</div>', unsafe_allow_html=True)
        selected_bid = p3_slider_with_number(
            "Your bid p",
            250.0,
            319.0,
            303.0,
            1.0,
            "p3_r3_selected_bid",
            "Your chosen reserve-price bid.",
        )
        average_bid = p3_slider_with_number(
            "Crowd average μ",
            250.0,
            319.0,
            287.0,
            1.0,
            "p3_r3_average_bid",
            "Estimated average bid from all participating teams.",
        )
        seller_count = p3_slider_with_number(
            "Number of sellers N",
            1.0,
            500.0,
            100.0,
            1.0,
            "p3_r3_seller_count",
            "Scales expected payoff. It does not change the optimal bid.",
        )
        penalty_exponent = p3_slider_with_number(
            "Penalty exponent",
            1.0,
            6.0,
            3.0,
            0.25,
            "p3_r3_penalty_exponent",
            "The official writeup used a cubic-looking penalty. Higher means harsher punishment below average.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="mc-panel">
              <div class="mc-section-title">Core Equations</div>
              <div class="mc-note">
                Reserve-only payoff:<br>
                <code>Π(p) = N · ((p - 250) / 70) · (320 - p)</code>
                <br><br>
                Strategic penalty:<br>
                <code>S(p, μ) = min(((320 - μ) / (320 - p))³, 1)</code>
                <br><br>
                Final game payoff:<br>
                <code>Π(p, μ) = N · ((p - 250) / 70) · (320 - p) · S(p, μ)</code>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    frame = p3_r3_payoff_frame(seller_count, average_bid, penalty_exponent)
    selected_row = frame.iloc[(frame["Bid"] - selected_bid).abs().argsort()[:1]].iloc[0]
    best_row = frame.sort_values("Game payoff", ascending=False).iloc[0]
    reserve_only_best = frame.sort_values("No-penalty payoff", ascending=False).iloc[0]

    with main_panel:
        card_a, card_b, card_c, card_d = st.columns(4)
        with card_a:
            mc_card("Selected payoff", fmt_number(selected_row["Game payoff"], 2), f"bid {int(selected_row['Bid'])}")
        with card_b:
            mc_card("Best strategic bid", str(int(best_row["Bid"])), f"payoff {fmt_number(best_row['Game payoff'], 2)}")
        with card_c:
            mc_card("Reserve-only optimum", str(int(reserve_only_best["Bid"])), f"payoff {fmt_number(reserve_only_best['No-penalty payoff'], 2)}")
        with card_d:
            mc_card("Penalty at selected", f"{100.0 * selected_row['Penalty factor']:.1f}%", f"fill {selected_row['Reserve fill %']:.1f}%")

        st.plotly_chart(p3_r3_payoff_chart(frame, selected_bid), use_container_width=True)
        lower_left, lower_right = st.columns([1.0, 1.15], gap="medium")
        with lower_left:
            st.plotly_chart(p3_r3_penalty_chart(frame, selected_bid), use_container_width=True)
        with lower_right:
            st.markdown('<div class="mc-panel"><div class="mc-section-title">Best Bids Under Current Assumptions</div>', unsafe_allow_html=True)
            mc_table(p3_r3_top_bid_table(frame, selected_bid))
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="mc-panel">
              <div class="mc-section-title">Intuition</div>
              <div class="mc-note">
                If you bid too low, you keep a large per-unit edge but very few sellers accept. If you bid too high, many sellers accept but your resale margin disappears. Without game theory, the maximum sits near the midpoint of the reserve distribution. With the average-bid penalty, being too far below the crowd average can be worse than giving up some margin by bidding higher.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="mc-panel">
              <div class="mc-section-title">Observed Human Distribution</div>
              <div class="mc-note">
                The realized crowd was not spread smoothly across the reserve-price range. First bids clustered extremely hard around <b>200</b>, while second bids clustered around the high-280s, especially <b>286–290</b>. That is a nice example of a manual round where the first decision was mostly solved analytically, but the second decision was shaped much more by focal points and social coordination.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        try:
            st.image(
                str(P3_R3_HUMAN_DISTRIBUTION_IMAGE),
                caption="Prosperity 3 Round 3 human bid distribution for first and second bids.",
                use_container_width=True,
            )
        except Exception:
            st.info("Human-distribution screenshot is referenced here, but the local image could not be loaded in this environment.")


def p3_r3_option_smile_chart(
    scatter: pd.DataFrame,
    options: pd.DataFrame,
    published_coeffs: np.ndarray,
    dynamic_coeffs: np.ndarray,
    fit_mode: str,
) -> go.Figure:
    fig = go.Figure()
    strike_colors = {
        9500: "#5b7cfa",
        9750: "#e76f51",
        10000: "#59c17a",
        10250: "#8d70ff",
        10500: "#f4a261",
    }
    for strike in P3_R3_OPTION_STRIKES:
        frame = scatter[scatter["strike"] == strike]
        if frame.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=frame["moneyness"],
                y=frame["market_iv"],
                mode="markers",
                name=f"strike={strike}",
                marker={"size": 5, "opacity": 0.55, "color": strike_colors.get(strike, "#93a1b2")},
                hovertemplate="m=%{x:.3f}<br>IV=%{y:.3f}<extra></extra>",
            )
        )

    domain = options["moneyness"].replace([np.inf, -np.inf], np.nan).dropna()
    if not domain.empty:
        xs = np.linspace(float(domain.min()), float(domain.max()), 250)
        published_poly = np.poly1d(published_coeffs)
        dynamic_poly = np.poly1d(dynamic_coeffs)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=published_poly(xs),
                mode="lines",
                name="Frankfurt published fit",
                line={"color": "#111111", "width": 3},
                hovertemplate="published fit<br>m=%{x:.3f}<br>IV=%{y:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=dynamic_poly(xs),
                mode="lines",
                name="Dashboard refit",
                line={"color": "#f6c85f", "width": 2, "dash": "dash"},
                visible=True if fit_mode == "Dashboard refit" else "legendonly",
                hovertemplate="dashboard refit<br>m=%{x:.3f}<br>IV=%{y:.3f}<extra></extra>",
            )
        )
    fig.update_xaxes(title="moneyness  log(K / S) / sqrt(tau)")
    fig.update_yaxes(title="implied volatility")
    return apply_mc_chart_layout(fig, "Figure 6a: Volatility Smile", height=420)


def p3_r3_multi_strike_timeseries_chart(
    frame: pd.DataFrame,
    value_column: str,
    title: str,
    y_title: str,
    zero_line: bool = True,
) -> go.Figure:
    fig = go.Figure()
    strike_colors = {
        9500: "#5b7cfa",
        9750: "#e76f51",
        10000: "#59c17a",
        10250: "#8d70ff",
        10500: "#f4a261",
    }
    for strike in P3_R3_OPTION_STRIKES:
        sub = frame[frame["strike"] == strike].sort_values("timestamp")
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["timestamp"],
                y=sub[value_column],
                mode="lines",
                name=f"strike={strike}",
                line={"width": 1.7, "color": strike_colors.get(strike, "#93a1b2")},
                hovertemplate="t=%{x}<br>value=%{y:.3f}<extra></extra>",
            )
        )
    if zero_line:
        fig.add_hline(y=0.0, line={"color": "#8792a2", "width": 1, "dash": "dot"})
    fig.update_xaxes(title="timestamp")
    fig.update_yaxes(title=y_title)
    return apply_mc_chart_layout(fig, title, height=360)


def p3_r3_focus_price_chart(
    focus: pd.DataFrame,
    focus_trades: pd.DataFrame,
    focus_strike: int,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=focus["timestamp"],
            y=focus["option_wall_mid"],
            mode="lines+markers",
            name="Observed voucher price",
            line={"color": "#5b7cfa", "width": 2},
            marker={"size": 4, "color": "#355cfa"},
            hovertemplate="t=%{x}<br>market=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=focus["timestamp"],
            y=focus["fair_price"],
            mode="lines",
            name="Theoretical price",
            line={"color": "#f4a261", "width": 2},
            hovertemplate="t=%{x}<br>theo=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=focus["timestamp"],
            y=focus["bid_price_1"],
            mode="markers",
            name="Best bid",
            marker={"size": 7, "symbol": "circle", "color": "#2448ff", "opacity": 0.8},
            hovertemplate="t=%{x}<br>bid=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=focus["timestamp"],
            y=focus["ask_price_1"],
            mode="markers",
            name="Best ask",
            marker={"size": 7, "symbol": "circle", "color": "#ff5f5f", "opacity": 0.8},
            hovertemplate="t=%{x}<br>ask=%{y:.2f}<extra></extra>",
        )
    )
    if not focus_trades.empty:
        fig.add_trace(
            go.Scatter(
                x=focus_trades["timestamp"],
                y=focus_trades["price"],
                mode="markers",
                name="Public trades",
                marker={
                    "size": np.clip(np.abs(focus_trades["quantity"]) * 1.4, 7, 18),
                    "symbol": "cross",
                    "color": "#ffd166",
                    "line": {"color": "#f0f0f0", "width": 0.6},
                },
                hovertemplate="t=%{x}<br>trade=%{y:.2f}<br>qty=%{marker.size:.0f}<extra></extra>",
            )
        )
    fig.update_xaxes(title="timestamp")
    fig.update_yaxes(title=f"voucher_{focus_strike} price")
    return apply_mc_chart_layout(fig, "Figure 7a: Focus Strike Price Fluctuations", height=390)


def p3_r3_focus_normalized_chart(focus: pd.DataFrame, focus_strike: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=focus["timestamp"],
            y=focus["normalized_deviation_pct"],
            mode="lines+markers",
            name="Normalized deviation %",
            line={"color": "#5b7cfa", "width": 2},
            marker={"size": 4, "color": "#355cfa"},
            hovertemplate="t=%{x}<br>norm dev=%{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=focus["timestamp"],
            y=100.0 * focus["mean_theo_diff"] / focus["fair_price"].replace(0, pd.NA),
            mode="lines",
            name="EMA(20) baseline %",
            line={"color": "#f4a261", "width": 2},
            hovertemplate="t=%{x}<br>ema=%{y:.2f}%<extra></extra>",
        )
    )
    fig.add_hline(y=0.0, line={"color": "#8792a2", "width": 1, "dash": "dot"})
    fig.update_xaxes(title="timestamp")
    fig.update_yaxes(title=f"voucher_{focus_strike} normalized deviation %")
    return apply_mc_chart_layout(fig, "Figure 7b: Focus Strike Price Fluctuations (Normalized)", height=360)


def p3_r3_autocorrelation_chart(acf_frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if acf_frame.empty:
        return apply_mc_chart_layout(fig, "Figure 8: Autocorrelation Plot for Volcanic Rock", height=360)

    for series_name, sub in acf_frame.groupby("series", sort=False):
        if series_name == "VOLCANIC_ROCK":
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["lag"],
                y=sub["value"],
                mode="lines",
                name=series_name,
                line={"color": "rgba(190,190,190,0.16)", "width": 1},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    real = acf_frame[acf_frame["series"] == "VOLCANIC_ROCK"].sort_values("lag")
    fig.add_trace(
        go.Scatter(
            x=real["lag"],
            y=real["value"],
            mode="lines",
            name="VOLCANIC_ROCK",
            line={"color": "#ff5f5f", "width": 3},
            hovertemplate="lag=%{x}<br>acf=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0.0, line={"color": "#8792a2", "width": 1, "dash": "dot"})
    fig.update_xaxes(title="lag window")
    fig.update_yaxes(title="autocorrelation")
    return apply_mc_chart_layout(fig, "Figure 8: Autocorrelation Plot for Volcanic Rock", height=360)


def p3_r3_gamma_convexity_chart(
    spot: float,
    strike: float,
    tau_days: float,
    sigma: float,
    move_pct: float,
) -> go.Figure:
    tau = max(float(tau_days), 0.05) / 365.0
    _, delta, _, _ = p3_r3_bs_call_metrics(spot, strike, tau, sigma)
    pct_moves = np.linspace(-float(move_pct), float(move_pct), 121)
    pnl = []
    for pct in pct_moves:
        shifted = spot * (1.0 + pct / 100.0)
        base, _, _, _ = p3_r3_bs_call_metrics(spot, strike, tau, sigma)
        nxt, _, _, _ = p3_r3_bs_call_metrics(shifted, strike, tau, sigma)
        pnl.append(nxt - base - delta * (shifted - spot))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pct_moves,
            y=pnl,
            mode="lines",
            name="delta-hedged option PnL",
            line={"color": "#59c17a", "width": 3},
            hovertemplate="move=%{x:.2f}%<br>PnL=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0.0, line={"color": "#8792a2", "width": 1, "dash": "dot"})
    fig.update_xaxes(title="underlying move %")
    fig.update_yaxes(title="delta-hedged voucher PnL")
    return apply_mc_chart_layout(fig, "Gamma Scalping Convexity Sketch", height=320)


def render_p3_round3_options_page() -> None:
    options_raw, underlying_raw, trades_raw = load_p3_r3_option_market()
    if options_raw.empty or underlying_raw.empty:
        st.warning("Round 3 options data was not found in the local Prosperity 3 resources.")
        return

    available_days = sorted(int(day) for day in options_raw["day"].dropna().unique())
    top_row_left, top_row_right = st.columns([0.95, 1.45], gap="medium")
    with top_row_left:
        st.markdown(
            """
            <div class="mc-panel">
              <div class="mc-section-title">What Frankfurt Hedgehogs Actually Did</div>
              <div class="mc-note">
                They did <b>not</b> treat each voucher as an isolated price series. They first compressed the entire option surface into a single object: a volatility smile. Then they traded the <b>residual mispricing</b> of each strike relative to that smile, converted back into price space through Black-Scholes, and only switched the scalper on when the deviation process was lively enough to overcome spread and noise.
                <br><br>
                The published README explains the logic; the polished trader reveals the concrete implementation. This dashboard combines both: README-sourced intuition, plus code-level details such as the exact smile coefficients, windows, thresholds, and the fact that the trading rule is based on <b>executable bid/ask mispricing</b>, not midpoint fantasy.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected_day = int(
            st.selectbox(
                "Historical day",
                available_days,
                index=0,
                format_func=lambda value: f"Round 3 day {value}",
                key="p3_r3_algo_day",
            )
        )
        days_to_expiry_open = p3_slider_with_number(
            "Days to expiry at day open",
            3.0,
            7.0,
            5.0,
            0.25,
            "p3_r3_algo_days_to_expiry",
            "Reasonable reconstruction of the remaining time to expiry during the Round 3 option round.",
        )
        extrinsic_floor = p3_slider_with_number(
            "Outlier filter: minimum extrinsic value",
            0.0,
            25.0,
            5.0,
            0.5,
            "p3_r3_algo_extrinsic_floor",
            "They explicitly ignored bottom-left smile outliers with too little extrinsic value.",
        )
        scatter_points_per_strike = int(
            p3_slider_with_number(
                "Smile scatter sample per strike",
                150.0,
                1200.0,
                400.0,
                50.0,
                "p3_r3_algo_scatter_points",
                "Downsamples the smile cloud so the scatter stays readable and fast.",
            )
        )
    with top_row_right:
        st.markdown(
            """
            <div class="mc-panel">
              <div class="mc-section-title">Production Parameters Reverse-Engineered from the Polished Trader</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fit_mode = st.selectbox(
            "Smile model used for pricing plots",
            ("Published Frankfurt fit", "Dashboard refit"),
            index=0,
            key="p3_r3_algo_fit_mode",
        )
        focus_strike = int(
            st.selectbox(
                "Focus strike for price-action view",
                P3_R3_OPTION_STRIKES,
                index=2,
                key="p3_r3_algo_focus_strike",
            )
        )
        random_sample_count = int(
            p3_slider_with_number(
                "Random baselines in autocorrelation plot",
                10.0,
                120.0,
                40.0,
                5.0,
                "p3_r3_algo_random_samples",
                "More baselines make the significance picture stronger, but cost more render time.",
            )
        )
        max_lag = int(
            p3_slider_with_number(
                "Max lag window in autocorrelation plot",
                20.0,
                150.0,
                100.0,
                5.0,
                "p3_r3_algo_max_lag",
                "Longer windows show whether the negative autocorrelation survives beyond the shortest horizons.",
            )
        )

    analysis = build_p3_r3_option_analysis(
        selected_day=selected_day,
        days_to_expiry_open=days_to_expiry_open,
        extrinsic_floor=extrinsic_floor,
        fit_mode=fit_mode,
        scatter_points_per_strike=scatter_points_per_strike,
        random_sample_count=random_sample_count,
        max_lag=max_lag,
    )
    options = analysis["options"]
    underlying = analysis["underlying"]
    trades = analysis["trades"]
    scatter = analysis["scatter"]
    focus_symbol = f"VOLCANIC_ROCK_VOUCHER_{focus_strike}"
    focus = options[options["product"] == focus_symbol].sort_values("timestamp").copy()
    focus_trades = trades[trades["symbol"] == focus_symbol].copy()

    published_coeffs = np.asarray(analysis["published_coeffs"], dtype=float)
    dynamic_coeffs = np.asarray(analysis["dynamic_coeffs"], dtype=float)

    if options.empty:
        st.warning("No option rows were available for the chosen settings.")
        return

    summary_cards = st.columns(4)
    with summary_cards[0]:
        mc_card("Published smile fit", f"{published_coeffs[0]:.4f} m^2 + {published_coeffs[1]:.4f} m + {published_coeffs[2]:.4f}", "Hardcoded in their polished file")
    with summary_cards[1]:
        mc_card("Dashboard refit", f"{dynamic_coeffs[0]:.4f} m^2 + {dynamic_coeffs[1]:.4f} m + {dynamic_coeffs[2]:.4f}", "Refit from the selected historical day")
    with summary_cards[2]:
        mc_card("Scalping windows", f"EMA {P3_R3_THEO_NORM_WINDOW} / switch {P3_R3_IV_SCALPING_WINDOW}", f"activation threshold {P3_R3_IV_SCALPING_THR:.1f}")
    with summary_cards[3]:
        mc_card("Mean reversion sleeve", f"VR thr {P3_R3_UNDERLYING_MR_THR:.0f} | 9500 thr {P3_R3_OPTIONS_MR_THR:.0f}", f"windows {P3_R3_UNDERLYING_MR_WINDOW} and {P3_R3_OPTIONS_MR_WINDOW}")

    st.markdown(
        """
        <div class="mc-panel">
          <div class="mc-section-title">Deep Walkthrough</div>
          <div class="mc-note">
            <b>1. Re-parameterize the problem in implied-vol space.</b> Voucher prices at different strikes are not directly comparable, but their implied vols are. By mapping each option into <code>(moneyness, IV)</code>, they turned five moving products into one structural curve.
            <br><br>
            <b>2. Fit the smile, then trade the residual.</b> The parabola removes the predictable cross-strike shape. What remains is the part they cared about: transient, local richness or cheapness relative to the smile.
            <br><br>
            <b>3. Convert back into executable price space.</b> Their production rule is not “mid is rich, so sell”. It is effectively “if the <b>best bid</b> is rich enough relative to theoretical value plus the EMA baseline, hit it”. Same idea on the ask for buys. That means their model already includes spread and execution friction.
            <br><br>
            <b>4. Only scalp when the tape is lively.</b> They maintain an EMA of the absolute residual deviation. If that choppiness proxy is below <code>0.7</code>, they switch the scalper off and flatten rather than force trades in dead regimes.
            <br><br>
            <b>5. Keep the mean-reversion sleeve deliberately simple.</b> Instead of overfitting a fancy stochastic-vol model, they used fast EMAs and fixed thresholds. The deepest ITM call (<code>9500</code>) served as the options-side mean-reversion instrument because it had the highest delta, so it behaved most like leveraged Volcanic Rock.
            <br><br>
            <b>6. Gamma scalping was a positive-EV fallback, not the main engine.</b> They believed convexity plus re-hedging had positive expectancy, but the absolute returns were too small relative to the bigger IV-residual opportunity, so it stayed secondary.
            <br><br>
            <b>7. Final portfolio logic was relative-risk aware.</b> Late in the contest they kept some mean-reversion exposure not because it was obviously the highest standalone EV, but because it reduced regret if rival teams were leaning hard into that same angle.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_chart, right_chart = st.columns([1.1, 1.1], gap="medium")
    with left_chart:
        st.plotly_chart(
            p3_r3_option_smile_chart(scatter, options, published_coeffs, dynamic_coeffs, fit_mode),
            use_container_width=True,
        )
    with right_chart:
        st.markdown('<div class="mc-panel"><div class="mc-section-title">Parameter Choices and Why They Matter</div>', unsafe_allow_html=True)
        parameter_table = pd.DataFrame(
            [
                {"Component": "Smile coefficients", "Value": "0.2736, 0.0101, 0.1488", "Interpretation": "Quadratic smile fit hardcoded into Black-Scholes pricing."},
                {"Component": "THR_OPEN / THR_CLOSE", "Value": "0.5 / 0.0", "Interpretation": "Open only with real edge through the spread; close as soon as the edge is gone."},
                {"Component": "LOW_VEGA_THR_ADJ", "Value": "0.5", "Interpretation": "Extra caution for low-vega wings where IV estimates are noisy."},
                {"Component": "THEO_NORM_WINDOW", "Value": "20", "Interpretation": "Fast enough to center short-lived mispricing without following every tick."},
                {"Component": "IV_SCALPING_WINDOW / THR", "Value": "100 / 0.7", "Interpretation": "Measures whether the deviation process is active enough to scalp."},
                {"Component": "Scalped strikes", "Value": "9750, 10000, 10250, 10500", "Interpretation": "They left 9500 out of the scalper and used it for the mean-reversion sleeve."},
                {"Component": "VR / 9500 MR thresholds", "Value": "15 / 5", "Interpretation": "Much larger threshold on the underlying than on the deep ITM call because the scales differ."},
            ]
        )
        mc_table(parameter_table)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="mc-panel"><div class="mc-section-title">Per-Strike Diagnostics</div>', unsafe_allow_html=True)
        mc_table(analysis["insight_table"])
        st.markdown("</div>", unsafe_allow_html=True)

    ts_left, ts_right = st.columns(2, gap="medium")
    with ts_left:
        st.plotly_chart(
            p3_r3_multi_strike_timeseries_chart(
                options,
                "iv_deviation",
                "Figure 6b: IV Deviations over Time",
                "market IV - smile IV",
            ),
            use_container_width=True,
        )
    with ts_right:
        st.plotly_chart(
            p3_r3_multi_strike_timeseries_chart(
                options,
                "price_deviation",
                "Figure 6c: Price Deviations over Time",
                "market price - theoretical price",
            ),
            use_container_width=True,
        )

    focus_left, focus_right = st.columns(2, gap="medium")
    with focus_left:
        st.plotly_chart(
            p3_r3_focus_price_chart(focus, focus_trades, focus_strike),
            use_container_width=True,
        )
    with focus_right:
        st.plotly_chart(
            p3_r3_focus_normalized_chart(focus, focus_strike),
            use_container_width=True,
        )
        st.markdown(
            """
            <div class="mc-panel">
              <div class="mc-section-title">How Their Scalper Really Fires</div>
              <div class="mc-note">
                For a sell, the operative object is roughly <code>best_bid - fair_price - EMA(price_deviation)</code>. For a buy, it is the same expression at the ask. If the switch variable is below threshold, they do <b>nothing</b> except flatten leftovers. This is why their scalper is selective instead of always-on.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    acf_left, acf_right = st.columns([1.15, 0.95], gap="medium")
    with acf_left:
        st.plotly_chart(p3_r3_autocorrelation_chart(analysis["acf_frame"]), use_container_width=True)
    with acf_right:
        st.markdown(
            """
            <div class="mc-panel">
              <div class="mc-section-title">Mean-Reversion Read</div>
              <div class="mc-note">
                Their README frames this as a robust sanity check: compare Volcanic Rock’s return autocorrelation to many random baselines. The point was not to estimate a perfect stochastic process. It was to answer a simpler question: “Is negative short-horizon autocorrelation strong enough that a lightweight EMA-threshold trader is defensible?” Their answer was yes — but only as a moderate sleeve, not as the whole book.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="mc-panel">
              <div class="mc-section-title">What They Tried But Down-Weighted</div>
              <div class="mc-note">
                The README explicitly says gamma scalping looked steadily positive but too small, while pure mean reversion looked promising but riskier and less certain with only a few historical days. That is why the final portfolio is a hybrid: a large, theory-backed IV-residual engine plus a smaller, regret-aware mean-reversion hedge.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    sim_spot = float(
        p3_slider_with_number(
            "Gamma sketch: underlying spot",
            9800.0,
            10800.0,
            10325.0,
            5.0,
            "p3_r3_gamma_spot",
            "Interactive convexity sketch for the gamma-scalping idea they discussed but did not prioritize.",
        )
    )
    sim_iv = float(
        p3_slider_with_number(
            "Gamma sketch: implied volatility",
            0.10,
            0.40,
            0.20,
            0.005,
            "p3_r3_gamma_iv",
            "Use the smile level you think is relevant for the strike being studied.",
        )
    )
    sim_days = float(
        p3_slider_with_number(
            "Gamma sketch: days to expiry",
            1.0,
            7.0,
            5.0,
            0.25,
            "p3_r3_gamma_days",
            "Shorter expiry makes the option more convex but also more exposed to theta decay.",
        )
    )
    sim_move = float(
        p3_slider_with_number(
            "Gamma sketch: symmetric move band %",
            0.25,
            6.0,
            2.5,
            0.25,
            "p3_r3_gamma_move",
            "This shows the delta-hedged convexity profile around the current spot.",
        )
    )

    gamma_price, gamma_delta, gamma_gamma, gamma_vega = p3_r3_bs_call_metrics(
        sim_spot,
        float(focus_strike),
        sim_days / 365.0,
        sim_iv,
    )
    gamma_cards = st.columns(4)
    with gamma_cards[0]:
        mc_card("Gamma sketch price", fmt_number(gamma_price, 2), f"strike {focus_strike}")
    with gamma_cards[1]:
        mc_card("Delta", fmt_number(gamma_delta, 3), "Why 9500 behaves most like the underlying")
    with gamma_cards[2]:
        mc_card("Gamma", fmt_number(gamma_gamma, 6), "Convexity that powers gamma scalping")
    with gamma_cards[3]:
        mc_card("Vega", fmt_number(gamma_vega, 3), "Low-vega wings deserve extra caution")

    st.plotly_chart(
        p3_r3_gamma_convexity_chart(sim_spot, float(focus_strike), sim_days, sim_iv, sim_move),
        use_container_width=True,
    )

    st.markdown(
        """
        <div class="mc-panel">
          <div class="mc-section-title">Bottom Line</div>
          <div class="mc-note">
            The deepest insight is that Frankfurt Hedgehogs separated <b>structural pricing</b> from <b>trading execution</b>. Structural pricing came from the smile and Black-Scholes. Execution came from comparing theoretical value to what could actually be hit or lifted. Then they overlaid a regime switch so the scalper only traded when microstructure noise was strong enough, and paired it with a deliberately modest mean-reversion book to hedge outcome risk rather than pretend they could perfectly delta-hedge everything for free.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_p3_reserve_price_page() -> None:
    st.markdown(
        '<div class="mc-title">Prosperity 3 Round 3 <span class="mc-chip">manual + algo deep dive</span></div>',
        unsafe_allow_html=True,
    )
    manual_tab, algo_tab = st.tabs(["Manual: Reserve Price", "Algo: Frankfurt Hedgehogs Options"])
    with manual_tab:
        render_p3_reserve_manual_page()
    with algo_tab:
        render_p3_round3_options_page()


def render_p3_container_page() -> None:
    st.markdown(
        '<div class="mc-title">Prosperity 3 Round 2 <span class="mc-chip">container game model</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="mc-panel">
          <div class="mc-section-title">Calibrated Priors Model</div>
          <div class="mc-note">
            This model is a small mixture of human strategies, not a curve-fit with lots of knobs. Each strategy creates its own distribution across the multipliers, then the strategy weights combine into the predicted crowd. R4-style inverse Nash and payoff-rank fitting are intentionally excluded here because this page is modeling first-play R2 behavior.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    p3_controls, p3_charts = st.columns([1.0, 2.15], gap="medium")
    with p3_controls:
        st.markdown('<div class="mc-panel"><div class="mc-section-title">Strategy Priors</div>', unsafe_allow_html=True)
        default_rank_tuple = p3_first_order_rank_tuple()
        st.session_state.setdefault("p3_known_rank_tuple", default_rank_tuple)
        for key, default_value in {
            "p3_nash_weight": 20.0,
            "p3_high_number_weight": 20.0,
            "p3_high_number_exponent": 3.0,
            "p3_avoid_famous_weight": 10.0,
            "p3_random_weight": 15.0,
            "p3_nice_numbers_weight": 20.0,
            "p3_value_hunters_weight": 20.0,
            "p3_first_order_pnl_weight": 15.0,
        }.items():
            st.session_state.setdefault(f"{key}_slider", float(default_value))
            st.session_state.setdefault(f"{key}_number", float(default_value))
        for key in [
            "p3_nash_enabled",
            "p3_high_number_enabled",
            "p3_avoid_famous_enabled",
            "p3_random_enabled",
            "p3_nice_numbers_enabled",
            "p3_value_hunters_enabled",
            "p3_first_order_pnl_enabled",
        ]:
            st.session_state.setdefault(key, True)

        enabled_map = {
            "nash": bool(st.session_state["p3_nash_enabled"]),
            "high_number": bool(st.session_state["p3_high_number_enabled"]),
            "avoid_famous": bool(st.session_state["p3_avoid_famous_enabled"]),
            "random": bool(st.session_state["p3_random_enabled"]),
            "nice_numbers": bool(st.session_state["p3_nice_numbers_enabled"]),
            "value_hunters": bool(st.session_state["p3_value_hunters_enabled"]),
            "first_order_pnl": bool(st.session_state["p3_first_order_pnl_enabled"]),
        }
        if st.button("Auto fit % people", type="primary", use_container_width=True):
            enabled_strategies = tuple(name for name, is_enabled in enabled_map.items() if is_enabled)
            fit = p3_auto_fit_container_model(
                enabled_strategies,
                tuple(int(value) for value in st.session_state["p3_known_rank_tuple"]),
                "density",
            )
            for fit_key in [
                "p3_nash_weight",
                "p3_high_number_weight",
                "p3_high_number_exponent",
                "p3_avoid_famous_weight",
                "p3_random_weight",
                "p3_nice_numbers_weight",
                "p3_value_hunters_weight",
                "p3_first_order_pnl_weight",
            ]:
                fit_value = round(float(fit[fit_key.replace("p3_", "")]), 2)
                p3_set_synced_param(fit_key, fit_value)

        fit_rank_left, fit_rank_right = st.columns(2, gap="small")
        with fit_rank_left:
            if st.button("Auto fit top 5 ranks", use_container_width=True):
                enabled_strategies = tuple(name for name, is_enabled in enabled_map.items() if is_enabled)
                fit = p3_auto_fit_container_model(
                    enabled_strategies,
                    tuple(int(value) for value in st.session_state["p3_known_rank_tuple"]),
                    "rank_top5",
                )
                for fit_key in [
                    "p3_nash_weight",
                    "p3_high_number_weight",
                    "p3_high_number_exponent",
                    "p3_avoid_famous_weight",
                    "p3_random_weight",
                    "p3_nice_numbers_weight",
                    "p3_value_hunters_weight",
                    "p3_first_order_pnl_weight",
                ]:
                    fit_value = round(float(fit[fit_key.replace("p3_", "")]), 2)
                    p3_set_synced_param(fit_key, fit_value)
        with fit_rank_right:
            if st.button("Auto fit profit", use_container_width=True):
                enabled_strategies = tuple(name for name, is_enabled in enabled_map.items() if is_enabled)
                fit = p3_auto_fit_container_model(
                    enabled_strategies,
                    tuple(int(value) for value in st.session_state["p3_known_rank_tuple"]),
                    "profit",
                )
                for fit_key in [
                    "p3_nash_weight",
                    "p3_high_number_weight",
                    "p3_high_number_exponent",
                    "p3_avoid_famous_weight",
                    "p3_random_weight",
                    "p3_nice_numbers_weight",
                    "p3_value_hunters_weight",
                    "p3_first_order_pnl_weight",
                ]:
                    fit_value = round(float(fit[fit_key.replace("p3_", "")]), 2)
                    p3_set_synced_param(fit_key, fit_value)

        st.caption("Turn a strategy off to exclude it from both the live model and the auto-fit.")
        st.caption("The current x-rank list stays fixed; the three autofit buttons optimize weights against different targets.")

        p3_nash_enabled = st.toggle("Use Nash", key="p3_nash_enabled")
        p3_nash_weight = p3_slider_with_number(
            "Nash",
            0.0,
            100.0,
            20.0,
            0.5,
            "p3_nash_weight",
            disabled=not p3_nash_enabled,
        )

        p3_high_number_enabled = st.toggle("Use High-number exponential", key="p3_high_number_enabled")
        p3_high_number_weight = p3_slider_with_number(
            "High-number exponential",
            0.0,
            100.0,
            20.0,
            0.5,
            "p3_high_number_weight",
            "People who mostly look at the larger multipliers.",
            disabled=not p3_high_number_enabled,
        )
        p3_high_number_exponent = p3_slider_with_number(
            "High-number exponent",
            0.5,
            8.0,
            3.0,
            0.25,
            "p3_high_number_exponent",
            "Higher exponent concentrates this strategy harder into the biggest multipliers.",
            disabled=not p3_high_number_enabled,
        )
        p3_avoid_famous_enabled = st.toggle("Use Famous-number avoiders", key="p3_avoid_famous_enabled")
        p3_avoid_famous_weight = p3_slider_with_number(
            "Famous-number avoiders",
            0.0,
            100.0,
            10.0,
            0.5,
            "p3_avoid_famous_weight",
            "People avoiding obvious/suspicious non-random numbers: 10, 20, 37, 50, 90.",
            disabled=not p3_avoid_famous_enabled,
        )
        p3_random_enabled = st.toggle("Use Random", key="p3_random_enabled")
        p3_random_weight = p3_slider_with_number(
            "Random",
            0.0,
            100.0,
            15.0,
            0.5,
            "p3_random_weight",
            disabled=not p3_random_enabled,
        )
        p3_nice_numbers_enabled = st.toggle("Use Nice numbers", key="p3_nice_numbers_enabled")
        p3_nice_numbers_weight = p3_slider_with_number(
            "Nice numbers",
            0.0,
            100.0,
            20.0,
            0.5,
            "p3_nice_numbers_weight",
            disabled=not p3_nice_numbers_enabled,
        )
        p3_value_hunters_enabled = st.toggle("Use Value hunters", key="p3_value_hunters_enabled")
        p3_value_hunters_weight = p3_slider_with_number(
            "Value hunters",
            0.0,
            100.0,
            20.0,
            0.5,
            "p3_value_hunters_weight",
            disabled=not p3_value_hunters_enabled,
        )
        p3_first_order_pnl_enabled = st.toggle("Use infer p_f from known payoff rank", key="p3_first_order_pnl_enabled")
        p3_first_order_pnl_weight = p3_slider_with_number(
            "Infer p_f from known payoff rank",
            0.0,
            100.0,
            15.0,
            0.5,
            "p3_first_order_pnl_weight",
            "Use the known final payoff ordering and invert the payoff formula to estimate crowd densities p_f that would produce that ranking.",
            disabled=not p3_first_order_pnl_enabled,
        )
        if not any([
            p3_nash_enabled,
            p3_high_number_enabled,
            p3_avoid_famous_enabled,
            p3_random_enabled,
            p3_nice_numbers_enabled,
            p3_value_hunters_enabled,
            p3_first_order_pnl_enabled,
        ]):
            st.warning("All strategies are off, so the chart falls back to a neutral random mix for display.")
        p3_model = p3_container_modeled_frame(
            p3_nash_weight if p3_nash_enabled else 0.0,
            p3_high_number_weight if p3_high_number_enabled else 0.0,
            p3_avoid_famous_weight if p3_avoid_famous_enabled else 0.0,
            p3_random_weight if p3_random_enabled else 0.0,
            p3_nice_numbers_weight if p3_nice_numbers_enabled else 0.0,
            p3_value_hunters_weight if p3_value_hunters_enabled else 0.0,
            p3_first_order_pnl_weight if p3_first_order_pnl_enabled else 0.0,
            p3_high_number_exponent,
            tuple(int(value) for value in st.session_state["p3_known_rank_tuple"]),
        )
        strategy_weights = {
            "nash_weight": float(p3_model["nash_strategy_weight"].iloc[0]),
            "high_number_weight": float(p3_model["high_number_strategy_weight"].iloc[0]),
            "avoid_famous_weight": float(p3_model["avoid_famous_strategy_weight"].iloc[0]),
            "random_weight": float(p3_model["random_strategy_weight"].iloc[0]),
            "nice_numbers_weight": float(p3_model["nice_numbers_strategy_weight"].iloc[0]),
            "value_hunters_weight": float(p3_model["value_hunters_strategy_weight"].iloc[0]),
            "first_order_pnl_weight": float(p3_model["first_order_pnl_strategy_weight"].iloc[0]),
        }
        mae = float(p3_model["absolute_error"].mean())
        rmse = math.sqrt(float((p3_model["model_error"] ** 2).mean()))
        best_container = p3_model.sort_values("modeled_density", ascending=False).iloc[0]
        mc_card(
            "Normalized total",
            "100%",
            "Entered strategy weights are normalized automatically.",
        )
        mc_card(
            "Mean absolute error",
            f"{mae:.2f} pp",
            f"RMSE {rmse:.2f} percentage points.",
        )
        mc_card(
            "Model crowd favorite",
            f"x{int(best_container['multiplier'])}",
            f"Modeled density {float(best_container['modeled_density']):.2f}%.",
        )
        st.caption("Each slider is a strategy share. The exact boxes let you type weights directly; the model normalizes all weights to 100%.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="mc-panel"><div class="mc-section-title">Component Meaning</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="mc-note">
            <b>Nash:</b> players follow the equilibrium table.<br>
            <b>High-number exponential:</b> players increasingly favor larger multipliers; the exponent controls concentration.<br>
            <b>Famous-number avoiders:</b> players avoid known non-random/famous numbers: 10, 20, 37, 50, 90.<br>
            <b>Random:</b> uniform first-order randomness.<br>
            <b>Nice numbers:</b> 3/7/73 psychological pull.<br>
            <b>Value hunters:</b> high multiplier per fixed inhabitant.<br>
            <b>Infer p_f from known payoff rank:</b> players assume the final payoff ordering is already known. They keep only that order, fit a simple monotone payoff ladder across ranks, and invert the payoff formula to estimate the crowd densities <b>p_f</b> behind that ranking.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="mc-panel"><div class="mc-section-title">Calibrated Prior Weights</div>', unsafe_allow_html=True)
        mc_table(p3_strategy_prior_table(strategy_weights))
        st.markdown("</div>", unsafe_allow_html=True)

    with p3_charts:
        st.plotly_chart(
            p3_container_chart(p3_model),
            use_container_width=True,
            config={"displaylogo": False},
        )
        st.markdown('<div class="mc-panel"><div class="mc-section-title">Strategy Weights By Multiplier</div>', unsafe_allow_html=True)
        mc_table(p3_strategy_multiplier_table(p3_model))
        st.markdown(
            '<div class="mc-note">Each strategy column is the distribution that strategy assigns across multipliers before the global strategy weights are applied.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="mc-panel">
              <div class="mc-section-title">Payoff Function</div>
              <div class="mc-note">
                For a container with multiplier <b>M</b>, fixed inhabitants <b>I</b>, and crowd density <b>d%</b>:
                <br><b>Payoff = M × 10,000 / (d + I)</b>.
                Lower crowd density increases payoff; higher multiplier increases payoff.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="mc-panel"><div class="mc-section-title">Naive First-Order Table</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="mc-note">Assumption: the crowd share on each island is directly proportional to <b>M_f</b>, so <b>p_f = 100 × M_f / ΣM_f</b>. This gives a first-pass payoff table. From that table, keep only the payoff ordering; that ordering is treated as the known final payoff ranking for this category.</div>',
            unsafe_allow_html=True,
        )
        mc_table(p3_first_order_payoff_table(p3_model))
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="mc-panel"><div class="mc-section-title">Reverse-Engineered Crowd For This Strategy</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="mc-note">This table estimates how many people chose each multiplier once the final payoff ranking is assumed known. The estimator uses that ranking only, assigns linear payoff tiers across the ranks, and then inverts <b>PnL = M_f × 10,000 / (p_f + I_f)</b> to solve for the implied crowd shares <b>p_f</b>.</div>',
            unsafe_allow_html=True,
        )
        mc_table(p3_rank_implied_crowd_table(p3_model))
        st.markdown("</div>", unsafe_allow_html=True)
        st.plotly_chart(
            p3_payout_proxy_chart(p3_model),
            use_container_width=True,
            config={"displaylogo": False},
        )
        st.markdown('<div class="mc-panel"><div class="mc-section-title">Predicted Payoffs vs Actual Payoffs</div>', unsafe_allow_html=True)
        mc_table(p3_payoff_comparison_table(p3_model))
        st.markdown("</div>", unsafe_allow_html=True)
        p3_error_left, p3_table_right = st.columns([1.0, 1.25], gap="medium")
        with p3_error_left:
            st.plotly_chart(
                p3_container_error_chart(p3_model),
                use_container_width=True,
                config={"displaylogo": False},
            )
        with p3_table_right:
            st.markdown('<div class="mc-panel"><div class="mc-section-title">Container Table</div>', unsafe_allow_html=True)
            mc_table(p3_container_table(p3_model))
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="mc-title">Full-Width Behavior Readout <span class="mc-chip">model anatomy</span></div>', unsafe_allow_html=True)
    more_left, more_right = st.columns([1, 1], gap="medium")
    with more_left:
        st.plotly_chart(
            p3_component_contribution_chart(p3_model),
            use_container_width=True,
            config={"displaylogo": False},
        )
    with more_right:
        st.plotly_chart(
            p3_payoff_difference_chart(p3_model),
            use_container_width=True,
            config={"displaylogo": False},
        )
    st.plotly_chart(
        p3_human_vs_nash_difference_chart(p3_model),
        use_container_width=True,
        config={"displaylogo": False},
    )
    st.markdown(
        '<div class="mc-title">Correlation Lab <span class="mc-chip">behavior features</span></div>',
        unsafe_allow_html=True,
    )
    target_options = {
        "Actual human density": "actual_density",
        "Human - Nash difference": "human_minus_nash",
        "Model predicted density": "model_predicted",
        "Model - actual error": "model_minus_actual",
    }
    selected_target_label = st.selectbox(
        "Correlation target",
        list(target_options.keys()),
        index=1,
        key="p3_correlation_target",
    )
    correlation_frame = p3_correlation_table(p3_model, target_options[selected_target_label])
    st.plotly_chart(
        p3_correlation_chart(correlation_frame, selected_target_label),
        use_container_width=True,
        config={"displaylogo": False},
    )
    st.markdown('<div class="mc-panel"><div class="mc-section-title">Correlation Table</div>', unsafe_allow_html=True)
    mc_table(p3_correlation_display_table(correlation_frame))
    st.markdown(
        '<div class="mc-note">Only 10 containers exist, so treat these as directional clues, not proof. Pearson checks linear relation; Spearman checks rank relation.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="mc-panel"><div class="mc-section-title">What The Correlations Mean</div>', unsafe_allow_html=True)
    mc_table(p3_correlation_interpretation_table(correlation_frame, selected_target_label))
    st.markdown("</div>", unsafe_allow_html=True)


def p3_r4_suitcase_base_frame() -> pd.DataFrame:
    rows = [
        {"suitcase_id": "D2", "multiplier": 79, "inhabitants": 5, "nash_density": 8.954},
        {"suitcase_id": "C3", "multiplier": 73, "inhabitants": 4, "nash_density": 8.894},
        {"suitcase_id": "B4", "multiplier": 70, "inhabitants": 4, "nash_density": 8.364},
        {"suitcase_id": "A1", "multiplier": 80, "inhabitants": 6, "nash_density": 8.131},
        {"suitcase_id": "B1", "multiplier": 89, "inhabitants": 8, "nash_density": 7.720},
        {"suitcase_id": "A3", "multiplier": 83, "inhabitants": 7, "nash_density": 7.660},
        {"suitcase_id": "A5", "multiplier": 60, "inhabitants": 4, "nash_density": 6.598},
        {"suitcase_id": "B5", "multiplier": 90, "inhabitants": 10, "nash_density": 5.897},
        {"suitcase_id": "D4", "multiplier": 47, "inhabitants": 3, "nash_density": 5.302},
        {"suitcase_id": "A2", "multiplier": 50, "inhabitants": 4, "nash_density": 4.832},
        {"suitcase_id": "D1", "multiplier": 41, "inhabitants": 3, "nash_density": 4.242},
        {"suitcase_id": "C2", "multiplier": 40, "inhabitants": 3, "nash_density": 4.066},
        {"suitcase_id": "B3", "multiplier": 37, "inhabitants": 3, "nash_density": 3.536},
        {"suitcase_id": "A4", "multiplier": 31, "inhabitants": 2, "nash_density": 3.476},
        {"suitcase_id": "D5", "multiplier": 30, "inhabitants": 2, "nash_density": 3.299},
        {"suitcase_id": "C4", "multiplier": 100, "inhabitants": 15, "nash_density": 2.663},
        {"suitcase_id": "D3", "multiplier": 23, "inhabitants": 2, "nash_density": 2.063},
        {"suitcase_id": "C1", "multiplier": 17, "inhabitants": 1, "nash_density": 2.003},
        {"suitcase_id": "C5", "multiplier": 20, "inhabitants": 2, "nash_density": 1.533},
        {"suitcase_id": "B2", "multiplier": 10, "inhabitants": 1, "nash_density": 0.767},
    ]
    frame = pd.DataFrame(rows)
    frame["nash_density"] = 100.0 * frame["nash_density"] / float(frame["nash_density"].sum())
    frame["nash_payoff"] = p3_container_payoff(
        frame["multiplier"],
        frame["nash_density"],
        frame["inhabitants"],
    )
    frame["nash_rank"] = frame["nash_payoff"].rank(ascending=False, method="min").astype(int)
    return frame


def p3_r4_suitcase_components(concentrated_exponent: float = 1.6) -> pd.DataFrame:
    frame = p3_r4_suitcase_base_frame().copy()
    multipliers = pd.to_numeric(frame["multiplier"], errors="coerce").astype(float)
    nash_density = pd.to_numeric(frame["nash_density"], errors="coerce").astype(float)

    frame["nash_component"] = normalize_component(nash_density)
    frame["conc_nash_component"] = normalize_component(
        np.power(nash_density.clip(lower=0.0001), max(0.25, float(concentrated_exponent)))
    )
    frame["inverse_nash_component"] = normalize_component(1.0 / nash_density.clip(lower=0.15))
    frame["random_component"] = normalize_component(np.ones(len(frame)))
    frame["nice_component"] = normalize_component(p3_digit_bias_scores(frame["multiplier"]))
    frame["high_multiplier_component"] = normalize_component(np.power(multipliers, 1.15))
    return frame


def p3_r4_modeled_frame(
    nash_weight: float,
    conc_nash_weight: float,
    inverse_nash_weight: float,
    random_weight: float,
    nice_weight: float,
    high_multiplier_weight: float,
    concentrated_exponent: float,
) -> pd.DataFrame:
    frame = p3_r4_suitcase_components(concentrated_exponent).copy()
    raw_weights = {
        "nash": max(0.0, float(nash_weight)),
        "conc_nash": max(0.0, float(conc_nash_weight)),
        "inverse_nash": max(0.0, float(inverse_nash_weight)),
        "random": max(0.0, float(random_weight)),
        "nice": max(0.0, float(nice_weight)),
        "high_multiplier": max(0.0, float(high_multiplier_weight)),
    }
    total_weight = float(sum(raw_weights.values()))
    if total_weight <= 0.0:
        raw_weights["random"] = 1.0
        total_weight = 1.0

    modeled_density = pd.Series([0.0] * len(frame), index=frame.index, dtype=float)
    for strategy_name, component_name in [
        ("nash", "nash_component"),
        ("conc_nash", "conc_nash_component"),
        ("inverse_nash", "inverse_nash_component"),
        ("random", "random_component"),
        ("nice", "nice_component"),
        ("high_multiplier", "high_multiplier_component"),
    ]:
        normalized_weight = raw_weights[strategy_name] / total_weight
        frame[f"{strategy_name}_strategy_weight"] = normalized_weight * 100.0
        frame[f"{strategy_name}_contribution"] = normalized_weight * frame[component_name]
        modeled_density += frame[f"{strategy_name}_contribution"]

    frame["modeled_density"] = modeled_density
    frame["modeled_payoff"] = p3_container_payoff(
        frame["multiplier"],
        frame["modeled_density"],
        frame["inhabitants"],
    )
    frame["modeled_rank"] = frame["modeled_payoff"].rank(ascending=False, method="min").astype(int)
    frame["nash_vs_model_density_pp"] = frame["modeled_density"] - frame["nash_density"]
    frame["nash_vs_model_payoff"] = frame["modeled_payoff"] - frame["nash_payoff"]
    return frame.sort_values(["nash_density", "multiplier"], ascending=[False, False]).reset_index(drop=True)


def p3_r4_strategy_prior_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "Strategy": "Nash",
            "R4 weight %": frame["nash_strategy_weight"].iloc[0],
            "Logic": "Players mostly follow the equilibrium allocation.",
        },
        {
            "Strategy": "Concentrated Nash",
            "R4 weight %": frame["conc_nash_strategy_weight"].iloc[0],
            "Logic": "Some people over-index the already-crowded Nash favorites.",
        },
        {
            "Strategy": "Inverse Nash",
            "R4 weight %": frame["inverse_nash_strategy_weight"].iloc[0],
            "Logic": "Some players over-correct and hunt the Nash under-owned tail.",
        },
        {
            "Strategy": "Random",
            "R4 weight %": frame["random_strategy_weight"].iloc[0],
            "Logic": "Pure first-order randomness across the suitcase grid.",
        },
        {
            "Strategy": "Nice numbers",
            "R4 weight %": frame["nice_strategy_weight"].iloc[0],
            "Logic": "Memorable 3/7-style numbers and psychologically sticky multipliers.",
        },
        {
            "Strategy": "High multipliers",
            "R4 weight %": frame["high_multiplier_strategy_weight"].iloc[0],
            "Logic": "Greedy pull toward the largest raw multipliers.",
        },
    ]
    output = pd.DataFrame(rows)
    output["R4 weight %"] = output["R4 weight %"].map(lambda value: f"{float(value):.1f}")
    return output


def p3_r4_suitcase_table(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[
        [
            "suitcase_id",
            "multiplier",
            "inhabitants",
            "nash_density",
            "modeled_density",
            "nash_payoff",
            "modeled_payoff",
            "nash_rank",
            "modeled_rank",
        ]
    ].copy()
    output.columns = [
        "id",
        "mult",
        "inhab",
        "Nash %",
        "Pred %",
        "Nash payoff",
        "Pred payoff",
        "Nash rank",
        "Pred rank",
    ]
    for column in ["Nash %", "Pred %"]:
        output[column] = output[column].map(lambda value: f"{float(value):.2f}")
    for column in ["Nash payoff", "Pred payoff"]:
        output[column] = output[column].map(lambda value: f"{float(value):,.0f}")
    return output


def p3_r4_component_table(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[
        [
            "suitcase_id",
            "nash_component",
            "conc_nash_component",
            "inverse_nash_component",
            "random_component",
            "nice_component",
            "high_multiplier_component",
            "modeled_density",
        ]
    ].copy()
    output.columns = [
        "id",
        "Nash %",
        "Conc Nash %",
        "Inv Nash %",
        "Random %",
        "Nice %",
        "High mult %",
        "Model %",
    ]
    for column in output.columns:
        if column != "id":
            output[column] = output[column].map(lambda value: f"{float(value):.2f}")
    return output


def p3_r4_strategy_distribution_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    traces = [
        ("Nash", "nash_component", "#4e79a7"),
        ("Concentrated Nash", "conc_nash_component", "#f28e2b"),
        ("Inverse Nash", "inverse_nash_component", "#59a14f"),
        ("Random", "random_component", "#8a8f98"),
        ("Nice numbers", "nice_component", "#b07aa1"),
        ("Model predicted", "modeled_density", "#e15759"),
    ]
    for name, column, color in traces:
        fig.add_trace(
            go.Bar(
                x=frame["suitcase_id"],
                y=frame[column],
                name=name,
                marker_color=color,
                hovertemplate=f"{name}<br>%{{x}}<br>%{{y:.2f}}%<extra></extra>",
            )
        )
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Suitcase ID (sorted by Nash density)")
    fig.update_yaxes(title="Probability / density (%)")
    return apply_mc_chart_layout(fig, "Prosperity 3 Round 4 · Calibrated Priors", height=460)


def p3_r4_payoff_chart(frame: pd.DataFrame) -> go.Figure:
    ranked = frame.sort_values(["modeled_payoff", "multiplier"], ascending=[False, False]).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=ranked["suitcase_id"],
            y=ranked["nash_payoff"],
            name="Nash payoff",
            marker_color="#4e79a7",
            hovertemplate="%{x}<br>Nash payoff %{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=ranked["suitcase_id"],
            y=ranked["modeled_payoff"],
            name="Predicted payoff",
            marker_color="#e15759",
            text=[f"#{rank}" for rank in ranked["modeled_rank"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>Pred payoff %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Suitcase ID (sorted by predicted payoff)")
    fig.update_yaxes(title="Payoff")
    return apply_mc_chart_layout(fig, "Predicted Payoff By Suitcase", height=420)


def p3_r4_combo_frame(frame: pd.DataFrame, pick_count: int) -> pd.DataFrame:
    if pick_count not in {1, 2, 3}:
        raise ValueError("pick_count must be 1, 2, or 3")
    rows = []
    opening_cost = 0 if pick_count == 1 else 50_000 if pick_count == 2 else 150_000
    for combo in itertools.combinations(frame.itertuples(index=False), pick_count):
        ids = ", ".join(item.suitcase_id for item in combo)
        multipliers = ", ".join(f"x{int(item.multiplier)}" for item in combo)
        nash_gross = float(sum(float(item.nash_payoff) for item in combo))
        modeled_gross = float(sum(float(item.modeled_payoff) for item in combo))
        rows.append(
            {
                "Suitcases": ids,
                "Multipliers": multipliers,
                "Nash gross": nash_gross,
                "Pred gross": modeled_gross,
                "Open cost": float(opening_cost),
                "Nash net": nash_gross - opening_cost,
                "Pred net": modeled_gross - opening_cost,
            }
        )
    return pd.DataFrame(rows).sort_values(["Pred net", "Nash net"], ascending=[False, False]).reset_index(drop=True)


def p3_r4_combo_display_table(frame: pd.DataFrame, pick_count: int, limit: int = 12) -> pd.DataFrame:
    combos = p3_r4_combo_frame(frame, pick_count).head(limit).copy()
    for column in ["Nash gross", "Pred gross", "Open cost", "Nash net", "Pred net"]:
        combos[column] = combos[column].map(lambda value: f"{float(value):,.0f}")
    return combos


def p3_r4_notebook_model_coefficients() -> np.ndarray:
    training = np.array(
        [
            [10, 1, 0.00998],
            [80, 6, 0.18178],
            [37, 3, 0.05118],
            [17, 1, 0.07539],
            [90, 10, 0.11807],
            [31, 2, 0.06987],
            [50, 4, 0.08516],
            [20, 2, 0.01614],
            [73, 4, 0.24060],
            [89, 8, 0.15184],
            [100, 8, 0.04900],
        ],
        dtype=float,
    )
    x = np.column_stack([np.ones(len(training)), training[:, 0], training[:, 1]])
    y = training[:, 2]
    return np.linalg.lstsq(x, y, rcond=None)[0]


def p3_r4_notebook_predict_share(multiplier: pd.Series, inhabitants: pd.Series) -> pd.Series:
    beta = p3_r4_notebook_model_coefficients()
    predicted = beta[0] + beta[1] * pd.to_numeric(multiplier, errors="coerce") + beta[2] * pd.to_numeric(
        inhabitants,
        errors="coerce",
    )
    return pd.Series(predicted, dtype=float).clip(lower=0.0)


def p3_r4_notebook_results_frame() -> pd.DataFrame:
    frame = p3_r4_suitcase_base_frame().copy()
    frame["notebook_share"] = p3_r4_notebook_predict_share(frame["multiplier"], frame["inhabitants"])
    frame["notebook_density"] = 100.0 * frame["notebook_share"]
    frame["notebook_payoff"] = p3_container_payoff(
        frame["multiplier"],
        frame["notebook_density"],
        frame["inhabitants"],
    )
    frame["notebook_rank"] = frame["notebook_payoff"].rank(ascending=False, method="min").astype(int)
    frame["notebook_minus_nash_pp"] = frame["notebook_density"] - frame["nash_density"]
    return frame.sort_values(["notebook_payoff", "multiplier"], ascending=[False, False]).reset_index(drop=True)


def p3_r4_notebook_accuracy_table() -> pd.DataFrame:
    beta = p3_r4_notebook_model_coefficients()
    train = pd.DataFrame(
        [
            [10, 1, 0.00998],
            [80, 6, 0.18178],
            [37, 3, 0.05118],
            [17, 1, 0.07539],
            [90, 10, 0.11807],
            [31, 2, 0.06987],
            [50, 4, 0.08516],
            [20, 2, 0.01614],
            [73, 4, 0.24060],
            [89, 8, 0.15184],
            [100, 8, 0.04900],
        ],
        columns=["multiplier", "inhabitants", "actual_share"],
    )
    old = pd.DataFrame(
        [
            [24, 2, 0.015],
            [70, 4, 0.082],
            [41, 3, 0.019],
            [21, 2, 0.000],
            [60, 4, 0.037],
            [47, 3, 0.030],
            [82, 5, 0.062],
            [87, 5, 0.098],
            [80, 5, 0.041],
            [35, 3, 0.012],
            [73, 4, 0.113],
            [89, 5, 0.108],
            [100, 8, 0.049],
            [90, 7, 0.034],
            [17, 2, 0.006],
            [77, 5, 0.046],
            [83, 5, 0.054],
            [85, 5, 0.065],
            [79, 5, 0.054],
            [55, 4, 0.026],
            [12, 2, 0.000],
            [27, 3, 0.000],
            [52, 4, 0.019],
            [15, 2, 0.000],
            [30, 3, 0.000],
        ],
        columns=["multiplier", "inhabitants", "actual_share"],
    )

    rows = []
    for label, data in [("R2 training fit", train), ("Old suitcase validation", old)]:
        predicted = beta[0] + beta[1] * data["multiplier"] + beta[2] * data["inhabitants"]
        predicted = predicted.clip(lower=0.0)
        err = predicted - data["actual_share"]
        mae = float(err.abs().mean())
        rmse = float(np.sqrt(np.mean(np.square(err))))
        rows.append(
            {
                "Check": label,
                "MAE": f"{mae * 100.0:.2f} pp",
                "RMSE": f"{rmse * 100.0:.2f} pp",
                "Meaning": "Average miss in crowd-share percentage points.",
            }
        )
    rows.append(
        {
            "Check": "Linear prior",
            "MAE": f"share = {beta[0]:+.4f} {beta[1]:+.4f}*M {beta[2]:+.4f}*I",
            "RMSE": "",
            "Meaning": "Same model as Manual R4 notebook, recomputed without sklearn.",
        }
    )
    return pd.DataFrame(rows)


def p3_r4_notebook_suitcase_table(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[
        [
            "suitcase_id",
            "multiplier",
            "inhabitants",
            "nash_density",
            "notebook_density",
            "notebook_minus_nash_pp",
            "notebook_payoff",
            "notebook_rank",
        ]
    ].copy()
    output.columns = [
        "id",
        "mult",
        "inhab",
        "Nash %",
        "Notebook %",
        "Notebook - Nash pp",
        "Notebook payoff",
        "Notebook rank",
    ]
    for column in ["Nash %", "Notebook %", "Notebook - Nash pp"]:
        output[column] = output[column].map(lambda value: f"{float(value):+.2f}" if "minus" in column.lower() or "pp" in column else f"{float(value):.2f}")
    output["Notebook payoff"] = output["Notebook payoff"].map(lambda value: f"{float(value):,.0f}")
    return output


def p3_r4_notebook_combo_frame(frame: pd.DataFrame, pick_count: int) -> pd.DataFrame:
    rows = []
    opening_cost = 0 if pick_count == 1 else 50_000 if pick_count == 2 else 150_000
    for combo in itertools.combinations(frame.itertuples(index=False), pick_count):
        ids = ", ".join(item.suitcase_id for item in combo)
        multipliers = ", ".join(f"x{int(item.multiplier)}" for item in combo)
        gross = float(sum(float(item.notebook_payoff) for item in combo))
        rows.append(
            {
                "Suitcases": ids,
                "Multipliers": multipliers,
                "Notebook gross": gross,
                "Open cost": float(opening_cost),
                "Notebook net": gross - opening_cost,
            }
        )
    return pd.DataFrame(rows).sort_values("Notebook net", ascending=False).reset_index(drop=True)


def p3_r4_notebook_combo_display_table(frame: pd.DataFrame, pick_count: int, limit: int = 10) -> pd.DataFrame:
    output = p3_r4_notebook_combo_frame(frame, pick_count).head(limit).copy()
    for column in ["Notebook gross", "Open cost", "Notebook net"]:
        output[column] = output[column].map(lambda value: f"{float(value):,.0f}")
    return output


def p3_r4_notebook_density_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    ranked = frame.sort_values(["notebook_payoff", "multiplier"], ascending=[False, False])
    fig.add_trace(
        go.Bar(
            x=ranked["suitcase_id"],
            y=ranked["nash_density"],
            name="Nash %",
            marker_color="#4e79a7",
            hovertemplate="%{x}<br>Nash %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=ranked["suitcase_id"],
            y=ranked["notebook_density"],
            name="Notebook prior %",
            marker_color="#f2c14e",
            hovertemplate="%{x}<br>Notebook %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Suitcase ID (sorted by notebook payoff)")
    fig.update_yaxes(title="Crowd share (%)")
    return apply_mc_chart_layout(fig, "Manual R4 Notebook Prior vs Nash", height=420)


def p3_r4_notebook_payoff_chart(frame: pd.DataFrame) -> go.Figure:
    ranked = frame.sort_values(["notebook_payoff", "multiplier"], ascending=[False, False])
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=ranked["suitcase_id"],
            y=ranked["notebook_payoff"],
            marker_color="#6ccf9c",
            text=[f"#{rank}" for rank in ranked["notebook_rank"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>Notebook payoff %{y:,.0f}<extra></extra>",
            name="Notebook payoff",
        )
    )
    fig.update_xaxes(title="Suitcase ID")
    fig.update_yaxes(title="Payoff before opening cost")
    return apply_mc_chart_layout(fig, "Manual R4 Notebook Result Payoffs", height=420)


def volterra_kernel_equation(kernel: str) -> tuple[str, str]:
    if kernel == "Fractional / rough":
        return (
            r"K(u)=\frac{u^{H-\frac12}}{\Gamma(H+\frac12)}",
            "Power-law memory. Small H makes the process rough and strongly sensitive to the most recent shocks.",
        )
    if kernel == "Exponential":
        return (
            r"K(u)=e^{-\lambda u}",
            "Short-memory kernel. The influence of old shocks decays exponentially fast.",
        )
    if kernel == "Mixed fractional + exponential":
        return (
            r"K(u)=0.65\frac{u^{H-\frac12}}{\Gamma(H+\frac12)}+0.35e^{-\lambda u}",
            "Hybrid kernel: rough near zero, but with an explicit finite-memory exponential tail.",
        )
    return (
        r"K(u)=\text{user-defined expression}",
        "Custom kernel entered directly in terms of u, H, and lambda_decay.",
    )


def volterra_custom_kernel_values(expression: str, lags: np.ndarray, hurst: float, decay: float) -> np.ndarray:
    safe_globals = {"__builtins__": {}}
    gamma_fn = np.vectorize(math.gamma, otypes=[float])
    safe_locals = {
        "u": lags,
        "H": float(hurst),
        "lambda_decay": float(decay),
        "lam": float(decay),
        "np": np,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "log": np.log,
        "sin": np.sin,
        "cos": np.cos,
        "abs": np.abs,
        "minimum": np.minimum,
        "maximum": np.maximum,
        "clip": np.clip,
        "where": np.where,
        "gamma": gamma_fn,
        "pi": math.pi,
    }
    fallback_expression = "u**(H-0.5)/gamma(H+0.5)"
    fallback_values = np.power(lags, float(hurst) - 0.5) / math.gamma(float(hurst) + 0.5)

    try:
        values = eval(expression, safe_globals, safe_locals)
        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            array = np.full_like(lags, float(array), dtype=float)
        if array.shape != lags.shape:
            array = np.broadcast_to(array, lags.shape).astype(float)
        if not np.all(np.isfinite(array)):
            raise ValueError("Custom kernel produced NaN or inf values.")
        st.session_state["volterra_custom_kernel_error"] = ""
        return array
    except Exception as exc:
        st.session_state["volterra_custom_kernel_error"] = (
            f"Invalid custom kernel expression `{expression}`. "
            f"Falling back to `{fallback_expression}`. "
            f"Details: {exc}"
        )
        return fallback_values


def volterra_kernel_values(
    kernel: str,
    dt: float,
    steps: int,
    hurst: float,
    decay: float,
    custom_expression: str | None = None,
) -> np.ndarray:
    lags = np.arange(1, steps + 1, dtype=float) * float(dt)
    if kernel == "Fractional / rough":
        exponent = float(hurst) - 0.5
        return np.power(lags, exponent) / math.gamma(float(hurst) + 0.5)
    if kernel == "Exponential":
        return np.exp(-float(decay) * lags)
    if kernel == "Custom expression":
        expression = (custom_expression or "").strip()
        if not expression:
            expression = "u**(H-0.5)/gamma(H+0.5)"
        return volterra_custom_kernel_values(expression, lags, hurst, decay)
    fractional = np.power(lags, float(hurst) - 0.5) / math.gamma(float(hurst) + 0.5)
    exponential = np.exp(-float(decay) * lags)
    return 0.65 * fractional + 0.35 * exponential


def volterra_drift(x_values: np.ndarray, kappa: float, theta: float) -> np.ndarray:
    return float(kappa) * (float(theta) - x_values)


def volterra_diffusion(x_values: np.ndarray, vol: float, model: str) -> np.ndarray:
    if model == "Constant sigma":
        return np.full_like(x_values, float(vol), dtype=float)
    if model == "Square-root safe":
        return float(vol) * np.sqrt(np.maximum(x_values, 0.0) + 1e-8)
    return float(vol) * np.exp(0.5 * np.clip(x_values, -8.0, 8.0))


def simulate_volterra_paths(
    *,
    kernel: str,
    model: str,
    steps: int,
    paths: int,
    horizon: float,
    x0: float,
    kappa: float,
    theta: float,
    vol: float,
    hurst: float,
    decay: float,
    custom_kernel_expression: str | None,
    seed: int,
) -> pd.DataFrame:
    steps = int(max(2, steps))
    paths = int(max(1, paths))
    dt = float(horizon) / float(steps)
    kernel_lags = volterra_kernel_values(kernel, dt, steps, hurst, decay, custom_kernel_expression)
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, float | int]] = []
    time_grid = np.linspace(0.0, float(horizon), steps + 1)

    for path_id in range(paths):
        x = np.zeros(steps + 1, dtype=float)
        x[0] = float(x0)
        d_w = rng.normal(0.0, math.sqrt(dt), size=steps)
        for n in range(1, steps + 1):
            previous = x[:n]
            lags = kernel_lags[n - 1 :: -1]
            drift_terms = lags * volterra_drift(previous, kappa, theta) * dt
            diffusion_terms = lags * volterra_diffusion(previous, vol, model) * d_w[:n]
            x[n] = float(x0) + float(drift_terms.sum()) + float(diffusion_terms.sum())
        for time_value, x_value in zip(time_grid, x):
            rows.append({"path": path_id + 1, "time": time_value, "x": x_value})
    return pd.DataFrame(rows)


def volterra_kernel_chart(
    kernel: str,
    horizon: float,
    steps: int,
    hurst: float,
    decay: float,
    custom_kernel_expression: str | None,
) -> go.Figure:
    dt = float(horizon) / float(max(2, steps))
    values = volterra_kernel_values(kernel, dt, int(max(2, steps)), hurst, decay, custom_kernel_expression)
    times = np.arange(1, len(values) + 1, dtype=float) * dt
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=values,
            mode="lines",
            line={"color": "#f2c14e", "width": 3.0},
            name="K lag",
            hovertemplate="lag %{x:.4f}<br>K %{y:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Lag t-s")
    fig.update_yaxes(title="Kernel weight K(t-s)")
    return apply_mc_chart_layout(fig, "Volterra Memory Kernel", height=360)


def volterra_paths_chart(paths_frame: pd.DataFrame, max_paths: int = 12) -> go.Figure:
    fig = go.Figure()
    visible_paths = sorted(paths_frame["path"].unique())[: int(max_paths)]
    for path_id in visible_paths:
        path_frame = paths_frame[paths_frame["path"] == path_id]
        fig.add_trace(
            go.Scatter(
                x=path_frame["time"],
                y=path_frame["x"],
                mode="lines",
                line={"width": 1.7},
                name=f"path {int(path_id)}",
                opacity=0.78,
                hovertemplate="t %{x:.3f}<br>X %{y:.4f}<extra></extra>",
            )
        )
    mean_path = paths_frame.groupby("time", as_index=False)["x"].mean()
    fig.add_trace(
        go.Scatter(
            x=mean_path["time"],
            y=mean_path["x"],
            mode="lines",
            line={"color": "#ffffff", "width": 4.0},
            name="mean path",
            hovertemplate="t %{x:.3f}<br>mean X %{y:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="time")
    fig.update_yaxes(title="X(t)")
    return apply_mc_chart_layout(fig, "Simulated Volterra Paths", height=470)


def volterra_terminal_chart(paths_frame: pd.DataFrame) -> go.Figure:
    final_time = float(paths_frame["time"].max())
    terminal = paths_frame[np.isclose(paths_frame["time"], final_time)]["x"]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=terminal,
            nbinsx=24,
            marker_color="#6ccf9c",
            name="terminal X",
            hovertemplate="X %{x:.4f}<br>count %{y}<extra></extra>",
        )
    )
    fig.add_vline(x=float(terminal.mean()), line={"color": "#e15759", "width": 2.6, "dash": "dash"})
    fig.update_xaxes(title="X(T)")
    fig.update_yaxes(title="count")
    return apply_mc_chart_layout(fig, "Terminal Distribution", height=360)


def volterra_summary_table(paths_frame: pd.DataFrame) -> pd.DataFrame:
    final_time = float(paths_frame["time"].max())
    terminal = paths_frame[np.isclose(paths_frame["time"], final_time)]["x"]
    increments = paths_frame.sort_values(["path", "time"]).groupby("path")["x"].diff().dropna()
    rows = [
        {"Metric": "Mean X(T)", "Value": f"{float(terminal.mean()):.5f}"},
        {"Metric": "Std X(T)", "Value": f"{float(terminal.std(ddof=1)):.5f}"},
        {"Metric": "P05 X(T)", "Value": f"{float(terminal.quantile(0.05)):.5f}"},
        {"Metric": "Median X(T)", "Value": f"{float(terminal.median()):.5f}"},
        {"Metric": "P95 X(T)", "Value": f"{float(terminal.quantile(0.95)):.5f}"},
        {"Metric": "Mean absolute step move", "Value": f"{float(increments.abs().mean()):.5f}"},
    ]
    return pd.DataFrame(rows)


def render_volterra_tool_page() -> None:
    st.markdown(
        '<div class="mc-title">Volterra Process <span class="mc-chip">derivation + simulation</span></div>',
        unsafe_allow_html=True,
    )

    intro_left, intro_right = st.columns([1.1, 1.0], gap="medium")
    with intro_left:
        st.markdown(
            r"""
            <div class="mc-panel">
              <div class="mc-section-title">Continuous-Time Object</div>
              <div class="mc-note">
                A Volterra process is like an SDE with memory. Instead of only reacting to the current Brownian shock, it remembers old shocks through a kernel <b>K(t-s)</b>. Recent shocks usually matter more; old shocks fade depending on the kernel.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.latex(r"X_t = X_0 + \int_0^t K(t-s)b(X_s)\,ds + \int_0^t K(t-s)\sigma(X_s)\,dW_s")
        st.markdown(
            r"""
            <div class="mc-panel">
              <div class="mc-section-title">Left-Point Discretization</div>
              <div class="mc-note">
                Put a grid <b>t_n=n\Delta t</b>. Approximate the deterministic integral with a Riemann sum and the stochastic integral with Brownian increments
                <b>&Delta;W_j = W<sub>t_{j+1}</sub> - W<sub>t_j</sub></b>.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.latex(
            r"X_n \approx X_0 + \sum_{j=0}^{n-1}K((n-j)\Delta t)b(X_j)\Delta t"
            r" + \sum_{j=0}^{n-1}K((n-j)\Delta t)\sigma(X_j)\Delta W_j"
        )
    with intro_right:
        st.markdown(
            r"""
            <div class="mc-panel">
              <div class="mc-section-title">Simulation Recipe</div>
              <div class="mc-note">
                <b>1.</b> Choose a kernel K and grid size.<br>
                <b>2.</b> Draw Brownian increments &Delta;W<sub>j</sub> ~ N(0,&Delta;t).<br>
                <b>3.</b> For every time n, reuse all previous values X<sub>j</sub> with lag weights K((n-j)&Delta;t).<br>
                <b>4.</b> Repeat for many paths and inspect the terminal distribution.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            r"""
            <div class="mc-panel">
              <div class="mc-section-title">Interpretation</div>
              <div class="mc-note">
                The only new ingredient versus a Markov SDE is the kernel. The drift and diffusion are still local in <b>X<sub>s</sub></b>, but their effect is filtered through memory weights before it reaches time <b>t</b>.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    spec_col, control_col = st.columns([1.05, 0.95], gap="medium")
    with spec_col:
        st.markdown(
            """
            <div class="mc-panel">
              <div class="mc-section-title">Preset Kernel Library</div>
              <div class="mc-note">
                Use a preset if you want a standard Volterra memory shape, or switch to a custom expression if you want to prototype your own kernel directly.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        preset_a, preset_b, preset_c = st.columns(3, gap="small")
        with preset_a:
            st.markdown('<div class="mc-panel"><div class="mc-section-title">Fractional / rough</div></div>', unsafe_allow_html=True)
            st.latex(r"K(u)=\frac{u^{H-\frac12}}{\Gamma(H+\frac12)}")
        with preset_b:
            st.markdown('<div class="mc-panel"><div class="mc-section-title">Exponential</div></div>', unsafe_allow_html=True)
            st.latex(r"K(u)=e^{-\lambda u}")
        with preset_c:
            st.markdown('<div class="mc-panel"><div class="mc-section-title">Mixed</div></div>', unsafe_allow_html=True)
            st.latex(r"K(u)=0.65\frac{u^{H-\frac12}}{\Gamma(H+\frac12)}+0.35e^{-\lambda u}")

    chart_col = None
    control_chart = st.columns([0.9, 2.0], gap="medium")
    control_col, chart_col = control_chart
    with control_col:
        st.markdown('<div class="mc-panel"><div class="mc-section-title">Model Specification</div>', unsafe_allow_html=True)
        kernel = st.selectbox(
            "Kernel",
            ["Fractional / rough", "Exponential", "Mixed fractional + exponential", "Custom expression"],
            index=0,
            help="The kernel controls how strongly old shocks affect today's value.",
        )
        equation, interpretation = volterra_kernel_equation(kernel)
        st.latex(equation)
        st.caption(interpretation)
        custom_kernel_expression = ""
        if kernel == "Custom expression":
            custom_kernel_expression = st.text_input(
                "Custom kernel K(u)",
                value="u**(H-0.5)/gamma(H+0.5)",
                help="Allowed symbols: u, H, lambda_decay, lam, exp, sqrt, log, sin, cos, abs, minimum, maximum, clip, where, gamma, np.",
            )
            kernel_warning = st.session_state.get("volterra_custom_kernel_error", "")
            if kernel_warning:
                st.warning(kernel_warning)
        model = st.selectbox(
            "Diffusion model",
            ["Constant sigma", "Square-root safe", "Lognormal safe"],
            index=0,
            help="Choose how sigma changes with X.",
        )
        steps = int(
            p3_slider_with_number(
                "Steps",
                50.0,
                600.0,
                240.0,
                10.0,
                "volterra_steps",
                "More steps are smoother but slower because the naive Volterra scheme is O(N^2).",
            )
        )
        paths = int(
            p3_slider_with_number(
                "Paths",
                5.0,
                120.0,
                40.0,
                5.0,
                "volterra_paths",
                "Number of Monte Carlo paths.",
            )
        )
        horizon = p3_slider_with_number("Horizon T", 0.25, 5.0, 1.0, 0.25, "volterra_horizon")
        x0 = p3_slider_with_number("Initial X0", -2.0, 2.0, 0.0, 0.05, "volterra_x0")
        kappa = p3_slider_with_number("Mean reversion kappa", 0.0, 8.0, 1.2, 0.1, "volterra_kappa")
        theta = p3_slider_with_number("Long-run theta", -2.0, 2.0, 0.0, 0.05, "volterra_theta")
        vol = p3_slider_with_number("Volatility nu", 0.0, 3.0, 0.6, 0.05, "volterra_vol")
        hurst = p3_slider_with_number("Hurst H", 0.05, 0.95, 0.15, 0.01, "volterra_hurst")
        decay = p3_slider_with_number("Exponential decay lambda", 0.0, 8.0, 1.5, 0.1, "volterra_decay")
        seed = int(p3_slider_with_number("Random seed", 1.0, 9999.0, 42.0, 1.0, "volterra_seed"))
        st.markdown(
            """
            <div class="mc-note">
              Preset equations are shown above. For a custom kernel, enter an expression in <b>u</b> where <b>u=t-s</b>. Example:
              <code>exp(-lambda_decay*u) * u**(H-0.5)</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    kernel_error = None
    try:
        paths_frame = simulate_volterra_paths(
            kernel=kernel,
            model=model,
            steps=steps,
            paths=paths,
            horizon=horizon,
            x0=x0,
            kappa=kappa,
            theta=theta,
            vol=vol,
            hurst=hurst,
            decay=decay,
            custom_kernel_expression=custom_kernel_expression,
            seed=seed,
        )
    except Exception as exc:
        kernel_error = str(exc)
        paths_frame = simulate_volterra_paths(
            kernel="Fractional / rough",
            model=model,
            steps=steps,
            paths=paths,
            horizon=horizon,
            x0=x0,
            kappa=kappa,
            theta=theta,
            vol=vol,
            hurst=hurst,
            decay=decay,
            custom_kernel_expression="",
            seed=seed,
        )

    with chart_col:
        st.markdown(
            """
            <div class="mc-panel">
              <div class="mc-section-title">Simulation Output</div>
              <div class="mc-note">
                This is the explicit Volterra convolution scheme: every new point reuses the whole past path with lag-dependent kernel weights. That is why the naive simulator is quadratic in the number of time steps.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if kernel_error:
            st.warning(f"Custom kernel could not be evaluated, so I fell back to the fractional preset. Details: {kernel_error}")
        st.plotly_chart(volterra_paths_chart(paths_frame), use_container_width=True, config={"displaylogo": False})
        kernel_col, terminal_col = st.columns(2, gap="medium")
        with kernel_col:
            st.plotly_chart(
                volterra_kernel_chart(kernel, horizon, steps, hurst, decay, custom_kernel_expression),
                use_container_width=True,
                config={"displaylogo": False},
            )
        with terminal_col:
            st.plotly_chart(volterra_terminal_chart(paths_frame), use_container_width=True, config={"displaylogo": False})

    table_left, table_right = st.columns([0.9, 1.2], gap="medium")
    with table_left:
        st.markdown('<div class="mc-panel"><div class="mc-section-title">Simulation Summary</div>', unsafe_allow_html=True)
        mc_table(volterra_summary_table(paths_frame))
        st.markdown("</div>", unsafe_allow_html=True)
    with table_right:
        st.markdown(
            """
            <div class="mc-panel">
              <div class="mc-section-title">Practical Reading Guide</div>
              <div class="mc-note">
                If <b>H</b> is small, the kernel spikes near zero and the paths get rougher. If <b>&lambda;</b> is large, old information dies fast. If <b>&kappa;</b> is large, the process snaps back to <b>&theta;</b>. A large <b>&nu;</b> widens the terminal distribution. The custom kernel box is useful when you want to test memory laws that are not exactly fractional or exponential.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_p3_suitcase_page() -> None:
    st.markdown(
        '<div class="mc-title">Prosperity 3 Round 4 <span class="mc-chip">suitcase game</span></div>',
        unsafe_allow_html=True,
    )

    default_r4_params = {
        "p3_r4_nash_weight": 55.0,
        "p3_r4_conc_nash_weight": 12.5,
        "p3_r4_inverse_nash_weight": 5.0,
        "p3_r4_random_weight": 15.0,
        "p3_r4_nice_weight": 12.5,
        "p3_r4_high_multiplier_weight": 0.0,
        "p3_r4_conc_exponent": 1.6,
    }
    for key, value in default_r4_params.items():
        st.session_state.setdefault(f"{key}_slider", float(value))
        st.session_state.setdefault(f"{key}_number", float(value))

    intro_left, intro_right = st.columns([1.2, 1.0], gap="medium")
    with intro_left:
        st.markdown(
            """
            <div class="mc-panel">
              <div class="mc-section-title">The Suitcase Game</div>
              <div class="mc-note">
                Round 4 is the larger version of the Round 2 container game: there are 20 suitcases instead of 10, you may open up to 3, the first is free, the second costs <b>50,000</b>, and the third costs <b>100,000</b>. Each suitcase still pays
                <br><br><b>PnL = M<sub>f</sub> × 10,000 / (p<sub>f</sub> × 100 + I<sub>f</sub>)</b>
                <br><br>where <b>M<sub>f</sub></b> is the multiplier, <b>I<sub>f</sub></b> the fixed inhabitants, and <b>p<sub>f</sub></b> the share of teams choosing that suitcase.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="mc-panel">
              <div class="mc-section-title">Why Round 4 Is Different</div>
              <div class="mc-note">
                The payoff formula is the same as Round 2, but now we have <b>real behavioral evidence from Round 2</b>. So instead of guessing that humans are random, we can blend a few interpretable priors: mostly Nash, some crowding into already-popular picks, some over-correction into low-Nash suitcases, some randomness, and some nice-number bias.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with intro_right:
        base = p3_r4_suitcase_base_frame()
        nash_ev = float(base["nash_payoff"].mean())
        second_suitcase_net = format(nash_ev - 50_000.0, "+,.0f")
        third_suitcase_net = format(nash_ev - 100_000.0, "+,.0f")
        mc_card("Nash EV", f"{nash_ev:,.0f}", "At equilibrium, all suitcases cluster near the same payoff.")
        mc_card("Second suitcase", second_suitcase_net, "Average extra Nash EV minus the 50k opening cost.")
        mc_card("Third suitcase", third_suitcase_net, "Average extra Nash EV minus the 100k third-suitcase cost.")

    controls_col, charts_col = st.columns([1.0, 2.15], gap="medium")
    with controls_col:
        st.markdown('<div class="mc-panel"><div class="mc-section-title">Calibrated Priors</div>', unsafe_allow_html=True)
        p3_r4_nash_weight = p3_slider_with_number(
            "Nash",
            0.0,
            100.0,
            55.0,
            0.5,
            "p3_r4_nash_weight",
            "Main rational-crowd prior from the equilibrium table.",
        )
        p3_r4_conc_nash_weight = p3_slider_with_number(
            "Concentrated Nash",
            0.0,
            100.0,
            12.5,
            0.5,
            "p3_r4_conc_nash_weight",
            "Over-index on already-popular Nash favorites.",
        )
        p3_r4_inverse_nash_weight = p3_slider_with_number(
            "Inverse Nash",
            0.0,
            100.0,
            5.0,
            0.5,
            "p3_r4_inverse_nash_weight",
            "Players who over-correct into the Nash under-owned tail.",
        )
        p3_r4_random_weight = p3_slider_with_number(
            "Random",
            0.0,
            100.0,
            15.0,
            0.5,
            "p3_r4_random_weight",
            "Uniform unexplained first-order randomness.",
        )
        p3_r4_nice_weight = p3_slider_with_number(
            "Nice numbers",
            0.0,
            100.0,
            12.5,
            0.5,
            "p3_r4_nice_weight",
            "Memorable multipliers with a 3/7-style pull.",
        )
        p3_r4_high_multiplier_weight = p3_slider_with_number(
            "High multipliers",
            0.0,
            100.0,
            0.0,
            0.5,
            "p3_r4_high_multiplier_weight",
            "Optional greed tilt toward the biggest raw multipliers.",
        )
        p3_r4_conc_exponent = p3_slider_with_number(
            "Concentrated Nash exponent",
            0.5,
            4.0,
            1.6,
            0.1,
            "p3_r4_conc_exponent",
            "Higher values pile more mass into the already-crowded Nash leaders.",
        )
        p3_r4_model = p3_r4_modeled_frame(
            p3_r4_nash_weight,
            p3_r4_conc_nash_weight,
            p3_r4_inverse_nash_weight,
            p3_r4_random_weight,
            p3_r4_nice_weight,
            p3_r4_high_multiplier_weight,
            p3_r4_conc_exponent,
        )
        st.caption("The weights are normalized to 100% internally, so you can think in rough shares rather than exact totals.")
        best_single = p3_r4_combo_frame(p3_r4_model, 1).iloc[0]
        best_pair = p3_r4_combo_frame(p3_r4_model, 2).iloc[0]
        best_triple = p3_r4_combo_frame(p3_r4_model, 3).iloc[0]
        mc_card("Best single", str(best_single["Suitcases"]), f"Pred net {best_single['Pred net']:,.0f}")
        mc_card("Best pair", str(best_pair["Suitcases"]), f"Pred net {best_pair['Pred net']:,.0f}")
        mc_card("Best triple", str(best_triple["Suitcases"]), f"Pred net {best_triple['Pred net']:,.0f}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="mc-panel"><div class="mc-section-title">Strategy Logic</div>', unsafe_allow_html=True)
        mc_table(p3_r4_strategy_prior_table(p3_r4_model))
        st.markdown("</div>", unsafe_allow_html=True)

    with charts_col:
        st.plotly_chart(
            p3_r4_strategy_distribution_chart(p3_r4_model),
            use_container_width=True,
            config={"displaylogo": False},
        )
        top_left, top_right = st.columns([1.1, 1.0], gap="medium")
        with top_left:
            st.markdown('<div class="mc-panel"><div class="mc-section-title">Nash Equilibrium Table</div>', unsafe_allow_html=True)
            mc_table(p3_r4_suitcase_table(p3_r4_model))
            st.markdown("</div>", unsafe_allow_html=True)
        with top_right:
            st.markdown('<div class="mc-panel"><div class="mc-section-title">Strategy Distributions By Suitcase</div>', unsafe_allow_html=True)
            mc_table(p3_r4_component_table(p3_r4_model))
            st.markdown("</div>", unsafe_allow_html=True)

    st.plotly_chart(
        p3_r4_payoff_chart(p3_r4_model),
        use_container_width=True,
        config={"displaylogo": False},
    )

    combo_left, combo_mid, combo_right = st.columns(3, gap="medium")
    with combo_left:
        st.markdown('<div class="mc-panel"><div class="mc-section-title">Best 1-Pick Ideas</div>', unsafe_allow_html=True)
        mc_table(p3_r4_combo_display_table(p3_r4_model, 1, limit=10))
        st.markdown("</div>", unsafe_allow_html=True)
    with combo_mid:
        st.markdown('<div class="mc-panel"><div class="mc-section-title">Best 2-Pick Ideas</div>', unsafe_allow_html=True)
        mc_table(p3_r4_combo_display_table(p3_r4_model, 2, limit=12))
        st.markdown("</div>", unsafe_allow_html=True)
    with combo_right:
        st.markdown('<div class="mc-panel"><div class="mc-section-title">Best 3-Pick Ideas</div>', unsafe_allow_html=True)
        mc_table(p3_r4_combo_display_table(p3_r4_model, 3, limit=12))
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="mc-title">Manual R4 Notebook Results <span class="mc-chip">recomputed</span></div>',
        unsafe_allow_html=True,
    )
    notebook_frame = p3_r4_notebook_results_frame()
    notebook_single = p3_r4_notebook_combo_frame(notebook_frame, 1).iloc[0]
    notebook_pair = p3_r4_notebook_combo_frame(notebook_frame, 2).iloc[0]
    notebook_triple = p3_r4_notebook_combo_frame(notebook_frame, 3).iloc[0]
    result_a, result_b, result_c, result_d = st.columns(4, gap="medium")
    with result_a:
        mc_card(
            "Notebook best single",
            str(notebook_single["Suitcases"]),
            f"Net {notebook_single['Notebook net']:,.0f} XIRECs",
        )
    with result_b:
        mc_card(
            "Notebook best pair",
            str(notebook_pair["Suitcases"]),
            f"Net {notebook_pair['Notebook net']:,.0f} XIRECs",
        )
    with result_c:
        mc_card(
            "Notebook best triple",
            str(notebook_triple["Suitcases"]),
            f"Net {notebook_triple['Notebook net']:,.0f} XIRECs",
        )
    with result_d:
        recommended = notebook_pair if notebook_pair["Notebook net"] >= notebook_single["Notebook net"] else notebook_single
        mc_card(
            "Notebook recommendation",
            str(recommended["Suitcases"]),
            "Third suitcase is not worth the 100k extra fee under this prior.",
        )

    st.markdown(
        """
        <div class="mc-panel">
          <div class="mc-section-title">What This Notebook Result Means</div>
          <div class="mc-note">
            The notebook fits a simple linear crowd model from earlier behavior:
            predicted crowd share depends only on the suitcase multiplier and fixed inhabitants. I recompute its outputs here in official XIREC units, then rank every 1, 2, and 3 suitcase choice after opening costs.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    notebook_chart_left, notebook_chart_right = st.columns(2, gap="medium")
    with notebook_chart_left:
        st.plotly_chart(
            p3_r4_notebook_density_chart(notebook_frame),
            use_container_width=True,
            config={"displaylogo": False},
        )
    with notebook_chart_right:
        st.plotly_chart(
            p3_r4_notebook_payoff_chart(notebook_frame),
            use_container_width=True,
            config={"displaylogo": False},
        )

    notebook_table_left, notebook_table_right = st.columns([1.35, 1.0], gap="medium")
    with notebook_table_left:
        st.markdown('<div class="mc-panel"><div class="mc-section-title">Notebook Predicted Suitcase Results</div>', unsafe_allow_html=True)
        mc_table(p3_r4_notebook_suitcase_table(notebook_frame))
        st.markdown("</div>", unsafe_allow_html=True)
    with notebook_table_right:
        st.markdown('<div class="mc-panel"><div class="mc-section-title">Notebook Model Checks</div>', unsafe_allow_html=True)
        mc_table(p3_r4_notebook_accuracy_table())
        st.markdown("</div>", unsafe_allow_html=True)

    notebook_combo_left, notebook_combo_mid, notebook_combo_right = st.columns(3, gap="medium")
    with notebook_combo_left:
        st.markdown('<div class="mc-panel"><div class="mc-section-title">Notebook Best 1-Pick</div>', unsafe_allow_html=True)
        mc_table(p3_r4_notebook_combo_display_table(notebook_frame, 1, limit=8))
        st.markdown("</div>", unsafe_allow_html=True)
    with notebook_combo_mid:
        st.markdown('<div class="mc-panel"><div class="mc-section-title">Notebook Best 2-Pick</div>', unsafe_allow_html=True)
        mc_table(p3_r4_notebook_combo_display_table(notebook_frame, 2, limit=10))
        st.markdown("</div>", unsafe_allow_html=True)
    with notebook_combo_right:
        st.markdown('<div class="mc-panel"><div class="mc-section-title">Notebook Best 3-Pick</div>', unsafe_allow_html=True)
        mc_table(p3_r4_notebook_combo_display_table(notebook_frame, 3, limit=10))
        st.markdown("</div>", unsafe_allow_html=True)


def rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def sidebar_round_selector() -> str:
    st.session_state.setdefault("active_round_page", "p4_r3" if ROUND3_DATA_DIR.exists() else "p4_r2")
    pages = {
        "p4_r1": "Round 1",
        "p4_r2": "Round 2",
        "p4_r3": "Round 3",
        "p3_r1": "Round 1",
        "p3_r2": "Round 2",
        "p3_r3": "Round 3",
        "p3_r4": "Round 4",
    }
    if st.session_state["active_round_page"] not in pages:
        st.session_state["active_round_page"] = "p4_r3" if ROUND3_DATA_DIR.exists() else "p4_r2"

    def nav_button(page_key: str) -> None:
        active = st.session_state["active_round_page"] == page_key
        label = pages[page_key]
        if st.sidebar.button(label, key=f"nav_{page_key}", type="primary" if active else "secondary", use_container_width=True):
            st.session_state["active_round_page"] = page_key

    st.sidebar.markdown("### Prosperity 4")
    nav_button("p4_r1")
    nav_button("p4_r2")
    nav_button("p4_r3")
    st.sidebar.markdown("<hr style='margin:1rem 0;border:0;border-top:1px solid rgba(160,166,178,0.35);'>", unsafe_allow_html=True)
    st.sidebar.markdown("### Prosperity 3")
    nav_button("p3_r1")
    nav_button("p3_r2")
    nav_button("p3_r3")
    nav_button("p3_r4")
    return st.session_state["active_round_page"]


def render_empty_p3_round(round_number: int) -> None:
    st.markdown(
        f'<div class="mc-title">Prosperity 3 Round {round_number} <span class="mc-chip">empty</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="mc-panel">
          <div class="mc-section-title">Nothing here yet</div>
          <div class="mc-note">This page is reserved for previous-year analysis. Round 2 has the manual container-game model so far.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Prosperity Order Book Visualizer", layout="wide")
    install_style()

    page = sidebar_round_selector()
    if page == "p3_r2":
        render_p3_container_page()
        return
    if page == "p3_r3":
        render_p3_reserve_price_page()
        return
    if page == "p3_r4":
        render_p3_suitcase_page()
        return
    if page.startswith("p3_"):
        render_empty_p3_round(int(page[-1]))
        return

    available_datasets = [
        label
        for label, folder_paths in DATASET_DIRS.items()
        if discover_dataset_files(tuple(str(path) for path in folder_paths)).price_paths
    ]
    if not available_datasets:
        st.error("No price CSVs found in the configured Prosperity 4 round folders.")
        return

    if page == "p4_r1":
        selected_dataset_label = "Round 1"
    elif page == "p4_r2":
        selected_dataset_label = "Round 2"
    else:
        selected_dataset_label = "Round 3"
    if selected_dataset_label not in available_datasets:
        st.error(f"No CSV files found for {selected_dataset_label}.")
        return
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
                key=f"submission_backtest_upload_{selected_dataset_label}",
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
                    key=f"submission_submit_button_{selected_dataset_label}",
                ):
                    st.session_state["submitted_submission_key"] = submission_key
                    st.session_state["submitted_submission_name"] = submission_upload.name
                    st.session_state["submitted_submission_code"] = submission_code
                    st.session_state["submitted_pnl_basis_label"] = pnl_basis_label
                    st.session_state["submitted_submission_event_id"] = str(time.time_ns())
                    st.session_state["submitted_submission_source"] = "upload"
                    st.rerun()
                if st.session_state.get("submitted_submission_key") != submission_key:
                    st.caption("Click Submit / Run Backtest when you want to run this file.")
                else:
                    st.success("Submission loaded. Backtest results are shown below.")

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

        if selected_dataset_label == "Round 3":
            p4_r3_analysis = build_p4_r3_option_analysis(int(plot_day))
            p4_r3_options = p4_r3_analysis["options"]
            p4_r3_scatter = p4_r3_analysis["scatter"]
            p4_r3_coeffs = p4_r3_analysis["coeffs"]
            p4_r3_strike_table = p4_r3_analysis["strike_table"]
            if isinstance(p4_r3_options, pd.DataFrame) and not p4_r3_options.empty:
                st.markdown(
                    """
                    <div class="mc-panel">
                      <div class="mc-section-title">Round 3 Option Smile: Quadratic Fit + Detrending</div>
                      <div class="mc-note">
                        We infer market implied volatility for every <b>VEV strike</b>, fit a parabola in <b>log-moneyness</b>, and then subtract that fitted smile from observed IV. That removes the structural strike curvature and leaves the <b>relative mispricing residual</b> — the part that is actually interesting for cross-strike trading.
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                smile_left, smile_right = st.columns([1.2, 1.0], gap="medium")
                with smile_left:
                    st.plotly_chart(
                        p4_r3_smile_fit_chart(
                            p4_r3_scatter,
                            p4_r3_options,
                            np.asarray(p4_r3_coeffs, dtype=float),
                            int(focus_timestamp),
                        ),
                        use_container_width=True,
                        config={"displaylogo": False},
                    )
                with smile_right:
                    coeffs = np.asarray(p4_r3_coeffs, dtype=float)
                    mc_card(
                        "Quadratic smile fit",
                        f"{coeffs[0]:.4f} m^2 + {coeffs[1]:.4f} m + {coeffs[2]:.4f}",
                        "m = log(K / S), fit on observed voucher IVs for the selected day",
                    )
                    st.plotly_chart(
                        p4_r3_snapshot_residual_chart(p4_r3_options, int(focus_timestamp)),
                        use_container_width=True,
                        config={"displaylogo": False},
                    )

                residual_left, residual_right = st.columns([1.25, 0.95], gap="medium")
                with residual_left:
                    st.plotly_chart(
                        p4_r3_iv_residual_chart(p4_r3_options),
                        use_container_width=True,
                        config={"displaylogo": False},
                    )
                with residual_right:
                    strike_table = p4_r3_strike_table.copy()
                    if not strike_table.empty:
                        strike_table = strike_table.rename(
                            columns={
                                "product": "Voucher",
                                "strike": "Strike",
                                "mean_iv": "Mean IV",
                                "mean_smile_iv": "Smile IV",
                                "mean_residual": "Mean residual",
                                "residual_std": "Residual std",
                                "mean_price": "Mean price",
                            }
                        )
                        for column in ["Mean IV", "Smile IV", "Mean residual", "Residual std", "Mean price"]:
                            strike_table[column] = strike_table[column].map(
                                lambda value: f"{float(value):.4f}" if pd.notna(value) else ""
                            )
                    st.markdown('<div class="mc-panel"><div class="mc-section-title">Per-Strike Smile Diagnostics</div>', unsafe_allow_html=True)
                    mc_table(strike_table)
                    st.markdown("</div>", unsafe_allow_html=True)

                delta1_diag = build_p4_r3_delta1_diagnostics(int(plot_day))
                if delta1_diag:
                    st.markdown(
                        """
                        <div class="mc-panel">
                          <div class="mc-section-title">Hydrogel vs Velvetfruit Diagnostics</div>
                          <div class="mc-note">
                            These panels test whether the two delta-1 products exhibit standalone mean reversion, whether they share a stable equilibrium spread, and whether one product tends to move first while the other follows.
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    diag_cards = st.columns(3)
                    coint_info = delta1_diag["cointegration"]
                    with diag_cards[0]:
                        mc_card(
                            "Cointegration p-value",
                            f"{coint_info['p_value']:.6f}",
                            f"t-stat {coint_info['t_stat']:.3f} | 5% crit {coint_info['crit_5']:.3f}",
                        )
                    with diag_cards[1]:
                        mc_card(
                            "Static beta",
                            f"{coint_info['beta']:.4f}",
                            "Log-price hedge ratio: Velvet regressed on Hydrogel",
                        )
                    with diag_cards[2]:
                        strongest = delta1_diag["lead_lag_frame"].iloc[
                            delta1_diag["lead_lag_frame"]["corr"].abs().idxmax()
                        ]
                        mc_card(
                            "Strongest lead-lag",
                            f"{strongest['corr']:.4f}",
                            f"{strongest['pair']} at lag {int(strongest['lag'])}",
                        )

                    delta_left, delta_right = st.columns([1.25, 0.95], gap="medium")
                    with delta_left:
                        st.plotly_chart(
                            p4_r3_rolling_autocorr_chart(delta1_diag["autocorr_frame"]),
                            use_container_width=True,
                            config={"displaylogo": False},
                        )
                        st.plotly_chart(
                            p4_r3_rolling_beta_spread_chart(delta1_diag["wide"]),
                            use_container_width=True,
                            config={"displaylogo": False},
                        )
                    with delta_right:
                        st.plotly_chart(
                            p4_r3_lead_lag_heatmap(delta1_diag["lead_lag_frame"]),
                            use_container_width=True,
                            config={"displaylogo": False},
                        )
                        st.plotly_chart(
                            p4_r3_delta1_signal_chart(delta1_diag["signal_frame"]),
                            use_container_width=True,
                            config={"displaylogo": False},
                        )
                        st.plotly_chart(
                            p4_r3_depth_regime_chart(delta1_diag["regime_frame"]),
                            use_container_width=True,
                            config={"displaylogo": False},
                        )
                        st.markdown('<div class="mc-panel"><div class="mc-section-title">Diagnostic Summary</div>', unsafe_allow_html=True)
                        mc_table(delta1_diag["summary"])
                        st.markdown("</div>", unsafe_allow_html=True)

                option_micro = build_p4_r3_option_microstructure_diagnostics(int(plot_day))
                if option_micro:
                    st.markdown(
                        """
                        <div class="mc-panel">
                          <div class="mc-section-title">Voucher Relative-Value and Bot-Template Diagnostics</div>
                          <div class="mc-note">
                            Here we test whether smile-detrended option residuals actually mean-revert in price space, whether the underlying leads the vouchers, and whether the ten strikes are being quoted by what looks like one synchronized template bot rather than ten independent books.
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    micro_left, micro_right = st.columns([1.15, 1.0], gap="medium")
                    with micro_left:
                        st.plotly_chart(
                            p4_r3_option_residual_signal_chart(option_micro["residual_signal"]),
                            use_container_width=True,
                            config={"displaylogo": False},
                        )
                        st.plotly_chart(
                            p4_r3_underlying_option_lag_chart(option_micro["under_option_lag"]),
                            use_container_width=True,
                            config={"displaylogo": False},
                        )
                    with micro_right:
                        st.plotly_chart(
                            p4_r3_voucher_sync_chart(option_micro["sync_frame"]),
                            use_container_width=True,
                            config={"displaylogo": False},
                        )
                        st.plotly_chart(
                            p4_r3_voucher_corr_heatmap(option_micro["imbalance_corr"]),
                            use_container_width=True,
                            config={"displaylogo": False},
                        )
                        st.markdown('<div class="mc-panel"><div class="mc-section-title">Bot-Template Summary</div>', unsafe_allow_html=True)
                        mc_table(option_micro["bot_summary"])
                        st.markdown("</div>", unsafe_allow_html=True)
                        template_table = option_micro["template_table"].copy()
                        if not template_table.empty:
                            template_table["Share"] = template_table["Share"].map(lambda value: f"{float(value):.2%}")
                        st.markdown('<div class="mc-panel"><div class="mc-section-title">Most Common L1 Size Templates</div>', unsafe_allow_html=True)
                        mc_table(template_table)
                        st.markdown("</div>", unsafe_allow_html=True)

                render_p4_r3_manual_bid_panel()

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
        submitted_name = st.session_state.get("submitted_submission_name")
        submitted_code = st.session_state.get("submitted_submission_code")
        active_submission_ready = bool(submitted_key and submitted_name and submitted_code)
        basis_label = st.session_state.get(
            "submission_pnl_basis",
            st.session_state.get("submitted_pnl_basis_label", "100,000 ticks"),
        )
        normalization_ticks = 1_000_000 if "1,000,000" in str(basis_label) else 100_000
        history_all = submission_history_dataframe(normalization_ticks)
        history_dataset = submission_history_dataframe(normalization_ticks, selected_dataset_label)

        if not history_all.empty:
            best_overall = history_all.iloc[0]
            history_left, history_mid, history_right = st.columns([1, 1, 1], gap="small")
            with history_left:
                mc_card(
                    f"Best PnL So Far / {normalization_ticks:,} ticks",
                    fmt_money(best_overall["Total PnL"], 0),
                    f"{best_overall['File']} · {best_overall['Dataset']}",
                )
            with history_mid:
                if not history_dataset.empty:
                    best_dataset = history_dataset.iloc[0]
                    mc_card(
                        f"Best {selected_dataset_label} Run",
                        fmt_money(best_dataset["Total PnL"], 0),
                        f"{best_dataset['File']} · {best_dataset['When']}",
                    )
                else:
                    mc_card(
                        f"Best {selected_dataset_label} Run",
                        "-",
                        "No saved runs yet for this dataset/basis.",
                    )
            with history_right:
                mc_card(
                    "Saved Submissions",
                    f"{len(history_all):,}",
                    "Code snapshots are stored locally and ranked by total PnL.",
                )
            with st.expander("Submission history and saved code snapshots", expanded=not active_submission_ready):
                history_view = history_all.copy()
                history_view["Total PnL"] = history_view["Total PnL"].map(lambda value: fmt_money(value, 0))
                history_view["Mean PnL"] = history_view["Mean PnL"].map(lambda value: fmt_money(value, 0))
                history_view["1σ"] = history_view["1σ"].map(lambda value: fmt_money(value, 0))
                st.dataframe(history_view, width="stretch", hide_index=True, height=260)
                restore_options = [
                    f"{row['When']} | {row['File']} | {fmt_money(row['Total PnL'], 0)}"
                    for _, row in history_all.iterrows()
                ]
                selected_restore_label = st.selectbox(
                    "Restore a saved submission",
                    restore_options,
                    key=f"restore_submission_{selected_dataset_label}_{normalization_ticks}",
                )
                selected_restore_idx = restore_options.index(selected_restore_label)
                selected_restore_row = history_all.iloc[selected_restore_idx]
                restore_cols = st.columns([1, 1], gap="small")
                with restore_cols[0]:
                    if st.button(
                        "Load selected submission",
                        use_container_width=True,
                        key=f"restore_submission_button_{selected_dataset_label}_{normalization_ticks}",
                    ):
                        try:
                            restored_code = Path(str(selected_restore_row["Code path"])).read_text()
                        except Exception as exc:
                            st.error(f"Could not restore saved code: {exc}")
                        else:
                            st.session_state["submitted_submission_key"] = f"restored:{selected_restore_row['Hash']}"
                            st.session_state["submitted_submission_name"] = str(selected_restore_row["File"])
                            st.session_state["submitted_submission_code"] = restored_code
                            st.session_state["submitted_submission_event_id"] = str(time.time_ns())
                            st.session_state["submitted_submission_source"] = "history"
                            st.rerun()
                with restore_cols[1]:
                    code_path = str(selected_restore_row["Code path"])
                    st.caption(f"Snapshot: `{Path(code_path).name}`")
                family_board = submission_family_leaderboard(history_all)
                if not family_board.empty:
                    family_view = family_board.copy()
                    family_view["Best_PnL"] = family_view["Best_PnL"].map(lambda value: fmt_money(value, 0))
                    family_view["Mean_PnL"] = family_view["Mean_PnL"].map(lambda value: fmt_money(value, 0))
                    st.markdown("**Leaderboard by strategy family / filename prefix**")
                    st.dataframe(family_view, width="stretch", hide_index=True, height=220)

        elif not active_submission_ready:
            st.info("No active submission loaded yet. Saved PnL history will appear here once you backtest something from this dashboard.")

        if active_submission_ready:
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
                record_submission_history(
                    event_id=str(st.session_state.get("submitted_submission_event_id", "")),
                    submitted_name=submitted_name,
                    submitted_code=submitted_code,
                    dataset_label=selected_dataset_label,
                    normalization_ticks=normalization_ticks,
                    total_pnl=total_sum,
                    mean_pnl=total_mean,
                    sd_pnl=total_sd,
                    own_trades=own_trades,
                    day_count=len(daily_report),
                )
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

        st.markdown('<div class="mc-title">Actual 2026 Speed Field <span class="mc-chip">empirical crowd</span></div>', unsafe_allow_html=True)
        actual_speed_pct = st.slider(
            "Your Speed % under actual human data",
            0.0,
            100.0,
            46.0,
            step=1.0,
            key="manual_actual_speed_pct",
        )
        actual_speed = empirical_speed_stats(actual_speed_pct)
        actual_remaining = max(0.0, 100.0 - actual_speed_pct)
        actual_research, actual_scale, actual_edge = optimal_research_scale_split(actual_remaining)
        actual_gross_below = actual_edge * actual_speed["multiplier_below"]
        actual_pnl_below = actual_gross_below - investment_budget_used(
            actual_research,
            actual_scale,
            actual_speed_pct,
        )
        actual_ratio = actual_scale / actual_research if actual_research > 0 else float("inf")

        actual_full_left, actual_full_right = st.columns([0.95, 1.55], gap="medium")
        with actual_full_left:
            mc_big_result(
                "PnL from actual speed crowd",
                fmt_money(actual_pnl_below, 0),
                (
                    f"Speed {actual_speed['speed_int']:.0f}% was above "
                    f"{100 * actual_speed['pct_below']:.1f}% of real players "
                    f"in the finished Round 2 manual data."
                ),
            )
            actual_metrics_top = st.columns(2, gap="small")
            with actual_metrics_top[0]:
                mc_card(
                    "Exact crowd at this speed",
                    f"{int(actual_speed['count'])}",
                    f"{100 * actual_speed['pct_exact']:.2f}% of {int(actual_speed['field_size'])} submissions.",
                )
            with actual_metrics_top[1]:
                mc_card(
                    "Empirical percentile",
                    f"{100 * actual_speed['pct_below']:.1f}%",
                    f"At or below: {100 * actual_speed['pct_at_or_below']:.1f}%",
                )
            actual_metrics_bottom = st.columns(2, gap="small")
            with actual_metrics_bottom[0]:
                mc_card(
                    "Implied speed multiplier",
                    f"{actual_speed['multiplier_below']:.3f}",
                    "Uses the exact finished crowd, not a normal approximation.",
                )
            with actual_metrics_bottom[1]:
                mc_card(
                    "Optimal R/S and PnL",
                    f"R {actual_research:.1f}% · S {actual_scale:.1f}%",
                    f"S:R {actual_ratio:.2f}:1 · PnL {fmt_money(actual_pnl_below, 0)}",
                )
            st.caption("Percentile here means strictly below your chosen speed. So if many people tied at your speed, they are not counted as beaten.")
        with actual_full_right:
            st.plotly_chart(
                empirical_speed_percentile_chart(actual_speed_pct),
                use_container_width=True,
                config={"displaylogo": False},
            )
            st.plotly_chart(
                empirical_speed_pnl_chart(actual_speed_pct),
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
