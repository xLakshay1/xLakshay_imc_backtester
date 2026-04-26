from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import math
import sys
import types
from typing import Any

import pandas as pd


DEFAULT_POSITION_LIMIT = 80
POSITION_LIMITS = {
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
}


@dataclass
class ProductState:
    position: int = 0
    cash: float = 0.0
    fills: int = 0
    buy_qty: int = 0
    sell_qty: int = 0
    buy_value: float = 0.0
    sell_value: float = 0.0

    @property
    def gross_qty(self) -> int:
        return self.buy_qty + self.sell_qty

    @property
    def gross_value(self) -> float:
        return self.buy_value + self.sell_value


@dataclass
class BacktestResult:
    trace: pd.DataFrame
    summary: pd.DataFrame
    daily: pd.DataFrame
    product_pnl: pd.DataFrame
    stats: pd.DataFrame
    maf_bid: int | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


def day_label(day: int) -> str:
    return f"D{day}" if day < 0 else f"day-{day}"


def day_label_map_from_prices(prices: pd.DataFrame) -> dict[int, str]:
    if {"day", "session_label"}.issubset(set(prices.columns)):
        labels = (
            prices[["day", "session_label"]]
            .dropna(subset=["day"])
            .drop_duplicates("day")
            .set_index("day")["session_label"]
            .to_dict()
        )
        return {int(day): str(label) for day, label in labels.items() if str(label)}
    return {
        int(day): day_label(int(day))
        for day in pd.to_numeric(prices["day"], errors="coerce").dropna().unique()
    }


def product_limit(product: str) -> int:
    return int(POSITION_LIMITS.get(product, DEFAULT_POSITION_LIMIT))


def day_tick_span(day_prices: pd.DataFrame) -> int:
    timestamps = (
        pd.to_numeric(day_prices["timestamp"], errors="coerce")
        .dropna()
        .sort_values()
        .unique()
    )
    if len(timestamps) == 0:
        return 0
    if len(timestamps) == 1:
        return 1
    diffs = pd.Series(timestamps).diff().dropna()
    step = int(diffs[diffs > 0].min()) if not diffs.empty else 1
    return int(timestamps[-1] - timestamps[0] + step)


def visible_levels(row: pd.Series, side: str) -> list[list[int]]:
    levels: list[list[int]] = []
    for level in range(1, 4):
        price = row.get(f"{side}_price_{level}")
        volume = row.get(f"{side}_volume_{level}")
        if pd.isna(price) or pd.isna(volume):
            continue
        volume_int = int(abs(float(volume)))
        if volume_int <= 0:
            continue
        levels.append([int(round(float(price))), volume_int])
    reverse = side == "bid"
    return sorted(levels, key=lambda item: item[0], reverse=reverse)


def row_order_depth(row: pd.Series, datamodel_module: Any) -> Any:
    depth = datamodel_module.OrderDepth()
    for price, volume in visible_levels(row, "bid"):
        depth.buy_orders[int(price)] = int(volume)
    for price, volume in visible_levels(row, "ask"):
        depth.sell_orders[int(price)] = -int(volume)
    return depth


def mutable_books(stamp_frame: pd.DataFrame) -> dict[str, dict[str, list[list[int]]]]:
    books: dict[str, dict[str, list[list[int]]]] = {}
    for _, row in stamp_frame.iterrows():
        product = str(row["product"])
        books[product] = {
            "bid": visible_levels(row, "bid"),
            "ask": visible_levels(row, "ask"),
        }
    return books


def prosperity_datamodel(strategy_root: Path) -> Any:
    backtester_root = strategy_root / "backtester"
    for path in (backtester_root, strategy_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

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

    sys.modules["datamodel"] = datamodel_module
    sys.modules.setdefault("prosperity3bt.datamodel", datamodel_module)
    sys.modules.setdefault("prosperity4mcbt.datamodel", datamodel_module)
    return datamodel_module


def compile_trader(trader_code: str, strategy_root: Path) -> tuple[Any, Any]:
    datamodel_module = prosperity_datamodel(strategy_root)
    namespace: dict[str, Any] = {
        "__name__": "dashboard_uploaded_strategy",
        "math": math,
        "json": json,
    }
    exec(trader_code, namespace)
    trader_cls = namespace.get("Trader")
    if trader_cls is None:
        raise ValueError("Uploaded file must define `class Trader`.")
    return trader_cls(), datamodel_module


def extract_maf_bid(trader_code: str, strategy_root: Path) -> tuple[int | None, str | None]:
    try:
        trader, _datamodel_module = compile_trader(trader_code, strategy_root)
        bid_func = getattr(trader, "bid", None)
        if not callable(bid_func):
            return None, None
        bid_value = bid_func()
        if bid_value is None:
            return None, None
        return int(bid_value), None
    except Exception as exc:
        return None, f"Could not evaluate Trader.bid(): {exc}"


def build_market_trade_map(
    trades: pd.DataFrame,
    products: tuple[str, ...],
    days: tuple[int, ...],
    datamodel_module: Any,
) -> dict[tuple[int, int], dict[str, list[Any]]]:
    empty = {product: [] for product in products}
    if trades.empty:
        return {}

    required = {"day", "timestamp", "symbol", "price", "quantity"}
    if not required.issubset(set(trades.columns)):
        return {}

    filtered = trades[
        trades["symbol"].isin(products) & trades["day"].isin(days)
    ].copy()
    if filtered.empty:
        return {}

    for column in ("day", "timestamp", "price", "quantity"):
        filtered[column] = pd.to_numeric(filtered[column], errors="coerce")
    filtered = filtered.dropna(subset=["day", "timestamp", "price", "quantity"])
    for column in ("buyer", "seller"):
        if column not in filtered:
            filtered[column] = ""
        filtered[column] = filtered[column].fillna("").astype(str)

    trade_map: dict[tuple[int, int], dict[str, list[Any]]] = {}
    for row in filtered.itertuples(index=False):
        day = int(getattr(row, "day"))
        timestamp = int(getattr(row, "timestamp"))
        symbol = str(getattr(row, "symbol"))
        key = (day, timestamp)
        bucket = trade_map.setdefault(key, {product: [] for product in products})
        bucket.setdefault(symbol, []).append(
            datamodel_module.Trade(
                symbol,
                int(round(float(getattr(row, "price")))),
                int(abs(float(getattr(row, "quantity")))),
                str(getattr(row, "buyer", "")),
                str(getattr(row, "seller", "")),
                timestamp,
            )
        )
    for bucket in trade_map.values():
        for product in products:
            bucket.setdefault(product, [])
    return trade_map


def build_trading_state(
    timestamp: int,
    stamp_frame: pd.DataFrame,
    products: tuple[str, ...],
    states: dict[str, ProductState],
    trader_data: str,
    previous_own_trades: dict[str, list[Any]],
    market_trades: dict[str, list[Any]],
    datamodel_module: Any,
) -> Any:
    listings = {
        product: datamodel_module.Listing(product, product, "XIRECS")
        for product in products
    }
    order_depths: dict[str, Any] = {}
    for _, row in stamp_frame.iterrows():
        product = str(row["product"])
        order_depths[product] = row_order_depth(row, datamodel_module)

    position = {product: int(states[product].position) for product in products}
    observations = datamodel_module.Observation({}, {})
    return datamodel_module.TradingState(
        traderData=trader_data,
        timestamp=int(timestamp),
        listings=listings,
        order_depths=order_depths,
        own_trades={product: list(previous_own_trades.get(product, [])) for product in products},
        market_trades={product: list(market_trades.get(product, [])) for product in products},
        position=position,
        observations=observations,
    )


def fill_buy_order(
    product: str,
    limit_price: int,
    desired_qty: int,
    book: dict[str, list[list[int]]],
    state: ProductState,
    timestamp: int,
    datamodel_module: Any,
) -> list[Any]:
    own_trades: list[Any] = []
    remaining_room = product_limit(product) - state.position
    remaining_qty = min(max(0, int(desired_qty)), max(0, remaining_room))
    if remaining_qty <= 0:
        return own_trades

    for level in book.get("ask", []):
        ask_price, available = int(level[0]), int(level[1])
        if ask_price > limit_price or remaining_qty <= 0:
            break
        fill_qty = min(remaining_qty, available)
        if fill_qty <= 0:
            continue
        level[1] -= fill_qty
        remaining_qty -= fill_qty
        value = fill_qty * ask_price
        state.position += fill_qty
        state.cash -= value
        state.fills += 1
        state.buy_qty += fill_qty
        state.buy_value += value
        own_trades.append(
            datamodel_module.Trade(
                product,
                ask_price,
                fill_qty,
                "SUBMISSION",
                "",
                timestamp,
            )
        )
    book["ask"] = [level for level in book.get("ask", []) if level[1] > 0]
    return own_trades


def fill_sell_order(
    product: str,
    limit_price: int,
    desired_qty: int,
    book: dict[str, list[list[int]]],
    state: ProductState,
    timestamp: int,
    datamodel_module: Any,
) -> list[Any]:
    own_trades: list[Any] = []
    remaining_room = product_limit(product) + state.position
    remaining_qty = min(max(0, int(desired_qty)), max(0, remaining_room))
    if remaining_qty <= 0:
        return own_trades

    for level in book.get("bid", []):
        bid_price, available = int(level[0]), int(level[1])
        if bid_price < limit_price or remaining_qty <= 0:
            break
        fill_qty = min(remaining_qty, available)
        if fill_qty <= 0:
            continue
        level[1] -= fill_qty
        remaining_qty -= fill_qty
        value = fill_qty * bid_price
        state.position -= fill_qty
        state.cash += value
        state.fills += 1
        state.sell_qty += fill_qty
        state.sell_value += value
        own_trades.append(
            datamodel_module.Trade(
                product,
                bid_price,
                fill_qty,
                "",
                "SUBMISSION",
                timestamp,
            )
        )
    book["bid"] = [level for level in book.get("bid", []) if level[1] > 0]
    return own_trades


def apply_orders(
    orders_by_product: Any,
    books: dict[str, dict[str, list[list[int]]]],
    states: dict[str, ProductState],
    products: tuple[str, ...],
    timestamp: int,
    datamodel_module: Any,
) -> dict[str, list[Any]]:
    own_trades = {product: [] for product in products}
    if not isinstance(orders_by_product, dict):
        return own_trades

    for raw_product, raw_orders in orders_by_product.items():
        product = str(raw_product)
        if product not in states or product not in books or raw_orders is None:
            continue
        if not isinstance(raw_orders, list):
            raw_orders = [raw_orders]
        for order in raw_orders:
            quantity = int(getattr(order, "quantity", 0) or 0)
            if quantity == 0:
                continue
            try:
                price = int(round(float(getattr(order, "price"))))
            except (TypeError, ValueError, OverflowError):
                continue
            if quantity > 0:
                fills = fill_buy_order(
                    product,
                    price,
                    quantity,
                    books[product],
                    states[product],
                    timestamp,
                    datamodel_module,
                )
            else:
                fills = fill_sell_order(
                    product,
                    price,
                    abs(quantity),
                    books[product],
                    states[product],
                    timestamp,
                    datamodel_module,
                )
            own_trades[product].extend(fills)
    return own_trades


def marked_pnl(state: ProductState, mid_price: float | None) -> float:
    pnl = float(state.cash)
    if mid_price is not None and not pd.isna(mid_price):
        pnl += float(state.position) * float(mid_price)
    return pnl


def trace_rows(
    states: dict[str, ProductState],
    last_mid: dict[str, float],
    day: int,
    timestamp: int,
    global_timestamp: int,
    completed_pnl_by_product: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = 0.0
    for product, state in states.items():
        mid = last_mid.get(product)
        pnl = marked_pnl(state, mid) + float(completed_pnl_by_product.get(product, 0.0))
        total += pnl
        rows.append(
            {
                "day": int(day),
                "timestamp": int(timestamp),
                "global_timestamp": int(global_timestamp),
                "product": product,
                "position": int(state.position),
                "cash": round(float(state.cash), 2),
                "mid_price": float(mid) if mid is not None and not pd.isna(mid) else float("nan"),
                "pnl": round(float(pnl), 2),
            }
        )
    rows.append(
        {
            "day": int(day),
            "timestamp": int(timestamp),
            "global_timestamp": int(global_timestamp),
            "product": "TOTAL",
            "position": 0,
            "cash": 0.0,
            "mid_price": float("nan"),
            "pnl": round(float(total), 2),
        }
    )
    return rows


def summary_rows_for_day(
    states: dict[str, ProductState],
    last_mid: dict[str, float],
    day: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for product, state in states.items():
        mid = last_mid.get(product, float("nan"))
        pnl = marked_pnl(state, mid)
        rows.append(
            {
                "day": int(day),
                "product": product,
                "final_position": int(state.position),
                "cash": round(float(state.cash), 2),
                "mark_mid": round(float(mid), 2) if not pd.isna(mid) else float("nan"),
                "pnl": round(float(pnl), 2),
                "fills": int(state.fills),
                "buy_qty": int(state.buy_qty),
                "sell_qty": int(state.sell_qty),
                "gross_qty": int(state.gross_qty),
                "buy_value": round(float(state.buy_value), 2),
                "sell_value": round(float(state.sell_value), 2),
                "gross_value": round(float(state.gross_value), 2),
                "avg_buy": round(float(state.buy_value / state.buy_qty), 2)
                if state.buy_qty
                else float("nan"),
                "avg_sell": round(float(state.sell_value / state.sell_qty), 2)
                if state.sell_qty
                else float("nan"),
            }
        )
    return rows


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    totals = (
        summary.groupby("product", as_index=False)
        .agg(
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
    )
    totals["day"] = "ALL"
    totals["final_position"] = float("nan")
    totals["cash"] = float("nan")
    totals["mark_mid"] = float("nan")
    totals["avg_buy"] = totals["buy_value"] / totals["buy_qty"].replace(0, pd.NA)
    totals["avg_sell"] = totals["sell_value"] / totals["sell_qty"].replace(0, pd.NA)
    for column in ("pnl", "buy_value", "sell_value", "gross_value", "avg_buy", "avg_sell"):
        totals[column] = pd.to_numeric(totals[column], errors="coerce").round(2)
    ordered = list(summary.columns)
    output = pd.concat([summary, totals[ordered]], ignore_index=True)
    output["day"] = output["day"].astype(str)
    return output


def daily_report_from_summary(prices: pd.DataFrame, summary: pd.DataFrame, days: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    day_summary = summary[summary["day"] != "ALL"].copy()
    label_map = day_label_map_from_prices(prices)
    for day in days:
        product_rows = day_summary[day_summary["day"].astype(int) == int(day)]
        day_prices = prices[prices["day"] == day]
        rows.append(
            {
                "SET": label_map.get(int(day), day_label(int(day))),
                "DAY": int(day),
                "TICKS": int(day_tick_span(day_prices)),
                "SNAPSHOTS": int(day_prices["timestamp"].nunique()),
                "OWN_TRADES": int(pd.to_numeric(product_rows["fills"], errors="coerce").sum()),
                "FINAL_PNL": round(float(pd.to_numeric(product_rows["pnl"], errors="coerce").sum()), 2),
            }
        )
    return pd.DataFrame(rows)


def product_pnl_report(summary: pd.DataFrame, days: tuple[int, ...], prices: pd.DataFrame) -> pd.DataFrame:
    day_summary = summary[summary["day"] != "ALL"].copy()
    if day_summary.empty:
        return pd.DataFrame()
    label_map = day_label_map_from_prices(prices)
    day_summary["SET"] = day_summary["day"].astype(int).map(
        lambda day: label_map.get(int(day), day_label(int(day)))
    )
    product_pnl = (
        day_summary.pivot_table(index="product", columns="SET", values="pnl", aggfunc="sum")
        .reset_index()
        .rename(columns={"product": "PRODUCT"})
    )
    ordered_columns = ["PRODUCT"] + [
        label_map.get(int(day), day_label(int(day)))
        for day in days
        if label_map.get(int(day), day_label(int(day))) in set(product_pnl.columns)
    ]
    return product_pnl[ordered_columns]


def stats_report(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    pnl = pd.to_numeric(daily["FINAL_PNL"], errors="coerce")
    return pd.DataFrame(
        [
            {"METRIC": "daily_pnl_mean", "VALUE": round(float(pnl.mean()), 2)},
            {"METRIC": "daily_pnl_sd", "VALUE": round(float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0, 2)},
            {"METRIC": "daily_pnl_min", "VALUE": round(float(pnl.min()), 2)},
            {"METRIC": "daily_pnl_max", "VALUE": round(float(pnl.max()), 2)},
            {"METRIC": "sum_final_pnl", "VALUE": round(float(pnl.sum()), 2)},
            {"METRIC": "own_trades_sum", "VALUE": int(daily["OWN_TRADES"].sum())},
        ]
    )


def run_submission_backtest(
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    trader_code: str,
    strategy_root: Path,
) -> BacktestResult:
    products = tuple(sorted(str(product) for product in prices["product"].dropna().unique()))
    days = tuple(sorted(int(day) for day in prices["day"].dropna().unique()))
    if not products or not days:
        empty = pd.DataFrame()
        return BacktestResult(empty, empty, empty, empty, empty)

    maf_bid, bid_warning = extract_maf_bid(trader_code, strategy_root)
    warnings: list[str] = []
    if bid_warning:
        warnings.append(bid_warning)

    trace: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    global_offset = 0
    completed_pnl_by_product = {product: 0.0 for product in products}

    for day in days:
        day_frame = (
            prices[prices["day"] == day]
            .sort_values(["timestamp", "product"])
            .copy()
        )
        if day_frame.empty:
            continue

        trader, datamodel_module = compile_trader(trader_code, strategy_root)
        market_trade_map = build_market_trade_map(trades, products, (day,), datamodel_module)
        trader_data = ""
        states = {product: ProductState() for product in products}
        last_mid: dict[str, float] = {}
        previous_own_trades = {product: [] for product in products}

        for timestamp, stamp_frame in day_frame.groupby("timestamp", sort=True):
            timestamp_int = int(timestamp)
            for _, row in stamp_frame.iterrows():
                product = str(row["product"])
                mid = row.get("mid_price")
                if not pd.isna(mid) and float(mid) > 0:
                    last_mid[product] = float(mid)

            state = build_trading_state(
                timestamp_int,
                stamp_frame,
                products,
                states,
                trader_data,
                previous_own_trades,
                market_trade_map.get((int(day), timestamp_int), {}),
                datamodel_module,
            )
            with redirect_stdout(io.StringIO()):
                result = trader.run(state)

            if isinstance(result, tuple):
                orders_by_product = result[0] if len(result) >= 1 else {}
                if len(result) >= 3:
                    trader_data = "" if result[2] is None else str(result[2])
            else:
                orders_by_product = result

            books = mutable_books(stamp_frame)
            previous_own_trades = apply_orders(
                orders_by_product,
                books,
                states,
                products,
                timestamp_int,
                datamodel_module,
            )
            trace.extend(
                trace_rows(
                    states,
                    last_mid,
                    int(day),
                    timestamp_int,
                    int(global_offset + timestamp_int - int(day_frame["timestamp"].min())),
                    completed_pnl_by_product,
                )
            )

        day_summary_rows = summary_rows_for_day(states, last_mid, int(day))
        summary_rows.extend(day_summary_rows)
        for row in day_summary_rows:
            completed_pnl_by_product[str(row["product"])] += float(row["pnl"])
        global_offset += day_tick_span(day_frame)

    trace_frame = pd.DataFrame(trace)
    summary_frame = aggregate_summary(pd.DataFrame(summary_rows))
    daily = daily_report_from_summary(prices, summary_frame, days)
    product_pnl = product_pnl_report(summary_frame, days, prices)
    stats = stats_report(daily)

    for frame in (trace_frame, summary_frame, daily, product_pnl, stats):
        frame.attrs["maf_bid"] = maf_bid
        frame.attrs["warnings"] = tuple(warnings)
        frame.attrs["backtester"] = "deterministic_visible_book_crossing_v2"

    return BacktestResult(
        trace=trace_frame,
        summary=summary_frame,
        daily=daily,
        product_pnl=product_pnl,
        stats=stats,
        maf_bid=maf_bid,
        warnings=tuple(warnings),
    )
