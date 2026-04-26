#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd


PRODUCTS = ("ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT")
TIMESTAMP_STEP = 100
DEFAULT_DATA_DIR = Path("/Users/lakshaykumar/Downloads/ROUND1")


@dataclass
class Level:
    price: int
    quantity: int
    owner: str = "BOT"


@dataclass
class Book:
    bids: list[Level]
    asks: list[Level]


@dataclass
class Fill:
    timestamp: int
    symbol: str
    price: int
    quantity: int
    buyer: str | None
    seller: str | None


@dataclass
class Ledger:
    position: int = 0
    cash: float = 0.0


@dataclass
class ProductTrace:
    fair: list[float] = field(default_factory=list)
    position: list[int] = field(default_factory=list)
    cash: list[float] = field(default_factory=list)
    mtm_pnl: list[float] = field(default_factory=list)


class StrategyWorker:
    def __init__(self, root: Path, strategy: Path, python_bin: str, timeout_ms: int) -> None:
        self.timeout_ms = timeout_ms
        worker = root / "scripts" / "python_strategy_worker.py"
        self.process = subprocess.Popen(
            [python_bin, str(worker), str(strategy)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("failed to open strategy worker pipes")
        self.stdin = self.process.stdin
        self.stdout = self.process.stdout

    def close(self) -> None:
        self.process.kill()
        self.process.wait()

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.stdin.flush()
        line = self.stdout.readline()
        if not line:
            raise RuntimeError("strategy worker exited unexpectedly")
        response = json.loads(line)
        if response.get("error"):
            raise RuntimeError(response["error"])
        return response

    def reset(self) -> None:
        self.request({"type": "reset"})

    def run(
        self,
        timestamp: int,
        trader_data: str,
        books: dict[str, Book],
        own_trades: dict[str, list[Fill]],
        market_trades: dict[str, list[Fill]],
        ledgers: dict[str, Ledger],
    ) -> dict[str, Any]:
        return self.request(
            {
                "type": "run",
                "timestamp": timestamp,
                "timeout_ms": self.timeout_ms,
                "trader_data": trader_data,
                "order_depths": {
                    product: {
                        "buy_orders": {str(level.price): level.quantity for level in book.bids},
                        "sell_orders": {str(level.price): -level.quantity for level in book.asks},
                    }
                    for product, book in books.items()
                },
                "own_trades": {
                    product: [fill_to_payload(fill) for fill in fills]
                    for product, fills in own_trades.items()
                },
                "market_trades": {
                    product: [fill_to_payload(fill) for fill in fills]
                    for product, fills in market_trades.items()
                },
                "position": {product: ledger.position for product, ledger in ledgers.items()},
            }
        )


def fill_to_payload(fill: Fill) -> dict[str, Any]:
    return {
        "symbol": fill.symbol,
        "price": fill.price,
        "quantity": fill.quantity,
        "buyer": fill.buyer,
        "seller": fill.seller,
        "timestamp": fill.timestamp,
    }


def fill_to_trade_row(fill: Fill) -> dict[str, Any]:
    return {
        "timestamp": fill.timestamp,
        "buyer": fill.buyer or "",
        "seller": fill.seller or "",
        "symbol": fill.symbol,
        "currency": "XIRECS",
        "price": float(fill.price),
        "quantity": fill.quantity,
    }


def read_round1(data_dir: Path) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    prices_by_day: dict[int, pd.DataFrame] = {}
    trades_by_day: dict[int, pd.DataFrame] = {}
    for path in sorted(data_dir.glob("prices_round_1_day_*.csv")):
        day = int(path.stem.split("_")[-1])
        frame = pd.read_csv(path, sep=";")
        frame = frame[frame["product"].isin(PRODUCTS)].copy()
        prices_by_day[day] = frame.sort_values(["timestamp", "product"])
    for path in sorted(data_dir.glob("trades_round_1_day_*.csv")):
        day = int(path.stem.split("_")[-1])
        frame = pd.read_csv(path, sep=";")
        frame = frame[frame["symbol"].isin(PRODUCTS)].copy()
        trades_by_day[day] = frame.sort_values(["timestamp", "symbol"])
    if not prices_by_day:
        raise FileNotFoundError(f"no round 1 price files found in {data_dir}")
    return prices_by_day, trades_by_day


def row_to_book(row: pd.Series) -> Book:
    bids: list[Level] = []
    asks: list[Level] = []
    for level in (1, 2, 3):
        bid_price = row.get(f"bid_price_{level}")
        bid_volume = row.get(f"bid_volume_{level}")
        ask_price = row.get(f"ask_price_{level}")
        ask_volume = row.get(f"ask_volume_{level}")
        if pd.notna(bid_price) and pd.notna(bid_volume):
            bids.append(Level(int(bid_price), abs(int(bid_volume))))
        if pd.notna(ask_price) and pd.notna(ask_volume):
            asks.append(Level(int(ask_price), abs(int(ask_volume))))
    bids.sort(key=lambda level: level.price, reverse=True)
    asks.sort(key=lambda level: level.price)
    return Book(bids=bids, asks=asks)


def insert_level(levels: list[Level], level: Level, descending: bool) -> None:
    for existing in levels:
        if existing.price == level.price and existing.owner == level.owner:
            existing.quantity += level.quantity
            break
    else:
        levels.append(level)
    levels.sort(key=lambda item: (item.price, item.owner != "BOT"), reverse=descending)
    if not descending:
        levels.sort(key=lambda item: (item.price, item.owner != "BOT"))


def execute_strategy_orders(
    product: str,
    timestamp: int,
    book: Book,
    ledger: Ledger,
    orders: list[dict[str, Any]],
) -> list[Fill]:
    fills: list[Fill] = []
    for order in orders:
        price = int(order["price"])
        quantity = int(order["quantity"])
        if quantity > 0:
            remaining = quantity
            while remaining > 0 and book.asks and book.asks[0].price <= price:
                level = book.asks[0]
                fill_qty = min(remaining, level.quantity)
                fills.append(Fill(timestamp, product, level.price, fill_qty, "SUBMISSION", level.owner))
                ledger.position += fill_qty
                ledger.cash -= level.price * fill_qty
                remaining -= fill_qty
                level.quantity -= fill_qty
                if level.quantity == 0:
                    book.asks.pop(0)
            if remaining > 0:
                insert_level(book.bids, Level(price, remaining, "SUBMISSION"), descending=True)
        elif quantity < 0:
            remaining = -quantity
            while remaining > 0 and book.bids and book.bids[0].price >= price:
                level = book.bids[0]
                fill_qty = min(remaining, level.quantity)
                fills.append(Fill(timestamp, product, level.price, fill_qty, level.owner, "SUBMISSION"))
                ledger.position -= fill_qty
                ledger.cash += level.price * fill_qty
                remaining -= fill_qty
                level.quantity -= fill_qty
                if level.quantity == 0:
                    book.bids.pop(0)
            if remaining > 0:
                insert_level(book.asks, Level(price, remaining, "SUBMISSION"), descending=False)
    return fills


def execute_public_trade(
    product: str,
    timestamp: int,
    price: int,
    quantity: int,
    market_buy: bool,
    book: Book,
    ledger: Ledger,
) -> tuple[list[Fill], list[Fill]]:
    own: list[Fill] = []
    market: list[Fill] = []
    side = book.asks if market_buy else book.bids
    remaining = quantity
    while remaining > 0 and side:
        level = side[0]
        fill_qty = min(remaining, level.quantity)
        if market_buy:
            fill = Fill(timestamp, product, level.price, fill_qty, "PUBLIC_TAKER", level.owner)
            if level.owner == "SUBMISSION":
                ledger.position -= fill_qty
                ledger.cash += level.price * fill_qty
                own.append(fill)
            else:
                market.append(fill)
        else:
            fill = Fill(timestamp, product, level.price, fill_qty, level.owner, "PUBLIC_TAKER")
            if level.owner == "SUBMISSION":
                ledger.position += fill_qty
                ledger.cash -= level.price * fill_qty
                own.append(fill)
            else:
                market.append(fill)
        remaining -= fill_qty
        level.quantity -= fill_qty
        if level.quantity == 0:
            side.pop(0)
    return own, market


def infer_public_trade_side(trade: pd.Series, book: Book, mid_price: float) -> bool:
    price = float(trade["price"])
    if book.asks and price >= book.asks[0].price:
        return True
    if book.bids and price <= book.bids[0].price:
        return False
    return price >= mid_price


def enforce_limits(
    orders: dict[str, list[dict[str, Any]]],
    ledgers: dict[str, Ledger],
    position_limit: int,
) -> dict[str, list[dict[str, Any]]]:
    accepted: dict[str, list[dict[str, Any]]] = {}
    for product in PRODUCTS:
        product_orders = orders.get(product, [])
        buys = sum(int(order["quantity"]) for order in product_orders if int(order["quantity"]) > 0)
        sells = sum(-int(order["quantity"]) for order in product_orders if int(order["quantity"]) < 0)
        position = ledgers[product].position
        if position + buys > position_limit or position - sells < -position_limit:
            accepted[product] = []
        else:
            accepted[product] = product_orders
    return accepted


def run_session(
    session_id: int,
    day: int,
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    root: Path,
    strategy: Path,
    args: argparse.Namespace,
    capture: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    worker = StrategyWorker(root, strategy, args.python_bin, args.timeout_ms)
    worker.reset()
    try:
        ledgers = {product: Ledger() for product in PRODUCTS}
        trader_data = ""
        prev_own_trades = {product: [] for product in PRODUCTS}
        prev_market_trades = {product: [] for product in PRODUCTS}
        trace_rows: list[dict[str, Any]] = []
        trade_rows: list[dict[str, Any]] = []
        price_rows: list[dict[str, Any]] = []
        timestamps = sorted(prices["timestamp"].unique())

        for timestamp in timestamps:
            snapshot = prices[prices["timestamp"] == timestamp]
            books = {
                product: row_to_book(snapshot[snapshot["product"] == product].iloc[0])
                for product in PRODUCTS
                if not snapshot[snapshot["product"] == product].empty
            }
            if set(books) != set(PRODUCTS):
                continue

            response = worker.run(
                int(timestamp),
                trader_data,
                books,
                prev_own_trades,
                prev_market_trades,
                ledgers,
            )
            trader_data = response.get("trader_data") or ""
            strategy_orders = enforce_limits(response.get("orders") or {}, ledgers, args.position_limit)

            own_this_tick = {product: [] for product in PRODUCTS}
            market_this_tick = {product: [] for product in PRODUCTS}

            for product in PRODUCTS:
                fills = execute_strategy_orders(
                    product,
                    int(timestamp),
                    books[product],
                    ledgers[product],
                    strategy_orders.get(product, []),
                )
                own_this_tick[product].extend(fills)
                if capture:
                    trade_rows.extend(fill_to_trade_row(fill) for fill in fills)

            tick_trades = trades[trades["timestamp"] == timestamp]
            for _, trade in tick_trades.iterrows():
                product = str(trade["symbol"])
                if product not in books:
                    continue
                price_row = snapshot[snapshot["product"] == product].iloc[0]
                market_buy = infer_public_trade_side(trade, books[product], float(price_row["mid_price"]))
                own, market = execute_public_trade(
                    product,
                    int(timestamp),
                    int(round(float(trade["price"]))),
                    int(trade["quantity"]),
                    market_buy,
                    books[product],
                    ledgers[product],
                )
                own_this_tick[product].extend(own)
                market_this_tick[product].extend(market)
                if capture:
                    trade_rows.extend(fill_to_trade_row(fill) for fill in own + market)

            if capture:
                for _, row in snapshot.iterrows():
                    price_rows.append(row.to_dict())
                for product in PRODUCTS:
                    row = snapshot[snapshot["product"] == product].iloc[0]
                    fair = float(row["mid_price"])
                    ledger = ledgers[product]
                    trace_rows.append(
                        {
                            "day": day,
                            "timestamp": int(timestamp),
                            "product": product,
                            "fair_value": fair,
                            "position": ledger.position,
                            "cash": ledger.cash,
                            "mtm_pnl": ledger.cash + ledger.position * fair,
                        }
                    )

            prev_own_trades = own_this_tick
            prev_market_trades = market_this_tick

        final_rows = prices.groupby("product").tail(1).set_index("product")
        product_pnls = {
            product: ledgers[product].cash + ledgers[product].position * float(final_rows.loc[product, "mid_price"])
            for product in PRODUCTS
        }
        summary = {
            "session_id": session_id,
            "day": day,
            "total_pnl": sum(product_pnls.values()),
            "osmium_pnl": product_pnls["ASH_COATED_OSMIUM"],
            "pepper_root_pnl": product_pnls["INTARIAN_PEPPER_ROOT"],
            "osmium_position": ledgers["ASH_COATED_OSMIUM"].position,
            "pepper_root_position": ledgers["INTARIAN_PEPPER_ROOT"].position,
            "osmium_cash": ledgers["ASH_COATED_OSMIUM"].cash,
            "pepper_root_cash": ledgers["INTARIAN_PEPPER_ROOT"].cash,
        }
        captured = {
            "prices": price_rows,
            "trades": trade_rows,
            "trace": trace_rows,
        }
        return summary, dict(summary), captured
    finally:
        worker.close()


def write_semicolon_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo replay backtester for Round 1 osmium/pepper roots.")
    parser.add_argument("strategy", help="Path to strategy Python file defining Trader")
    parser.add_argument("--sessions", type=int, default=100)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=Path("tmp/round1_mc_backtests"))
    parser.add_argument("--seed", type=int, default=20260414)
    parser.add_argument("--position-limit", type=int, default=50)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--timeout-ms", type=int, default=900)
    parser.add_argument("--write-session-limit", type=int, default=3)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    strategy = Path(args.strategy).expanduser().resolve()
    output_dir = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    prices_by_day, trades_by_day = read_round1(args.data_dir)
    days = sorted(prices_by_day)
    rng = random.Random(args.seed)
    summaries: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []

    for session_id in range(args.sessions):
        day = rng.choice(days)
        capture = session_id < args.write_session_limit
        summary, run_summary, captured = run_session(
            session_id,
            day,
            prices_by_day[day],
            trades_by_day.get(day, pd.DataFrame()),
            root,
            strategy,
            args,
            capture,
        )
        summaries.append(summary)
        run_summaries.append(run_summary)

        if capture:
            session_dir = output_dir / "sessions" / f"session_{session_id:05}" / "round1"
            write_semicolon_csv(session_dir / f"prices_round_1_day_{day}.csv", captured["prices"])
            write_semicolon_csv(session_dir / f"trades_round_1_day_{day}.csv", captured["trades"])
            write_semicolon_csv(session_dir / f"trace_round_1_day_{day}.csv", captured["trace"])

    summary_frame = pd.DataFrame(summaries)
    run_frame = pd.DataFrame(run_summaries)
    summary_frame.to_csv(output_dir / "session_summary.csv", index=False)
    run_frame.to_csv(output_dir / "run_summary.csv", index=False)
    (output_dir / "run.log").write_text(
        "\n".join(
            [
                f"strategy={strategy}",
                f"data_dir={args.data_dir}",
                f"sessions={args.sessions}",
                f"seed={args.seed}",
                f"position_limit={args.position_limit}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"sessions={len(summary_frame)}")
    print(f"output={output_dir}")
    print(summary_frame["total_pnl"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(4).to_string())
    print()
    print("Per-product mean PnL:")
    print(f"  ASH_COATED_OSMIUM: {mean(summary_frame['osmium_pnl']):.2f}")
    print(f"  INTARIAN_PEPPER_ROOT: {mean(summary_frame['pepper_root_pnl']):.2f}")


if __name__ == "__main__":
    main()
