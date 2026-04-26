from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

ROOT = Path("/Users/lakshaykumar/Documents/Playground/imc-prosperity-4-fresh")
sys.path.insert(0, str(ROOT / "scripts"))

from run_round1_monte_carlo_backtest import (  # noqa: E402
    Ledger,
    StrategyWorker,
    enforce_limits,
    execute_public_trade,
    execute_strategy_orders,
    infer_public_trade_side,
    row_to_book,
)

PRODUCTS = ("ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT")
DATASETS = {
    "R1": Path("/Users/lakshaykumar/Downloads/ROUND1"),
    "R2": Path("/Users/lakshaykumar/Downloads/ROUND2"),
}


def parse_day(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def load_dataset(data_dir: Path) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    prices_by_day: dict[int, pd.DataFrame] = {}
    trades_by_day: dict[int, pd.DataFrame] = {}

    for path in sorted(data_dir.glob("prices_round_*_day_*.csv")):
        day = parse_day(path)
        frame = pd.read_csv(path, sep=";")
        frame = frame[frame["product"].isin(PRODUCTS)].copy()
        prices_by_day[day] = frame.sort_values(["timestamp", "product"])

    for path in sorted(data_dir.glob("trades_round_*_day_*.csv")):
        day = parse_day(path)
        frame = pd.read_csv(path, sep=";")
        frame = frame[frame["symbol"].isin(PRODUCTS)].copy()
        trades_by_day[day] = frame.sort_values(["timestamp", "symbol"])

    if not prices_by_day:
        raise FileNotFoundError(f"no price files found in {data_dir}")

    return prices_by_day, trades_by_day


def validate_strategy(strategy: Path, max_timestamp: int | None) -> list[dict[str, Any]]:
    args = SimpleNamespace(
        python_bin=sys.executable,
        timeout_ms=900,
        position_limit=80,
    )
    rows: list[dict[str, Any]] = []
    session_id = 0

    for dataset, data_dir in DATASETS.items():
        prices_by_day, trades_by_day = load_dataset(data_dir)
        for day in sorted(prices_by_day):
            prices = prices_by_day[day]
            trades = trades_by_day.get(day, pd.DataFrame())
            if max_timestamp is not None:
                prices = prices[prices["timestamp"] <= max_timestamp].copy()
                if not trades.empty:
                    trades = trades[trades["timestamp"] <= max_timestamp].copy()

            summary = run_session_fast(
                session_id=session_id,
                day=day,
                prices=prices,
                trades=trades,
                root=ROOT,
                strategy=strategy,
                args=args,
            )
            row = {
                "strategy": strategy.name,
                "dataset": dataset,
                "day": day,
                "total_pnl": summary["total_pnl"],
                "osmium_pnl": summary["osmium_pnl"],
                "pepper_root_pnl": summary["pepper_root_pnl"],
                "osmium_position": summary["osmium_position"],
                "pepper_root_position": summary["pepper_root_position"],
            }
            rows.append(row)
            session_id += 1
    return rows


def run_session_fast(
    session_id: int,
    day: int,
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    root: Path,
    strategy: Path,
    args: Any,
) -> dict[str, Any]:
    worker = StrategyWorker(root, strategy, args.python_bin, args.timeout_ms)
    worker.reset()
    try:
        ledgers = {product: Ledger() for product in PRODUCTS}
        trader_data = ""
        prev_own_trades = {product: [] for product in PRODUCTS}
        prev_market_trades = {product: [] for product in PRODUCTS}

        snapshots: dict[int, dict[str, pd.Series]] = {}
        for timestamp, frame in prices.groupby("timestamp", sort=True):
            product_rows: dict[str, pd.Series] = {}
            for _, row in frame.iterrows():
                product_rows[str(row["product"])] = row
            if set(product_rows) == set(PRODUCTS):
                snapshots[int(timestamp)] = product_rows

        trades_by_timestamp = {
            int(timestamp): frame
            for timestamp, frame in trades.groupby("timestamp", sort=True)
        } if not trades.empty else {}

        for timestamp in sorted(snapshots):
            snapshot = snapshots[timestamp]
            books = {
                product: row_to_book(snapshot[product])
                for product in PRODUCTS
            }

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

            tick_trades = trades_by_timestamp.get(timestamp)
            if tick_trades is not None:
                for _, trade in tick_trades.iterrows():
                    product = str(trade["symbol"])
                    if product not in books:
                        continue
                    price_row = snapshot[product]
                    market_buy = infer_public_trade_side(
                        trade,
                        books[product],
                        float(price_row["mid_price"]),
                    )
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

            prev_own_trades = own_this_tick
            prev_market_trades = market_this_tick

        final_rows = prices.groupby("product").tail(1).set_index("product")
        product_pnls = {
            product: ledgers[product].cash
            + ledgers[product].position * float(final_rows.loc[product, "mid_price"])
            for product in PRODUCTS
        }
        return {
            "session_id": session_id,
            "day": day,
            "total_pnl": sum(product_pnls.values()),
            "osmium_pnl": product_pnls["ASH_COATED_OSMIUM"],
            "pepper_root_pnl": product_pnls["INTARIAN_PEPPER_ROOT"],
            "osmium_position": ledgers["ASH_COATED_OSMIUM"].position,
            "pepper_root_position": ledgers["INTARIAN_PEPPER_ROOT"].position,
        }
    finally:
        worker.close()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strategy",
                "dataset",
                "day",
                "total_pnl",
                "osmium_pnl",
                "pepper_root_pnl",
                "osmium_position",
                "pepper_root_position",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    print()
    grouped = frame.groupby(["strategy", "dataset"])[
        ["total_pnl", "osmium_pnl", "pepper_root_pnl"]
    ].mean()
    print(grouped.round(2).to_string())
    print()
    all_grouped = frame.groupby("strategy")[
        ["total_pnl", "osmium_pnl", "pepper_root_pnl"]
    ].mean()
    print(all_grouped.round(2).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("strategies", nargs="+")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "strategy_search_results" / "local_9700_candidate_scores.csv",
    )
    parser.add_argument("--max-timestamp", type=int)
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    for name in args.strategies:
        strategy = (ROOT / name).resolve()
        rows = validate_strategy(strategy, args.max_timestamp)
        write_rows(args.out, rows)
        all_rows.extend(rows)

    print_summary(all_rows)


if __name__ == "__main__":
    main()
