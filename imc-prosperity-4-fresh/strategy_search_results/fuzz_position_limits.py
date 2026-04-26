from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path("/Users/lakshaykumar/Documents/Playground/imc-prosperity-4-fresh")
WORKER = ROOT / "scripts" / "python_strategy_worker.py"
PRODUCTS = ("ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT")


@dataclass
class FuzzStats:
    ticks: int = 0
    post_wrapper_rejections: int = 0
    trim_events: int = 0
    worker_errors: int = 0
    max_buy_overrun: int = 0
    max_sell_overrun: int = 0


class StrategyProcess:
    def __init__(self, strategy: Path, timeout_ms: int) -> None:
        self.timeout_ms = timeout_ms
        self.process = subprocess.Popen(
            [sys.executable, str(WORKER), str(strategy)],
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("strategy worker pipe is closed")
        payload.setdefault("timeout_ms", self.timeout_ms)
        self.process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            stderr = ""
            if self.process.stderr is not None:
                stderr = self.process.stderr.read()
            raise RuntimeError(f"strategy worker stopped unexpectedly: {stderr}")
        return json.loads(line)

    def reset(self) -> None:
        response = self.call({"type": "reset"})
        if response.get("error"):
            raise RuntimeError(response["error"])


def random_book(
    rng: random.Random,
    product: str,
    timestamp: int,
    session: int,
) -> dict[str, dict[str, int]]:
    if product == "ASH_COATED_OSMIUM":
        base = 10_000 + rng.randint(-90, 90)
    else:
        base = 10_000 + session * 80 + timestamp * 0.001 + rng.randint(-8, 8)

    spread = rng.choice([2, 2, 3, 4, 5, 6, 8, 10, 14, 18])
    mid = int(round(base))
    best_bid = mid - spread // 2
    best_ask = best_bid + spread

    bid_levels = rng.choices([0, 1, 2, 3], weights=[8, 45, 35, 12], k=1)[0]
    ask_levels = rng.choices([0, 1, 2, 3], weights=[8, 45, 35, 12], k=1)[0]

    if bid_levels == 0 and ask_levels == 0:
        if rng.random() < 0.5:
            bid_levels = 1
        else:
            ask_levels = 1

    buy_orders: dict[str, int] = {}
    sell_orders: dict[str, int] = {}

    for level in range(bid_levels):
        price = best_bid - level * rng.choice([1, 1, 2, 3])
        buy_orders[str(price)] = rng.randint(1, 45)

    for level in range(ask_levels):
        price = best_ask + level * rng.choice([1, 1, 2, 3])
        sell_orders[str(price)] = -rng.randint(1, 45)

    return {"buy_orders": buy_orders, "sell_orders": sell_orders}


def random_position(rng: random.Random, previous: int, limit: int, tick: int) -> int:
    if tick % 53 == 0:
        return rng.choice([-limit, -limit + 1, -limit + 5, 0, limit - 5, limit - 1, limit])
    if rng.random() < 0.08:
        return rng.randint(-limit, limit)
    step = rng.randint(-12, 12)
    return max(-limit, min(limit, previous + step))


def validate_orders(
    orders: dict[str, list[dict[str, Any]]],
    positions: dict[str, int],
    limit: int,
) -> tuple[bool, dict[str, int]]:
    worst = {"buy_overrun": 0, "sell_overrun": 0}
    ok = True

    for product in PRODUCTS:
        position = positions.get(product, 0)
        buy_qty = 0
        sell_qty = 0
        for order in orders.get(product, []):
            quantity = int(order["quantity"])
            if quantity > 0:
                buy_qty += quantity
            elif quantity < 0:
                sell_qty += -quantity

        buy_overrun = max(0, position + buy_qty - limit)
        sell_overrun = max(0, -limit - (position - sell_qty))
        worst["buy_overrun"] = max(worst["buy_overrun"], buy_overrun)
        worst["sell_overrun"] = max(worst["sell_overrun"], sell_overrun)
        if buy_overrun or sell_overrun:
            ok = False

    return ok, worst


def run_fuzz(
    strategy: Path,
    sessions: int,
    ticks: int,
    seed: int,
    limit: int,
    timeout_ms: int,
    show_failures: int,
) -> FuzzStats:
    rng = random.Random(seed)
    stats = FuzzStats()
    failures: list[dict[str, Any]] = []

    worker = StrategyProcess(strategy, timeout_ms)
    try:
        for session in range(sessions):
            worker.reset()
            trader_data = ""
            positions = {product: 0 for product in PRODUCTS}

            for tick in range(ticks):
                timestamp = tick * 100
                for product in PRODUCTS:
                    positions[product] = random_position(
                        rng,
                        positions[product],
                        limit,
                        tick,
                    )

                payload = {
                    "type": "run",
                    "timestamp": timestamp,
                    "trader_data": trader_data,
                    "order_depths": {
                        product: random_book(rng, product, timestamp, session)
                        for product in PRODUCTS
                    },
                    "own_trades": {product: [] for product in PRODUCTS},
                    "market_trades": {product: [] for product in PRODUCTS},
                    "position": positions,
                    "observations": {},
                    "timeout_ms": timeout_ms,
                }
                response = worker.call(payload)
                stats.ticks += 1

                if response.get("error"):
                    stats.worker_errors += 1
                    if len(failures) < show_failures:
                        failures.append(
                            {
                                "session": session,
                                "tick": tick,
                                "kind": "worker_error",
                                "error": response["error"],
                            }
                        )
                    continue

                trader_data = response.get("trader_data") or ""
                stdout = response.get("stdout") or ""
                stats.trim_events += stdout.count("SAFETY_TRIM")

                ok, worst = validate_orders(response.get("orders") or {}, positions, limit)
                stats.max_buy_overrun = max(stats.max_buy_overrun, worst["buy_overrun"])
                stats.max_sell_overrun = max(stats.max_sell_overrun, worst["sell_overrun"])

                if not ok:
                    stats.post_wrapper_rejections += 1
                    if len(failures) < show_failures:
                        failures.append(
                            {
                                "session": session,
                                "tick": tick,
                                "timestamp": timestamp,
                                "positions": positions.copy(),
                                "orders": response.get("orders") or {},
                                "worst": worst,
                            }
                        )
    finally:
        worker.close()

    if failures:
        print("First failures:")
        for failure in failures:
            print(json.dumps(failure, indent=2, sort_keys=True))
        print()

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuzz a Prosperity trader for silent engine order-limit rejections."
    )
    parser.add_argument("strategy", help="Strategy file relative to repo root or absolute path")
    parser.add_argument("--sessions", type=int, default=100)
    parser.add_argument("--ticks", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260418)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--timeout-ms", type=int, default=900)
    parser.add_argument("--show-failures", type=int, default=3)
    args = parser.parse_args()

    strategy = Path(args.strategy)
    if not strategy.is_absolute():
        strategy = ROOT / strategy
    strategy = strategy.resolve()
    if not strategy.exists():
        raise FileNotFoundError(strategy)

    stats = run_fuzz(
        strategy=strategy,
        sessions=args.sessions,
        ticks=args.ticks,
        seed=args.seed,
        limit=args.limit,
        timeout_ms=args.timeout_ms,
        show_failures=args.show_failures,
    )

    print(f"strategy: {strategy.name}")
    print(f"sessions: {args.sessions}")
    print(f"ticks: {stats.ticks}")
    print(f"post_wrapper_rejections: {stats.post_wrapper_rejections}")
    print(f"worker_errors: {stats.worker_errors}")
    print(f"trim_events: {stats.trim_events}")
    print(f"max_buy_overrun: {stats.max_buy_overrun}")
    print(f"max_sell_overrun: {stats.max_sell_overrun}")

    if stats.post_wrapper_rejections or stats.worker_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
