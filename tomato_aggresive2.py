from typing import Any
import json
import math
from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = max(0, (self.max_log_length - base_length) // 3)

        print(
            self.to_json(
                [
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        return {
            s: [od.buy_orders, od.sell_orders]
            for s, od in order_depths.items()
        }

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [
            [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
            for arr in trades.values()
            for t in arr
        ]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conv = {
            p: [
                o.bidPrice,
                o.askPrice,
                o.transportFees,
                o.exportTariff,
                o.importTariff,
                o.sugarPrice,
                o.sunlightIndex,
            ]
            for p, o in observations.conversionObservations.items()
        }
        return [observations.plainValueObservations, conv]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [
            [o.symbol, o.price, o.quantity]
            for arr in orders.values()
            for o in arr
        ]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if max_length <= 0:
            return ""
        return value if len(value) <= max_length else value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def __init__(self):
        self.limits = {"EMERALDS": 80, "TOMATOES": 80}

        self.orders = {}
        self.conversions = 0
        self.traderData = "AGGRESSIVE_V1"

        self.emeralds_position = 0
        self.emeralds_buy_orders = 0
        self.emeralds_sell_orders = 0

        self.tomatoes_position = 0
        self.tomatoes_buy_orders = 0
        self.tomatoes_sell_orders = 0

        # Aggressive params
        self.alpha = 0.30
        self.vol_alpha = 0.20
        self.base_spread = 0.8
        self.take_size = 20
        self.quote_size = 25

        self.ema = None
        self.last_mid = None
        self.vol = None

    def send_buy(self, p, price, qty):
        if qty > 0:
            self.orders[p].append(Order(p, int(price), qty))

    def send_sell(self, p, price, qty):
        if qty > 0:
            self.orders[p].append(Order(p, int(price), -qty))

    def best_prices(self, state, p):
        od = state.order_depths[p]
        if not od.buy_orders or not od.sell_orders:
            return None, None
        return max(od.buy_orders), min(od.sell_orders)

    def trade_emeralds(self, state):
        p = "EMERALDS"
        bid, ask = self.best_prices(state, p)
        if bid is None:
            return

        fair = 10000
        pos = state.position.get(p, 0)

        if ask < fair:
            self.send_buy(p, ask, min(20, 80 - pos))
        if bid > fair:
            self.send_sell(p, bid, min(20, pos + 80))

        self.send_buy(p, bid + 1, min(20, 80 - pos))
        self.send_sell(p, ask - 1, min(20, pos + 80))

    def trade_tomatoes(self, state):
        p = "TOMATOES"
        bid, ask = self.best_prices(state, p)
        if bid is None:
            return

        pos = state.position.get(p, 0)
        mid = (bid + ask) / 2

        if self.ema is None:
            self.ema = mid
        else:
            self.ema = self.alpha * mid + (1 - self.alpha) * self.ema

        if self.last_mid is None:
            change = 0
        else:
            change = abs(mid - self.last_mid)

        if self.vol is None:
            self.vol = change
        else:
            self.vol = self.vol_alpha * change + (1 - self.vol_alpha) * self.vol

        self.last_mid = mid

        spread = self.base_spread + self.vol

        # TAKE aggressively
        if ask < self.ema - spread:
            self.send_buy(p, ask, min(self.take_size, 80 - pos))

        if bid > self.ema + spread:
            self.send_sell(p, bid, min(self.take_size, pos + 80))

        # momentum push
        momentum = self.ema - mid
        bias = 1 if momentum > 0 else -1 if momentum < 0 else 0

        buy_price = bid + 1 + max(0, bias)
        sell_price = ask - 1 - max(0, -bias)

        size = self.quote_size
        if abs(pos) > 50:
            size = 10

        self.send_buy(p, buy_price, min(size, 80 - pos))
        self.send_sell(p, sell_price, min(size, pos + 80))

    def reset(self, state):
        self.orders = {p: [] for p in state.order_depths}
        self.emeralds_position = state.position.get("EMERALDS", 0)
        self.tomatoes_position = state.position.get("TOMATOES", 0)

    def run(self, state: TradingState):
        self.reset(state)

        if "EMERALDS" in state.order_depths:
            self.trade_emeralds(state)

        if "TOMATOES" in state.order_depths:
            self.trade_tomatoes(state)

        logger.flush(state, self.orders, self.conversions, self.traderData)
        return self.orders, self.conversions, self.traderData