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
        return {s: [d.buy_orders, d.sell_orders] for s, d in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        out = []
        for arr in trades.values():
            for t in arr:
                out.append(
                    [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                )
        return out

    def compress_observations(self, observations: Observation) -> list[Any]:
        conv = {}
        for p, o in observations.conversionObservations.items():
            conv[p] = [
                o.bidPrice,
                o.askPrice,
                o.transportFees,
                o.exportTariff,
                o.importTariff,
                o.sugarPrice,
                o.sunlightIndex,
            ]
        return [observations.plainValueObservations, conv]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        out = []
        for arr in orders.values():
            for o in arr:
                out.append([o.symbol, o.price, o.quantity])
        return out

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if max_length <= 0:
            return ""
        return value if len(value) <= max_length else value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def __init__(self):
        self.limits = {
            "EMERALDS": 80,
            "TOMATOES": 80,
        }

        self.orders: dict[Symbol, list[Order]] = {}
        self.conversions = 0
        self.traderData = "EMA_MICRO_OPT_V1"

        self.emeralds_position = 0
        self.emeralds_buy_orders = 0
        self.emeralds_sell_orders = 0

        self.tomatoes_position = 0
        self.tomatoes_buy_orders = 0
        self.tomatoes_sell_orders = 0

        # Tomatoes state
        self.tomatoes_ema = None

        # Tuned parameters
        self.tomatoes_alpha = 0.18
        self.tomatoes_imbalance_weight = 0.75
        self.tomatoes_base_take_threshold = 1.0
        self.tomatoes_min_take_threshold = 0.35
        self.tomatoes_base_quote_offset = 2.0
        self.tomatoes_min_quote_size = 4
        self.tomatoes_max_quote_size = 16

    def send_sell_order(self, product: str, price: int, amount: int, msg=None):
        if amount < 0:
            self.orders[product].append(Order(product, int(price), amount))
            if msg is not None:
                logger.print(msg)

    def send_buy_order(self, product: str, price: int, amount: int, msg=None):
        if amount > 0:
            self.orders[product].append(Order(product, int(price), amount))
            if msg is not None:
                logger.print(msg)

    def get_product_pos(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def get_best_bid_ask(self, state: TradingState, product: str):
        order_depth = state.order_depths[product]
        if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
            return None, None
        return max(order_depth.buy_orders.keys()), min(order_depth.sell_orders.keys())

    def get_order_book_imbalance(self, state: TradingState, product: str) -> float:
        order_depth = state.order_depths[product]
        if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
            return 0.0

        bid_vol = sum(order_depth.buy_orders.values())
        ask_vol = sum(abs(v) for v in order_depth.sell_orders.values())
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0

        return (bid_vol - ask_vol) / total

    def search_buys(
        self, state: TradingState, product: str, acceptable_price: float, depth: int = 1
    ):
        order_depth = state.order_depths[product]
        if len(order_depth.sell_orders) == 0:
            return

        sell_side = sorted(order_depth.sell_orders.items())
        for ask, amount in sell_side[: min(depth, len(sell_side))]:
            pos = self.get_product_pos(state, product)

            should_buy = (
                ask < acceptable_price
                or (
                    abs(ask - acceptable_price) < 1
                    and pos < 0
                    and abs(pos - amount) < abs(pos)
                )
            )

            if not should_buy:
                continue

            if product == "EMERALDS":
                size = min(
                    self.limits["EMERALDS"]
                    - self.emeralds_position
                    - self.emeralds_buy_orders,
                    -amount,
                )
                if size > 0:
                    self.emeralds_buy_orders += size
                    self.send_buy_order(
                        product, ask, size, msg=f"TAKE BUY {size} {product} @ {ask}"
                    )

            elif product == "TOMATOES":
                size = min(
                    self.limits["TOMATOES"]
                    - self.tomatoes_position
                    - self.tomatoes_buy_orders,
                    -amount,
                )
                if size > 0:
                    self.tomatoes_buy_orders += size
                    self.send_buy_order(
                        product, ask, size, msg=f"TAKE BUY {size} {product} @ {ask}"
                    )

    def search_sells(
        self, state: TradingState, product: str, acceptable_price: float, depth: int = 1
    ):
        order_depth = state.order_depths[product]
        if len(order_depth.buy_orders) == 0:
            return

        buy_side = sorted(order_depth.buy_orders.items(), reverse=True)
        for bid, amount in buy_side[: min(depth, len(buy_side))]:
            pos = self.get_product_pos(state, product)

            should_sell = (
                bid > acceptable_price
                or (
                    abs(bid - acceptable_price) < 1
                    and pos > 0
                    and abs(pos - amount) < abs(pos)
                )
            )

            if not should_sell:
                continue

            if product == "EMERALDS":
                size = min(
                    self.emeralds_position
                    + self.limits["EMERALDS"]
                    - self.emeralds_sell_orders,
                    amount,
                )
                if size > 0:
                    self.emeralds_sell_orders += size
                    self.send_sell_order(
                        product, bid, -size, msg=f"TAKE SELL {-size} {product} @ {bid}"
                    )

            elif product == "TOMATOES":
                size = min(
                    self.tomatoes_position
                    + self.limits["TOMATOES"]
                    - self.tomatoes_sell_orders,
                    amount,
                )
                if size > 0:
                    self.tomatoes_sell_orders += size
                    self.send_sell_order(
                        product, bid, -size, msg=f"TAKE SELL {-size} {product} @ {bid}"
                    )

    def trade_emeralds(self, state: TradingState):
        product = "EMERALDS"
        fair_value = 10000

        best_bid, best_ask = self.get_best_bid_ask(state, product)
        if best_bid is None or best_ask is None:
            return

        position = self.get_product_pos(state, product)

        self.search_buys(state, product, fair_value, depth=3)
        self.search_sells(state, product, fair_value, depth=3)

        skew = position // 20
        buy_price = min(best_bid + 1, fair_value - 1 - skew)
        sell_price = max(best_ask - 1, fair_value + 1 - skew)

        max_buy = (
            self.limits[product]
            - self.emeralds_position
            - self.emeralds_buy_orders
        )
        max_sell = (
            self.emeralds_position
            + self.limits[product]
            - self.emeralds_sell_orders
        )

        max_buy = max(0, max_buy)
        max_sell = max(0, max_sell)

        quote_size = 20
        if abs(position) > 50:
            quote_size = 10
        elif abs(position) > 30:
            quote_size = 15

        max_buy = min(max_buy, quote_size)
        max_sell = min(max_sell, quote_size)

        if max_buy > 0:
            self.send_buy_order(
                product, buy_price, max_buy, msg=f"{product} MM BUY {max_buy} @ {buy_price}"
            )
        if max_sell > 0:
            self.send_sell_order(
                product, sell_price, -max_sell, msg=f"{product} MM SELL {max_sell} @ {sell_price}"
            )

    def trade_tomatoes(self, state: TradingState):
        product = "TOMATOES"

        best_bid, best_ask = self.get_best_bid_ask(state, product)
        if best_bid is None or best_ask is None:
            return

        position = self.get_product_pos(state, product)
        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid

        # EMA baseline
        if self.tomatoes_ema is None:
            self.tomatoes_ema = mid
        else:
            self.tomatoes_ema = (
                self.tomatoes_alpha * mid
                + (1 - self.tomatoes_alpha) * self.tomatoes_ema
            )

        imbalance = self.get_order_book_imbalance(state, product)
        confidence = abs(imbalance)

        # Stable fair value + modest microstructure tilt
        fair_value = self.tomatoes_ema + self.tomatoes_imbalance_weight * imbalance

        # Adaptive take threshold
        take_threshold = max(
            self.tomatoes_min_take_threshold,
            self.tomatoes_base_take_threshold - 0.35 * confidence
        )

        # Only be aggressive if there is enough spread to earn
        can_quote = spread >= 2

        # Adaptive passive offset
        if spread >= 5:
            adaptive_quote_offset = 1.5
        elif spread >= 3:
            adaptive_quote_offset = 2.0
        else:
            adaptive_quote_offset = 2.5

        # Conservative size scaling
        raw_size = self.tomatoes_min_quote_size + int(8 * confidence)
        quote_size = max(self.tomatoes_min_quote_size, min(self.tomatoes_max_quote_size, raw_size))

        if abs(position) > 50:
            quote_size = min(6, quote_size)
        elif abs(position) > 30:
            quote_size = min(10, quote_size)

        logger.print(
            f"TOMATOES mid={mid:.2f} ema={self.tomatoes_ema:.2f} fv={fair_value:.2f} "
            f"imb={imbalance:.3f} conf={confidence:.3f} thr={take_threshold:.2f} "
            f"spread={spread} pos={position}"
        )

        # Aggressive taking only on clear edge
        buy_take_price = fair_value - take_threshold
        sell_take_price = fair_value + take_threshold

        self.search_buys(state, product, buy_take_price, depth=2)
        self.search_sells(state, product, sell_take_price, depth=2)

        if not can_quote:
            return

        # Stronger inventory control
        inventory_skew = position / 15.0
        buy_offset = adaptive_quote_offset + max(0.0, inventory_skew)
        sell_offset = adaptive_quote_offset + max(0.0, -inventory_skew)

        buy_price = min(best_bid + 1, math.floor(fair_value - buy_offset))
        sell_price = max(best_ask - 1, math.ceil(fair_value + sell_offset))

        max_buy = (
            self.limits[product]
            - self.tomatoes_position
            - self.tomatoes_buy_orders
        )
        max_sell = (
            self.tomatoes_position
            + self.limits[product]
            - self.tomatoes_sell_orders
        )

        max_buy = max(0, max_buy)
        max_sell = max(0, max_sell)

        max_buy = min(max_buy, quote_size)
        max_sell = min(max_sell, quote_size)

        # Avoid crossing accidentally
        if buy_price < best_ask and max_buy > 0:
            self.send_buy_order(
                product, buy_price, max_buy, msg=f"{product} MM BUY {max_buy} @ {buy_price}"
            )

        if sell_price > best_bid and max_sell > 0:
            self.send_sell_order(
                product, sell_price, -max_sell, msg=f"{product} MM SELL {max_sell} @ {sell_price}"
            )

    def reset_orders(self, state: TradingState):
        self.orders = {}
        self.conversions = 0

        self.emeralds_position = self.get_product_pos(state, "EMERALDS")
        self.emeralds_buy_orders = 0
        self.emeralds_sell_orders = 0

        self.tomatoes_position = self.get_product_pos(state, "TOMATOES")
        self.tomatoes_buy_orders = 0
        self.tomatoes_sell_orders = 0

        for product in state.order_depths:
            self.orders[product] = []

    def run(self, state: TradingState):
        self.reset_orders(state)

        if "EMERALDS" in state.order_depths:
            self.trade_emeralds(state)

        if "TOMATOES" in state.order_depths:
            self.trade_tomatoes(state)

        logger.flush(state, self.orders, self.conversions, self.traderData)
        return self.orders, self.conversions, self.traderData