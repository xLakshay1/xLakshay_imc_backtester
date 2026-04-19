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
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if max_length <= 0:
            return ""
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def __init__(self):
        self.limits = {
            "EMERALDS": 80,
            "TOMATOES": 80,
        }

        self.orders: dict[Symbol, list[Order]] = {}
        self.conversions = 0
        self.traderData = "EMA_SIMPLE_INV_V1"

        # Position/order tracking
        self.emeralds_position = 0
        self.emeralds_buy_orders = 0
        self.emeralds_sell_orders = 0

        self.tomatoes_position = 0
        self.tomatoes_buy_orders = 0
        self.tomatoes_sell_orders = 0

        # Tomatoes EMA state
        self.tomatoes_ema = None

        # Hyperparameters
        self.tomatoes_alpha = 0.34
        self.tomatoes_take_threshold = 1.1
        self.tomatoes_base_quote_offset = 2.0

        # Inventory management knobs
        self.tomatoes_soft_inventory = 30
        self.tomatoes_hard_inventory = 55
        self.tomatoes_inventory_skew_strength = 1.8
        self.tomatoes_take_inventory_penalty = 0.8

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
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return best_bid, best_ask

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
                product,
                buy_price,
                max_buy,
                msg=f"{product} MM BUY {max_buy} @ {buy_price}",
            )
        if max_sell > 0:
            self.send_sell_order(
                product,
                sell_price,
                -max_sell,
                msg=f"{product} MM SELL {max_sell} @ {sell_price}",
            )

    def trade_tomatoes(self, state: TradingState):
        product = "TOMATOES"

        best_bid, best_ask = self.get_best_bid_ask(state, product)
        if best_bid is None or best_ask is None:
            return

        position = self.get_product_pos(state, product)
        mid = (best_bid + best_ask) / 2.0

        # EMA fair value
        if self.tomatoes_ema is None:
            self.tomatoes_ema = mid
        else:
            self.tomatoes_ema = (
                self.tomatoes_alpha * mid
                + (1 - self.tomatoes_alpha) * self.tomatoes_ema
            )

        fair_value = self.tomatoes_ema

        # Inventory-aware take thresholds
        inv_ratio = position / self.limits[product]

        buy_take_threshold = self.tomatoes_take_threshold + max(
            0.0, inv_ratio * self.tomatoes_take_inventory_penalty
        )
        sell_take_threshold = self.tomatoes_take_threshold + max(
            0.0, -inv_ratio * self.tomatoes_take_inventory_penalty
        )

        logger.print(
            f"TOMATOES mid={mid:.2f} fair={fair_value:.2f} pos={position} "
            f"bthr={buy_take_threshold:.2f} sthr={sell_take_threshold:.2f}"
        )

        # Corrected take logic + inventory aware
        self.search_buys(
            state,
            product,
            fair_value - buy_take_threshold,
            depth=3,
        )
        self.search_sells(
            state,
            product,
            fair_value + sell_take_threshold,
            depth=3,
        )

        # Stronger inventory-aware quoting
        inventory_skew = inv_ratio * self.tomatoes_inventory_skew_strength
        buy_offset = self.tomatoes_base_quote_offset + max(0.0, inventory_skew)
        sell_offset = self.tomatoes_base_quote_offset + max(0.0, -inventory_skew)

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

        # Inventory-aware sizes
        quote_size = 20
        if abs(position) > self.tomatoes_hard_inventory:
            quote_size = 4
        elif abs(position) > self.tomatoes_soft_inventory:
            quote_size = 8
        elif abs(position) > 20:
            quote_size = 12

        buy_quote_size = quote_size
        sell_quote_size = quote_size

        # If already long, shrink buy size and enlarge sell side a bit
        if position > self.tomatoes_soft_inventory:
            buy_quote_size = max(1, int(quote_size * 0.3))
            sell_quote_size = max(2, int(quote_size * 1.2))
        elif position < -self.tomatoes_soft_inventory:
            sell_quote_size = max(1, int(quote_size * 0.3))
            buy_quote_size = max(2, int(quote_size * 1.2))

        max_buy = min(max_buy, buy_quote_size)
        max_sell = min(max_sell, sell_quote_size)

        # Soft inventory reduction mode
        if position >= self.tomatoes_hard_inventory:
            reduce_sell = min(max_sell, max(1, int(position * 0.25)))
            if reduce_sell > 0:
                self.send_sell_order(
                    product,
                    max(best_ask - 1, best_bid),
                    -reduce_sell,
                    msg=f"{product} REDUCE LONG {reduce_sell}",
                )
            return

        if position <= -self.tomatoes_hard_inventory:
            reduce_buy = min(max_buy, max(1, int(abs(position) * 0.25)))
            if reduce_buy > 0:
                self.send_buy_order(
                    product,
                    min(best_bid + 1, best_ask),
                    reduce_buy,
                    msg=f"{product} REDUCE SHORT {reduce_buy}",
                )
            return

        if max_buy > 0:
            self.send_buy_order(
                product,
                buy_price,
                max_buy,
                msg=f"{product} MM BUY {max_buy} @ {buy_price}",
            )
        if max_sell > 0:
            self.send_sell_order(
                product,
                sell_price,
                -max_sell,
                msg=f"{product} MM SELL {max_sell} @ {sell_price}",
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