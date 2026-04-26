from typing import Any, Dict, Optional
import json
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
        orders: Dict[Symbol, list[Order]],
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

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(
        self, order_depths: Dict[Symbol, OrderDepth]
    ) -> Dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: Dict[Symbol, list[Trade]]) -> list[list[Any]]:
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

    def compress_orders(self, orders: Dict[Symbol, list[Order]]) -> list[list[Any]]:
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
        self.product = "HYDROGEL_PACK"
        self.limit = 200

        self.initial_day_mean = 9991.0
        self.entry_threshold = 22.0
        self.flatten_threshold = 5.0
        self.extreme_size = 40
        self.max_extreme_inventory = 120

        self.orders: Dict[Symbol, list[Order]] = {}
        self.buy_reserved: Dict[Symbol, int] = {}
        self.sell_reserved: Dict[Symbol, int] = {}
        self.conversions = 0
        self.traderData = "R3_HYDROGEL_ONLY_PREVDAY_EXTREMES_V1"

    def parse_memory(self, trader_data: str) -> Dict[str, Any]:
        memory = {
            "anchor": self.initial_day_mean,
            "day_sum": 0.0,
            "day_count": 0,
            "last_timestamp": None,
        }
        if not trader_data:
            return memory
        try:
            raw = json.loads(trader_data)
        except Exception:
            return memory
        memory["anchor"] = float(raw.get("anchor", self.initial_day_mean))
        memory["day_sum"] = float(raw.get("day_sum", 0.0))
        memory["day_count"] = int(raw.get("day_count", 0))
        memory["last_timestamp"] = raw.get("last_timestamp")
        return memory

    def encode_memory(self, memory: Dict[str, Any]) -> str:
        return json.dumps(memory, separators=(",", ":"))

    def reset_orders(self, state: TradingState) -> None:
        self.orders = {product: [] for product in state.order_depths}
        self.buy_reserved = {product: 0 for product in state.order_depths}
        self.sell_reserved = {product: 0 for product in state.order_depths}
        self.conversions = 0

    def get_position(self, state: TradingState) -> int:
        return int(state.position.get(self.product, 0))

    def remaining_buy(self, state: TradingState) -> int:
        return max(0, self.limit - self.get_position(state) - self.buy_reserved.get(self.product, 0))

    def remaining_sell(self, state: TradingState) -> int:
        return max(0, self.limit + self.get_position(state) - self.sell_reserved.get(self.product, 0))

    def place_buy(self, state: TradingState, price: int, quantity: int, msg: str) -> None:
        qty = min(max(0, int(quantity)), self.remaining_buy(state))
        if qty <= 0:
            return
        self.orders[self.product].append(Order(self.product, int(price), qty))
        self.buy_reserved[self.product] += qty
        logger.print(msg)

    def place_sell(self, state: TradingState, price: int, quantity: int, msg: str) -> None:
        qty = min(max(0, int(quantity)), self.remaining_sell(state))
        if qty <= 0:
            return
        self.orders[self.product].append(Order(self.product, int(price), -qty))
        self.sell_reserved[self.product] += qty
        logger.print(msg)

    def best_bid_ask(self, order_depth: OrderDepth) -> tuple[Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def wall_mid(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask = self.best_bid_ask(order_depth)
        if best_bid is not None and best_ask is not None:
            return 0.5 * (best_bid + best_ask)
        if best_ask is not None:
            return best_ask - 0.5
        if best_bid is not None:
            return best_bid + 0.5
        return None

    def update_day_anchor(self, memory: Dict[str, Any], timestamp: int, mid: float) -> None:
        last_timestamp = memory["last_timestamp"]
        if last_timestamp is not None and int(timestamp) < int(last_timestamp):
            if int(memory["day_count"]) > 0:
                memory["anchor"] = float(memory["day_sum"]) / float(memory["day_count"])
            memory["day_sum"] = 0.0
            memory["day_count"] = 0

        memory["day_sum"] = float(memory["day_sum"]) + mid
        memory["day_count"] = int(memory["day_count"]) + 1
        memory["last_timestamp"] = int(timestamp)

    def trade_hydrogel(self, state: TradingState, memory: Dict[str, Any]) -> None:
        if self.product not in state.order_depths:
            return

        order_depth = state.order_depths[self.product]
        best_bid, best_ask = self.best_bid_ask(order_depth)
        mid = self.wall_mid(order_depth)
        if mid is None or best_bid is None or best_ask is None:
            return

        self.update_day_anchor(memory, state.timestamp, mid)

        anchor = float(memory["anchor"])
        position = self.get_position(state)
        deviation = mid - anchor

        if abs(deviation) <= self.flatten_threshold:
            if position > 0:
                self.place_sell(
                    state,
                    best_bid,
                    min(position, self.extreme_size),
                    f"HYDRO FLATTEN SELL @ {best_bid} anchor={anchor:.2f} dev={deviation:.2f}",
                )
            elif position < 0:
                self.place_buy(
                    state,
                    best_ask,
                    min(-position, self.extreme_size),
                    f"HYDRO FLATTEN BUY @ {best_ask} anchor={anchor:.2f} dev={deviation:.2f}",
                )
        elif deviation <= -self.entry_threshold and position < self.max_extreme_inventory:
            buy_size = min(
                self.extreme_size,
                self.max_extreme_inventory - position,
                abs(int(order_depth.sell_orders.get(best_ask, 0))),
            )
            self.place_buy(
                state,
                best_ask,
                buy_size,
                f"HYDRO EXTREME BUY @ {best_ask} anchor={anchor:.2f} dev={deviation:.2f}",
            )
        elif deviation >= self.entry_threshold and position > -self.max_extreme_inventory:
            sell_size = min(
                self.extreme_size,
                self.max_extreme_inventory + position,
                abs(int(order_depth.buy_orders.get(best_bid, 0))),
            )
            self.place_sell(
                state,
                best_bid,
                sell_size,
                f"HYDRO EXTREME SELL @ {best_bid} anchor={anchor:.2f} dev={deviation:.2f}",
            )

        logger.print(
            f"mid={mid:.2f} anchor={anchor:.2f} dev={deviation:.2f} "
            f"pos={position} day_count={int(memory['day_count'])}"
        )

    def run(self, state: TradingState):
        self.reset_orders(state)
        memory = self.parse_memory(state.traderData)
        self.trade_hydrogel(state, memory)
        trader_data = self.encode_memory(memory)
        logger.flush(state, self.orders, self.conversions, trader_data)
        return self.orders, self.conversions, trader_data
