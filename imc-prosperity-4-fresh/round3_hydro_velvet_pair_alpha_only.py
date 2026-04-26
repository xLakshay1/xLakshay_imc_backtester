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
        return [[listing.symbol, listing.product, listing.denomination] for listing in listings.values()]

    def compress_order_depths(
        self, order_depths: Dict[Symbol, OrderDepth]
    ) -> Dict[Symbol, list[Any]]:
        return {
            symbol: [order_depth.buy_orders, order_depth.sell_orders]
            for symbol, order_depth in order_depths.items()
        }

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
    def __init__(self) -> None:
        self.hydro = "HYDROGEL_PACK"
        self.velvet = "VELVETFRUIT_EXTRACT"
        self.hydro_limit = 200
        self.velvet_limit = 200

        self.hydro_initial_anchor = 9991.0
        self.velvet_initial_anchor = 5260.0
        self.hydro_scale = 32.0
        self.velvet_scale = 15.0

        self.entry_threshold = 0.90
        self.flatten_threshold = 0.25
        self.max_hydro_target = 90
        self.max_velvet_target = 170
        self.min_signal_gap = 0.10

        self.orders: Dict[Symbol, list[Order]] = {}
        self.buy_reserved: Dict[Symbol, int] = {}
        self.sell_reserved: Dict[Symbol, int] = {}
        self.conversions = 0
        self.traderData = "R3_HYDRO_VELVET_PAIR_ALPHA_ONLY_V1"

    def parse_memory(self, trader_data: str) -> Dict[str, Any]:
        memory = {
            "last_timestamp": None,
            "hydro_anchor": self.hydro_initial_anchor,
            "velvet_anchor": self.velvet_initial_anchor,
            "hydro_day_sum": 0.0,
            "hydro_day_count": 0,
            "velvet_day_sum": 0.0,
            "velvet_day_count": 0,
        }
        if not trader_data:
            return memory
        try:
            raw = json.loads(trader_data)
        except Exception:
            return memory
        for key, default in memory.items():
            value = raw.get(key, default)
            if isinstance(default, float):
                memory[key] = float(value)
            elif isinstance(default, int):
                memory[key] = int(value)
            else:
                memory[key] = value
        return memory

    def encode_memory(self, memory: Dict[str, Any]) -> str:
        return json.dumps(memory, separators=(",", ":"))

    def reset_orders(self, state: TradingState) -> None:
        self.orders = {product: [] for product in state.order_depths}
        self.buy_reserved = {product: 0 for product in state.order_depths}
        self.sell_reserved = {product: 0 for product in state.order_depths}
        self.conversions = 0

    def get_position(self, state: TradingState, product: str) -> int:
        return int(state.position.get(product, 0))

    def limit_for(self, product: str) -> int:
        return self.hydro_limit if product == self.hydro else self.velvet_limit

    def remaining_buy(self, state: TradingState, product: str) -> int:
        return max(
            0,
            self.limit_for(product)
            - self.get_position(state, product)
            - self.buy_reserved.get(product, 0),
        )

    def remaining_sell(self, state: TradingState, product: str) -> int:
        return max(
            0,
            self.limit_for(product)
            + self.get_position(state, product)
            - self.sell_reserved.get(product, 0),
        )

    def place_buy(self, state: TradingState, product: str, price: int, quantity: int, msg: str) -> None:
        qty = min(max(0, int(quantity)), self.remaining_buy(state, product))
        if qty <= 0:
            return
        self.orders[product].append(Order(product, int(price), qty))
        self.buy_reserved[product] += qty
        logger.print(msg)

    def place_sell(self, state: TradingState, product: str, price: int, quantity: int, msg: str) -> None:
        qty = min(max(0, int(quantity)), self.remaining_sell(state, product))
        if qty <= 0:
            return
        self.orders[product].append(Order(product, int(price), -qty))
        self.sell_reserved[product] += qty
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

    def signal_for(self, order_depth: OrderDepth) -> Dict[str, Optional[float]]:
        best_bid, best_ask = self.best_bid_ask(order_depth)
        mid = self.wall_mid(order_depth)
        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid": mid,
        }

    def update_day_anchors(
        self,
        memory: Dict[str, Any],
        timestamp: int,
        hydro_mid: float,
        velvet_mid: float,
    ) -> None:
        last_timestamp = memory["last_timestamp"]
        if last_timestamp is not None and int(timestamp) < int(last_timestamp):
            if int(memory["hydro_day_count"]) > 0:
                memory["hydro_anchor"] = float(memory["hydro_day_sum"]) / float(memory["hydro_day_count"])
            if int(memory["velvet_day_count"]) > 0:
                memory["velvet_anchor"] = float(memory["velvet_day_sum"]) / float(memory["velvet_day_count"])
            memory["hydro_day_sum"] = 0.0
            memory["hydro_day_count"] = 0
            memory["velvet_day_sum"] = 0.0
            memory["velvet_day_count"] = 0

        memory["hydro_day_sum"] = float(memory["hydro_day_sum"]) + hydro_mid
        memory["hydro_day_count"] = int(memory["hydro_day_count"]) + 1
        memory["velvet_day_sum"] = float(memory["velvet_day_sum"]) + velvet_mid
        memory["velvet_day_count"] = int(memory["velvet_day_count"]) + 1
        memory["last_timestamp"] = int(timestamp)

    def target_positions(self, signal: float) -> tuple[int, int]:
        if abs(signal) <= self.flatten_threshold:
            return 0, 0
        if abs(signal) < self.entry_threshold:
            return 0, 0

        edge = abs(signal) - self.entry_threshold
        strength = min(1.0, edge / 1.5)
        hydro_qty = max(10, int(round(self.max_hydro_target * strength)))
        velvet_qty = max(20, int(round(self.max_velvet_target * strength)))

        if signal > 0:
            return -hydro_qty, velvet_qty
        return hydro_qty, -velvet_qty

    def trade_to_target(
        self,
        state: TradingState,
        product: str,
        best_bid: int,
        best_ask: int,
        target: int,
    ) -> None:
        current = self.get_position(state, product)
        delta = target - current
        if abs(delta) < 1:
            return
        if delta > 0:
            self.place_buy(
                state,
                product,
                best_ask,
                delta,
                f"{product} PAIR BUY @ {best_ask} target={target} current={current}",
            )
        else:
            self.place_sell(
                state,
                product,
                best_bid,
                -delta,
                f"{product} PAIR SELL @ {best_bid} target={target} current={current}",
            )

    def run_pair_alpha(self, state: TradingState, memory: Dict[str, Any]) -> None:
        if self.hydro not in state.order_depths or self.velvet not in state.order_depths:
            return

        hydro_signal = self.signal_for(state.order_depths[self.hydro])
        velvet_signal = self.signal_for(state.order_depths[self.velvet])
        hydro_mid = hydro_signal["mid"]
        velvet_mid = velvet_signal["mid"]
        hydro_bid = hydro_signal["best_bid"]
        hydro_ask = hydro_signal["best_ask"]
        velvet_bid = velvet_signal["best_bid"]
        velvet_ask = velvet_signal["best_ask"]

        if None in (hydro_mid, velvet_mid, hydro_bid, hydro_ask, velvet_bid, velvet_ask):
            return

        hydro_mid = float(hydro_mid)
        velvet_mid = float(velvet_mid)
        hydro_bid = int(hydro_bid)
        hydro_ask = int(hydro_ask)
        velvet_bid = int(velvet_bid)
        velvet_ask = int(velvet_ask)

        self.update_day_anchors(memory, state.timestamp, hydro_mid, velvet_mid)

        hydro_anchor = float(memory["hydro_anchor"])
        velvet_anchor = float(memory["velvet_anchor"])
        hydro_z = (hydro_mid - hydro_anchor) / self.hydro_scale
        velvet_z = (velvet_mid - velvet_anchor) / self.velvet_scale
        pair_signal = hydro_z - velvet_z

        hydro_target, velvet_target = self.target_positions(pair_signal)
        if abs(pair_signal) < self.min_signal_gap:
            hydro_target, velvet_target = 0, 0

        self.trade_to_target(state, self.hydro, hydro_bid, hydro_ask, hydro_target)
        self.trade_to_target(state, self.velvet, velvet_bid, velvet_ask, velvet_target)

        logger.print(
            f"pair_signal={pair_signal:.3f} hydro_z={hydro_z:.3f} velvet_z={velvet_z:.3f} "
            f"hydro_anchor={hydro_anchor:.2f} velvet_anchor={velvet_anchor:.2f} "
            f"hydro_target={hydro_target} velvet_target={velvet_target} "
            f"hydro_pos={self.get_position(state, self.hydro)} velvet_pos={self.get_position(state, self.velvet)}"
        )

    def run(self, state: TradingState):
        self.reset_orders(state)
        memory = self.parse_memory(state.traderData)
        self.run_pair_alpha(state, memory)
        trader_data = self.encode_memory(memory)
        logger.flush(state, self.orders, self.conversions, trader_data)
        return self.orders, self.conversions, trader_data
