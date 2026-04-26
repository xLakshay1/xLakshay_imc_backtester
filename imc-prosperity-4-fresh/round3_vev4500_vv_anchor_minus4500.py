from typing import Any, Dict, Optional
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

    def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[Symbol, list[Any]]:
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
    def __init__(self):
        self.underlying = "VELVETFRUIT_EXTRACT"
        self.option = "VEV_4500"
        self.limit = 300

        self.initial_vv_mean = 5266.0
        self.day_shift = 20.0
        self.synthetic_strike_gap = 4500.0
        self.ret_weight = 0.18
        self.book_weight = 0.90
        self.l1_weight = 0.35
        self.inventory_skew = 0.03
        self.base_quote_size = 16
        self.take_size = 24
        self.depth_min = 12.0
        self.spread_cap = 24.0
        self.depth_p25 = 35.0
        self.depth_p75 = 55.0
        self.traderData = "R3_VEV4500_VV_ANCHOR_MINUS4500_V1"

        self.orders: Dict[Symbol, list[Order]] = {}
        self.buy_reserved: Dict[Symbol, int] = {}
        self.sell_reserved: Dict[Symbol, int] = {}
        self.conversions = 0

    def reset_orders(self, state: TradingState) -> None:
        self.orders = {product: [] for product in state.order_depths}
        self.buy_reserved = {product: 0 for product in state.order_depths}
        self.sell_reserved = {product: 0 for product in state.order_depths}
        self.conversions = 0

    def parse_memory(self, trader_data: str) -> Dict[str, Any]:
        memory = {
            "vv_anchor": self.initial_vv_mean,
            "day_sum": 0.0,
            "day_count": 0,
            "last_timestamp": None,
            "prev_vv_mid": None,
        }
        if not trader_data:
            return memory
        try:
            raw = json.loads(trader_data)
        except Exception:
            return memory
        memory["vv_anchor"] = float(raw.get("vv_anchor", self.initial_vv_mean))
        memory["day_sum"] = float(raw.get("day_sum", 0.0))
        memory["day_count"] = int(raw.get("day_count", 0))
        memory["last_timestamp"] = raw.get("last_timestamp")
        memory["prev_vv_mid"] = raw.get("prev_vv_mid")
        return memory

    def encode_memory(self, memory: Dict[str, Any]) -> str:
        return json.dumps(memory, separators=(",", ":"))

    def get_position(self, state: TradingState) -> int:
        return int(state.position.get(self.option, 0))

    def remaining_buy(self, state: TradingState) -> int:
        return max(0, self.limit - self.get_position(state) - self.buy_reserved.get(self.option, 0))

    def remaining_sell(self, state: TradingState) -> int:
        return max(0, self.limit + self.get_position(state) - self.sell_reserved.get(self.option, 0))

    def place_buy(self, state: TradingState, price: int, quantity: int, msg: str) -> None:
        qty = min(max(0, int(quantity)), self.remaining_buy(state))
        if qty <= 0:
            return
        self.orders[self.option].append(Order(self.option, int(price), qty))
        self.buy_reserved[self.option] += qty
        logger.print(msg)

    def place_sell(self, state: TradingState, price: int, quantity: int, msg: str) -> None:
        qty = min(max(0, int(quantity)), self.remaining_sell(state))
        if qty <= 0:
            return
        self.orders[self.option].append(Order(self.option, int(price), -qty))
        self.sell_reserved[self.option] += qty
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

    def option_signal(self, order_depth: OrderDepth) -> Dict[str, Any]:
        best_bid, best_ask = self.best_bid_ask(order_depth)
        mid = self.wall_mid(order_depth)
        spread = float(best_ask - best_bid) if best_bid is not None and best_ask is not None else 8.0

        bid_levels = sorted(order_depth.buy_orders.items(), reverse=True)
        ask_levels = sorted(order_depth.sell_orders.items())
        bid_total = 0
        ask_total = 0
        for level in range(1, 4):
            bid_total += abs(int(bid_levels[level - 1][1])) if len(bid_levels) >= level else 0
            ask_total += abs(int(ask_levels[level - 1][1])) if len(ask_levels) >= level else 0

        best_bid_volume = abs(int(order_depth.buy_orders.get(best_bid, 0))) if best_bid is not None else 0
        best_ask_volume = abs(int(order_depth.sell_orders.get(best_ask, 0))) if best_ask is not None else 0
        depth_total = bid_total + ask_total
        l1_total = best_bid_volume + best_ask_volume
        imbalance = (bid_total - ask_total) / depth_total if depth_total > 0 else 0.0
        l1_imbalance = (best_bid_volume - best_ask_volume) / l1_total if l1_total > 0 else 0.0
        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid": mid,
            "spread": spread,
            "imbalance": imbalance,
            "l1_imbalance": l1_imbalance,
            "depth_total": depth_total,
            "best_bid_volume": best_bid_volume,
            "best_ask_volume": best_ask_volume,
        }

    def regime_scale(self, depth_total: float) -> float:
        if depth_total >= self.depth_p75:
            return 1.20
        if depth_total <= self.depth_p25:
            return 0.80
        fraction = (depth_total - self.depth_p25) / (self.depth_p75 - self.depth_p25)
        return 0.80 + 0.40 * fraction

    def update_vv_anchor(self, memory: Dict[str, Any], timestamp: int, vv_mid: float) -> None:
        last_timestamp = memory["last_timestamp"]
        if last_timestamp is not None and int(timestamp) < int(last_timestamp):
            if int(memory["day_count"]) > 0:
                memory["vv_anchor"] = float(memory["day_sum"]) / float(memory["day_count"]) + self.day_shift
            memory["day_sum"] = 0.0
            memory["day_count"] = 0
            memory["prev_vv_mid"] = None

        memory["day_sum"] = float(memory["day_sum"]) + vv_mid
        memory["day_count"] = int(memory["day_count"]) + 1
        memory["last_timestamp"] = int(timestamp)

    def trade_option(self, state: TradingState, memory: Dict[str, Any]) -> None:
        if self.option not in state.order_depths or self.underlying not in state.order_depths:
            return

        vv_mid = self.wall_mid(state.order_depths[self.underlying])
        signal = self.option_signal(state.order_depths[self.option])
        option_mid = signal["mid"]
        best_bid = signal["best_bid"]
        best_ask = signal["best_ask"]
        if vv_mid is None or option_mid is None or best_bid is None or best_ask is None:
            return

        self.update_vv_anchor(memory, state.timestamp, vv_mid)

        prev_vv_mid = memory["prev_vv_mid"]
        vv_move = 0.0 if prev_vv_mid is None else vv_mid - float(prev_vv_mid)
        memory["prev_vv_mid"] = vv_mid

        vv_anchor = float(memory["vv_anchor"])
        fair = vv_anchor - self.synthetic_strike_gap
        fair -= self.ret_weight * vv_move

        spread = float(signal["spread"])
        depth_total = float(signal["depth_total"])
        if depth_total < self.depth_min or spread > self.spread_cap:
            logger.print(
                f"VEV4500 skip depth={depth_total:.0f} spread={spread:.1f} fair={fair:.2f} mid={option_mid:.2f}"
            )
            return

        regime_scale = self.regime_scale(depth_total)
        edge = fair - option_mid
        edge += regime_scale * spread * (
            -self.book_weight * float(signal["imbalance"])
            + self.l1_weight * float(signal["l1_imbalance"])
        )

        position = self.get_position(state)
        quote_fair = fair - self.inventory_skew * position
        take_threshold = max(6.0, 0.45 * spread / max(0.90, regime_scale))
        passive_offset = max(1.0, 0.35 * spread / max(0.90, regime_scale))
        quote_size = max(6, int(round(self.base_quote_size * regime_scale)))
        if abs(position) > 0.60 * self.limit:
            quote_size = max(4, quote_size // 2)

        if edge >= take_threshold:
            self.place_buy(
                state,
                best_ask,
                min(int(signal["best_ask_volume"]), self.take_size),
                f"VEV4500 TAKE BUY edge={edge:.2f} fair={fair:.2f} vv_anchor={vv_anchor:.2f}",
            )
        if edge <= -take_threshold:
            self.place_sell(
                state,
                best_bid,
                min(int(signal["best_bid_volume"]), self.take_size),
                f"VEV4500 TAKE SELL edge={edge:.2f} fair={fair:.2f} vv_anchor={vv_anchor:.2f}",
            )

        buy_price = min(best_bid + 1, int(math.floor(quote_fair - passive_offset)))
        sell_price = max(best_ask - 1, int(math.ceil(quote_fair + passive_offset)))
        if buy_price >= sell_price:
            buy_price = min(best_bid, sell_price - 1)

        if edge > take_threshold:
            self.place_buy(
                state,
                buy_price,
                quote_size,
                f"VEV4500 MM BUY q={quote_size} fair={fair:.2f} vv_anchor={vv_anchor:.2f}",
            )
        if edge < -take_threshold:
            self.place_sell(
                state,
                sell_price,
                quote_size,
                f"VEV4500 MM SELL q={quote_size} fair={fair:.2f} vv_anchor={vv_anchor:.2f}",
            )

        logger.print(
            f"vv_mid={vv_mid:.2f} vv_anchor={vv_anchor:.2f} vev4500_mid={option_mid:.2f} fair={fair:.2f} "
            f"edge={edge:.2f} gap={vv_mid - option_mid:.2f} depth={depth_total:.0f} pos={position}"
        )

    def run(self, state: TradingState):
        self.reset_orders(state)
        memory = self.parse_memory(state.traderData)
        self.trade_option(state, memory)
        trader_data = self.encode_memory(memory)
        logger.flush(state, self.orders, self.conversions, trader_data)
        return self.orders, self.conversions, trader_data
