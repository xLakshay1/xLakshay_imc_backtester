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
        self.product = "VELVETFRUIT_EXTRACT"
        self.limit = 200

        self.base_mean = 5260.0
        self.anchor_min = 5200.0
        self.anchor_max = 5320.0

        # Very strict regime-shift logic:
        # the anchor only moves after both a slow and a fast EMA sit away from 5260
        # in the same direction for many consecutive ticks.
        self.fast_alpha = 0.10
        self.slow_alpha = 0.01
        self.shift_fast_gap = 10.0
        self.shift_slow_gap = 18.0
        self.shift_confirm_ticks = 45
        self.shift_step_cap = 0.80
        self.shift_nudge = 0.03

        # Delta-1 microstructure signal from the Round 3 analysis.
        self.ret_weight = 0.30
        self.total_imb_weight = 1.00
        self.l1_imb_weight = 0.85
        self.inventory_skew = 0.85
        self.base_quote_size = 32
        self.depth_p25 = 110.0
        self.depth_p75 = 132.0

        self.orders: Dict[Symbol, list[Order]] = {}
        self.buy_reserved: Dict[Symbol, int] = {}
        self.sell_reserved: Dict[Symbol, int] = {}
        self.conversions = 0
        self.traderData = "R3_VELVET_ONLY_STRICT_MEAN_V1"

    def clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def parse_memory(self, trader_data: str) -> Dict[str, Any]:
        memory = {
            "anchor": self.base_mean,
            "fast_ema": None,
            "slow_ema": None,
            "prev_mid": None,
            "confirm_count": 0,
        }
        if not trader_data:
            return memory
        try:
            raw = json.loads(trader_data)
        except Exception:
            return memory

        memory["anchor"] = float(raw.get("anchor", self.base_mean))
        memory["fast_ema"] = raw.get("fast_ema")
        memory["slow_ema"] = raw.get("slow_ema")
        memory["prev_mid"] = raw.get("prev_mid")
        memory["confirm_count"] = int(raw.get("confirm_count", 0))
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

    def remaining_buy(self, state: TradingState, product: str) -> int:
        return max(
            0,
            self.limit - self.get_position(state, product) - self.buy_reserved.get(product, 0),
        )

    def remaining_sell(self, state: TradingState, product: str) -> int:
        return max(
            0,
            self.limit + self.get_position(state, product) - self.sell_reserved.get(product, 0),
        )

    def place_buy(self, state: TradingState, price: int, quantity: int, msg: str) -> None:
        qty = min(max(0, int(quantity)), self.remaining_buy(state, self.product))
        if qty <= 0:
            return
        self.orders[self.product].append(Order(self.product, int(price), qty))
        self.buy_reserved[self.product] += qty
        logger.print(msg)

    def place_sell(self, state: TradingState, price: int, quantity: int, msg: str) -> None:
        qty = min(max(0, int(quantity)), self.remaining_sell(state, self.product))
        if qty <= 0:
            return
        self.orders[self.product].append(Order(self.product, int(price), -qty))
        self.sell_reserved[self.product] += qty
        logger.print(msg)

    def ordered_bid_levels(self, order_depth: OrderDepth) -> list[tuple[int, int]]:
        return sorted(order_depth.buy_orders.items(), reverse=True)

    def ordered_ask_levels(self, order_depth: OrderDepth) -> list[tuple[int, int]]:
        return sorted(order_depth.sell_orders.items())

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

    def book_signal(self, order_depth: OrderDepth) -> Dict[str, Any]:
        best_bid, best_ask = self.best_bid_ask(order_depth)
        mid = self.wall_mid(order_depth)
        spread = float(best_ask - best_bid) if best_bid is not None and best_ask is not None else 6.0

        bid_levels = self.ordered_bid_levels(order_depth)
        ask_levels = self.ordered_ask_levels(order_depth)
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

    def update_anchor(self, memory: Dict[str, Any], mid: float, spread: float) -> None:
        fast_ema = mid if memory["fast_ema"] is None else self.fast_alpha * mid + (1 - self.fast_alpha) * float(memory["fast_ema"])
        slow_ema = mid if memory["slow_ema"] is None else self.slow_alpha * mid + (1 - self.slow_alpha) * float(memory["slow_ema"])
        anchor = float(memory["anchor"])

        fast_gap = fast_ema - anchor
        slow_gap = slow_ema - anchor
        same_direction = fast_gap * slow_gap > 0
        stable_shift = (
            same_direction
            and abs(fast_gap) >= self.shift_fast_gap
            and abs(slow_gap) >= self.shift_slow_gap
            and spread <= 6.0
        )

        if stable_shift:
            memory["confirm_count"] = int(memory["confirm_count"]) + 1
        else:
            memory["confirm_count"] = max(0, int(memory["confirm_count"]) - 2)

        if int(memory["confirm_count"]) >= self.shift_confirm_ticks:
            nudge = self.clamp(self.shift_nudge * slow_gap, -self.shift_step_cap, self.shift_step_cap)
            anchor = self.clamp(anchor + nudge, self.anchor_min, self.anchor_max)

        memory["anchor"] = anchor
        memory["fast_ema"] = fast_ema
        memory["slow_ema"] = slow_ema

    def trade_velvet(self, state: TradingState, memory: Dict[str, Any]) -> None:
        if self.product not in state.order_depths:
            return

        signal = self.book_signal(state.order_depths[self.product])
        mid = signal["mid"]
        best_bid = signal["best_bid"]
        best_ask = signal["best_ask"]
        if mid is None or best_bid is None or best_ask is None:
            return

        self.update_anchor(memory, mid, float(signal["spread"]))

        prev_mid = memory["prev_mid"]
        recent_move = 0.0 if prev_mid is None else mid - float(prev_mid)
        memory["prev_mid"] = mid

        anchor = float(memory["anchor"])
        regime_scale = self.regime_scale(float(signal["depth_total"]))
        fair = anchor
        fair -= regime_scale * self.ret_weight * recent_move
        fair += regime_scale * float(signal["spread"]) * (
            -self.total_imb_weight * float(signal["imbalance"])
            + self.l1_imb_weight * float(signal["l1_imbalance"])
        )

        position = self.get_position(state, self.product)
        position_ratio = position / float(self.limit)
        quote_fair = fair - self.inventory_skew * position_ratio * float(signal["spread"])

        take_threshold = max(1.0, 0.55 * float(signal["spread"]) / max(0.85, regime_scale))
        passive_offset = max(1.0, 0.45 * float(signal["spread"]) / max(0.85, regime_scale))

        if best_ask <= fair - take_threshold:
            self.place_buy(
                state,
                best_ask,
                min(int(signal["best_ask_volume"]), max(8, int(round(18 * regime_scale)))),
                f"VELVET TAKE BUY @ {best_ask} fair={fair:.2f} anchor={anchor:.2f}",
            )
        if best_bid >= fair + take_threshold:
            self.place_sell(
                state,
                best_bid,
                min(int(signal["best_bid_volume"]), max(8, int(round(18 * regime_scale)))),
                f"VELVET TAKE SELL @ {best_bid} fair={fair:.2f} anchor={anchor:.2f}",
            )

        buy_price = min(best_bid + 1, int(math.floor(quote_fair - passive_offset)))
        sell_price = max(best_ask - 1, int(math.ceil(quote_fair + passive_offset)))
        if buy_price >= sell_price:
            buy_price = min(best_bid, sell_price - 1)

        quote_size = max(8, int(round(self.base_quote_size * regime_scale)))
        if abs(position) > 0.60 * self.limit:
            quote_size = max(4, quote_size // 2)

        self.place_buy(
            state,
            buy_price,
            quote_size,
            f"VELVET MM BUY {quote_size} @ {buy_price} fair={fair:.2f} anchor={anchor:.2f}",
        )
        self.place_sell(
            state,
            sell_price,
            quote_size,
            f"VELVET MM SELL {quote_size} @ {sell_price} fair={fair:.2f} anchor={anchor:.2f}",
        )

        logger.print(
            f"mid={mid:.2f} anchor={anchor:.2f} fast={float(memory['fast_ema']):.2f} "
            f"slow={float(memory['slow_ema']):.2f} confirm={int(memory['confirm_count'])} "
            f"ret={recent_move:.2f} imb={float(signal['imbalance']):.3f} "
            f"l1={float(signal['l1_imbalance']):.3f} pos={position}"
        )

    def run(self, state: TradingState):
        self.reset_orders(state)
        memory = self.parse_memory(state.traderData)

        self.trade_velvet(state, memory)

        self.traderData = self.encode_memory(memory)
        logger.flush(state, self.orders, self.conversions, self.traderData)
        return self.orders, self.conversions, self.traderData
