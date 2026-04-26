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
        self.product = "HYDROGEL_PACK"
        self.limit = 200

        self.initial_day_mean = 9991.0
        self.fast_ema_alpha = 0.12
        self.ret_weight = 0.18
        self.total_imb_weight = 1.30
        self.inventory_skew = 0.80
        self.base_quote_size = 16
        self.depth_p25 = 70.0
        self.depth_p75 = 80.0

        self.orders: Dict[Symbol, list[Order]] = {}
        self.buy_reserved: Dict[Symbol, int] = {}
        self.sell_reserved: Dict[Symbol, int] = {}
        self.conversions = 0
        self.trade_alpha_weight = 0.18
        self.traderData = "R3_HYDROGEL_HYBRID_DEPTH_ANCHOR_TAKER_V1"

    def parse_memory(self, trader_data: str) -> Dict[str, Any]:
        memory = {
            "prev_day_anchor": self.initial_day_mean,
            "day_sum": 0.0,
            "day_count": 0,
            "last_timestamp": None,
            "ema": self.initial_day_mean,
            "prev_mid": None,
            "recent_mids": [],
        }
        if not trader_data:
            return memory
        try:
            raw = json.loads(trader_data)
        except Exception:
            return memory
        memory["prev_day_anchor"] = float(raw.get("prev_day_anchor", self.initial_day_mean))
        memory["day_sum"] = float(raw.get("day_sum", 0.0))
        memory["day_count"] = int(raw.get("day_count", 0))
        memory["last_timestamp"] = raw.get("last_timestamp")
        memory["ema"] = float(raw.get("ema", memory["prev_day_anchor"]))
        memory["prev_mid"] = raw.get("prev_mid")
        mids = raw.get("recent_mids", [])
        if isinstance(mids, list):
            memory["recent_mids"] = [float(x) for x in mids[-3:]]
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
        return max(
            0,
            self.limit - self.get_position(state) - self.buy_reserved.get(self.product, 0),
        )

    def remaining_sell(self, state: TradingState) -> int:
        return max(
            0,
            self.limit + self.get_position(state) - self.sell_reserved.get(self.product, 0),
        )

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
        spread = (
            float(best_ask - best_bid)
            if best_bid is not None and best_ask is not None
            else 16.0
        )

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
        imbalance = (bid_total - ask_total) / depth_total if depth_total > 0 else 0.0

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid": mid,
            "spread": spread,
            "imbalance": imbalance,
            "depth_total": depth_total,
            "best_bid_volume": best_bid_volume,
            "best_ask_volume": best_ask_volume,
        }

    def regime_scale(self, depth_total: float) -> float:
        if depth_total >= self.depth_p75:
            return 1.25
        if depth_total <= self.depth_p25:
            return 0.80
        fraction = (depth_total - self.depth_p25) / (self.depth_p75 - self.depth_p25)
        return 0.80 + 0.45 * fraction

    def trade_flow_signal(self, state: TradingState, signal: Dict[str, Any]) -> Dict[str, float]:
        best_bid = signal["best_bid"]
        best_ask = signal["best_ask"]
        if best_bid is None or best_ask is None:
            return {"buy_small": 0.0, "sell_small": 0.0, "score": 0.0}

        buy_small = 0.0
        sell_small = 0.0
        for trade in state.market_trades.get(self.product, []):
            qty = abs(int(trade.quantity))
            if qty <= 0:
                continue
            if qty >= 6:
                continue
            price = float(trade.price)
            if price >= best_ask:
                buy_small += qty
            elif price <= best_bid:
                sell_small += qty
            else:
                if abs(price - best_ask) < abs(price - best_bid):
                    buy_small += qty
                else:
                    sell_small += qty

        score = max(-8.0, min(8.0, buy_small - sell_small))
        return {
            "buy_small": buy_small,
            "sell_small": sell_small,
            "score": score,
        }

    def update_day_state(self, memory: Dict[str, Any], timestamp: int, mid: float) -> None:
        last_timestamp = memory["last_timestamp"]
        if last_timestamp is not None and int(timestamp) < int(last_timestamp):
            if int(memory["day_count"]) > 0:
                memory["prev_day_anchor"] = float(memory["day_sum"]) / float(memory["day_count"])
            memory["day_sum"] = 0.0
            memory["day_count"] = 0
            memory["prev_mid"] = None
            memory["ema"] = float(memory["prev_day_anchor"])
            memory["recent_mids"] = []

        memory["day_sum"] = float(memory["day_sum"]) + mid
        memory["day_count"] = int(memory["day_count"]) + 1
        memory["last_timestamp"] = int(timestamp)

    def update_recent_mids(self, memory: Dict[str, Any], mid: float) -> None:
        mids = list(memory.get("recent_mids", []))
        mids.append(float(mid))
        memory["recent_mids"] = mids[-3:]

    def three_trade_reversal_signal(self, memory: Dict[str, Any]) -> int:
        mids = memory.get("recent_mids", [])
        if len(mids) < 3:
            return 0
        a, b, c = float(mids[-3]), float(mids[-2]), float(mids[-1])
        if a > b < c:
            return 1
        if a < b > c:
            return -1
        return 0

    def trade_hydrogel(self, state: TradingState, memory: Dict[str, Any]) -> None:
        if self.product not in state.order_depths:
            return

        signal = self.book_signal(state.order_depths[self.product])
        mid = signal["mid"]
        best_bid = signal["best_bid"]
        best_ask = signal["best_ask"]
        if mid is None or best_bid is None or best_ask is None:
            return

        self.update_day_state(memory, state.timestamp, mid)

        prev_mid = memory["prev_mid"]
        recent_move = 0.0 if prev_mid is None else mid - float(prev_mid)
        memory["prev_mid"] = mid

        previous_ema = float(memory["ema"])
        ema = self.fast_ema_alpha * mid + (1.0 - self.fast_ema_alpha) * previous_ema
        memory["ema"] = ema
        self.update_recent_mids(memory, mid)

        depth_total = float(signal["depth_total"])
        regime_scale = self.regime_scale(depth_total)
        thick_book = depth_total >= self.depth_p75
        anchor = ema if thick_book else float(memory["prev_day_anchor"])
        trade_flow = self.trade_flow_signal(state, signal)
        reversal_signal = self.three_trade_reversal_signal(memory)

        fair = anchor
        if thick_book:
            fair -= regime_scale * self.ret_weight * recent_move
            fair += self.trade_alpha_weight * float(trade_flow["score"])
        fair += regime_scale * float(signal["spread"]) * (
            -self.total_imb_weight * float(signal["imbalance"])
        )

        position = self.get_position(state)
        position_ratio = position / float(self.limit)
        quote_fair = fair - self.inventory_skew * position_ratio * float(signal["spread"])

        take_threshold = max(2.0, 0.45 * float(signal["spread"]) / max(0.90, regime_scale))
        passive_offset = max(2.0, 0.40 * float(signal["spread"]) / max(0.90, regime_scale))

        take_size = max(6, int(round((12 if thick_book else 8) * regime_scale)))
        quote_size = max(6, int(round(self.base_quote_size * regime_scale)))
        if abs(position) > 0.60 * self.limit:
            quote_size = max(4, quote_size // 2)

        if best_ask <= fair - take_threshold:
            self.place_buy(
                state,
                best_ask,
                min(int(signal["best_ask_volume"]), take_size),
                f"HYDRO TAKE BUY @ {best_ask} fair={fair:.2f} anchor={anchor:.2f}",
            )
        if best_bid >= fair + take_threshold:
            self.place_sell(
                state,
                best_bid,
                min(int(signal["best_bid_volume"]), take_size),
                f"HYDRO TAKE SELL @ {best_bid} fair={fair:.2f} anchor={anchor:.2f}",
            )

        buy_price = min(best_bid + 1, int(math.floor(quote_fair - passive_offset)))
        sell_price = max(best_ask - 1, int(math.ceil(quote_fair + passive_offset)))
        if buy_price >= sell_price:
            buy_price = min(best_bid, sell_price - 1)

        self.place_buy(
            state,
            buy_price,
            quote_size,
            f"HYDRO MM BUY {quote_size} @ {buy_price} fair={fair:.2f} anchor={anchor:.2f}",
        )
        self.place_sell(
            state,
            sell_price,
            quote_size,
            f"HYDRO MM SELL {quote_size} @ {sell_price} fair={fair:.2f} anchor={anchor:.2f}",
        )

        reversal_room = 6
        if reversal_signal > 0 and best_ask <= math.floor(fair):
            self.place_buy(
                state,
                best_ask,
                reversal_room,
                f"HYDRO REVERSAL BUY @ {best_ask} fair={fair:.2f} mids={memory['recent_mids']}",
            )
        elif reversal_signal < 0 and best_bid >= math.ceil(fair):
            self.place_sell(
                state,
                best_bid,
                reversal_room,
                f"HYDRO REVERSAL SELL @ {best_bid} fair={fair:.2f} mids={memory['recent_mids']}",
            )

        mode = "ema-thick" if thick_book else "prevday-thin"
        logger.print(
            f"mode={mode} mid={mid:.2f} anchor={anchor:.2f} ema={ema:.2f} "
            f"ret={recent_move:.2f} imb={float(signal['imbalance']):.3f} "
            f"depth={depth_total:.0f} tf={float(trade_flow['score']):.1f} "
            f"rev={reversal_signal} "
            f"bs={float(trade_flow['buy_small']):.0f} ss={float(trade_flow['sell_small']):.0f} "
            f"pos={position}"
        )

    def run(self, state: TradingState):
        self.reset_orders(state)
        memory = self.parse_memory(state.traderData)
        self.trade_hydrogel(state, memory)
        trader_data = self.encode_memory(memory)
        logger.flush(state, self.orders, self.conversions, trader_data)
        return self.orders, self.conversions, trader_data
