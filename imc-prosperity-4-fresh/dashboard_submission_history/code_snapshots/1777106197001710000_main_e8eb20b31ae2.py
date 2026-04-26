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
        self.hydro = {
            "product": "HYDROGEL_PACK",
            "limit": 200,
            "initial_day_mean": 9991.0,
            "ret_weight": 0.18,
            "total_imb_weight": 1.30,
            "inventory_skew": 0.80,
            "base_quote_size": 16,
            "depth_p25": 70.0,
            "depth_p75": 80.0,
            "early_ema_alpha": 0.05,
            "late_ema_alpha": 0.16,
            "early_take_mult": 0.80,
            "late_take_mult": 0.35,
            "early_passive_mult": 0.65,
            "late_passive_mult": 0.28,
        }
        self.velvet = {
            "product": "VELVETFRUIT_EXTRACT",
            "limit": 200,
            "initial_day_mean": 5266.0,
            "day_shift": 20.0,
            "ret_weight": 0.30,
            "total_imb_weight": 1.00,
            "l1_imb_weight": 0.85,
            "inventory_skew": 0.85,
            "base_quote_size": 32,
            "depth_p25": 110.0,
            "depth_p75": 132.0,
        }

        self.orders: Dict[Symbol, list[Order]] = {}
        self.buy_reserved: Dict[Symbol, int] = {}
        self.sell_reserved: Dict[Symbol, int] = {}
        self.conversions = 0
        self.traderData = "R3_MAIN_HYDRO_TIME_DECAY_VELVET_PREVDAY_PLUS20_V1"

    def parse_memory(self, trader_data: str) -> Dict[str, Any]:
        memory = {
            "hydro": {
                "prev_day_anchor": self.hydro["initial_day_mean"],
                "day_sum": 0.0,
                "day_count": 0,
                "last_timestamp": None,
                "ema": self.hydro["initial_day_mean"],
                "prev_mid": None,
            },
            "velvet": {
                "anchor": self.velvet["initial_day_mean"],
                "day_sum": 0.0,
                "day_count": 0,
                "prev_mid": None,
                "last_timestamp": None,
            },
        }
        if not trader_data:
            return memory
        try:
            raw = json.loads(trader_data)
        except Exception:
            return memory

        hydro_raw = raw.get("hydro", {})
        velvet_raw = raw.get("velvet", {})
        memory["hydro"]["prev_day_anchor"] = float(
            hydro_raw.get("prev_day_anchor", self.hydro["initial_day_mean"])
        )
        memory["hydro"]["day_sum"] = float(hydro_raw.get("day_sum", 0.0))
        memory["hydro"]["day_count"] = int(hydro_raw.get("day_count", 0))
        memory["hydro"]["last_timestamp"] = hydro_raw.get("last_timestamp")
        memory["hydro"]["ema"] = float(
            hydro_raw.get("ema", memory["hydro"]["prev_day_anchor"])
        )
        memory["hydro"]["prev_mid"] = hydro_raw.get("prev_mid")

        memory["velvet"]["anchor"] = float(
            velvet_raw.get("anchor", self.velvet["initial_day_mean"])
        )
        memory["velvet"]["day_sum"] = float(velvet_raw.get("day_sum", 0.0))
        memory["velvet"]["day_count"] = int(velvet_raw.get("day_count", 0))
        memory["velvet"]["prev_mid"] = velvet_raw.get("prev_mid")
        memory["velvet"]["last_timestamp"] = velvet_raw.get("last_timestamp")
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

    def remaining_buy(self, state: TradingState, product: str, limit: int) -> int:
        return max(0, limit - self.get_position(state, product) - self.buy_reserved.get(product, 0))

    def remaining_sell(self, state: TradingState, product: str, limit: int) -> int:
        return max(0, limit + self.get_position(state, product) - self.sell_reserved.get(product, 0))

    def place_buy(
        self, state: TradingState, product: str, limit: int, price: int, quantity: int, msg: str
    ) -> None:
        qty = min(max(0, int(quantity)), self.remaining_buy(state, product, limit))
        if qty <= 0:
            return
        self.orders[product].append(Order(product, int(price), qty))
        self.buy_reserved[product] += qty
        logger.print(msg)

    def place_sell(
        self, state: TradingState, product: str, limit: int, price: int, quantity: int, msg: str
    ) -> None:
        qty = min(max(0, int(quantity)), self.remaining_sell(state, product, limit))
        if qty <= 0:
            return
        self.orders[product].append(Order(product, int(price), -qty))
        self.sell_reserved[product] += qty
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

    def book_signal(self, order_depth: OrderDepth, fallback_spread: float) -> Dict[str, Any]:
        best_bid, best_ask = self.best_bid_ask(order_depth)
        mid = self.wall_mid(order_depth)
        spread = (
            float(best_ask - best_bid)
            if best_bid is not None and best_ask is not None
            else fallback_spread
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

    def regime_scale(self, depth_total: float, depth_p25: float, depth_p75: float, high: float = 1.20) -> float:
        if depth_total >= depth_p75:
            return high
        if depth_total <= depth_p25:
            return 0.80
        fraction = (depth_total - depth_p25) / (depth_p75 - depth_p25)
        return 0.80 + (high - 0.80) * fraction

    def intraday_progress(self, timestamp: int) -> float:
        return min(1.0, max(0.0, float(timestamp) / 1_000_000.0))

    def update_hydro_day_state(self, memory: Dict[str, Any], timestamp: int, mid: float) -> None:
        last_timestamp = memory["last_timestamp"]
        if last_timestamp is not None and int(timestamp) < int(last_timestamp):
            if int(memory["day_count"]) > 0:
                memory["prev_day_anchor"] = float(memory["day_sum"]) / float(memory["day_count"])
            memory["day_sum"] = 0.0
            memory["day_count"] = 0
            memory["prev_mid"] = None
            memory["ema"] = float(memory["prev_day_anchor"])
        memory["day_sum"] = float(memory["day_sum"]) + mid
        memory["day_count"] = int(memory["day_count"]) + 1
        memory["last_timestamp"] = int(timestamp)

    def update_velvet_day_anchor(self, memory: Dict[str, Any], timestamp: int, mid: float) -> None:
        last_timestamp = memory["last_timestamp"]
        if last_timestamp is not None and int(timestamp) < int(last_timestamp):
            if int(memory["day_count"]) > 0:
                memory["anchor"] = float(memory["day_sum"]) / float(memory["day_count"]) + float(self.velvet["day_shift"])
            memory["day_sum"] = 0.0
            memory["day_count"] = 0
            memory["prev_mid"] = None
        memory["day_sum"] = float(memory["day_sum"]) + mid
        memory["day_count"] = int(memory["day_count"]) + 1
        memory["last_timestamp"] = int(timestamp)

    def trade_hydrogel(self, state: TradingState, memory: Dict[str, Any]) -> None:
        product = str(self.hydro["product"])
        limit = int(self.hydro["limit"])
        if product not in state.order_depths:
            return

        signal = self.book_signal(state.order_depths[product], 16.0)
        mid = signal["mid"]
        best_bid = signal["best_bid"]
        best_ask = signal["best_ask"]
        if mid is None or best_bid is None or best_ask is None:
            return

        self.update_hydro_day_state(memory, state.timestamp, mid)
        progress = self.intraday_progress(state.timestamp)
        effective_ema_alpha = (1.0 - progress) * float(self.hydro["early_ema_alpha"]) + progress * float(self.hydro["late_ema_alpha"])

        prev_mid = memory["prev_mid"]
        recent_move = 0.0 if prev_mid is None else mid - float(prev_mid)
        memory["prev_mid"] = mid

        previous_ema = float(memory["ema"])
        ema = effective_ema_alpha * mid + (1.0 - effective_ema_alpha) * previous_ema
        memory["ema"] = ema

        depth_total = float(signal["depth_total"])
        regime_scale = self.regime_scale(
            depth_total, float(self.hydro["depth_p25"]), float(self.hydro["depth_p75"]), 1.25
        )
        thick_book = depth_total >= float(self.hydro["depth_p75"])
        anchor = ema if thick_book else float(memory["prev_day_anchor"])

        fair = anchor
        if thick_book:
            fair -= regime_scale * float(self.hydro["ret_weight"]) * recent_move
        fair += regime_scale * float(signal["spread"]) * (
            -float(self.hydro["total_imb_weight"]) * float(signal["imbalance"])
        )

        position = self.get_position(state, product)
        position_ratio = position / float(limit)
        quote_fair = fair - float(self.hydro["inventory_skew"]) * position_ratio * float(signal["spread"])

        take_mult = (1.0 - progress) * float(self.hydro["early_take_mult"]) + progress * float(self.hydro["late_take_mult"])
        passive_mult = (1.0 - progress) * float(self.hydro["early_passive_mult"]) + progress * float(self.hydro["late_passive_mult"])
        take_threshold = max(2.0, take_mult * float(signal["spread"]) / max(0.90, regime_scale))
        passive_offset = max(2.0, passive_mult * float(signal["spread"]) / max(0.90, regime_scale))

        take_size = max(6, int(round((12 if thick_book else 8) * regime_scale)))
        quote_size = max(6, int(round(float(self.hydro["base_quote_size"]) * regime_scale)))
        if abs(position) > 0.60 * limit:
            quote_size = max(4, quote_size // 2)

        if best_ask <= fair - take_threshold:
            self.place_buy(
                state, product, limit, best_ask,
                min(int(signal["best_ask_volume"]), take_size),
                f"HYDRO TAKE BUY @ {best_ask} fair={fair:.2f} anchor={anchor:.2f}",
            )
        if best_bid >= fair + take_threshold:
            self.place_sell(
                state, product, limit, best_bid,
                min(int(signal["best_bid_volume"]), take_size),
                f"HYDRO TAKE SELL @ {best_bid} fair={fair:.2f} anchor={anchor:.2f}",
            )

        buy_price = min(best_bid + 1, int(math.floor(quote_fair - passive_offset)))
        sell_price = max(best_ask - 1, int(math.ceil(quote_fair + passive_offset)))
        if buy_price >= sell_price:
            buy_price = min(best_bid, sell_price - 1)

        self.place_buy(
            state, product, limit, buy_price, quote_size,
            f"HYDRO MM BUY {quote_size} @ {buy_price} fair={fair:.2f} anchor={anchor:.2f}",
        )
        self.place_sell(
            state, product, limit, sell_price, quote_size,
            f"HYDRO MM SELL {quote_size} @ {sell_price} fair={fair:.2f} anchor={anchor:.2f}",
        )

        mode = "ema-thick" if thick_book else "prevday-thin"
        logger.print(
            f"hydro_mode={mode} mid={mid:.2f} anchor={anchor:.2f} ema={ema:.2f} "
            f"ret={recent_move:.2f} imb={float(signal['imbalance']):.3f} depth={depth_total:.0f} "
            f"pos={position} progress={progress:.3f}"
        )

    def trade_velvet(self, state: TradingState, memory: Dict[str, Any]) -> None:
        product = str(self.velvet["product"])
        limit = int(self.velvet["limit"])
        if product not in state.order_depths:
            return

        signal = self.book_signal(state.order_depths[product], 6.0)
        mid = signal["mid"]
        best_bid = signal["best_bid"]
        best_ask = signal["best_ask"]
        if mid is None or best_bid is None or best_ask is None:
            return

        self.update_velvet_day_anchor(memory, state.timestamp, mid)

        prev_mid = memory["prev_mid"]
        recent_move = 0.0 if prev_mid is None else mid - float(prev_mid)
        memory["prev_mid"] = mid

        anchor = float(memory["anchor"])
        regime_scale = self.regime_scale(
            float(signal["depth_total"]),
            float(self.velvet["depth_p25"]),
            float(self.velvet["depth_p75"]),
            1.20,
        )
        fair = anchor
        fair -= regime_scale * float(self.velvet["ret_weight"]) * recent_move
        fair += regime_scale * float(signal["spread"]) * (
            -float(self.velvet["total_imb_weight"]) * float(signal["imbalance"])
            + float(self.velvet["l1_imb_weight"]) * float(signal["l1_imbalance"])
        )

        position = self.get_position(state, product)
        position_ratio = position / float(limit)
        quote_fair = fair - float(self.velvet["inventory_skew"]) * position_ratio * float(signal["spread"])

        take_threshold = max(1.0, 0.55 * float(signal["spread"]) / max(0.85, regime_scale))
        passive_offset = max(1.0, 0.45 * float(signal["spread"]) / max(0.85, regime_scale))

        if best_ask <= fair - take_threshold:
            self.place_buy(
                state, product, limit, best_ask,
                min(int(signal["best_ask_volume"]), max(8, int(round(18 * regime_scale)))),
                f"VELVET TAKE BUY @ {best_ask} fair={fair:.2f} anchor={anchor:.2f}",
            )
        if best_bid >= fair + take_threshold:
            self.place_sell(
                state, product, limit, best_bid,
                min(int(signal["best_bid_volume"]), max(8, int(round(18 * regime_scale)))),
                f"VELVET TAKE SELL @ {best_bid} fair={fair:.2f} anchor={anchor:.2f}",
            )

        buy_price = min(best_bid + 1, int(math.floor(quote_fair - passive_offset)))
        sell_price = max(best_ask - 1, int(math.ceil(quote_fair + passive_offset)))
        if buy_price >= sell_price:
            buy_price = min(best_bid, sell_price - 1)

        quote_size = max(8, int(round(float(self.velvet["base_quote_size"]) * regime_scale)))
        if abs(position) > 0.60 * limit:
            quote_size = max(4, quote_size // 2)

        self.place_buy(
            state, product, limit, buy_price, quote_size,
            f"VELVET MM BUY {quote_size} @ {buy_price} fair={fair:.2f} anchor={anchor:.2f}",
        )
        self.place_sell(
            state, product, limit, sell_price, quote_size,
            f"VELVET MM SELL {quote_size} @ {sell_price} fair={fair:.2f} anchor={anchor:.2f}",
        )

        logger.print(
            f"velvet mid={mid:.2f} anchor={anchor:.2f} day_count={int(memory['day_count'])} "
            f"ret={recent_move:.2f} imb={float(signal['imbalance']):.3f} "
            f"l1={float(signal['l1_imbalance']):.3f} pos={position}"
        )

    def run(self, state: TradingState):
        self.reset_orders(state)
        memory = self.parse_memory(state.traderData)
        self.trade_hydrogel(state, memory["hydro"])
        self.trade_velvet(state, memory["velvet"])
        trader_data = self.encode_memory(memory)
        logger.flush(state, self.orders, self.conversions, trader_data)
        return self.orders, self.conversions, trader_data
