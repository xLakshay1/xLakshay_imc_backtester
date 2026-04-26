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
            "ASH_COATED_OSMIUM": 80,
            "INTARIAN_PEPPER_ROOT": 80,
        }

        self.orders: dict[Symbol, list[Order]] = {}
        self.conversions = 0
        self.traderData = "CAND_NOMD_TOP05_EMA005"

        self.osmium_position = 0
        self.osmium_buy_orders = 0
        self.osmium_sell_orders = 0

        self.pepper_position = 0
        self.pepper_buy_orders = 0
        self.pepper_sell_orders = 0

        # Osmium market-making state plus live-safe alpha signals.
        self.osmium_fair_value = None
        self.osmium_quote_size = 20
        self.osmium_prev_mid = None
        self.osmium_mid_history: list[float] = []
        self.osmium_depth_vwap_history: list[float] = []
        self.osmium_depth_vwap_live_trend = None
        self.osmium_depth_vwap_median_window = 90
        self.osmium_depth_vwap_min_points = 15
        self.osmium_depth_vwap_ema_alpha = 2 / 81

        # Conservative weights from Round 2 alpha tests. Keep the total adjustment clamped,
        # because the signals predict direction but are smaller than the spread.
        self.osmium_micro_weight = 0.00
        self.osmium_depth_trend_weight = 0.00
        self.osmium_top_imbalance_weight = 0.50
        self.osmium_recent_move_weight = 0.25
        self.osmium_ret5_weight = 0.08
        self.osmium_ret5_clip = 10.0
        self.osmium_liquidity_pressure_weight = 4.0
        self.osmium_liquidity_pressure_clip = 0.20
        self.osmium_alpha_clip = 3.0

        # Pepper: buy everything near the beginning, then hold forever.
        self.pepper_limit = 80
        self.pepper_clip_size = 80

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

    def clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def median(self, values: list[float]) -> float:
        ordered = sorted(values)
        n = len(ordered)
        if n == 0:
            return 0.0
        middle = n // 2
        if n % 2 == 1:
            return ordered[middle]
        return 0.5 * (ordered[middle - 1] + ordered[middle])

    def update_osmium_depth_vwap_live_trend(self, depth_vwap: float) -> float:
        self.osmium_depth_vwap_history.append(depth_vwap)
        max_history = self.osmium_depth_vwap_median_window
        if len(self.osmium_depth_vwap_history) > max_history:
            self.osmium_depth_vwap_history = self.osmium_depth_vwap_history[-max_history:]

        if len(self.osmium_depth_vwap_history) >= self.osmium_depth_vwap_min_points:
            median_line = self.median(self.osmium_depth_vwap_history)
        else:
            median_line = depth_vwap

        if self.osmium_depth_vwap_live_trend is None:
            self.osmium_depth_vwap_live_trend = median_line
        else:
            alpha = self.osmium_depth_vwap_ema_alpha
            self.osmium_depth_vwap_live_trend = (
                alpha * median_line + (1 - alpha) * self.osmium_depth_vwap_live_trend
            )
        return self.osmium_depth_vwap_live_trend

    def get_osmium_alpha_features(self, state: TradingState, product: str):
        order_depth = state.order_depths[product]
        best_bid, best_ask = self.get_best_bid_ask(state, product)
        if best_bid is None or best_ask is None:
            return None

        bid_volume = abs(order_depth.buy_orders.get(best_bid, 0))
        ask_volume = abs(order_depth.sell_orders.get(best_ask, 0))
        top_depth = bid_volume + ask_volume
        if top_depth <= 0:
            return None

        mid = (best_bid + best_ask) / 2.0
        microprice = (best_ask * bid_volume + best_bid * ask_volume) / top_depth
        top_imbalance = (bid_volume - ask_volume) / top_depth
        liquidity_pressure = self.clamp(
            1.0 / (ask_volume + 1.0) - 1.0 / (bid_volume + 1.0),
            -self.osmium_liquidity_pressure_clip,
            self.osmium_liquidity_pressure_clip,
        )

        total_depth = 0
        weighted_price = 0.0
        for price, volume in order_depth.buy_orders.items():
            abs_volume = abs(volume)
            if abs_volume <= 0:
                continue
            total_depth += abs_volume
            weighted_price += price * abs_volume
        for price, volume in order_depth.sell_orders.items():
            abs_volume = abs(volume)
            if abs_volume <= 0:
                continue
            total_depth += abs_volume
            weighted_price += price * abs_volume
        depth_vwap = weighted_price / total_depth if total_depth > 0 else mid
        depth_vwap_live_trend = self.update_osmium_depth_vwap_live_trend(depth_vwap)

        recent_move = 0.0 if self.osmium_prev_mid is None else mid - self.osmium_prev_mid
        self.osmium_prev_mid = mid
        ret_5 = 0.0
        if len(self.osmium_mid_history) >= 5:
            ret_5 = mid - self.osmium_mid_history[-5]
        self.osmium_mid_history.append(mid)
        if len(self.osmium_mid_history) > 20:
            self.osmium_mid_history = self.osmium_mid_history[-20:]

        return {
            "mid": mid,
            "microprice": microprice,
            "top_imbalance": top_imbalance,
            "depth_vwap": depth_vwap,
            "depth_vwap_live_trend": depth_vwap_live_trend,
            "recent_move": recent_move,
            "ret_5": ret_5,
            "liquidity_pressure": liquidity_pressure,
        }

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

            if product == "ASH_COATED_OSMIUM":
                size = min(
                    self.limits[product] - self.osmium_position - self.osmium_buy_orders,
                    -amount,
                )
                if size > 0:
                    self.osmium_buy_orders += size
                    self.send_buy_order(product, ask, size, f"TAKE BUY {size} {product} @ {ask}")
            elif product == "INTARIAN_PEPPER_ROOT":
                size = min(
                    self.limits[product] - self.pepper_position - self.pepper_buy_orders,
                    -amount,
                )
                if size > 0:
                    self.pepper_buy_orders += size
                    self.send_buy_order(product, ask, size, f"TAKE BUY {size} {product} @ {ask}")

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

            if product == "ASH_COATED_OSMIUM":
                size = min(
                    self.osmium_position + self.limits[product] - self.osmium_sell_orders,
                    amount,
                )
                if size > 0:
                    self.osmium_sell_orders += size
                    self.send_sell_order(product, bid, -size, f"TAKE SELL {-size} {product} @ {bid}")
            elif product == "INTARIAN_PEPPER_ROOT":
                size = min(
                    self.pepper_position + self.limits[product] - self.pepper_sell_orders,
                    amount,
                )
                if size > 0:
                    self.pepper_sell_orders += size
                    self.send_sell_order(product, bid, -size, f"TAKE SELL {-size} {product} @ {bid}")

    def trade_osmium(self, state: TradingState):
        product = "ASH_COATED_OSMIUM"

        best_bid, best_ask = self.get_best_bid_ask(state, product)
        if best_bid is None or best_ask is None:
            return

        position = self.get_product_pos(state, product)
        features = self.get_osmium_alpha_features(state, product)
        if features is None:
            return
        mid = features["mid"]

        if self.osmium_fair_value is None:
            self.osmium_fair_value = mid
        else:
            self.osmium_fair_value = 0.05 * mid + 0.95 * self.osmium_fair_value

        micro_signal = features["microprice"] - mid
        depth_signal = features["depth_vwap_live_trend"] - mid
        top_imbalance = features["top_imbalance"]
        recent_move = features["recent_move"]
        ret_5_signal = self.clamp(
            features["ret_5"],
            -self.osmium_ret5_clip,
            self.osmium_ret5_clip,
        )
        liquidity_pressure = features["liquidity_pressure"]
        alpha_adjustment = (
            self.osmium_micro_weight * micro_signal
            + self.osmium_depth_trend_weight * depth_signal
            + self.osmium_top_imbalance_weight * top_imbalance
            - self.osmium_recent_move_weight * recent_move
            - self.osmium_ret5_weight * ret_5_signal
            + self.osmium_liquidity_pressure_weight * liquidity_pressure
        )
        alpha_adjustment = self.clamp(
            alpha_adjustment,
            -self.osmium_alpha_clip,
            self.osmium_alpha_clip,
        )

        take_fair_value = self.osmium_fair_value
        quote_fair_value = self.osmium_fair_value + alpha_adjustment

        self.search_buys(state, product, take_fair_value, depth=2)
        self.search_sells(state, product, take_fair_value, depth=2)

        spread = best_ask - best_bid
        inv_ratio = position / self.limits[product]
        inventory_skew = inv_ratio * 2.0

        base_offset = 2.0 if spread >= 4 else 3.0
        buy_offset = base_offset + max(0.0, inventory_skew)
        sell_offset = base_offset + max(0.0, -inventory_skew)

        buy_price = min(best_bid + 1, math.floor(quote_fair_value - buy_offset))
        sell_price = max(best_ask - 1, math.ceil(quote_fair_value + sell_offset))

        max_buy = max(0, self.limits[product] - self.osmium_position - self.osmium_buy_orders)
        max_sell = max(0, self.osmium_position + self.limits[product] - self.osmium_sell_orders)

        quote_size = self.osmium_quote_size
        if abs(position) > 50:
            quote_size = 8
        elif abs(position) > 30:
            quote_size = 12

        buy_quote_size = quote_size
        sell_quote_size = quote_size
        if position > 30:
            buy_quote_size = max(1, int(quote_size * 0.4))
            sell_quote_size = max(2, int(quote_size * 1.2))
        elif position < -30:
            sell_quote_size = max(1, int(quote_size * 0.4))
            buy_quote_size = max(2, int(quote_size * 1.2))

        max_buy = min(max_buy, buy_quote_size)
        max_sell = min(max_sell, sell_quote_size)

        logger.print(
            f"OSMIUM_RET5_PRESSURE_ALPHA mid={mid:.2f} ema={self.osmium_fair_value:.2f} "
            f"micro_sig={micro_signal:.2f} depth_sig={depth_signal:.2f} "
            f"imb={top_imbalance:.2f} move={recent_move:.2f} ret5={ret_5_signal:.2f} "
            f"liq_pressure={liquidity_pressure:.3f} "
            f"alpha={alpha_adjustment:.2f} "
            f"take_fv={take_fair_value:.2f} quote_fv={quote_fair_value:.2f} "
            f"pos={position} bid={best_bid} ask={best_ask} "
            f"buy_px={buy_price} sell_px={sell_price}"
        )

        if max_buy > 0:
            self.send_buy_order(product, buy_price, max_buy, f"{product} MM BUY {max_buy} @ {buy_price}")
        if max_sell > 0:
            self.send_sell_order(product, sell_price, -max_sell, f"{product} MM SELL {max_sell} @ {sell_price}")

    def trade_pepper(self, state: TradingState):
        product = "INTARIAN_PEPPER_ROOT"

        if product not in state.order_depths:
            return

        best_bid, best_ask = self.get_best_bid_ask(state, product)
        if best_bid is None or best_ask is None:
            return

        position = self.get_product_pos(state, product)
        logger.print(
            f"PEPPER_MAINBH t={state.timestamp} bid={best_bid} ask={best_ask} pos={position}"
        )

        remaining_buy = self.pepper_limit - position - self.pepper_buy_orders
        size = min(remaining_buy, self.pepper_clip_size)
        if size > 0:
            self.pepper_buy_orders += size
            self.send_buy_order(product, best_ask, size, f"PEPPER START BUY {size} @ {best_ask}")

    def reset_orders(self, state: TradingState):
        self.orders = {}
        self.conversions = 0

        self.osmium_position = self.get_product_pos(state, "ASH_COATED_OSMIUM")
        self.osmium_buy_orders = 0
        self.osmium_sell_orders = 0

        self.pepper_position = self.get_product_pos(state, "INTARIAN_PEPPER_ROOT")
        self.pepper_buy_orders = 0
        self.pepper_sell_orders = 0

        for product in state.order_depths:
            self.orders[product] = []

    def run(self, state: TradingState):
        self.reset_orders(state)

        if "ASH_COATED_OSMIUM" in state.order_depths:
            self.trade_osmium(state)

        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            self.trade_pepper(state)

        logger.flush(state, self.orders, self.conversions, self.traderData)
        return self.orders, self.conversions, self.traderData
