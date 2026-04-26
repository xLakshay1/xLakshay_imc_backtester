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
        self.traderData = "OSMIUM_SAFE_MEANREV_PLUS_PEPPER_HOLD_V1"

        # Position/order tracking
        self.osmium_position = 0
        self.osmium_buy_orders = 0
        self.osmium_sell_orders = 0

        self.pepper_position = 0
        self.pepper_buy_orders = 0
        self.pepper_sell_orders = 0

        # Osmium state.
        # Fundamental read: ASH_COATED_OSMIUM is volatile, but centered near
        # 10000. The safer website strategy is to trade deviations from a
        # stable fair, not chase bursts.
        self.osmium_static_fair = 10000.0
        self.osmium_fair_value = None
        self.osmium_alpha = 0.03
        self.osmium_quote_size = 10
        self.osmium_take_edge = 6.0
        self.osmium_quote_edge = 4.0
        self.osmium_inventory_skew = 0.08
        self.osmium_microprice_weight = 0.12
        self.osmium_recent_mean_weight = 0.20
        self.osmium_mid_history: list[float] = []

        # Pepper hold strategy params
        self.pepper_limit = 80
        self.pepper_accumulation_end = 200_000
        self.pepper_endgame_start = 900_000
        self.pepper_clip_size = 20

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

    def rolling_slope_stats(self, window: int):
        if len(self.osmium_mid_history) < window:
            return None
        values = self.osmium_mid_history[-window:]
        x_mean = (window - 1) / 2
        y_mean = sum(values) / window
        denom = sum((i - x_mean) ** 2 for i in range(window))
        if denom <= 0:
            return None
        slope = sum((i - x_mean) * (values[i] - y_mean) for i in range(window)) / denom
        intercept = y_mean - slope * x_mean
        residual_ss = sum((values[i] - (intercept + slope * i)) ** 2 for i in range(window))
        stderr = math.sqrt(residual_ss / max(window - 2, 1) / denom) if residual_ss > 0 else 0.0
        tstat = slope / stderr if stderr > 0 else 0.0
        window_return = values[-1] - values[0]
        return slope, window_return, tstat

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
                    self.limits["ASH_COATED_OSMIUM"]
                    - self.osmium_position
                    - self.osmium_buy_orders,
                    -amount,
                )
                if size > 0:
                    self.osmium_buy_orders += size
                    self.send_buy_order(
                        product, ask, size, msg=f"TAKE BUY {size} {product} @ {ask}"
                    )

            elif product == "INTARIAN_PEPPER_ROOT":
                size = min(
                    self.limits["INTARIAN_PEPPER_ROOT"]
                    - self.pepper_position
                    - self.pepper_buy_orders,
                    -amount,
                )
                if size > 0:
                    self.pepper_buy_orders += size
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

            if product == "ASH_COATED_OSMIUM":
                size = min(
                    self.osmium_position
                    + self.limits["ASH_COATED_OSMIUM"]
                    - self.osmium_sell_orders,
                    amount,
                )
                if size > 0:
                    self.osmium_sell_orders += size
                    self.send_sell_order(
                        product, bid, -size, msg=f"TAKE SELL {-size} {product} @ {bid}"
                    )

            elif product == "INTARIAN_PEPPER_ROOT":
                size = min(
                    self.pepper_position
                    + self.limits["INTARIAN_PEPPER_ROOT"]
                    - self.pepper_sell_orders,
                    amount,
                )
                if size > 0:
                    self.pepper_sell_orders += size
                    self.send_sell_order(
                        product, bid, -size, msg=f"TAKE SELL {-size} {product} @ {bid}"
                    )

    def update_osmium_upburst_state(self, state: TradingState, mid: float):
        self.osmium_mid_history.append(mid)
        if len(self.osmium_mid_history) > 180:
            self.osmium_mid_history = self.osmium_mid_history[-180:]

        entry_stats = self.rolling_slope_stats(self.osmium_entry_window)
        strict_stats = self.rolling_slope_stats(self.osmium_strict_window)
        entry_signal = False
        strict_signal = False

        if entry_stats is not None:
            slope, window_return, tstat = entry_stats
            entry_signal = (
                slope >= self.osmium_entry_slope
                and window_return >= self.osmium_entry_return
                and tstat >= self.osmium_entry_tstat
            )
        if strict_stats is not None:
            slope, window_return, tstat = strict_stats
            strict_signal = (
                slope >= self.osmium_strict_slope
                and window_return >= self.osmium_strict_return
                and tstat >= self.osmium_strict_tstat
            )

        if (
            entry_signal
            and not self.osmium_upburst_active
            and state.timestamp - self.osmium_last_upburst_exit_ts >= self.osmium_upburst_cooldown
        ):
            self.osmium_upburst_active = True
            self.osmium_upburst_entry_ts = state.timestamp
            self.osmium_upburst_entry_mid = mid
            self.osmium_upburst_peak_mid = mid

        if self.osmium_upburst_active:
            self.osmium_upburst_peak_mid = max(self.osmium_upburst_peak_mid or mid, mid)
            hold_time = state.timestamp - self.osmium_upburst_entry_ts
            entry_mid = self.osmium_upburst_entry_mid or mid
            drawdown_from_peak = (self.osmium_upburst_peak_mid or mid) - mid
            peak_gain = (self.osmium_upburst_peak_mid or mid) - entry_mid

            exit_signal = (
                strict_signal
                or hold_time >= self.osmium_upburst_max_hold
                or (
                    peak_gain >= self.osmium_upburst_min_peak_gain
                    and drawdown_from_peak >= self.osmium_upburst_trailing_reversal
                )
            )
            if exit_signal:
                self.osmium_upburst_active = False
                self.osmium_last_upburst_exit_ts = state.timestamp

        return entry_signal, strict_signal

    def trade_osmium_upburst(self, state: TradingState, best_bid: int, best_ask: int, mid: float):
        product = "ASH_COATED_OSMIUM"
        position = self.get_product_pos(state, product)
        entry_signal, strict_signal = self.update_osmium_upburst_state(state, mid)

        # Strict signal is usually late; use it as an exhaustion / fade cue.
        entry_mid = self.osmium_upburst_entry_mid or mid
        peak_mid = self.osmium_upburst_peak_mid or mid
        peak_gain = peak_mid - entry_mid
        trailing_reversal = peak_mid - mid
        exhaustion_signal = strict_signal or (
            peak_gain >= self.osmium_upburst_min_peak_gain
            and trailing_reversal >= self.osmium_upburst_trailing_reversal
        )

        if exhaustion_signal:
            if position > 0:
                size = min(position, self.osmium_upburst_clip)
            else:
                short_room = position - self.osmium_upburst_max_short
                size = min(max(0, short_room), self.osmium_upburst_clip)

            if size > 0:
                self.osmium_sell_orders += size
                self.send_sell_order(
                    product,
                    best_bid,
                    -size,
                    msg=f"OSMIUM UPBURST FADE SELL {size} @ {best_bid}",
                )

        logger.print(
            f"OSMIUM_UPBURST active={self.osmium_upburst_active} "
            f"entry={entry_signal} strict={strict_signal} exhaust={exhaustion_signal} pos={position}"
        )

    def trade_osmium(self, state: TradingState):
        product = "ASH_COATED_OSMIUM"

        best_bid, best_ask = self.get_best_bid_ask(state, product)
        if best_bid is None or best_ask is None:
            return

        position = self.get_product_pos(state, product)
        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        order_depth = state.order_depths[product]

        self.osmium_mid_history.append(mid)
        if len(self.osmium_mid_history) > 80:
            self.osmium_mid_history = self.osmium_mid_history[-80:]

        # Very slow adaptive anchor, pulled toward the known fundamental fair.
        if self.osmium_fair_value is None:
            self.osmium_fair_value = self.osmium_static_fair
        else:
            target = 0.85 * self.osmium_static_fair + 0.15 * mid
            self.osmium_fair_value = (
                self.osmium_alpha * target
                + (1 - self.osmium_alpha) * self.osmium_fair_value
            )

        recent_mean = sum(self.osmium_mid_history[-30:]) / min(30, len(self.osmium_mid_history))

        bid_volume = abs(order_depth.buy_orders.get(best_bid, 0))
        ask_volume = abs(order_depth.sell_orders.get(best_ask, 0))
        if bid_volume + ask_volume > 0:
            microprice = (
                best_ask * bid_volume + best_bid * ask_volume
            ) / (bid_volume + ask_volume)
        else:
            microprice = mid

        fair_value = (
            (1 - self.osmium_recent_mean_weight - self.osmium_microprice_weight)
            * self.osmium_fair_value
            + self.osmium_recent_mean_weight * recent_mean
            + self.osmium_microprice_weight * microprice
        )

        # Inventory skew lowers fair when long and raises fair when short.
        fair_value -= self.osmium_inventory_skew * position

        # Take only large, obvious dislocations. This is intentionally
        # conservative for the official website fill model.
        self.search_buys(state, product, fair_value - self.osmium_take_edge, depth=1)
        self.search_sells(state, product, fair_value + self.osmium_take_edge, depth=1)

        # Passive quotes: mean-reversion around stable fair, with inventory
        # controls. Quote inside only when the spread is wide enough.
        if spread >= 8:
            base_edge = 3
        elif spread >= 5:
            base_edge = 4
        else:
            base_edge = 5

        if abs(position) > 55:
            quote_size = 4
        elif abs(position) > 35:
            quote_size = 6
        else:
            quote_size = self.osmium_quote_size

        buy_size = quote_size
        sell_size = quote_size
        if position > 30:
            buy_size = max(1, quote_size // 3)
            sell_size = min(quote_size + 4, 14)
        elif position < -30:
            sell_size = max(1, quote_size // 3)
            buy_size = min(quote_size + 4, 14)

        max_buy = max(0, self.limits[product] - position - self.osmium_buy_orders)
        max_sell = max(0, position + self.limits[product] - self.osmium_sell_orders)
        buy_size = min(buy_size, max_buy)
        sell_size = min(sell_size, max_sell)

        buy_price = min(best_bid + 1, math.floor(fair_value - base_edge))
        sell_price = max(best_ask - 1, math.ceil(fair_value + base_edge))

        # If badly long/short, cross slightly more aggressively to reduce risk.
        if position > 60 and max_sell > 0:
            reduce_size = min(max_sell, 10)
            self.osmium_sell_orders += reduce_size
            self.send_sell_order(
                product,
                best_bid,
                -reduce_size,
                msg=f"OSMIUM RISK REDUCE SELL {reduce_size} @ {best_bid}",
            )
            sell_size = max(0, sell_size - reduce_size)
        elif position < -60 and max_buy > 0:
            reduce_size = min(max_buy, 10)
            self.osmium_buy_orders += reduce_size
            self.send_buy_order(
                product,
                best_ask,
                reduce_size,
                msg=f"OSMIUM RISK REDUCE BUY {reduce_size} @ {best_ask}",
            )
            buy_size = max(0, buy_size - reduce_size)

        # Avoid crossed or inverted quotes after skewing.
        if buy_price >= sell_price:
            buy_price = best_bid
            sell_price = best_ask

        logger.print(
            f"OSMIUM_SAFE mid={mid:.2f} fv={fair_value:.2f} micro={microprice:.2f} "
            f"recent={recent_mean:.2f} pos={position} spread={spread} "
            f"buy_px={buy_price} sell_px={sell_price}"
        )

        if buy_size > 0:
            self.send_buy_order(
                product,
                buy_price,
                buy_size,
                msg=f"{product} SAFE BUY {buy_size} @ {buy_price}",
            )
        if sell_size > 0:
            self.send_sell_order(
                product,
                sell_price,
                -sell_size,
                msg=f"{product} SAFE SELL {sell_size} @ {sell_price}",
            )

    def trade_pepper(self, state: TradingState):
        product = "INTARIAN_PEPPER_ROOT"

        if product not in state.order_depths:
            return

        best_bid, best_ask = self.get_best_bid_ask(state, product)
        if best_bid is None or best_ask is None:
            return

        position = self.get_product_pos(state, product)
        t = state.timestamp

        logger.print(
            f"PEPPER t={t} bid={best_bid} ask={best_ask} pos={position}"
        )

        # Phase 1: early accumulation
        if t < self.pepper_accumulation_end:
            remaining = self.pepper_limit - position
            if remaining <= 0:
                return

            size = min(remaining, self.pepper_clip_size)
            self.send_buy_order(
                product,
                best_ask,
                size,
                msg=f"PEPPER BUY {size} @ {best_ask}",
            )
            return

        # Phase 2: hold
        if t < self.pepper_endgame_start:
            return

        # Phase 3: late exit
        if position > 0:
            size = min(position, self.pepper_clip_size)
            self.send_sell_order(
                product,
                best_bid,
                -size,
                msg=f"PEPPER EXIT SELL {size} @ {best_bid}",
            )

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
