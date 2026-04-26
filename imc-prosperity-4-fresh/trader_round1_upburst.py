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
        self.traderData = "OSMIUM_UPBURST_FADE_PLUS_MM_PEPPER_HOLD_V2"

        # Position/order tracking
        self.osmium_position = 0
        self.osmium_buy_orders = 0
        self.osmium_sell_orders = 0

        self.pepper_position = 0
        self.pepper_buy_orders = 0
        self.pepper_sell_orders = 0

        # Osmium MM state
        self.osmium_fair_value = None
        self.osmium_quote_size = 20
        self.osmium_alpha = 0.08
        self.osmium_mean_reversion_strength = 0.6
        self.enable_osmium_upburst = True

        # Osmium upburst state.
        #
        # Website-safe interpretation:
        # The strict detector fires late, often near the peak. So we do NOT
        # chase the upburst with aggressive buys. Instead:
        # 1. early trigger = guard: stop fading too early / reduce sell quotes.
        # 2. strict trigger = exhaustion: sell/flatten into likely reversion.
        self.osmium_mid_history: list[float] = []
        self.osmium_upburst_active = False
        self.osmium_upburst_entry_ts = -1
        self.osmium_last_upburst_exit_ts = -1
        self.osmium_entry_window = 30
        self.osmium_strict_window = 100
        self.osmium_entry_slope = 0.080
        self.osmium_entry_return = 4.0
        self.osmium_entry_tstat = 2.5
        self.osmium_strict_slope = 0.078
        self.osmium_strict_return = 6.0
        self.osmium_strict_tstat = 3.0
        self.osmium_upburst_clip = 10
        self.osmium_upburst_max_short = -25
        self.osmium_upburst_cooldown = 45_000
        self.osmium_upburst_max_hold = 25_000
        self.osmium_upburst_min_peak_gain = 5.0
        self.osmium_upburst_trailing_reversal = 2.0
        self.osmium_upburst_entry_mid = None
        self.osmium_upburst_peak_mid = None

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
        if self.enable_osmium_upburst:
            self.trade_osmium_upburst(state, best_bid, best_ask, mid)
        else:
            self.osmium_mid_history.append(mid)
            if len(self.osmium_mid_history) > 180:
                self.osmium_mid_history = self.osmium_mid_history[-180:]

        # Slow EMA anchor
        if self.osmium_fair_value is None:
            self.osmium_fair_value = mid
        else:
            self.osmium_fair_value = (
                self.osmium_alpha * mid
                + (1 - self.osmium_alpha) * self.osmium_fair_value
            )

        ema_anchor = self.osmium_fair_value

        # Mean reversion adjustment
        deviation = mid - ema_anchor
        fair_value = ema_anchor - self.osmium_mean_reversion_strength * deviation

        # Do not fade a live upward burst too early.
        if not self.osmium_upburst_active:
            self.search_buys(state, product, fair_value, depth=2)
            self.search_sells(state, product, fair_value, depth=2)
        else:
            # During the early part of a burst, avoid aggressive selling, but
            # do not cross the spread to chase either.
            self.search_buys(state, product, fair_value, depth=1)

        spread = best_ask - best_bid
        inv_ratio = position / self.limits[product]
        inventory_skew = inv_ratio * 2.0

        # If spread is healthy, quote tighter. Otherwise stay a bit wider.
        base_offset = 2.0 if spread >= 4 else 3.0
        if self.osmium_upburst_active:
            base_offset = 3.0

        buy_offset = base_offset + max(0.0, inventory_skew)
        sell_offset = base_offset + max(0.0, -inventory_skew)

        buy_price = min(best_bid + 1, math.floor(fair_value - buy_offset))
        sell_price = max(best_ask - 1, math.ceil(fair_value + sell_offset))

        max_buy = (
            self.limits[product]
            - self.osmium_position
            - self.osmium_buy_orders
        )
        max_sell = (
            self.osmium_position
            + self.limits[product]
            - self.osmium_sell_orders
        )

        max_buy = max(0, max_buy)
        max_sell = max(0, max_sell)

        quote_size = self.osmium_quote_size
        if self.osmium_upburst_active:
            quote_size = 6
            max_sell = min(max_sell, 3)
        elif abs(position) > 50:
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
            f"OSMIUM mid={mid:.2f} ema={ema_anchor:.2f} dev={deviation:.2f} "
            f"fv={fair_value:.2f} pos={position} bid={best_bid} ask={best_ask} "
            f"buy_px={buy_price} sell_px={sell_price}"
        )

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
