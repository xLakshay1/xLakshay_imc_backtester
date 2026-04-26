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
    def __init__(self):
        self.underlying = "VELVETFRUIT_EXTRACT"
        self.vouchers = [
            "VEV_4000",
            "VEV_4500",
            "VEV_5000",
            "VEV_5100",
            "VEV_5200",
            "VEV_5300",
            "VEV_5400",
            "VEV_5500",
            "VEV_6000",
            "VEV_6500",
        ]
        self.trade_vouchers = ["VEV_4000", "VEV_4500"]
        self.limit = {symbol: 300 for symbol in self.vouchers}

        self.tte_start_days = 5.0
        self.iv_floor = 0.01
        self.iv_cap = 3.0
        self.extrinsic_floor = 0.25
        self.entry_threshold = 1.30
        self.take_threshold = 1.80
        self.inventory_skew = 0.006
        self.base_quote_size = 18
        self.depth_p25 = 55.0
        self.depth_p75 = 95.0
        self.book_weight = 0.70
        self.l1_weight = 0.35

        self.orders: Dict[Symbol, list[Order]] = {}
        self.buy_reserved: Dict[Symbol, int] = {}
        self.sell_reserved: Dict[Symbol, int] = {}
        self.conversions = 0
        self.traderData = "R3_OPTIONS_ONLY_SMILE_V1"

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
            self.limit[product] - self.get_position(state, product) - self.buy_reserved.get(product, 0),
        )

    def remaining_sell(self, state: TradingState, product: str) -> int:
        return max(
            0,
            self.limit[product] + self.get_position(state, product) - self.sell_reserved.get(product, 0),
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

    def option_mid(self, order_depth: OrderDepth) -> Optional[float]:
        return self.wall_mid(order_depth)

    def book_signal(self, order_depth: OrderDepth) -> Dict[str, Any]:
        best_bid, best_ask = self.best_bid_ask(order_depth)
        spread = float(best_ask - best_bid) if best_bid is not None and best_ask is not None else 6.0
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
        imbalance = (bid_total - ask_total) / depth_total if depth_total > 0 else 0.0
        l1_total = best_bid_volume + best_ask_volume
        l1_imbalance = (best_bid_volume - best_ask_volume) / l1_total if l1_total > 0 else 0.0
        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "depth_total": depth_total,
            "best_bid_volume": best_bid_volume,
            "best_ask_volume": best_ask_volume,
            "imbalance": imbalance,
            "l1_imbalance": l1_imbalance,
        }

    def regime_scale(self, depth_total: float) -> float:
        if depth_total >= self.depth_p75:
            return 1.20
        if depth_total <= self.depth_p25:
            return 0.80
        frac = (depth_total - self.depth_p25) / (self.depth_p75 - self.depth_p25)
        return 0.80 + 0.40 * frac

    def strike_from_symbol(self, symbol: str) -> float:
        return float(symbol.split("_")[1])

    def norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_call_price(self, spot: float, strike: float, tau: float, vol: float) -> float:
        intrinsic = max(0.0, spot - strike)
        if tau <= 0.0:
            return intrinsic
        vol = max(self.iv_floor, vol)
        root_tau = math.sqrt(tau)
        if root_tau <= 0.0:
            return intrinsic
        d1 = (math.log(max(spot, 1.0) / max(strike, 1.0)) + 0.5 * vol * vol * tau) / (vol * root_tau)
        d2 = d1 - vol * root_tau
        return spot * self.norm_cdf(d1) - strike * self.norm_cdf(d2)

    def implied_vol(self, spot: float, strike: float, tau: float, price: float) -> Optional[float]:
        intrinsic = max(0.0, spot - strike)
        extrinsic = price - intrinsic
        if tau <= 0.0 or extrinsic <= self.extrinsic_floor:
            return None
        low = self.iv_floor
        high = self.iv_cap
        high_price = self.bs_call_price(spot, strike, tau, high)
        if price > high_price:
            return None
        for _ in range(60):
            mid = 0.5 * (low + high)
            model = self.bs_call_price(spot, strike, tau, mid)
            if model > price:
                high = mid
            else:
                low = mid
        return 0.5 * (low + high)

    def solve_3x3(self, matrix: list[list[float]], rhs: list[float]) -> Optional[list[float]]:
        a = [row[:] for row in matrix]
        b = rhs[:]
        n = 3
        for col in range(n):
            pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
            if abs(a[pivot][col]) < 1e-12:
                return None
            if pivot != col:
                a[col], a[pivot] = a[pivot], a[col]
                b[col], b[pivot] = b[pivot], b[col]
            scale = a[col][col]
            for j in range(col, n):
                a[col][j] /= scale
            b[col] /= scale
            for row in range(n):
                if row == col:
                    continue
                factor = a[row][col]
                for j in range(col, n):
                    a[row][j] -= factor * a[col][j]
                b[row] -= factor * b[col]
        return b

    def quadratic_smile_fit(self, xs: list[float], ys: list[float]) -> Optional[tuple[float, float, float]]:
        if len(xs) < 3:
            return None
        sx = sum(xs)
        sx2 = sum(x * x for x in xs)
        sx3 = sum(x * x * x for x in xs)
        sx4 = sum(x * x * x * x for x in xs)
        sy = sum(ys)
        sxy = sum(x * y for x, y in zip(xs, ys))
        sx2y = sum(x * x * y for x, y in zip(xs, ys))
        matrix = [
            [sx4, sx3, sx2],
            [sx3, sx2, sx],
            [sx2, sx, float(len(xs))],
        ]
        rhs = [sx2y, sxy, sy]
        sol = self.solve_3x3(matrix, rhs)
        if sol is None:
            return None
        return (sol[0], sol[1], sol[2])

    def tte_years(self, timestamp: int) -> float:
        days_left = max(0.25, self.tte_start_days - float(timestamp) / 1_000_000.0)
        return days_left / 365.0

    def trade_voucher(
        self,
        state: TradingState,
        product: str,
        fair_price: float,
        market_price: float,
        signal: Dict[str, Any],
    ) -> None:
        position = self.get_position(state, product)
        regime_scale = self.regime_scale(float(signal["depth_total"]))
        spread = float(signal["spread"])
        broad_imb = float(signal["imbalance"])
        l1_imb = float(signal["l1_imbalance"])

        adjusted_edge = (
            market_price
            - fair_price
            + self.book_weight * spread * broad_imb
            - self.l1_weight * spread * l1_imb
        )

        take_threshold = max(self.take_threshold, 0.85 * spread)
        passive_offset = max(1.0, 0.55 * spread / max(0.85, regime_scale))
        quote_size = max(6, int(round(self.base_quote_size * regime_scale)))
        if abs(position) > 0.60 * self.limit[product]:
            quote_size = max(4, quote_size // 2)

        if (
            signal["best_ask"] is not None
            and adjusted_edge < -take_threshold
            and signal["best_ask"] <= fair_price - 0.4
        ):
            self.place_buy(
                state,
                product,
                int(signal["best_ask"]),
                min(int(signal["best_ask_volume"]), max(8, quote_size)),
                f"{product} TAKE BUY @ {signal['best_ask']} fair={fair_price:.2f} edge={adjusted_edge:.2f}",
            )
        if (
            signal["best_bid"] is not None
            and adjusted_edge > take_threshold
            and signal["best_bid"] >= fair_price + 0.4
        ):
            self.place_sell(
                state,
                product,
                int(signal["best_bid"]),
                min(int(signal["best_bid_volume"]), max(8, quote_size)),
                f"{product} TAKE SELL @ {signal['best_bid']} fair={fair_price:.2f} edge={adjusted_edge:.2f}",
            )

        if abs(adjusted_edge) < self.entry_threshold:
            logger.print(
                f"{product} PASS fair={fair_price:.2f} mkt={market_price:.2f} "
                f"edge={adjusted_edge:.2f} imb={broad_imb:.3f} l1={l1_imb:.3f}"
            )
            return

        quote_fair = fair_price - adjusted_edge - self.inventory_skew * position
        buy_price = min(int(signal["best_bid"]) + 1, int(math.floor(quote_fair - passive_offset)))
        sell_price = max(int(signal["best_ask"]) - 1, int(math.ceil(quote_fair + passive_offset)))
        if buy_price >= sell_price:
            buy_price = min(int(signal["best_bid"]), sell_price - 1)

        self.place_buy(
            state,
            product,
            buy_price,
            quote_size,
            f"{product} MM BUY {quote_size} @ {buy_price} fair={fair_price:.2f} edge={adjusted_edge:.2f}",
        )
        self.place_sell(
            state,
            product,
            sell_price,
            quote_size,
            f"{product} MM SELL {quote_size} @ {sell_price} fair={fair_price:.2f} edge={adjusted_edge:.2f}",
        )

        logger.print(
            f"{product} fair={fair_price:.2f} mkt={market_price:.2f} edge={adjusted_edge:.2f} "
            f"imb={broad_imb:.3f} l1={l1_imb:.3f} pos={position}"
        )

    def run(self, state: TradingState):
        self.reset_orders(state)

        if self.underlying not in state.order_depths:
            logger.flush(state, self.orders, self.conversions, self.traderData)
            return self.orders, self.conversions, self.traderData

        underlying_mid = self.wall_mid(state.order_depths[self.underlying])
        if underlying_mid is None:
            logger.flush(state, self.orders, self.conversions, self.traderData)
            return self.orders, self.conversions, self.traderData

        tau = self.tte_years(state.timestamp)
        xs = []
        ys = []
        observed: Dict[str, Dict[str, float]] = {}

        for product in self.vouchers:
            if product not in state.order_depths:
                continue
            market_price = self.option_mid(state.order_depths[product])
            if market_price is None:
                continue
            strike = self.strike_from_symbol(product)
            iv = self.implied_vol(underlying_mid, strike, tau, market_price)
            if iv is None:
                continue
            m = math.log(strike / max(underlying_mid, 1.0))
            xs.append(m)
            ys.append(iv)
            observed[product] = {
                "market_price": market_price,
                "strike": strike,
                "iv": iv,
                "moneyness": m,
            }

        fit = self.quadratic_smile_fit(xs, ys)
        if fit is None:
            logger.print("SMILE FIT SKIP: not enough stable IV points")
            logger.flush(state, self.orders, self.conversions, self.traderData)
            return self.orders, self.conversions, self.traderData

        a, b, c = fit
        logger.print(
            f"SMILE FIT a={a:.4f} b={b:.4f} c={c:.4f} spot={underlying_mid:.2f} tau_d={tau * 365.0:.2f}"
        )

        for product in self.trade_vouchers:
            if product not in observed:
                logger.print(f"{product} SKIP: unstable IV or missing book")
                continue
            strike = observed[product]["strike"]
            m = observed[product]["moneyness"]
            fitted_iv = max(self.iv_floor, a * m * m + b * m + c)
            fair_price = self.bs_call_price(underlying_mid, strike, tau, fitted_iv)
            signal = self.book_signal(state.order_depths[product])
            self.trade_voucher(
                state,
                product,
                fair_price,
                observed[product]["market_price"],
                signal,
            )

        logger.flush(state, self.orders, self.conversions, self.traderData)
        return self.orders, self.conversions, self.traderData
