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
        self.limit = {symbol: 300 for symbol in self.vouchers}

        self.tte_start_days = 5.0
        self.iv_floor = 0.01
        self.iv_cap = 3.0
        self.extrinsic_floor = 0.20

        self.entry_threshold = 2.60
        self.take_threshold = 3.40
        self.inventory_skew = 0.007
        self.take_size = 22
        self.mm_size = 8
        self.depth_min = 16.0
        self.book_weight = 0.35
        self.l1_weight = 0.18
        self.passive_offset_mult = 0.55

        self.vv_window = 220
        self.default_slope = 1.0
        self.default_intercept = -4000.0
        self.vv4000_blend = 0.70
        self.traderData = "R3_OPTIONS_ALL_IN_ONE_V1"

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
        memory = {"pairs_4000": [], "last_timestamp": None}
        if not trader_data:
            return memory
        try:
            raw = json.loads(trader_data)
        except Exception:
            return memory
        pairs = raw.get("pairs_4000", [])
        if isinstance(pairs, list):
            cleaned = []
            for item in pairs[-self.vv_window :]:
                if isinstance(item, list) and len(item) == 2:
                    cleaned.append([float(item[0]), float(item[1])])
            memory["pairs_4000"] = cleaned
        memory["last_timestamp"] = raw.get("last_timestamp")
        return memory

    def encode_memory(self, memory: Dict[str, Any]) -> str:
        return json.dumps(memory, separators=(",", ":"))

    def get_position(self, state: TradingState, product: str) -> int:
        return int(state.position.get(product, 0))

    def remaining_buy(self, state: TradingState, product: str) -> int:
        return max(0, self.limit[product] - self.get_position(state, product) - self.buy_reserved.get(product, 0))

    def remaining_sell(self, state: TradingState, product: str) -> int:
        return max(0, self.limit[product] + self.get_position(state, product) - self.sell_reserved.get(product, 0))

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
        if price > self.bs_call_price(spot, strike, tau, high):
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
        return sol[0], sol[1], sol[2]

    def tte_years(self, timestamp: int) -> float:
        days_left = max(0.25, self.tte_start_days - float(timestamp) / 1_000_000.0)
        return days_left / 365.0

    def book_signal(self, order_depth: OrderDepth) -> Dict[str, Any]:
        best_bid, best_ask = self.best_bid_ask(order_depth)
        mid = self.wall_mid(order_depth)
        spread = float(best_ask - best_bid) if best_bid is not None and best_ask is not None else 99.0
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
            "best_bid_volume": best_bid_volume,
            "best_ask_volume": best_ask_volume,
            "mid": mid,
            "spread": spread,
            "depth_total": depth_total,
            "imbalance": imbalance,
            "l1_imbalance": l1_imbalance,
        }

    def dynamic_spread_cap(self, product: str, market_price: float, strike: float, spot: float) -> float:
        base = max(10.0, 0.02 * max(market_price, 1.0))
        intrinsic = max(0.0, spot - strike)
        if product == "VEV_4000":
            return max(30.0, base, 0.015 * max(intrinsic, 1.0))
        return max(10.0, base)

    def update_vv4000_pairs(self, memory: Dict[str, Any], timestamp: int, vv_mid: float, vev_mid: float) -> None:
        last_timestamp = memory["last_timestamp"]
        if last_timestamp is not None and int(timestamp) < int(last_timestamp):
            memory["pairs_4000"] = []
        pairs = list(memory["pairs_4000"])
        pairs.append([float(vv_mid), float(vev_mid)])
        memory["pairs_4000"] = pairs[-self.vv_window :]
        memory["last_timestamp"] = int(timestamp)

    def vv_regression_fair(self, pairs: list[list[float]], vv_mid: float) -> tuple[float, float, float]:
        if len(pairs) < 12:
            fair = self.default_intercept + self.default_slope * vv_mid
            return fair, self.default_slope, self.default_intercept
        xs = [row[0] for row in pairs]
        ys = [row[1] for row in pairs]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        denom = sum((x - x_mean) ** 2 for x in xs)
        if denom <= 1e-9:
            slope = self.default_slope
            intercept = y_mean - slope * x_mean
        else:
            slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
            intercept = y_mean - slope * x_mean
        fair = intercept + slope * vv_mid
        return fair, slope, intercept

    def blended_fair(
        self,
        product: str,
        smile_fair: float,
        vv_mid: float,
        memory: Dict[str, Any],
    ) -> tuple[float, str]:
        if product != "VEV_4000":
            return smile_fair, "smile"
        vv_fair, slope, intercept = self.vv_regression_fair(memory["pairs_4000"], vv_mid)
        fair = self.vv4000_blend * vv_fair + (1.0 - self.vv4000_blend) * smile_fair
        logger.print(
            f"VEV4000 fair blend vv={vv_fair:.2f} smile={smile_fair:.2f} slope={slope:.4f} intercept={intercept:.2f}"
        )
        return fair, "blend"

    def trade_voucher(
        self,
        state: TradingState,
        product: str,
        fair_price: float,
        signal: Dict[str, Any],
        common_imbalance: float,
        common_l1: float,
        source: str,
    ) -> None:
        best_bid = signal["best_bid"]
        best_ask = signal["best_ask"]
        market_price = signal["mid"]
        if best_bid is None or best_ask is None or market_price is None:
            logger.print(f"{product} SKIP no-two-sided-book")
            return

        strike = self.strike_from_symbol(product)
        spread = float(signal["spread"])
        depth_total = float(signal["depth_total"])
        spread_cap = self.dynamic_spread_cap(product, market_price, strike, fair_price + strike)
        if spread > spread_cap or depth_total < self.depth_min:
            logger.print(
                f"{product} SKIP spread={spread:.1f}/{spread_cap:.1f} depth={depth_total:.1f} source={source}"
            )
            return

        position = self.get_position(state, product)
        residual_imb = float(signal["imbalance"]) - common_imbalance
        residual_l1 = float(signal["l1_imbalance"]) - common_l1
        adjusted_edge = (
            market_price
            - fair_price
            + self.book_weight * spread * residual_imb
            - self.l1_weight * spread * residual_l1
        )
        take_threshold = max(self.entry_threshold, 0.45 * spread)
        passive_offset = max(1.0, self.passive_offset_mult * spread)
        mm_size = self.mm_size if abs(position) < 180 else max(4, self.mm_size // 2)
        quote_fair = fair_price - self.inventory_skew * position

        if adjusted_edge < -take_threshold:
            self.place_buy(
                state,
                product,
                int(best_ask),
                min(int(signal["best_ask_volume"]), self.take_size),
                f"{product} TAKE BUY @ {best_ask} fair={fair_price:.2f} edge={adjusted_edge:.2f} src={source}",
            )
        if adjusted_edge > take_threshold:
            self.place_sell(
                state,
                product,
                int(best_bid),
                min(int(signal["best_bid_volume"]), self.take_size),
                f"{product} TAKE SELL @ {best_bid} fair={fair_price:.2f} edge={adjusted_edge:.2f} src={source}",
            )

        buy_price = min(best_bid + 1, int(math.floor(quote_fair - passive_offset)))
        sell_price = max(best_ask - 1, int(math.ceil(quote_fair + passive_offset)))
        if buy_price >= sell_price:
            buy_price = min(best_bid, sell_price - 1)

        if adjusted_edge < -self.take_threshold:
            self.place_buy(
                state,
                product,
                buy_price,
                mm_size,
                f"{product} MM BUY {mm_size} @ {buy_price} fair={fair_price:.2f} edge={adjusted_edge:.2f} src={source}",
            )
        elif adjusted_edge > self.take_threshold:
            self.place_sell(
                state,
                product,
                sell_price,
                mm_size,
                f"{product} MM SELL {mm_size} @ {sell_price} fair={fair_price:.2f} edge={adjusted_edge:.2f} src={source}",
            )
        else:
            logger.print(
                f"{product} PASS fair={fair_price:.2f} mkt={market_price:.2f} edge={adjusted_edge:.2f} "
                f"spread={spread:.1f} depth={depth_total:.1f} src={source}"
            )

    def run(self, state: TradingState):
        self.reset_orders(state)
        memory = self.parse_memory(state.traderData)

        if self.underlying not in state.order_depths:
            trader_data = self.encode_memory(memory)
            logger.flush(state, self.orders, self.conversions, trader_data)
            return self.orders, self.conversions, trader_data

        underlying_mid = self.wall_mid(state.order_depths[self.underlying])
        if underlying_mid is None:
            trader_data = self.encode_memory(memory)
            logger.flush(state, self.orders, self.conversions, trader_data)
            return self.orders, self.conversions, trader_data

        tau = self.tte_years(state.timestamp)
        xs: list[float] = []
        ys: list[float] = []
        signals: Dict[str, Dict[str, Any]] = {}
        observed: Dict[str, Dict[str, float]] = {}

        for product in self.vouchers:
            if product not in state.order_depths:
                continue
            signal = self.book_signal(state.order_depths[product])
            if signal["mid"] is None:
                continue
            strike = self.strike_from_symbol(product)
            signals[product] = signal
            observed[product] = {
                "market_price": float(signal["mid"]),
                "strike": strike,
                "moneyness": math.log(strike / max(underlying_mid, 1.0)),
            }
            if product == "VEV_4000":
                self.update_vv4000_pairs(memory, state.timestamp, underlying_mid, float(signal["mid"]))
            iv = self.implied_vol(underlying_mid, strike, tau, float(signal["mid"]))
            if iv is not None:
                xs.append(observed[product]["moneyness"])
                ys.append(iv)
                observed[product]["iv"] = iv

        fit = self.quadratic_smile_fit(xs, ys)
        if fit is None:
            logger.print("OPTIONS SKIP: not enough stable smile points")
            trader_data = self.encode_memory(memory)
            logger.flush(state, self.orders, self.conversions, trader_data)
            return self.orders, self.conversions, trader_data

        a, b, c = fit
        logger.print(
            f"ALL-OPT SMILE a={a:.4f} b={b:.4f} c={c:.4f} spot={underlying_mid:.2f} tau_d={tau * 365.0:.2f}"
        )

        common_imbalance = 0.0
        common_l1 = 0.0
        if signals:
            common_imbalance = sum(float(signal["imbalance"]) for signal in signals.values()) / len(signals)
            common_l1 = sum(float(signal["l1_imbalance"]) for signal in signals.values()) / len(signals)

        for product in self.vouchers:
            if product not in observed or product not in signals:
                continue
            strike = observed[product]["strike"]
            m = observed[product]["moneyness"]
            fitted_iv = max(self.iv_floor, a * m * m + b * m + c)
            smile_fair = self.bs_call_price(underlying_mid, strike, tau, fitted_iv)
            fair_price, source = self.blended_fair(product, smile_fair, underlying_mid, memory)
            self.trade_voucher(
                state,
                product,
                fair_price,
                signals[product],
                common_imbalance,
                common_l1,
                source,
            )

        trader_data = self.encode_memory(memory)
        logger.flush(state, self.orders, self.conversions, trader_data)
        return self.orders, self.conversions, trader_data
