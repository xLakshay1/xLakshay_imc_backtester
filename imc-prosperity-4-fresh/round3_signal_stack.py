from typing import Any, Optional
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
        self.hydrogel = "HYDROGEL_PACK"
        self.velvet = "VELVETFRUIT_EXTRACT"
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
        self.strikes = {
            "VEV_4000": 4000,
            "VEV_4500": 4500,
            "VEV_5000": 5000,
            "VEV_5100": 5100,
            "VEV_5200": 5200,
            "VEV_5300": 5300,
            "VEV_5400": 5400,
            "VEV_5500": 5500,
            "VEV_6000": 6000,
            "VEV_6500": 6500,
        }
        self.limits = {
            self.hydrogel: 200,
            self.velvet: 200,
        }
        for voucher in self.vouchers:
            self.limits[voucher] = 300

        self.orders: dict[Symbol, list[Order]] = {}
        self.buy_reserved: dict[Symbol, int] = {}
        self.sell_reserved: dict[Symbol, int] = {}
        self.conversions = 0
        self.traderData = "R3_SIGNAL_STACK_V2"

        self.delta1_alpha = {
            self.hydrogel: {
                "ema_alpha": 0.18,
                "ema_weight": 0.28,
                "ret_weight": 0.18,
                "total_imb_weight": 1.10,
                "l1_imb_weight": 0.80,
                "take_threshold": 2.0,
                "passive_offset": 3.5,
                "quote_size": 18,
                "inventory_skew": 0.90,
                "depth_p25": 70.0,
                "depth_p75": 80.0,
            },
            self.velvet: {
                "ema_alpha": 0.25,
                "ema_weight": 0.40,
                "ret_weight": 0.30,
                "total_imb_weight": 1.00,
                "l1_imb_weight": 0.85,
                "take_threshold": 1.0,
                "passive_offset": 1.4,
                "quote_size": 30,
                "inventory_skew": 0.85,
                "depth_p25": 110.0,
                "depth_p75": 132.0,
            },
        }
        self.voucher_depth_regime = {
            "VEV_4000": {"p25": 60.0, "p75": 76.0},
            "VEV_4500": {"p25": 48.0, "p75": 60.0},
            "VEV_5000": {"p25": 48.0, "p75": 60.0},
            "VEV_5100": {"p25": 48.0, "p75": 60.0},
            "VEV_5200": {"p25": 48.0, "p75": 60.0},
            "VEV_5300": {"p25": 40.0, "p75": 50.0},
            "VEV_5400": {"p25": 40.0, "p75": 50.0},
            "VEV_5500": {"p25": 40.0, "p75": 50.0},
            "VEV_6000": {"p25": 40.0, "p75": 50.0},
            "VEV_6500": {"p25": 40.0, "p75": 50.0},
        }
        self.voucher_quote_size = {
            "VEV_4000": 16,
            "VEV_4500": 18,
            "VEV_5000": 20,
            "VEV_5100": 22,
            "VEV_5200": 24,
            "VEV_5300": 24,
            "VEV_5400": 22,
            "VEV_5500": 18,
            "VEV_6000": 0,
            "VEV_6500": 0,
        }
        self.voucher_residual_weight = {
            "VEV_4000": 1.30,
            "VEV_4500": 1.20,
            "VEV_5000": 1.00,
            "VEV_5100": 1.00,
            "VEV_5200": 1.00,
            "VEV_5300": 1.05,
            "VEV_5400": 1.05,
            "VEV_5500": 1.10,
            "VEV_6000": 0.00,
            "VEV_6500": 0.00,
        }
        self.voucher_total_imb_weight = 1.15
        self.voucher_l1_imb_weight = 0.70
        self.voucher_inventory_skew = 0.90
        self.velvet_partial_delta_hedge = 0.08

    def clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def parse_memory(self, trader_data: str) -> dict[str, Any]:
        default = {
            self.hydrogel: {"ema": None, "prev_mid": None},
            self.velvet: {"ema": None, "prev_mid": None},
        }
        if not trader_data:
            return default
        try:
            raw = json.loads(trader_data)
        except Exception:
            return default
        for product in [self.hydrogel, self.velvet]:
            product_state = raw.get(product, {})
            default[product]["ema"] = product_state.get("ema")
            default[product]["prev_mid"] = product_state.get("prev_mid")
        return default

    def encode_memory(self, memory: dict[str, Any]) -> str:
        return json.dumps(memory, separators=(",", ":"))

    def reset_orders(self, state: TradingState):
        self.orders = {product: [] for product in state.order_depths}
        self.buy_reserved = {product: 0 for product in state.order_depths}
        self.sell_reserved = {product: 0 for product in state.order_depths}
        self.conversions = 0

    def get_position(self, state: TradingState, product: str) -> int:
        return int(state.position.get(product, 0))

    def remaining_buy(self, state: TradingState, product: str) -> int:
        limit = self.limits.get(product, 0)
        return max(0, limit - self.get_position(state, product) - self.buy_reserved.get(product, 0))

    def remaining_sell(self, state: TradingState, product: str) -> int:
        limit = self.limits.get(product, 0)
        return max(0, limit + self.get_position(state, product) - self.sell_reserved.get(product, 0))

    def place_buy(self, state: TradingState, product: str, price: int, quantity: int, msg: str):
        qty = min(max(0, int(quantity)), self.remaining_buy(state, product))
        if qty <= 0:
            return
        self.orders[product].append(Order(product, int(price), qty))
        self.buy_reserved[product] += qty
        logger.print(msg)

    def place_sell(self, state: TradingState, product: str, price: int, quantity: int, msg: str):
        qty = min(max(0, int(quantity)), self.remaining_sell(state, product))
        if qty <= 0:
            return
        self.orders[product].append(Order(product, int(price), -qty))
        self.sell_reserved[product] += qty
        logger.print(msg)

    def best_bid_ask(self, order_depth: OrderDepth):
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def ordered_bid_levels(self, order_depth: OrderDepth) -> list[tuple[int, int]]:
        return sorted(order_depth.buy_orders.items(), reverse=True)

    def ordered_ask_levels(self, order_depth: OrderDepth) -> list[tuple[int, int]]:
        return sorted(order_depth.sell_orders.items())

    def wall_mid(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask = self.best_bid_ask(order_depth)
        if best_bid is not None and best_ask is not None:
            return 0.5 * (best_bid + best_ask)
        if best_ask is not None:
            return best_ask - 0.5
        if best_bid is not None:
            return best_bid + 0.5
        return None

    def book_signal(self, order_depth: OrderDepth) -> dict[str, Any]:
        best_bid, best_ask = self.best_bid_ask(order_depth)
        mid = self.wall_mid(order_depth)
        spread = float(best_ask - best_bid) if best_bid is not None and best_ask is not None else 2.0
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
            "bid_total": bid_total,
            "ask_total": ask_total,
            "best_bid_volume": best_bid_volume,
            "best_ask_volume": best_ask_volume,
            "depth_total": depth_total,
        }

    def regime_scale(self, depth_total: float, p25: float, p75: float) -> float:
        if depth_total >= p75:
            return 1.20
        if depth_total <= p25:
            return 0.80
        if p75 <= p25:
            return 1.00
        fraction = (depth_total - p25) / (p75 - p25)
        return 0.80 + 0.40 * fraction

    def solve_3x3(self, matrix: list[list[float]], vector: list[float]) -> Optional[tuple[float, float, float]]:
        augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
        n = 3
        for pivot in range(n):
            best_row = max(range(pivot, n), key=lambda row: abs(augmented[row][pivot]))
            if abs(augmented[best_row][pivot]) < 1e-10:
                return None
            if best_row != pivot:
                augmented[pivot], augmented[best_row] = augmented[best_row], augmented[pivot]

            pivot_value = augmented[pivot][pivot]
            for col in range(pivot, n + 1):
                augmented[pivot][col] /= pivot_value

            for row in range(n):
                if row == pivot:
                    continue
                factor = augmented[row][pivot]
                for col in range(pivot, n + 1):
                    augmented[row][col] -= factor * augmented[pivot][col]

        return tuple(augmented[row][n] for row in range(n))

    def quadratic_fit(self, points: list[tuple[float, float]]) -> tuple[float, float, float]:
        if len(points) >= 3:
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            s0 = float(len(xs))
            s1 = sum(xs)
            s2 = sum(x * x for x in xs)
            s3 = sum(x * x * x for x in xs)
            s4 = sum(x * x * x * x for x in xs)
            t0 = sum(ys)
            t1 = sum(x * y for x, y in points)
            t2 = sum(x * x * y for x, y in points)
            solution = self.solve_3x3(
                [
                    [s4, s3, s2],
                    [s3, s2, s1],
                    [s2, s1, s0],
                ],
                [t2, t1, t0],
            )
            if solution is not None:
                return solution
        if points:
            ordered_ivs = sorted(point[1] for point in points)
            median_iv = ordered_ivs[len(ordered_ivs) // 2]
            return 0.0, 0.0, float(median_iv)
        return 0.0, 0.0, 0.28

    def standard_normal_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_call_price_delta(self, spot: float, strike: float, tau: float, sigma: float):
        if spot <= 0 or strike <= 0:
            return 0.0, 0.0
        if tau <= 0 or sigma <= 0:
            intrinsic = max(spot - strike, 0.0)
            delta = 1.0 if spot > strike else 0.0
            return intrinsic, delta
        root_tau = math.sqrt(tau)
        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * tau) / (sigma * root_tau)
        d2 = d1 - sigma * root_tau
        call_value = spot * self.standard_normal_cdf(d1) - strike * self.standard_normal_cdf(d2)
        delta = self.standard_normal_cdf(d1)
        return call_value, delta

    def implied_volatility(self, spot: float, strike: float, tau: float, price: float) -> Optional[float]:
        intrinsic = max(spot - strike, 0.0)
        if spot <= 0 or strike <= 0 or tau <= 0 or price <= intrinsic + 1e-6:
            return None
        low = 1e-4
        high = 3.0
        for _ in range(60):
            mid = 0.5 * (low + high)
            theo, _ = self.bs_call_price_delta(spot, strike, tau, mid)
            if theo > price:
                high = mid
            else:
                low = mid
        return 0.5 * (low + high)

    def time_to_expiry_years(self, timestamp: int) -> float:
        tte_days = max(0.10, 5.0 - timestamp / 1_000_000.0)
        return tte_days / 365.0

    def fit_voucher_surface(self, state: TradingState, underlying_mid: float, timestamp: int):
        tau = self.time_to_expiry_years(timestamp)
        fit_points: list[tuple[float, float]] = []
        surface: dict[str, dict[str, float]] = {}

        for product in self.vouchers:
            if product not in state.order_depths:
                continue
            order_depth = state.order_depths[product]
            signal = self.book_signal(order_depth)
            market_mid = signal["mid"]
            if market_mid is None:
                continue
            strike = float(self.strikes[product])
            intrinsic = max(underlying_mid - strike, 0.0)
            extrinsic = market_mid - intrinsic
            iv = None
            if extrinsic > 0.50:
                iv = self.implied_volatility(underlying_mid, strike, tau, market_mid)
            if iv is not None and 0.01 < iv < 2.50:
                log_moneyness = math.log(strike / underlying_mid)
                fit_points.append((log_moneyness, iv))
            surface[product] = {
                "market_mid": market_mid,
                "intrinsic": intrinsic,
                "spread": max(1.0, float(signal["spread"])),
                "imbalance": float(signal["imbalance"]),
                "l1_imbalance": float(signal["l1_imbalance"]),
                "depth_total": float(signal["depth_total"]),
                "best_bid": signal["best_bid"],
                "best_ask": signal["best_ask"],
                "best_bid_volume": float(signal["best_bid_volume"]),
                "best_ask_volume": float(signal["best_ask_volume"]),
            }

        a, b, c = self.quadratic_fit(fit_points)

        for product, info in surface.items():
            strike = float(self.strikes[product])
            log_moneyness = math.log(strike / underlying_mid)
            smile_iv = self.clamp(a * log_moneyness * log_moneyness + b * log_moneyness + c, 0.03, 1.20)
            theo_price, delta = self.bs_call_price_delta(underlying_mid, strike, tau, smile_iv)
            info["smile_iv"] = smile_iv
            info["theo_price"] = theo_price
            info["delta"] = delta
            info["residual"] = info["market_mid"] - theo_price

        return surface

    def quote_delta_one(
        self,
        state: TradingState,
        product: str,
        memory: dict[str, Any],
        inventory_target_shift: float = 0.0,
    ):
        if product not in state.order_depths:
            return
        order_depth = state.order_depths[product]
        signal = self.book_signal(order_depth)
        mid = signal["mid"]
        best_bid = signal["best_bid"]
        best_ask = signal["best_ask"]
        if mid is None or best_bid is None or best_ask is None:
            return

        params = self.delta1_alpha[product]
        previous_ema = memory[product]["ema"]
        previous_mid = memory[product]["prev_mid"]
        ema = mid if previous_ema is None else params["ema_alpha"] * mid + (1 - params["ema_alpha"]) * previous_ema
        recent_move = 0.0 if previous_mid is None else mid - previous_mid

        fair = mid
        regime_scale = self.regime_scale(
            float(signal["depth_total"]),
            params["depth_p25"],
            params["depth_p75"],
        )

        fair += regime_scale * params["ema_weight"] * (ema - mid)
        fair -= regime_scale * params["ret_weight"] * recent_move
        fair += regime_scale * signal["spread"] * (
            -params["total_imb_weight"] * signal["imbalance"]
            + params["l1_imb_weight"] * signal["l1_imbalance"]
        )

        effective_position = self.get_position(state, product) + inventory_target_shift
        position_ratio = effective_position / max(1, self.limits[product])
        quote_fair = fair - params["inventory_skew"] * position_ratio * signal["spread"]

        take_buy_threshold = max(1.0, params["take_threshold"])
        take_sell_threshold = max(1.0, params["take_threshold"])
        if best_ask <= fair - take_buy_threshold:
            self.place_buy(
                state,
                product,
                best_ask,
                min(int(signal["best_ask_volume"]), params["quote_size"]),
                f"{product} TAKE BUY @ {best_ask}",
            )
        if best_bid >= fair + take_sell_threshold:
            self.place_sell(
                state,
                product,
                best_bid,
                min(int(signal["best_bid_volume"]), params["quote_size"]),
                f"{product} TAKE SELL @ {best_bid}",
            )

        passive_offset = params["passive_offset"] / max(0.85, regime_scale)
        buy_price = min(best_bid + 1, math.floor(quote_fair - passive_offset))
        sell_price = max(best_ask - 1, math.ceil(quote_fair + passive_offset))
        if buy_price >= sell_price:
            buy_price = min(best_bid, sell_price - 1)
        quote_size = max(4, int(round(params["quote_size"] * regime_scale)))
        if abs(effective_position) > 0.6 * self.limits[product]:
            quote_size = max(2, quote_size // 2)

        self.place_buy(
            state,
            product,
            buy_price,
            quote_size,
            f"{product} MM BUY {quote_size} @ {buy_price} fair={fair:.2f}",
        )
        self.place_sell(
            state,
            product,
            sell_price,
            quote_size,
            f"{product} MM SELL {quote_size} @ {sell_price} fair={fair:.2f}",
        )

        memory[product]["ema"] = ema
        memory[product]["prev_mid"] = mid

    def trade_voucher(self, state: TradingState, product: str, info: dict[str, float], underlying_mid: float):
        if product not in state.order_depths:
            return
        if self.voucher_quote_size.get(product, 0) <= 0:
            return
        best_bid = info["best_bid"]
        best_ask = info["best_ask"]
        if best_bid is None or best_ask is None:
            return

        spread = max(1.0, info["spread"])
        regime = self.voucher_depth_regime.get(product, {"p25": 40.0, "p75": 50.0})
        regime_scale = self.regime_scale(
            float(info["depth_total"]),
            regime["p25"],
            regime["p75"],
        )
        fair = info["theo_price"]
        fair += regime_scale * spread * (
            -self.voucher_total_imb_weight * info["imbalance"]
            + self.voucher_l1_imb_weight * info["l1_imbalance"]
        )
        signal_strength = self.voucher_residual_weight.get(product, 1.0)
        take_threshold = max(1.0, (0.45 * spread) / max(0.5, signal_strength * regime_scale))
        passive_offset = max(1.0, (0.35 * spread) / max(0.6, signal_strength))

        if best_ask <= fair - take_threshold:
            self.place_buy(
                state,
                product,
                int(best_ask),
                min(int(info["best_ask_volume"]), self.voucher_quote_size[product]),
                f"{product} TAKE BUY @ {best_ask} fair={fair:.2f} residual={info['residual']:.2f}",
            )
        if best_bid >= fair + take_threshold:
            self.place_sell(
                state,
                product,
                int(best_bid),
                min(int(info["best_bid_volume"]), self.voucher_quote_size[product]),
                f"{product} TAKE SELL @ {best_bid} fair={fair:.2f} residual={info['residual']:.2f}",
            )

        position_ratio = self.get_position(state, product) / max(1, self.limits[product])
        quote_fair = fair - self.voucher_inventory_skew * position_ratio * spread
        buy_price = min(int(best_bid) + 1, math.floor(quote_fair - passive_offset))
        sell_price = max(int(best_ask) - 1, math.ceil(quote_fair + passive_offset))
        if buy_price >= sell_price:
            buy_price = min(int(best_bid), sell_price - 1)
        quote_size = max(4, int(round(self.voucher_quote_size[product] * regime_scale)))
        if abs(self.get_position(state, product)) > 0.65 * self.limits[product]:
            quote_size = max(2, quote_size // 2)

        self.place_buy(
            state,
            product,
            int(buy_price),
            quote_size,
            f"{product} MM BUY {quote_size} @ {buy_price} theo={info['theo_price']:.2f}",
        )
        self.place_sell(
            state,
            product,
            int(sell_price),
            quote_size,
            f"{product} MM SELL {quote_size} @ {sell_price} theo={info['theo_price']:.2f}",
        )

    def run(self, state: TradingState):
        self.reset_orders(state)
        memory = self.parse_memory(state.traderData)

        velvet_mid = None
        if self.velvet in state.order_depths:
            velvet_mid = self.wall_mid(state.order_depths[self.velvet])

        voucher_surface = {}
        voucher_delta_exposure = 0.0
        if velvet_mid is not None and velvet_mid > 0:
            voucher_surface = self.fit_voucher_surface(state, velvet_mid, state.timestamp)
            for product, info in voucher_surface.items():
                voucher_delta_exposure += self.get_position(state, product) * info["delta"]

        self.quote_delta_one(state, self.hydrogel, memory, 0.0)
        self.quote_delta_one(
            state,
            self.velvet,
            memory,
            inventory_target_shift=self.velvet_partial_delta_hedge * voucher_delta_exposure,
        )

        for product in self.vouchers:
            info = voucher_surface.get(product)
            if info is None:
                continue
            if info["market_mid"] <= 1.0 and self.strikes[product] >= 6000:
                continue
            self.trade_voucher(state, product, info, velvet_mid if velvet_mid is not None else 0.0)

        self.traderData = self.encode_memory(memory)
        logger.print(
            f"VELVET_DELTA_EXPOSURE={voucher_delta_exposure:.2f} "
            f"velvet_pos={self.get_position(state, self.velvet)} "
            f"hydrogel_pos={self.get_position(state, self.hydrogel)}"
        )
        logger.flush(state, self.orders, self.conversions, self.traderData)
        return self.orders, self.conversions, self.traderData
