from __future__ import annotations

import re
from pathlib import Path

ROOT = Path("/Users/lakshaykumar/Documents/Playground/imc-prosperity-4-fresh")
BASE = ROOT / "cand_no_micro_depth_top05.py"


def set_trader_data(source: str, name: str) -> str:
    return re.sub(
        r'self\.traderData = "[^"]+"',
        f'self.traderData = "{name}"',
        source,
        count=1,
    )


def add_init_params(source: str, params: dict[str, str]) -> str:
    insertion = "".join(f"        self.{key} = {value}\n" for key, value in params.items())
    marker = "        self.osmium_alpha_clip = 3.0\n"
    if insertion.strip() and insertion not in source:
        source = source.replace(marker, marker + insertion, 1)
    return source


def apply_pepper_sweep(source: str) -> str:
    replacement = r'''
    def trade_pepper(self, state: TradingState):
        product = "INTARIAN_PEPPER_ROOT"

        if product not in state.order_depths:
            return

        order_depth = state.order_depths[product]
        if len(order_depth.sell_orders) == 0:
            return

        position = self.get_product_pos(state, product)
        remaining_buy = self.pepper_limit - position - self.pepper_buy_orders
        if remaining_buy <= 0:
            return

        sell_side = sorted(order_depth.sell_orders.items())
        # Pepper has a persistent structural upward drift. Paying through all visible asks
        # improves early fill probability and costs only a few ticks versus the daily move.
        limit_price = max(int(price) for price, _amount in sell_side[:3])
        size = min(remaining_buy, self.pepper_clip_size)
        self.pepper_buy_orders += size
        self.send_buy_order(
            product,
            limit_price,
            size,
            f"PEPPER SWEEP BUY {size} @ {limit_price}",
        )

'''
    return re.sub(
        r"    def trade_pepper\(self, state: TradingState\):\n.*?\n    def reset_orders",
        replacement + "    def reset_orders",
        source,
        flags=re.S,
        count=1,
    )


def apply_pepper_passive(source: str, deadline: int, bid_offset: int) -> str:
    replacement = f'''
    def trade_pepper(self, state: TradingState):
        product = "INTARIAN_PEPPER_ROOT"

        if product not in state.order_depths:
            return

        order_depth = state.order_depths[product]
        position = self.get_product_pos(state, product)
        remaining_buy = self.pepper_limit - position - self.pepper_buy_orders
        if remaining_buy <= 0:
            return

        best_bid = max(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) > 0 else None
        best_ask = min(order_depth.sell_orders.keys()) if len(order_depth.sell_orders) > 0 else None
        if best_bid is None and best_ask is None:
            return

        if state.timestamp < {deadline} and best_bid is not None:
            price = best_bid + {bid_offset}
            if best_ask is not None:
                price = min(price, best_ask - 1)
        elif best_ask is not None:
            price = best_ask
        elif best_bid is not None:
            price = best_bid + {bid_offset}
        else:
            return

        size = min(remaining_buy, self.pepper_clip_size)
        self.pepper_buy_orders += size
        self.send_buy_order(
            product,
            price,
            size,
            f"PEPPER PASSIVE BUY {{size}} @ {{price}}",
        )

'''
    return re.sub(
        r"    def trade_pepper\(self, state: TradingState\):\n.*?\n    def reset_orders",
        replacement + "    def reset_orders",
        source,
        flags=re.S,
        count=1,
    )


def apply_one_sided_osmium(source: str) -> str:
    old = '''        best_bid, best_ask = self.get_best_bid_ask(state, product)
        if best_bid is None or best_ask is None:
            return

        position = self.get_product_pos(state, product)
        features = self.get_osmium_alpha_features(state, product)
'''
    new = '''        order_depth = state.order_depths[product]
        best_bid = max(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) > 0 else None
        best_ask = min(order_depth.sell_orders.keys()) if len(order_depth.sell_orders) > 0 else None

        if best_bid is None or best_ask is None:
            if self.osmium_fair_value is None:
                return
            if best_ask is not None:
                max_buy = max(
                    0,
                    self.limits[product] - self.osmium_position - self.osmium_buy_orders,
                )
                available = -order_depth.sell_orders.get(best_ask, 0)
                size = min(max_buy, available)
                if size > 0 and best_ask <= math.floor(self.osmium_fair_value - 1.0):
                    self.osmium_buy_orders += size
                    self.send_buy_order(
                        product,
                        best_ask,
                        size,
                        f"{product} ONE-SIDED TAKE BUY {size} @ {best_ask}",
                    )
            if best_bid is not None:
                max_sell = max(
                    0,
                    self.osmium_position + self.limits[product] - self.osmium_sell_orders,
                )
                available = order_depth.buy_orders.get(best_bid, 0)
                size = min(max_sell, available)
                if size > 0 and best_bid >= math.ceil(self.osmium_fair_value + 1.0):
                    self.osmium_sell_orders += size
                    self.send_sell_order(
                        product,
                        best_bid,
                        -size,
                        f"{product} ONE-SIDED TAKE SELL {size} @ {best_bid}",
                    )
            return

        position = self.get_product_pos(state, product)
        features = self.get_osmium_alpha_features(state, product)
'''
    if old not in source:
        raise RuntimeError("trade_osmium opening block not found")
    return source.replace(old, new, 1)


def apply_wallmid_features(source: str) -> str:
    params = {
        "osmium_micro_sign_weight": "0.0",
        "osmium_void_sign_weight": "0.0",
        "osmium_wallmid_vote_weight": "0.0",
    }
    source = add_init_params(source, params)

    void_code = '''        bid_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
        ask_prices = sorted(order_depth.sell_orders.keys())
        bid_gap = best_bid - bid_prices[1] if len(bid_prices) > 1 else best_ask - best_bid
        ask_gap = ask_prices[1] - best_ask if len(ask_prices) > 1 else best_ask - best_bid
        void_skew = ask_gap - bid_gap

'''
    marker = "        total_depth = 0\n"
    if "void_skew = ask_gap - bid_gap" not in source:
        source = source.replace(marker, void_code + marker, 1)

    source = source.replace(
        '            "liquidity_pressure": liquidity_pressure,\n',
        '            "liquidity_pressure": liquidity_pressure,\n'
        '            "void_skew": void_skew,\n',
        1,
    )

    signal_code = '''        void_skew = features.get("void_skew", 0.0)
        micro_sign_signal = 1.0 if micro_signal > 0 else (-1.0 if micro_signal < 0 else 0.0)
        void_sign_signal = 1.0 if void_skew > 0 else (-1.0 if void_skew < 0 else 0.0)
        top_sign_signal = 1.0 if top_imbalance > 0 else (-1.0 if top_imbalance < 0 else 0.0)
        wallmid_vote_signal = self.clamp(
            micro_sign_signal + void_sign_signal + top_sign_signal,
            -2.0,
            2.0,
        )
'''
    marker = '        liquidity_pressure = features["liquidity_pressure"]\n'
    if "wallmid_vote_signal" not in source:
        source = source.replace(marker, marker + signal_code, 1)

    old = "            + self.osmium_liquidity_pressure_weight * liquidity_pressure\n        )"
    new = (
        "            + self.osmium_liquidity_pressure_weight * liquidity_pressure\n"
        "            + self.osmium_micro_sign_weight * micro_sign_signal\n"
        "            + self.osmium_void_sign_weight * void_sign_signal\n"
        "            + self.osmium_wallmid_vote_weight * wallmid_vote_signal\n"
        "        )"
    )
    if "self.osmium_wallmid_vote_weight * wallmid_vote_signal" not in source:
        source = source.replace(old, new, 1)
    return source


def set_param(source: str, param: str, value: str) -> str:
    return re.sub(
        rf"self\.{re.escape(param)} = [^\n]+",
        f"self.{param} = {value}",
        source,
        count=1,
    )


def apply_take_alpha(source: str, mode: str) -> str:
    if mode == "half":
        replacement = "        take_fair_value = self.osmium_fair_value + 0.50 * alpha_adjustment"
    elif mode == "full":
        replacement = "        take_fair_value = self.osmium_fair_value + alpha_adjustment"
    else:
        raise ValueError(mode)
    return source.replace("        take_fair_value = self.osmium_fair_value", replacement, 1)


def apply_gamma_target(source: str) -> str:
    old = '''        inv_ratio = position / self.limits[product]
        inventory_skew = inv_ratio * 2.0
'''
    new = '''        target_position = 0
        if ret_5_signal <= -2.0:
            target_position = 20
        elif ret_5_signal >= 2.0:
            target_position = -20
        if top_imbalance >= 0.35:
            target_position += 10
        elif top_imbalance <= -0.35:
            target_position -= 10
        target_position = int(self.clamp(target_position, -30, 30))

        inv_ratio = (position - target_position) / self.limits[product]
        inventory_skew = inv_ratio * 2.0
'''
    if old not in source:
        raise RuntimeError("inventory skew block not found")
    return source.replace(old, new, 1)


def build_candidate(name: str, transforms: list[tuple[str, object]]) -> None:
    source = BASE.read_text()
    source = set_trader_data(source, name.upper().replace(".PY", ""))

    for transform, value in transforms:
        if transform == "pepper_sweep":
            source = apply_pepper_sweep(source)
        elif transform == "pepper_passive":
            deadline, bid_offset = value
            source = apply_pepper_passive(source, int(deadline), int(bid_offset))
        elif transform == "one_sided_osmium":
            source = apply_one_sided_osmium(source)
        elif transform == "wallmid":
            source = apply_wallmid_features(source)
            for key, param_value in dict(value).items():
                source = set_param(source, key, str(param_value))
        elif transform == "take_alpha":
            source = apply_take_alpha(source, str(value))
        elif transform == "gamma_target":
            source = apply_gamma_target(source)
        else:
            raise ValueError(transform)

    path = ROOT / name
    path.write_text(source, encoding="utf-8")
    print(path.name)


def main() -> None:
    candidates = {
        "s9700_pepper_sweep.py": [("pepper_sweep", None)],
        "s9700_pepper_passive_5k_bid0.py": [("pepper_passive", (5_000, 0))],
        "s9700_pepper_passive_5k_bid1.py": [("pepper_passive", (5_000, 1))],
        "s9700_pepper_passive_10k_bid1.py": [("pepper_passive", (10_000, 1))],
        "s9700_pepper_passive_20k_bid1.py": [("pepper_passive", (20_000, 1))],
        "s9700_pepper_passive_10k_bid2.py": [("pepper_passive", (10_000, 2))],
        "s9700_one_sided.py": [("one_sided_osmium", None)],
        "s9700_pepper_sweep_one_sided.py": [
            ("pepper_sweep", None),
            ("one_sided_osmium", None),
        ],
        "s9700_wallmid_micro_sign.py": [
            ("wallmid", {"osmium_micro_sign_weight": 1.0}),
        ],
        "s9700_wallmid_void_sign.py": [
            ("wallmid", {"osmium_void_sign_weight": 1.0}),
        ],
        "s9700_wallmid_vote.py": [
            ("wallmid", {"osmium_wallmid_vote_weight": 1.0}),
        ],
        "s9700_wallmid_vote_take_half.py": [
            ("wallmid", {"osmium_wallmid_vote_weight": 1.0}),
            ("take_alpha", "half"),
        ],
        "s9700_wallmid_vote_take_full.py": [
            ("wallmid", {"osmium_wallmid_vote_weight": 1.0}),
            ("take_alpha", "full"),
        ],
        "s9700_gamma_target.py": [("gamma_target", None)],
        "s9700_gamma_wallmid.py": [
            ("wallmid", {"osmium_wallmid_vote_weight": 1.0}),
            ("gamma_target", None),
        ],
        "s9700_stack_structural.py": [
            ("pepper_sweep", None),
            ("one_sided_osmium", None),
            ("wallmid", {"osmium_wallmid_vote_weight": 1.0}),
            ("take_alpha", "half"),
            ("gamma_target", None),
        ],
    }
    for name, transforms in candidates.items():
        build_candidate(name, transforms)


if __name__ == "__main__":
    main()
