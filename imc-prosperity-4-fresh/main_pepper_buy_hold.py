from datamodel import Order, Symbol, TradingState


class Trader:
    def __init__(self) -> None:
        self.product = "INTARIAN_PEPPER_ROOT"
        self.limit = 80
        self.trader_data = "PEPPER_BUY_HOLD_ONLY_V1"

    def run(self, state: TradingState):
        orders: dict[Symbol, list[Order]] = {
            symbol: [] for symbol in state.order_depths
        }
        conversions = 0

        if self.product not in state.order_depths:
            return orders, conversions, self.trader_data

        position = state.position.get(self.product, 0)
        remaining = self.limit - position
        if remaining <= 0:
            return orders, conversions, self.trader_data

        order_depth = state.order_depths[self.product]
        if not order_depth.sell_orders:
            return orders, conversions, self.trader_data

        # Buy from the cheapest visible asks until we reach the +80 limit.
        for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
            if remaining <= 0:
                break

            available = -ask_volume
            if available <= 0:
                continue

            quantity = min(remaining, available)
            orders[self.product].append(Order(self.product, int(ask_price), int(quantity)))
            remaining -= quantity

        return orders, conversions, self.trader_data
