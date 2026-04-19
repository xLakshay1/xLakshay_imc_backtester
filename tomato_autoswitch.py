def trade_tomatoes(self, state: TradingState):
    product = "TOMATOES"

    best_bid, best_ask = self.get_best_bid_ask(state, product)
    if best_bid is None or best_ask is None:
        return

    position = self.get_product_pos(state, product)
    mid = (best_bid + best_ask) / 2.0

    # ===== EMA =====
    if self.tomatoes_ema is None:
        self.tomatoes_ema = mid
    else:
        self.tomatoes_ema = 0.2 * mid + 0.8 * self.tomatoes_ema

    fair_value = self.tomatoes_ema

    # ===== VOLATILITY =====
    if self.tomatoes_last_mid is None:
        change = 0
    else:
        change = abs(mid - self.tomatoes_last_mid)

    if self.tomatoes_volatility is None:
        self.tomatoes_volatility = change
    else:
        self.tomatoes_volatility = 0.1 * change + 0.9 * self.tomatoes_volatility

    self.tomatoes_last_mid = mid

    vol = self.tomatoes_volatility

    # ===== REGIME SWITCH =====
    if vol < 1.5:
        # AGGRESSIVE MODE
        spread = 0.8 + vol
        take_size = 20
        quote_size = 25

    elif vol > 3.0:
        # SAFE MODE
        spread = 2.5 + vol
        take_size = 8
        quote_size = 10

    else:
        # BLENDED MODE
        spread = 1.5 + vol
        take_size = 12
        quote_size = 15

    # ===== TAKE TRADES =====
    if best_ask < fair_value - spread:
        size = min(take_size, self.limits[product] - position)
        if size > 0:
            self.send_buy_order(product, best_ask, size)

    if best_bid > fair_value + spread:
        size = min(take_size, self.limits[product] + position)
        if size > 0:
            self.send_sell_order(product, best_bid, -size)

    # ===== INVENTORY SKEW =====
    skew = position / 20

    buy_price = min(
        best_bid + 1,
        math.floor(fair_value - spread - max(0, skew))
    )

    sell_price = max(
        best_ask - 1,
        math.ceil(fair_value + spread + max(0, -skew))
    )

    max_buy = self.limits[product] - position
    max_sell = self.limits[product] + position

    max_buy = max(0, max_buy)
    max_sell = max(0, max_sell)

    max_buy = min(max_buy, quote_size)
    max_sell = min(max_sell, quote_size)

    if max_buy > 0:
        self.send_buy_order(product, buy_price, max_buy)

    if max_sell > 0:
        self.send_sell_order(product, sell_price, -max_sell)

    logger.print(
        f"{product} vol={vol:.2f} spread={spread:.2f} pos={position}"
    )