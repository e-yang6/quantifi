"""
UTEFA QuantiFi - Contestant Template (Final ADX + ATR Strategy)

Core ideas:
- Trend timing from index EMA crossover (fast 20, slow 30)
- Cross-sectional 10 day momentum across 5 stocks
- ADX filter to include only strong trend names
- ATR, volatility and volume tilts to shape position sizing
- Trailing stop per stock, macro regime risk scaling
""" 

"""recommiting"""

import csv
import statistics
from typing import List


class Market:
    """
    Represents the stock market with current prices for all available stocks.

    Attributes:
        transaction_fee: Float representing the transaction fee (0.5% = 0.005)
        stocks: Dictionary mapping stock names to their current prices
    """
    transaction_fee = 0.005

    def __init__(self) -> None:
        # Initialize with 5 stocks
        # Prices will be set by the backtesting script from the CSV data
        self.stocks = {
            "Stock_A": 0.0,
            "Stock_B": 0.0,
            "Stock_C": 0.0,
            "Stock_D": 0.0,
            "Stock_E": 0.0,
        }

    def updateMarket(self):
        """
        Updates stock prices to reflect market changes.
        This function will be implemented during grading.
        DO NOT MODIFY THIS METHOD.
        """
        pass


class Portfolio:
    """
    Represents your investment portfolio containing shares and cash.

    Attributes:
        shares: Dictionary mapping stock names to number of shares owned
        cash: Float representing available cash balance
    """

    def __init__(self) -> None:
        # Start with no shares and 100,000 cash
        self.shares = {
            "Stock_A": 0.0,
            "Stock_B": 0.0,
            "Stock_C": 0.0,
            "Stock_D": 0.0,
            "Stock_E": 0.0,
        }
        self.cash = 100000.0

    def evaluate(self, curMarket: Market) -> float:
        """
        Calculate the total value of the portfolio (shares + cash).

        Args:
            curMarket: Current Market object with stock prices

        Returns:
            Float representing total portfolio value
        """
        total_value = self.cash

        for stock_name, num_shares in self.shares.items():
            total_value += num_shares * curMarket.stocks[stock_name]

        return total_value

    def sell(self, stock_name: str, shares_to_sell: float, curMarket: Market) -> None:
        """
        Sell shares of a specific stock.
        """
        if shares_to_sell <= 0:
            raise ValueError("Number of shares must be positive")

        if stock_name not in self.shares:
            raise ValueError(f"Invalid stock name: {stock_name}")

        if shares_to_sell > self.shares[stock_name]:
            raise ValueError(
                f"Attempted to sell {shares_to_sell} shares of {stock_name}, "
                f"but only {self.shares[stock_name]} available"
            )

        self.shares[stock_name] -= shares_to_sell
        sale_proceeds = (1 - Market.transaction_fee) * shares_to_sell * curMarket.stocks[stock_name]
        self.cash += sale_proceeds

    def buy(self, stock_name: str, shares_to_buy: float, curMarket: Market) -> None:
        """
        Buy shares of a specific stock.
        """
        if shares_to_buy <= 0:
            raise ValueError("Number of shares must be positive")

        if stock_name not in self.shares:
            raise ValueError(f"Invalid stock name: {stock_name}")

        cost = (1 + Market.transaction_fee) * shares_to_buy * curMarket.stocks[stock_name]

        if cost > self.cash + 0.01:
            raise ValueError(
                f"Attempted to spend ${cost:.2f}, but only ${self.cash:.2f} available"
            )

        self.shares[stock_name] += shares_to_buy
        self.cash -= cost

    def get_position_value(self, stock_name: str, curMarket: Market) -> float:
        """
        Helper method to get the current value of a specific position.
        """
        return self.shares[stock_name] * curMarket.stocks[stock_name]

    def get_max_buyable_shares(self, stock_name: str, curMarket: Market) -> float:
        """
        Helper method to calculate the maximum number of shares that can be bought.
        """
        price_per_share = curMarket.stocks[stock_name] * (1 + Market.transaction_fee)
        return self.cash / price_per_share if price_per_share > 0 else 0


class Context:
    """
    Holds all the state your strategy needs between days:
    - price history from Market (for trends, volatility, ATR, ADX)
    - CSV data (macro, volume, momentum)
    - current day index, trade count, and position state
    """

    def __init__(self) -> None:
        # Universe
        self.stocks = ["Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E"]

        # CSV path
        self.csv_path = "UTEFA_QuantiFi_Contestant_Dataset.csv"

        # Price history per stock
        self.price_history = {s: [] for s in self.stocks}

        # Index history
        self.index_history: List[float] = []

        # Macro series
        self.interest_history: List[float] = []
        self.growth_history: List[float] = []
        self.inflation_history: List[float] = []

        # Volume and 10 day momentum from CSV
        self.volume_history = {s: [] for s in self.stocks}
        self.momentum10_history = {s: [] for s in self.stocks}

        # ATR / ADX history
        self.atr_history = {s: [] for s in self.stocks}
        self.adx_history = {s: [] for s in self.stocks}

        # Time state
        self.day = 0
        self.num_rows = 0

        # Trading state
        self.in_position = False
        self.trade_count = 0

        # Hardcoded final parameters from optimization
        self.ma_type = "EMA"
        self.fast_window = 20
        self.slow_window = 30

        self.momentum_window = 10
        self.proportional_momentum = False
        self.num_stocks = 3

        self.use_trailing = True
        self.trailing_pct = 0.05

        self.vol_lookback = 20
        self.low_vol = 0.01
        self.high_vol = 0.03
        self.volume_lookback = 40

        self.atr_window = 10
        self.adx_window = 14
        self.adx_min = 20.0
        self.adx_max = 50.0

        self.macro_lower = 0.95
        self.macro_upper = 1.05

        self.max_trades = 20

        # Trailing peaks
        self.peak_price = {s: None for s in self.stocks}

        # Fee tracking for analysis
        self.total_fees = 0.0

        # Load CSV data once
        self.load_csv()

    # ---------------- CSV loading ---------------- #

    def load_csv(self):
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Macro
                self.interest_history.append(self.to_float(row.get("Interest_Rate")))
                self.growth_history.append(self.to_float(row.get("Economic_Growth")))
                self.inflation_history.append(self.to_float(row.get("Inflation")))

                # Per stock
                for stock in self.stocks:
                    vol_col = f"{stock}_Volume"
                    mom_col = f"{stock}_Momentum_10d"
                    self.volume_history[stock].append(self.to_float(row.get(vol_col)))
                    self.momentum10_history[stock].append(self.to_float(row.get(mom_col)))

                self.num_rows += 1

    def to_float(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def median_clean(self, values):
        clean = [v for v in values if v is not None]
        if not clean:
            return None
        return statistics.median(clean)

    # ---------------- Volatility / ATR / ADX helpers ---------------- #

    def rolling_volatility(self, stock: str, window: int = 30):
        prices = self.price_history[stock]
        if len(prices) < window + 1:
            return None

        recent = prices[-(window + 1):]
        returns = []
        for i in range(1, len(recent)):
            prev_price = recent[i - 1]
            curr_price = recent[i]
            if prev_price > 0:
                returns.append((curr_price / prev_price) - 1.0)

        if len(returns) < window:
            return None

        mean_r = sum(returns) / len(returns)
        var = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        return var ** 0.5

    def compute_atr(self, prices, window=None):
        """
        ATR proxy using close-only data:
        TR_t = |close_t - close_{t-1}|
        """
        if window is None:
            window = self.atr_window

        if len(prices) < window + 1:
            return None

        trs = []
        for i in range(-window, 0):
            tr = abs(prices[i] - prices[i - 1])
            trs.append(tr)

        if not trs:
            return None

        return sum(trs) / float(window)

    def compute_adx(self, prices, window=None):
        """
        ADX-style trend strength proxy using close-only data.
        Uses +DM / -DM from close changes and a simple averaging.
        Returns a value roughly in [0, 100].
        """
        if window is None:
            window = self.adx_window

        if len(prices) < window + 1:
            return None

        plus_dm = []
        minus_dm = []
        tr_list = []

        for i in range(-window, 0):
            diff = prices[i] - prices[i - 1]
            up_move = max(diff, 0.0)
            down_move = max(-diff, 0.0)
            tr = abs(diff)

            plus_dm.append(up_move)
            minus_dm.append(down_move)
            tr_list.append(tr)

        avg_tr = sum(tr_list) / float(window)
        if avg_tr <= 0:
            return None

        avg_plus_dm = sum(plus_dm) / float(window)
        avg_minus_dm = sum(minus_dm) / float(window)

        plus_di = 100.0 * (avg_plus_dm / avg_tr)
        minus_di = 100.0 * (avg_minus_dm / avg_tr)
        denom = plus_di + minus_di
        if denom <= 0:
            return None

        dx = 100.0 * abs(plus_di - minus_di) / denom
        return dx


def update_portfolio(curMarket: Market, curPortfolio: Portfolio, context: Context):
    """
    Runs once per trading day.

    Core idea:
    - Build an equal weight index from the 5 stocks.
    - Use EMA(20) and EMA(30) on that index for timing.
    - On a bullish cross (fast > slow and was not before), buy a basket
      of top momentum stocks that pass an ADX filter.
    - On a bearish cross (fast < slow and was not before), sell everything.
    - Per stock trailing stops and macro multiplier adjust risk.
    """

    stocks = context.stocks
    d = context.day

    # Stop if we run past CSV data
    if d >= context.num_rows:
        context.day += 1
        return

    fee = Market.transaction_fee

    # ----------------------------------------------------
    # 1) Update price history, ATR / ADX, and index
    # ----------------------------------------------------
    index_prices = []
    for s in stocks:
        px = curMarket.stocks[s]
        context.price_history[s].append(px)
        index_prices.append(px)

        atr_val = context.compute_atr(context.price_history[s], window=context.atr_window)
        adx_val = context.compute_adx(context.price_history[s], window=context.adx_window)

        context.atr_history[s].append(atr_val)
        context.adx_history[s].append(adx_val)

    index_level = sum(index_prices) / len(index_prices)
    context.index_history.append(index_level)

    # ----------------------------------------------------
    # 2) Moving averages on index (trend regime)
    # ----------------------------------------------------
    def ma(values, window):
        if len(values) < window:
            return None

        if context.ma_type == "SMA":
            return sum(values[-window:]) / float(window)
        else:
            alpha = 2.0 / (window + 1.0)
            ema = values[-window]
            for v in values[-window + 1:]:
                ema = alpha * v + (1.0 - alpha) * ema
            return ema

    fast_window = context.fast_window
    slow_window = context.slow_window

    fast_now = ma(context.index_history, fast_window)
    slow_now = ma(context.index_history, slow_window)
    fast_prev = ma(context.index_history[:-1], fast_window)
    slow_prev = ma(context.index_history[:-1], slow_window)

    if any(v is None for v in [fast_now, slow_now, fast_prev, slow_prev]):
        context.day += 1
        return

    risk_on_now = fast_now > slow_now
    risk_on_prev = fast_prev > slow_prev
    cross_up = (not risk_on_prev) and risk_on_now
    cross_down = risk_on_prev and (not risk_on_now)

    # ----------------------------------------------------
    # 3) Read CSV macro and momentum
    # ----------------------------------------------------
    interest = context.interest_history[d]
    growth = context.growth_history[d]
    inflation = context.inflation_history[d]

    momentum_today = {}
    for s in stocks:
        if context.momentum_window == 10:
            momentum_today[s] = context.momentum10_history[s][d]
        else:
            ph = context.price_history[s]
            if len(ph) > context.momentum_window:
                momentum_today[s] = ph[-1] / ph[-context.momentum_window] - 1.0
            else:
                momentum_today[s] = None

    # ----------------------------------------------------
    # 4) Macro risk multiplier
    # ----------------------------------------------------
    risk_mult = 1.0
    ir_med = context.median_clean(context.interest_history[: d + 1])
    eg_med = context.median_clean(context.growth_history[: d + 1])
    inf_med = context.median_clean(context.inflation_history[: d + 1])

    if interest is not None and inflation is not None:
        if ir_med is not None and inf_med is not None:
            if interest > ir_med and inflation > inf_med:
                risk_mult -= 0.05

        if growth is not None and eg_med is not None and inf_med is not None:
            if growth < eg_med and inflation > inf_med:
                risk_mult -= 0.05
            if growth > eg_med and inflation <= inf_med and ir_med is not None and interest <= ir_med:
                risk_mult += 0.05

    if risk_mult < context.macro_lower:
        risk_mult = context.macro_lower
    if risk_mult > context.macro_upper:
        risk_mult = context.macro_upper

    # ----------------------------------------------------
    # 5) Trailing stop exits (per stock)
    # ----------------------------------------------------
    if context.use_trailing and context.in_position:
        for s in stocks:
            px = curMarket.stocks[s]
            if curPortfolio.shares[s] > 0:
                if context.peak_price[s] is None:
                    context.peak_price[s] = px
                else:
                    context.peak_price[s] = max(context.peak_price[s], px)

                stop_price = context.peak_price[s] * (1.0 - context.trailing_pct)

                if px < stop_price and context.trade_count < context.max_trades:
                    shares = int(curPortfolio.shares[s])
                    if shares > 0:
                        try:
                            curPortfolio.sell(s, shares, curMarket)
                            context.trade_count += 1
                            context.total_fees += shares * px * fee
                        except Exception:
                            pass

        still = any(curPortfolio.shares[s] > 0 for s in stocks)
        if not still:
            context.in_position = False
            for s in stocks:
                context.peak_price[s] = None

    # ----------------------------------------------------
    # 6) Entry: trend flips off -> on
    # ----------------------------------------------------
    if cross_up and not context.in_position and context.trade_count < context.max_trades:

        # Momentum + ADX filter
        momentum_pairs = []
        for s in stocks:
            m = momentum_today[s]
            if m is None:
                continue

            adx_list = context.adx_history[s]
            adx_today = adx_list[-1] if adx_list else None
            if adx_today is None:
                continue

            # Strict ADX band filter
            if adx_today < context.adx_min:
                continue
            if adx_today > context.adx_max:
                continue

            momentum_pairs.append((s, m))

        if not momentum_pairs:
            context.day += 1
            return

        momentum_pairs.sort(key=lambda x: x[1], reverse=True)

        k = context.num_stocks
        top_stocks = [s for s, _ in momentum_pairs[:k]]

        if not top_stocks:
            context.day += 1
            return

        # Base weights
        if context.proportional_momentum:
            raw = {s: max(momentum_today[s], 0.0) for s in top_stocks}
            tot = sum(raw.values())
            if tot > 0:
                weights = {s: raw[s] / tot for s in top_stocks}
            else:
                weights = {s: 1.0 / float(k) for s in top_stocks}
        else:
            weights = {s: 1.0 / float(k) for s in top_stocks}

        # Volatility tilt
        for s in top_stocks:
            vol = context.rolling_volatility(s, context.vol_lookback)
            if vol is not None:
                if vol > context.high_vol:
                    weights[s] *= 0.97
                elif vol < context.low_vol:
                    weights[s] *= 1.03

        # ATR tilt relative to median among selected names
        atr_today_map = {}
        for s in top_stocks:
            hist = context.atr_history[s]
            atr_today_map[s] = hist[-1] if hist else None

        atr_vals = [v for v in atr_today_map.values() if v is not None]
        atr_med = context.median_clean(atr_vals) if atr_vals else None

        if atr_med is not None:
            for s in top_stocks:
                atr_val = atr_today_map.get(s)
                if atr_val is None:
                    continue
                if atr_val > atr_med:
                    weights[s] *= 0.99
                else:
                    weights[s] *= 1.01

        # Volume z score tilt
        if d + 1 >= context.volume_lookback:
            for s in top_stocks:
                recent = context.volume_history[s][d + 1 - context.volume_lookback : d + 1]
                clean = [v for v in recent if v is not None]
                if len(clean) >= context.volume_lookback:
                    mean_v = sum(clean) / float(len(clean))
                    var_v = sum((v - mean_v) ** 2 for v in clean) / float(len(clean))
                    std_v = (var_v ** 0.5) if var_v > 0 else None
                    if std_v:
                        z = (clean[-1] - mean_v) / std_v
                        if z > 2.0:
                            weights[s] *= 1.02
                        elif z < -2.0:
                            weights[s] *= 0.98

        totw = sum(weights.values())
        if totw <= 0:
            context.day += 1
            return
        weights = {s: w / totw for s, w in weights.items()}

        # Reset peaks only for bought names
        for s in stocks:
            context.peak_price[s] = curMarket.stocks[s] if s in top_stocks else None

        equity = curPortfolio.evaluate(curMarket)
        if equity <= 0 or curPortfolio.cash <= 0:
            context.day += 1
            return

        deploy = equity * risk_mult

        # Execute buys
        for s, w in weights.items():
            if context.trade_count >= context.max_trades:
                break

            px = curMarket.stocks[s]
            if px <= 0 or w <= 0:
                continue

            desired = deploy * w
            eff_px = px * (1.0 + fee)

            shares = int(min(desired // eff_px, curPortfolio.cash // eff_px))
            if shares > 0:
                try:
                    curPortfolio.buy(s, shares, curMarket)
                    context.trade_count += 1
                    context.total_fees += shares * px * fee
                except Exception:
                    pass

        context.in_position = any(curPortfolio.shares[s] > 0 for s in stocks)

    # ----------------------------------------------------
    # 7) Exit: trend flips on -> off
    # ----------------------------------------------------
    if cross_down and context.in_position and context.trade_count < context.max_trades:
        for s in stocks:
            if context.trade_count >= context.max_trades:
                break

            sh = int(curPortfolio.shares[s])
            if sh > 0:
                try:
                    px = curMarket.stocks[s]
                    curPortfolio.sell(s, sh, curMarket)
                    context.trade_count += 1
                    context.total_fees += sh * px * fee
                except Exception:
                    pass

        still = any(curPortfolio.shares[s] > 0 for s in stocks)
        if not still:
            context.in_position = False
            for s in stocks:
                context.peak_price[s] = None

    # ----------------------------------------------------
    # Next day
    # ----------------------------------------------------
    context.day += 1
