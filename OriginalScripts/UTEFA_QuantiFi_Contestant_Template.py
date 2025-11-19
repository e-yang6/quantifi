"""
UTEFA QuantiFi - Contestant Template

This template provides the structure for implementing your trading strategy.
Your goal is to maximize portfolio value over 252 (range 0 to 251) trading days.

IMPORTANT:
- Implement your strategy in the update_portfolio() function
- You can store any data you need in the Context class
- Transaction fees apply to both buying and selling (0.5%)
- Do not modify the Market or Portfolio class structures
"""

import csv
import statistics
from typing import Dict, List


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
            "Stock_E": 0.0
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
        # Start with no shares and $100,000 cash
        self.shares = {
            "Stock_A": 0.0,
            "Stock_B": 0.0,
            "Stock_C": 0.0,
            "Stock_D": 0.0,
            "Stock_E": 0.0
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
        
        Args:
            stock_name: Name of the stock to sell (must match keys in self.shares)
            shares_to_sell: Number of shares to sell (must be positive)
            curMarket: Current Market object with stock prices
            
        Raises:
            ValueError: If shares_to_sell is invalid or exceeds owned shares
        """
        if shares_to_sell <= 0:
            raise ValueError("Number of shares must be positive")

        if stock_name not in self.shares:
            raise ValueError(f"Invalid stock name: {stock_name}")

        if shares_to_sell > self.shares[stock_name]:
            raise ValueError(f"Attempted to sell {shares_to_sell} shares of {stock_name}, but only {self.shares[stock_name]} available")

        # Update portfolio
        self.shares[stock_name] -= shares_to_sell
        sale_proceeds = (1 - Market.transaction_fee) * shares_to_sell * curMarket.stocks[stock_name]
        self.cash += sale_proceeds

    def buy(self, stock_name: str, shares_to_buy: float, curMarket: Market) -> None:
        """
        Buy shares of a specific stock.
        
        Args:
            stock_name: Name of the stock to buy (must match keys in self.shares)
            shares_to_buy: Number of shares to buy (must be positive)
            curMarket: Current Market object with stock prices
            
        Raises:
            ValueError: If shares_to_buy is invalid or exceeds available cash
        """
        if shares_to_buy <= 0:
            raise ValueError("Number of shares must be positive")
        
        if stock_name not in self.shares:
            raise ValueError(f"Invalid stock name: {stock_name}")
        
        cost = (1 + Market.transaction_fee) * shares_to_buy * curMarket.stocks[stock_name]
        
        if cost > self.cash + 0.01:
            raise ValueError(f"Attempted to spend ${cost:.2f}, but only ${self.cash:.2f} available")

        # Update portfolio
        self.shares[stock_name] += shares_to_buy
        self.cash -= cost

    def get_position_value(self, stock_name: str, curMarket: Market) -> float:
        """
        Helper method to get the current value of a specific position.
        
        Args:
            stock_name: Name of the stock
            curMarket: Current Market object with stock prices
            
        Returns:
            Float representing the total value of owned shares for this stock
        """
        return self.shares[stock_name] * curMarket.stocks[stock_name]

    def get_max_buyable_shares(self, stock_name: str, curMarket: Market) -> float:
        """
        Helper method to calculate the maximum number of shares that can be bought.
        
        Args:
            stock_name: Name of the stock
            curMarket: Current Market object with stock prices
            
        Returns:
            Float representing maximum shares that can be purchased with available cash
        """
        price_per_share = curMarket.stocks[stock_name] * (1 + Market.transaction_fee)
        return self.cash / price_per_share if price_per_share > 0 else 0

class Context:
    """
    Holds all the state your strategy needs between days:
    - price history from Market (for trends & volatility)
    - CSV data (macro, volume, momentum)
    - current day index, trade count, and position flag
    """

    def __init__(self) -> None:
        
        # The five stocks the dataset gives us
        self.stocks = ["Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E"]

        # Name of the CSV file with extra data
        self.csv_path = "UTEFA_QuantiFi_Contestant_Dataset.csv"

        # Price history per stock, built from Market.stocks each day
        self.price_history = {s: [] for s in self.stocks}

        # History of a simple equal-weight index of the 5 stocks (for 20/50 MA)
        self.index_history = []

        # Macro series loaded from CSV (one value per day)
        self.interest_history = []    # Interest_Rate
        self.growth_history = []      # Economic_Growth
        self.inflation_history = []   # Inflation

        # Per-stock series from CSV (one value per day per stock)
        self.volume_history = {s: [] for s in self.stocks}
        self.momentum10_history = {s: [] for s in self.stocks}

        # Day index (which CSV row / day we’re on)
        self.day = 0
        self.num_rows = 0

        # Trading state
        self.in_position = False          # do we currently own any stocks?
        self.trade_count = 0             # total trades so far
        self.max_trades = 20             # cap from your requirement

        # Load all CSV data once at startup
        self.load_csv()

    # ---------------- CSV loading ---------------- #

    def load_csv(self):
        """Read the CSV and fill the macro / volume / momentum histories."""

        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Macro values for this day
                self.interest_history.append(self.to_float(row.get("Interest_Rate")))
                self.growth_history.append(self.to_float(row.get("Economic_Growth")))
                self.inflation_history.append(self.to_float(row.get("Inflation")))

                # Per-stock volume and 10-day momentum
                for stock in self.stocks:
                    vol_col = f"{stock}_Volume"
                    mom_col = f"{stock}_Momentum_10d"

                    self.volume_history[stock].append(self.to_float(row.get(vol_col)))
                    self.momentum10_history[stock].append(self.to_float(row.get(mom_col)))

                self.num_rows += 1

    def to_float(self, value):
        """Try to convert a CSV cell to float; return None if it's empty/bad."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def median_clean(self, values):
        """Median of a list, ignoring None. Returns None if everything is missing."""
        clean = [v for v in values if v is not None]
        if not clean:
            return None
        return statistics.median(clean)

    # ---------------- Volatility helper ---------------- #

    def rolling_volatility(self, stock, window=30):
        """
        Estimate how "jumpy" a stock has been over the last `window` days.
        Uses standard deviation of daily returns over that window.
        Returns None until we have enough history.
        """

        prices = self.price_history[stock]

        # Need at least window+1 prices to get window daily returns
        if len(prices) < window + 1:
            return None

        recent = prices[-(window + 1):]

        # Compute simple daily returns
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
def update_portfolio(curMarket: Market, curPortfolio: Portfolio, context: Context):
    """
    Runs once per trading day.

    Core idea:
    - Build a simple index from the 5 stocks.
    - Use 20/50 SMA cross on that index as the only timing rule.
    - On a bullish cross (20 > 50 and was not before), buy a small basket
      of top-momentum stocks.
    - On a bearish cross (20 < 50 and was not before), sell everything.

    CSV data (macro, momentum, volume) shapes *what* we buy and *how much*,
    but never decides *whether* we trade. Timing is purely price-based.
    """

    stocks = context.stocks
    day_idx = context.day

    # Stop if we've run past the CSV data
    if day_idx >= context.num_rows:
        context.day += 1
        return

    # ---------------- 1) Update price and index history ---------------- #

    index_prices = []

    for stock in stocks:
        price = curMarket.stocks[stock]              # today’s price
        context.price_history[stock].append(price)   # store for later indicators
        index_prices.append(price)

    # Equal-weight index = simple average of today's 5 stock prices
    index_level = sum(index_prices) / float(len(index_prices))
    context.index_history.append(index_level)

    # Need at least 50 days of index data before we can do 20/50 logic
    if len(context.index_history) < 51:
        context.day += 1
        return

    # Small local helper to compute a trailing moving average
    def moving_average(values, window):
        if len(values) < window:
            return None
        return sum(values[-window:]) / float(window)

    # Current 20d and 50d moving averages
    fast_now = moving_average(context.index_history, 20)
    slow_now = moving_average(context.index_history, 50)

    # Yesterday's 20d and 50d moving averages (drop today's value)
    fast_prev = moving_average(context.index_history[:-1], 20)
    slow_prev = moving_average(context.index_history[:-1], 50)

    # If something is missing, we just skip the day
    if fast_now is None or slow_now is None or fast_prev is None or slow_prev is None:
        context.day += 1
        return

    # Risk-on when short MA is above long MA
    risk_on_now = fast_now > slow_now
    risk_on_prev = fast_prev > slow_prev

    # Cross up: yesterday off, today on → buy signal
    cross_up = (not risk_on_prev) and risk_on_now

    # Cross down: yesterday on, today off → sell signal
    cross_down = risk_on_prev and (not risk_on_now)

    fee = Market.transaction_fee  # e.g. 0.005

    # ---------------- 2) Read today's CSV values ---------------- #

    # Current macro values
    interest = context.interest_history[day_idx]
    growth = context.growth_history[day_idx]
    inflation = context.inflation_history[day_idx]

    # Current per-stock values
    momentum10_today = {s: context.momentum10_history[s][day_idx] for s in stocks}
    # volume_today kept for clarity; z-scores below will use the history
    volume_today = {s: context.volume_history[s][day_idx] for s in stocks}

    # Compute medians using *only up to today* → no look-ahead
    ir_med = context.median_clean(context.interest_history[: day_idx + 1])
    eg_med = context.median_clean(context.growth_history[: day_idx + 1])
    inf_med = context.median_clean(context.inflation_history[: day_idx + 1])

    # ---------------- 3) Macro → small risk multiplier ---------------- #

    # Start assuming we use 100% of our equity
    risk_multiplier = 1.0

    # If rates and inflation are both high vs history, lean a bit conservative
    if (interest is not None and inflation is not None and
        ir_med is not None and inf_med is not None and
        interest > ir_med and inflation > inf_med):
        risk_multiplier -= 0.05

    # If growth is weak and inflation is high, shave off a bit more
    if (growth is not None and eg_med is not None and
        inflation is not None and inf_med is not None and
        growth < eg_med and inflation > inf_med):
        risk_multiplier -= 0.05

    # If growth is strong and both inflation and rates are not elevated,
    # allow a small risk increase
    if (growth is not None and eg_med is not None and
        inflation is not None and inf_med is not None and
        interest is not None and ir_med is not None and
        growth > eg_med and inflation <= inf_med and interest <= ir_med):
        risk_multiplier += 0.05

    # Keep this between 0.95 and 1.05 so macro never dominates the strategy
    if risk_multiplier < 0.95:
        risk_multiplier = 0.95
    if risk_multiplier > 1.05:
        risk_multiplier = 1.05

    # ---------------- 4) ENTRY: trend flips OFF → ON ---------------- #

    if (not context.in_position) and cross_up and context.trade_count < context.max_trades:

        # Rank stocks by 10-day momentum from the CSV
        momentum_pairs = []
        for stock in stocks:
            m = momentum10_today.get(stock)
            if m is not None:
                momentum_pairs.append((stock, m))

        # If we have no momentum info at all, just skip this signal
        if not momentum_pairs:
            context.day += 1
            return

        # Sort strongest first
        momentum_pairs.sort(key=lambda x: x[1], reverse=True)

        # Focus on the top 2 momentum names (adapts to which stocks are strong)
        top_stocks = [s for (s, _) in momentum_pairs[:2]]
        if not top_stocks:
            context.day += 1
            return

        # Start with equal weights among the chosen names
        base_weight = 1.0 / float(len(top_stocks))
        weights_raw = {}

        # Thresholds for calling volatility "low" or "high"
        low_vol = 0.01   # ~1% daily
        high_vol = 0.03  # ~3% daily

        # Build simple volume z-scores over the last 40 days (up to today)
        vol_lookback = 40
        volume_z = {}

        for stock in stocks:
            full_vol = context.volume_history[stock]

            # Need at least 40 days of volume history
            if day_idx + 1 < vol_lookback:
                continue

            recent = full_vol[day_idx + 1 - vol_lookback : day_idx + 1]
            clean = [v for v in recent if v is not None]
            if len(clean) < vol_lookback:
                continue

            mean_v = sum(clean) / float(len(clean))
            var_v = sum((v - mean_v) ** 2 for v in clean) / float(len(clean))
            std_v = var_v ** 0.5
            if std_v <= 0:
                continue

            today_v = recent[-1]
            if today_v is None:
                continue

            volume_z[stock] = (today_v - mean_v) / std_v

        # Tilt weights for the top stocks based on volatility and volume patterns
        for stock in top_stocks:
            w = base_weight

            # Volatility tilt: penalise very jumpy names, reward very calm ones a bit
            vol = context.rolling_volatility(stock, window=30)
            if vol is not None:
                if vol > high_vol:
                    w *= 0.97
                elif vol < low_vol:
                    w *= 1.03

            # Volume tilt: if volume is unusually high, nudge weight up;
            # if unusually low, nudge down
            vz = volume_z.get(stock)
            if vz is not None:
                if vz > 2.0:
                    w *= 1.02
                elif vz < -2.0:
                    w *= 0.98

            weights_raw[stock] = w

        # Normalise weights so they sum to 1.0
        total_w = sum(weights_raw.values())
        if total_w <= 0:
            context.day += 1
            return
        final_weights = {s: w / total_w for s, w in weights_raw.items()}

        # Total trades this entry would need: 1 buy per chosen stock
        needed_trades = len(final_weights)
        if context.trade_count > context.max_trades - needed_trades:
            context.day += 1
            return

        # Work out current equity (cash + holdings)
        try:
            equity = curPortfolio.evaluate(curMarket)
        except Exception:
            equity = curPortfolio.cash

        # If we have nothing to deploy, skip
        if equity <= 0 or curPortfolio.cash <= 0:
            context.day += 1
            return

        # Adjust equity slightly up or down based on macro regime
        effective_equity = equity * risk_multiplier

        # Place buys for each selected stock
        for stock, weight in final_weights.items():
            if context.trade_count >= context.max_trades:
                break

            price = curMarket.stocks[stock]
            if price <= 0 or weight <= 0:
                continue

            # Target dollar amount for this stock
            desired_notional = effective_equity * weight

            # Effective per-share cost including fee
            effective_price = price * (1.0 + fee)
            if effective_price <= 0:
                continue

            # Cap by both how much we want to invest and how much cash we have
            max_by_budget = int(desired_notional // effective_price)
            max_by_cash = int(curPortfolio.cash // effective_price)
            shares_to_buy = min(max_by_budget, max_by_cash)

            if shares_to_buy <= 0:
                continue

            try:
                curPortfolio.buy(stock, shares_to_buy, curMarket)
                context.trade_count += 1
            except Exception:
                # If something goes wrong with this leg, just move on
                pass

        # If we now hold anything at all, flip the in_position flag
        any_position = any(curPortfolio.shares.get(s, 0.0) > 0.0 for s in stocks)
        if any_position:
            context.in_position = True

    # ---------------- 5) EXIT: trend flips ON → OFF ---------------- #

    if context.in_position and cross_down and context.trade_count < context.max_trades:
        # Flatten the book: sell everything we own
        for stock in stocks:
            if context.trade_count >= context.max_trades:
                break

            current_shares = int(curPortfolio.shares.get(stock, 0.0))
            if current_shares <= 0:
                continue

            try:
                curPortfolio.sell(stock, current_shares, curMarket)
                context.trade_count += 1
            except Exception:
                # If a sell fails for one name, still try the others
                pass

        # If no shares are left, we are flat again
        any_left = any(curPortfolio.shares.get(s, 0.0) > 0.0 for s in stocks)
        if not any_left:
            context.in_position = False

    # ---------------- 6) Advance to next day ---------------- #

    context.day += 1