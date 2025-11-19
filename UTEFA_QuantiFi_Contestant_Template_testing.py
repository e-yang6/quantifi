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

# Global parameter dictionary, to be set by the optimizer
PARAMS: Dict = {}


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
        # Start with no shares and 100,000 cash
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
            raise ValueError(
                f"Attempted to sell {shares_to_sell} shares of {stock_name}, "
                f"but only {self.shares[stock_name]} available"
            )

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
            raise ValueError(
                f"Attempted to spend ${cost:.2f}, but only ${self.cash:.2f} available"
            )

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

# Global parameter dictionary, to be set by the optimizer
PARAMS: Dict = {}


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
    Store any data you need for your trading strategy.
    
    This class is completely customizable. Use it to track:
    - Historical prices
    - Calculated indicators (moving averages, momentum, etc.)
    - Trading signals
    - Strategy state
    
    Example usage:
        self.price_history = {stock: [] for stock in ["Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E"]}
        self.day_counter = 0
    """

    def __init__(self) -> None:
        # Parameters injected from optimizer
        self.params: Dict = PARAMS

        # The five stocks the dataset gives us
        self.stocks = ["Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E"]

        # Name of the CSV file with extra data
        self.csv_path = "UTEFA_QuantiFi_Contestant_Dataset.csv"

        # Price history per stock, built from Market.stocks each day
        self.price_history = {s: [] for s in self.stocks}

        # History of a simple equal-weight index of the 5 stocks
        self.index_history: List[float] = []

        # Macro series loaded from CSV (one value per day)
        self.interest_history: List[float] = []    # Interest_Rate
        self.growth_history: List[float] = []      # Economic_Growth
        self.inflation_history: List[float] = []   # Inflation

        # Per-stock series from CSV (one value per day per stock)
        self.volume_history = {s: [] for s in self.stocks}
        self.momentum10_history = {s: [] for s in self.stocks}

        # Day index (which CSV row / day we are on)
        self.day = 0
        self.num_rows = 0

        # Trading state
        self.in_position = False
        self.trade_count = 0
        self.max_trades = 20

        # Load CSV data once at startup
        self.load_csv()

        # Parameterized values
        self.fast_window = self.params.get("fast_ma_window", 20)
        self.slow_window = self.params.get("slow_ma_window", 50)
        self.momentum_window = self.params.get("momentum_window", 10)
        self.num_stocks = self.params.get("num_stocks", 2)
        self.use_trailing = self.params.get("use_trailing_stop", False)
        self.trailing_pct = self.params.get("trailing_stop_pct", 0.02)
        self.vol_lookback = self.params.get("vol_lookback", 30)
        self.low_vol = self.params.get("low_vol_threshold", 0.01)
        self.high_vol = self.params.get("high_vol_threshold", 0.03)
        self.volume_lookback = self.params.get("volume_lookback", 40)
        self.max_trades = self.params.get("max_trades", 20)

        #test

        # Track trailing stop peaks
        self.peak_price = {s: None for s in self.stocks}

        # fees
        self.total_fees = 0.0  

    def load_csv(self):

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
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def median_clean(self, values):
        clean = [v for v in values if v is not None]
        if not clean:
            return None
        return statistics.median(clean)

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


def update_portfolio(curMarket: Market, curPortfolio: Portfolio, context: Context):
    """
    Implement your trading strategy here.
    
    This function is called once per trading day, before the market updates.
    
    Args:
        curMarket: Current Market object with stock prices
        curPortfolio: Current Portfolio object with your holdings
        context: Context object for storing strategy data
    
    Example strategy (DO NOT USE THIS - IT'S JUST A PLACEHOLDER):
        # Track prices
        for stock in curMarket.stocks:
            context.price_history[stock].append(curMarket.stocks[stock])
        
        
        # Simple buy-and-hold: invest all cash on day 0
        if context.day == 0:
            for stock in curMarket.stocks:
                max_shares = curPortfolio.get_max_buyable_shares(stock, curMarket)
                if max_shares > 0:
                    curPortfolio.buy(stock, max_shares / 5, curMarket)  # Split equally
        
        context.day += 1
    """
    stocks = context.stocks
    d = context.day

    if d >= context.num_rows:
        context.day += 1
        return

    fee = Market.transaction_fee

    index_prices = []
    for s in stocks:
        px = curMarket.stocks[s]
        context.price_history[s].append(px)
        index_prices.append(px)

    index_level = sum(index_prices) / len(index_prices)
    context.index_history.append(index_level)

    # Need enough data for MA windows
    if len(context.index_history) < max(context.fast_window, context.slow_window) + 1:
        context.day += 1
        return

    def ma(values, window):
        if len(values) < window:
            return None

        ma_type = context.params.get("ma_type", "SMA")
        if ma_type == "SMA":
            return sum(values[-window:]) / float(window)
        else:
            alpha = 2 / (window + 1)
            ema = values[-window]
            for v in values[-window+1:]:
                ema = alpha * v + (1-alpha) * ema
            return ema

    fast_window = context.fast_window
    slow_window = context.slow_window

    fast_now  = ma(context.index_history, fast_window)
    slow_now  = ma(context.index_history, slow_window)
    fast_prev = ma(context.index_history[:-1], fast_window)
    slow_prev = ma(context.index_history[:-1], slow_window)

    if fast_now is None or slow_now is None:
        context.day += 1
        return

    risk_on_now  = fast_now > slow_now
    risk_on_prev = fast_prev > slow_prev
    cross_up = (not risk_on_prev) and risk_on_now
    cross_down = risk_on_prev and (not risk_on_now)

    interest  = context.interest_history[d]
    growth    = context.growth_history[d]
    inflation = context.inflation_history[d]

    # Momentum (parameterized)
    momentum_today = {}
    for s in stocks:
        if context.momentum_window == 10:
            momentum_today[s] = context.momentum10_history[s][d]
        else:
            ph = context.price_history[s]
            if len(ph) > context.momentum_window:
                momentum_today[s] = ph[-1] / ph[-context.momentum_window] - 1
            else:
                momentum_today[s] = None

    risk_mult = 1.0
    ir_med  = context.median_clean(context.interest_history[:d+1])
    eg_med  = context.median_clean(context.growth_history[:d+1])
    inf_med = context.median_clean(context.inflation_history[:d+1])

    if interest is not None and inflation is not None:
        if interest > ir_med and inflation > inf_med:
            risk_mult -= 0.05
        if growth < eg_med and inflation > inf_med:
            risk_mult -= 0.05
        if growth > eg_med and inflation <= inf_med and interest <= ir_med:
            risk_mult += 0.05

    risk_mult = max(context.params.get("macro_lower", 0.95),
                    min(risk_mult, context.params.get("macro_upper", 1.05)))

    if context.use_trailing and context.in_position:
        for s in stocks:
            px = curMarket.stocks[s]
            if curPortfolio.shares[s] > 0:
                if context.peak_price[s] is None:
                    context.peak_price[s] = px
                else:
                    context.peak_price[s] = max(context.peak_price[s], px)

                stop_price = context.peak_price[s] * (1 - context.trailing_pct)

                if px < stop_price and context.trade_count < context.max_trades:
                    shares = int(curPortfolio.shares[s])
                    try:
                        curPortfolio.sell(s, shares, curMarket)
                        context.trade_count += 1
                        context.total_fees += shares * px * fee
                    except:
                        pass

        still = any(curPortfolio.shares[s] > 0 for s in stocks)
        if not still:
            context.in_position = False
            for s in stocks:
                context.peak_price[s] = None

    if cross_up and not context.in_position and context.trade_count < context.max_trades:

        # Rank stocks by momentum
        momentum_pairs = [(s, momentum_today[s]) for s in stocks if momentum_today[s] is not None]
        if not momentum_pairs:
            context.day += 1
            return

        momentum_pairs.sort(key=lambda x: x[1], reverse=True)

        k = context.num_stocks
        top_stocks = [s for s, _ in momentum_pairs[:k]]

        # Weighting scheme
        if context.params.get("proportional_momentum", False):
            raw = {s: max(momentum_today[s], 0) for s in top_stocks}
            tot = sum(raw.values())
            weights = {s: raw[s] / tot if tot > 0 else 1/k for s in top_stocks}
        else:
            weights = {s: 1/k for s in top_stocks}

        # Volatility tilt
        for s in top_stocks:
            vol = context.rolling_volatility(s, context.vol_lookback)
            if vol is not None:
                if vol > context.high_vol:
                    weights[s] *= 0.97
                elif vol < context.low_vol:
                    weights[s] *= 1.03

        # Volume z-score tilt
        if d+1 >= context.volume_lookback:
            for s in top_stocks:
                recent = context.volume_history[s][d+1-context.volume_lookback:d+1]
                clean = [v for v in recent if v is not None]

                if len(clean) >= context.volume_lookback:
                    mean_v = sum(clean) / context.volume_lookback
                    var_v  = sum((v-mean_v)**2 for v in clean) / context.volume_lookback
                    std_v  = (var_v ** 0.5) if var_v > 0 else None

                    if std_v:
                        z = (clean[-1] - mean_v) / std_v
                        if z > 2:
                            weights[s] *= 1.02
                        elif z < -2:
                            weights[s] *= 0.98

        # Normalize weights
        totw = sum(weights.values())
        weights = {s: w/totw for s, w in weights.items()}

        # Reset trailing peaks only for bought stocks
        for s in stocks:
            context.peak_price[s] = curMarket.stocks[s] if s in top_stocks else None

        equity = curPortfolio.evaluate(curMarket)
        deploy = equity * risk_mult

        # Execute buys
        for s, w in weights.items():
            if context.trade_count >= context.max_trades:
                break

            px = curMarket.stocks[s]
            desired = deploy * w
            eff_px = px * (1 + fee)

            shares = int(min(desired // eff_px, curPortfolio.cash // eff_px))
            if shares > 0:
                try:
                    curPortfolio.buy(s, shares, curMarket)
                    context.trade_count += 1
                    context.total_fees += shares * px * fee
                except:
                    pass

        any_pos = any(curPortfolio.shares[s] > 0 for s in stocks)
        context.in_position = any_pos

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
                except:
                    pass

        still = any(curPortfolio.shares[s] > 0 for s in stocks)
        if not still:
            context.in_position = False
            for s in stocks:
                context.peak_price[s] = None

    context.day += 1
