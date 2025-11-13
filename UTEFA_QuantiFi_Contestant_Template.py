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

from math import floor
import statistics

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
    - Calculculated indicators (moving averages, momentum, etc.)
    - Trading signals
    - Strategy state
    
    Example usage:
        self.price_history = {stock: [] for stock in ["Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E"]}
        self.day_counter = 0
    """
    
    def __init__(self) -> None:
        self.price_history = {
            "Stock_A": [],
            "Stock_B": [],
            "Stock_C": [],
            "Stock_D": [],
            "Stock_E": []
        }
        self.day = 0
        self.VOL_WINDOW = 10
        self.MOM_FAST = 10
        self.MOM_SLOW = 20
        self.Z_WINDOW = 20
        self.VOL_TH_HI = 0.025
        self.VOL_TH_LO = 0.018
        self.BUY_TH = 0.75
        self.SELL_TH = -0.75
        self.CASH_BUFFER = 0.125
        self.MAX_POS = 0.30
        self.last_regime = "low"


def _pct_change(series):
    if len(series) < 2:
        return []
    out = []
    i = 1
    while i < len(series):
        if series[i-1] != 0:
            out.append(series[i]/series[i-1]-1.0)
        else:
            out.append(0.0)
        i += 1
    return out

def _stdev(values):
    try:
        if len(values) >= 2:
            return statistics.stdev(values)
        else:
            return 0.0
    except:
        return 0.0

def _eq_weight_vol(context: Context) -> float:
    w = context.VOL_WINDOW
    if w <= 1:
        return 0.0
    first = next(iter(context.price_history.values()))
    ln = len(first)
    eqrets = []
    t = 1
    while t < min(w+1, ln):
        daily = []
        for s in context.price_history:
            hist = context.price_history[s]
            if len(hist) >= t+1 and hist[-t-1] != 0:
                r = hist[-t]/hist[-t-1]-1.0
                daily.append(r)
        if len(daily) > 0:
            eqrets.append(sum(daily)/len(daily))
        t += 1
    return _stdev(eqrets)

def _detect_regime(context: Context) -> str:
    v = _eq_weight_vol(context)
    if v >= context.VOL_TH_HI:
        reg = "high"
    elif v <= context.VOL_TH_LO:
        reg = "low"
    else:
        reg = "transition"
    context.last_regime = reg
    return reg

def _momentum_score(stock: str, context: Context) -> float:
    h = context.price_history[stock]
    if len(h) < context.MOM_SLOW+1:
        return 0.0
    if h[-context.MOM_FAST] != 0:
        fast = h[-1]/h[-context.MOM_FAST]-1.0
    else:
        fast = 0.0
    if h[-context.MOM_SLOW] != 0:
        slow = h[-1]/h[-context.MOM_SLOW]-1.0
    else:
        slow = 0.0
    window = h[-context.Z_WINDOW:]
    vols = _pct_change(window)
    norm = _stdev(vols)
    if norm == 0:
        norm = 1e-6
    return (fast - slow)/norm

def _meanrev_score(stock: str, context: Context) -> float:
    h = context.price_history[stock]
    if len(h) < context.Z_WINDOW:
        return 0.0
    window = h[-context.Z_WINDOW:]
    ma = sum(window)/len(window)
    rets = _pct_change(window)
    vol = _stdev(rets)
    if vol == 0:
        vol = 1e-6
    z = (h[-1] - ma) / (ma * vol)
    return -z

def _stock_vol(stock: str, context: Context, lookback: int = 20) -> float:
    h = context.price_history[stock]
    r = _pct_change(h[-lookback:])
    v = _stdev(r)
    if v < 0:
        v = -v
    if v == 0:
        v = 1e-6
    return v

def _inv_vol_weights(cands: list[str], context: Context) -> dict:
    vols = {}
    for s in cands:
        vols[s] = _stock_vol(s, context)
    inv = {}
    tot = 0
    for s in cands:
        inv[s] = 1.0/vols[s]
        tot += inv[s]
    if tot == 0:
        tot = 1e-6
    w = {}
    for s in cands:
        w[s] = inv[s]/tot
    return w

def _target_shares(stock: str, w: float, curPortfolio: Portfolio, curMarket: Market, context: Context) -> int:
    total_val = curPortfolio.evaluate(curMarket)
    cash_keep = total_val * context.CASH_BUFFER
    spendable = curPortfolio.cash - cash_keep
    if spendable < 0:
        spendable = 0
    cap = total_val * context.MAX_POS
    desired_val = w * (total_val - cash_keep)
    if desired_val > cap:
        desired_val = cap
    px = curMarket.stocks[stock]
    if px <= 0 or desired_val <= 0:
        return 0
    cost_one = px * (1 + Market.transaction_fee)
    if cost_one <= 0:
        return 0
    sh = desired_val / cost_one
    sh = floor(sh)
    if sh < 0:
        sh = 0
    return sh

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
    for s in curMarket.stocks:
        context.price_history[s].append(curMarket.stocks[s])

    any_hist = next(iter(context.price_history.values()))
    if len(any_hist) <= max(context.MOM_SLOW, context.Z_WINDOW):
        context.day += 1
        return

    regime = _detect_regime(context)
    scores = {}
    for s in curMarket.stocks:
        if regime == "low":
            scores[s] = _momentum_score(s, context)
        elif regime == "high":
            scores[s] = _meanrev_score(s, context)
        else:
            a = _momentum_score(s, context)
            b = _meanrev_score(s, context)
            scores[s] = 0.5*(a+b)*0.5

    ranked = sorted(scores.items(), key=lambda k: k[1], reverse=True)
    longs = []
    for nm, sc in ranked:
        if sc >= context.BUY_TH:
            longs.append(nm)
    longs = longs[:2]

    exits = []
    tmp = []
    for nm, sc in ranked:
        if sc <= context.SELL_TH:
            tmp.append(nm)
    exits = tmp[-2:]

    for s in exits:
        held = curPortfolio.shares[s]
        if held > 0:
            try:
                curPortfolio.sell(s, held, curMarket)
            except:
                pass

    if len(longs) > 0:
        wts = _inv_vol_weights(longs, context)
        if regime == "transition":
            new_w = {}
            for s in wts:
                new_w[s] = 0.5*wts[s]
            wts = new_w
        for s in longs:
            tgt = _target_shares(s, wts[s], curPortfolio, curMarket, context)
            cur = floor(curPortfolio.shares[s])
            diff = tgt - cur
            if diff > 0:
                try:
                    curPortfolio.buy(s, diff, curMarket)
                except:
                    pass

    if regime == "high" and len(longs) == 0:
        tv = curPortfolio.evaluate(curMarket)
        cap = tv * context.MAX_POS
        for s in curMarket.stocks:
            pv = curPortfolio.get_position_value(s, curMarket)
            if pv > cap:
                px = curMarket.stocks[s]
                if px > 0:
                    extra = pv - cap
                    sale_px = px * (1 - Market.transaction_fee)
                    if sale_px > 0:
                        trim = floor(extra / sale_px)
                        if trim > 0:
                            if trim > curPortfolio.shares[s]:
                                trim = curPortfolio.shares[s]
                            try:
                                curPortfolio.sell(s, trim, curMarket)
                            except:
                                pass

    context.day += 1