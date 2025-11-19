"""
UTEFA QuantiFi – Final Contestant Strategy (ADX + ATR version)
Hard-coded winning parameters:
EMA(20/30), Momentum10, ADX filter, ATR/Vol/Volume tilts, Trailing Stop 5%, 3-stock basket
"""

import csv
import statistics
from typing import Dict, List

WIN_PARAMS = {
    "ma_type": "EMA",
    "fast_ma_window": 20,
    "slow_ma_window": 30,

    "momentum_window": 10,
    "num_stocks": 3,
    "proportional_momentum": False,

    "use_trailing_stop": True,
    "trailing_stop_pct": 0.05,

    "vol_lookback": 20,
    "low_vol_threshold": 0.01,
    "high_vol_threshold": 0.03,
    "volume_lookback": 40,

    "atr_window": 10,
    "adx_window": 14,
    "adx_min": 20.0,
    "adx_max": 50.0,

    "macro_lower": 0.95,
    "macro_upper": 1.05,

    "max_trades": 20,
}

PARAMS = WIN_PARAMS.copy()

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
    def __init__(self):
        self.params = PARAMS.copy()

        self.stocks = [f"Stock_{c}" for c in ["A","B","C","D","E"]]
        self.csv_path = "UTEFA_QuantiFi_Contestant_Dataset.csv"

        self.price_history = {s: [] for s in self.stocks}
        self.atr_history   = {s: [] for s in self.stocks}
        self.adx_history   = {s: [] for s in self.stocks}

        self.index_history = []

        self.interest_history   = []
        self.growth_history     = []
        self.inflation_history  = []
        self.volume_history     = {s: [] for s in self.stocks}
        self.momentum10_history = {s: [] for s in self.stocks}

        self.day = 0
        self.num_rows = 0

        # Trading state
        self.in_position = False
        self.trade_count = 0
        self.max_trades = self.params["max_trades"]
        self.peak_price = {s: None for s in self.stocks}

        # Fee tracking required for report
        self.total_fees = 0.0

        self.load_csv()

    def load_csv(self):
        with open(self.csv_path) as f:
            r = csv.DictReader(f)
            for row in r:
                self.interest_history.append(self.to_float(row["Interest_Rate"]))
                self.growth_history.append(self.to_float(row["Economic_Growth"]))
                self.inflation_history.append(self.to_float(row["Inflation"]))
                for s in self.stocks:
                    self.volume_history[s].append(self.to_float(row[f"{s}_Volume"]))
                    self.momentum10_history[s].append(self.to_float(row[f"{s}_Momentum_10d"]))
                self.num_rows += 1

    def to_float(self,v):
        try: return float(v)
        except: return None

    def median_clean(self, arr):
        x=[v for v in arr if v is not None]
        return statistics.median(x) if x else None

    def compute_atr(self, prices, w):
        if len(prices) < w+1: return None
        trs=[abs(prices[i]-prices[i-1]) for i in range(len(prices)-w, len(prices))]
        return sum(trs)/w

    def compute_adx(self, prices, w):
        if len(prices) < w+1: return None
        plus=[]; minus=[]; tr=[]
        for i in range(len(prices)-w, len(prices)):
            c = prices[i]; p = prices[i-1]
            dm_plus  = max(c-p, 0)
            dm_minus = max(p-c, 0)
            plus.append(dm_plus); minus.append(dm_minus)
            tr.append(abs(c-p))
        avgtr = sum(tr)/w
        if avgtr <= 0: return None
        plus_di  = 100*(sum(plus)/w)/avgtr
        minus_di = 100*(sum(minus)/w)/avgtr
        denom = plus_di+minus_di
        if denom <= 0: return None
        dx = 100*abs(plus_di-minus_di)/denom
        return dx

def update_portfolio(mkt, port, ctx):

    p = PARAMS  # shortcut
    fee = Market.transaction_fee
    d = ctx.day

    if d >= ctx.num_rows:
        ctx.day += 1
        return

    idx=[]
    for s in ctx.stocks:
        px = mkt.stocks[s]
        ctx.price_history[s].append(px)
        idx.append(px)

        atr = ctx.compute_atr(ctx.price_history[s], p["atr_window"])
        adx = ctx.compute_adx(ctx.price_history[s], p["adx_window"])
        ctx.atr_history[s].append(atr)
        ctx.adx_history[s].append(adx)

    ctx.index_history.append(sum(idx)/5.0)

    def ma(v, w):
        if len(v)<w: return None
        if p["ma_type"]=="SMA":
            return sum(v[-w:])/w
        alpha=2/(w+1)
        ema=v[-w]
        for x in v[-w+1:]:
            ema = alpha*x + (1-alpha)*ema
        return ema

    fast_now = ma(ctx.index_history, p["fast_ma_window"])
    slow_now = ma(ctx.index_history, p["slow_ma_window"])
    fast_prev = ma(ctx.index_history[:-1], p["fast_ma_window"])
    slow_prev = ma(ctx.index_history[:-1], p["slow_ma_window"])

    if any(x is None for x in [fast_now,slow_now,fast_prev,slow_prev]):
        ctx.day += 1
        return

    risk_on_now = fast_now > slow_now
    risk_on_prev = fast_prev > slow_prev
    cross_up   = (not risk_on_prev) and risk_on_now
    cross_down = risk_on_prev and (not risk_on_now)

    risk_mult = 1.0
    ir_med = ctx.median_clean(ctx.interest_history[:d+1])
    eg_med = ctx.median_clean(ctx.growth_history[:d+1])
    inf_med = ctx.median_clean(ctx.inflation_history[:d+1])

    ir = ctx.interest_history[d]
    eg = ctx.growth_history[d]
    inf= ctx.inflation_history[d]

    if ir is not None and inf is not None and ir_med and inf_med:
        if ir>ir_med and inf>inf_med:
            risk_mult -= 0.05

    if eg is not None and eg_med and inf_med:
        if eg<eg_med and inf>inf_med:
            risk_mult -= 0.05
        if eg>eg_med and inf<=inf_med and ir is not None and ir_med and ir<=ir_med:
            risk_mult += 0.05

    risk_mult = max(p["macro_lower"], min(risk_mult, p["macro_upper"]))

    if p["use_trailing_stop"] and ctx.in_position:
        for s in ctx.stocks:
            sh = port.shares[s]
            if sh>0:
                px = mkt.stocks[s]
                if ctx.peak_price[s] is None:
                    ctx.peak_price[s] = px
                ctx.peak_price[s] = max(ctx.peak_price[s],px)

                stop_price = ctx.peak_price[s]*(1-p["trailing_stop_pct"])
                if px < stop_price and ctx.trade_count<p["max_trades"]:
                    port.sell(s, sh, mkt)
                    ctx.trade_count += 1
                    ctx.total_fees += sh*px*fee

        if not any(port.shares[s]>0 for s in ctx.stocks):
            ctx.in_position=False
            for s in ctx.stocks: ctx.peak_price[s]=None

    if cross_up and not ctx.in_position and ctx.trade_count<p["max_trades"]:

        # momentum + ADX filter
        momentum=[]
        for s in ctx.stocks:
            m = ctx.momentum10_history[s][d]
            adx = ctx.adx_history[s][-1]
            if m is None or adx is None: continue
            if adx < p["adx_min"]: continue
            if adx > p["adx_max"]: continue
            momentum.append((s,m))

        if not momentum:
            ctx.day+=1; return

        momentum.sort(key=lambda x:x[1], reverse=True)
        top = [s for s,_ in momentum[:p["num_stocks"]]]

        # equal weights
        w = {s:1/len(top) for s in top}

        # vol tilt
        for s in top:
            vol = compute_vol(ctx.price_history[s], p["vol_lookback"])
            if vol:
                if vol>p["high_vol_threshold"]: w[s]*=0.97
                elif vol<p["low_vol_threshold"]: w[s]*=1.03

        # ATR tilt
        atrs=[ctx.atr_history[s][-1] for s in top if ctx.atr_history[s][-1] is not None]
        if atrs:
            atr_med=statistics.median(atrs)
            for s in top:
                atr=ctx.atr_history[s][-1]
                if atr:
                    if atr>atr_med: w[s]*=0.99
                    else: w[s]*=1.01

        # normalize
        tw=sum(w.values()); w={s:v/tw for s,v in w.items()}

        # reset peaks
        for s in ctx.stocks:
            ctx.peak_price[s]=mkt.stocks[s] if s in top else None

        eq = port.evaluate(mkt)
        deploy = eq*risk_mult

        for s,v in w.items():
            if ctx.trade_count>=p["max_trades"]: break
            px=mkt.stocks[s]
            eff=px*(1+fee)
            shares=int(min(deploy*v//eff, port.cash//eff))
            if shares>0:
                port.buy(s, shares, mkt)
                ctx.trade_count+=1
                ctx.total_fees+=shares*px*fee

        ctx.in_position = any(port.shares[s]>0 for s in ctx.stocks)

    if cross_down and ctx.in_position and ctx.trade_count<p["max_trades"]:
        for s in ctx.stocks:
            sh=int(port.shares[s])
            if sh>0:
                px=mkt.stocks[s]
                port.sell(s, sh, mkt)
                ctx.trade_count+=1
                ctx.total_fees+=sh*px*fee

        ctx.in_position = any(port.shares[s]>0 for s in ctx.stocks)
        if not ctx.in_position:
            for s in ctx.stocks: ctx.peak_price[s]=None

    ctx.day+=1


def compute_vol(prices, w):
    if len(prices)<w+1: return None
    rets=[]
    for i in range(-w,0):
        p0=prices[i-1]; p1=prices[i]
        if p0>0: rets.append(p1/p0-1)
    if len(rets)<w: return None
    m=sum(rets)/len(rets)
    var=sum((r-m)**2 for r in rets)/len(rets)
    return var**0.5
