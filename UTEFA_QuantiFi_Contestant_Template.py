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
        # PUT WHATEVER YOU WANT HERE
        # Example: Track price history for technical analysis
        self.price_history = {
            "Stock_A": [],
            "Stock_B": [],
            "Stock_C": [],
            "Stock_D": [],
            "Stock_E": []
        }
        self.day = 0


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
    # YOUR TRADING STRATEGY GOES HERE
    pass


###SIMULATION###
if __name__ == "__main__":
    market = Market()
    portfolio = Portfolio()
    context = Context()

    # Simulate 252 trading days (one trading year)
    for day in range(252):
        update_portfolio(market, portfolio, context)
        market.updateMarket()

    # Print final portfolio value
    final_value = portfolio.evaluate(market)
    print(f"Final Portfolio Value: ${final_value:,.2f}")
