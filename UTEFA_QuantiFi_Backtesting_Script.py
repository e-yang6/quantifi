"""
UTEFA QuantiFi - Backtesting Script


Usage in terminal:
    Have grading_script.py, contestant_file.py, and price_data.csv in the same directory.
    Then run in terminal: 
    python <grading_script.py> <contestant_file.py> <price_data.csv>
    OR
    python3 <grading_script.py> <contestant_file.py> <price_data.csv>

Example:
    python3 grading_script.py Team1_submission.py testing_data.csv
"""

import sys
import csv
import importlib.util
import traceback
from typing import Dict, List, Tuple
import signal
from contextlib import contextmanager


class TimeoutException(Exception):
    """Raised when code execution times out"""
    pass


@contextmanager
def time_limit(seconds):
    """Context manager to limit execution time"""
    def signal_handler(signum, frame):
        raise TimeoutException("Code execution timed out")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def load_price_data(csv_file: str) -> Dict[str, List[float]]:
    """
    Load historical price data from CSV file.
    
    Expected CSV format:
    Day,Stock_A,Stock_B,Stock_C,Stock_D,Stock_E
    0,100.0,100.0,100.0,100.0,100.0
    1,101.5,99.8,100.2,101.0,99.5
    ...
    
    Args:
        csv_file: Path to CSV file with price data
        
    Returns:
        Dictionary mapping stock names to lists of prices
        
    Raises:
        ValueError: If CSV format is invalid or data is incomplete
    """
    price_data = {
        "Stock_A": [],
        "Stock_B": [],
        "Stock_C": [],
        "Stock_D": [],
        "Stock_E": []
    }
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            
            # Verify headers
            expected_headers = ['Day', 'Stock_A', 'Stock_B', 'Stock_C', 'Stock_D', 'Stock_E']
            if not all(header in reader.fieldnames for header in expected_headers):
                raise ValueError(f"CSV must have headers: {expected_headers}")
            
            # Load data
            for row in reader:
                for stock in price_data.keys():
                    price_data[stock].append(float(row[stock]))
        
        # Verify we have 252 days of data
        for stock, prices in price_data.items():
            if len(prices) != 252:
                raise ValueError(f"{stock} has {len(prices)} days, expected 252")
        
        print(f"✓ Successfully loaded price data: 252 days for 5 stocks")
        return price_data
        
    except FileNotFoundError:
        print(f"ERROR: Price data file '{csv_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading price data: {e}")
        sys.exit(1)


def load_contestant_module(file_path: str):
    """
    Dynamically load a contestant's Python file as a module.
    
    Args:
        file_path: Path to contestant's Python file
        
    Returns:
        Loaded module object
        
    Raises:
        ImportError: If module cannot be loaded
    """
    try:
        spec = importlib.util.spec_from_file_location("contestant_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except FileNotFoundError:
        print(f"ERROR: Contestant file '{file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading contestant file: {e}")
        traceback.print_exc()
        sys.exit(1)


def update_market(market, day: int, price_data: Dict[str, List[float]]) -> None:
    """
    Updates the market prices for all stocks based on the provided price data.
    
    Args:
        market: The Market object to update
        day: Current trading day (0 to 251)
        price_data: Dictionary mapping stock names to lists of prices
    """
    for stock_name in market.stocks:
        if stock_name in price_data and day < len(price_data[stock_name]):
            market.stocks[stock_name] = price_data[stock_name][day]
        else:
            raise ValueError(f"Price data missing for {stock_name} on day {day}")


class TransactionTracker:
    """Tracks all trades made during the simulation"""
    
    def __init__(self):
        self.buy_count = 0
        self.sell_count = 0
        self.total_fees = 0.0
        self.transactions = []
    
    def record_buy(self, stock: str, shares: float, price: float, fee: float):
        """Record a buy transaction"""
        self.buy_count += 1
        self.total_fees += fee
        self.transactions.append({
            'type': 'BUY',
            'stock': stock,
            'shares': shares,
            'price': price,
            'fee': fee
        })
    
    def record_sell(self, stock: str, shares: float, price: float, fee: float):
        """Record a sell transaction"""
        self.sell_count += 1
        self.total_fees += fee
        self.transactions.append({
            'type': 'SELL',
            'stock': stock,
            'shares': shares,
            'price': price,
            'fee': fee
        })
    
    def get_summary(self) -> str:
        """Generate transaction summary report"""
        total_trades = self.buy_count + self.sell_count
        return f"""
Transaction Summary:
  Total Trades: {total_trades}
  - Buys: {self.buy_count}
  - Sells: {self.sell_count}
  Total Fees Paid: ${self.total_fees:,.2f}
"""


def wrap_portfolio_methods(portfolio, tracker: TransactionTracker):
    """Wrap portfolio buy/sell methods to track transactions"""
    original_buy = portfolio.buy
    original_sell = portfolio.sell
    
    def tracked_buy(stock_name: str, shares_to_buy: float, curMarket) -> None:
        original_buy(stock_name, shares_to_buy, curMarket)
        price = curMarket.stocks[stock_name]
        fee = shares_to_buy * price * curMarket.transaction_fee
        tracker.record_buy(stock_name, shares_to_buy, price, fee)
    
    def tracked_sell(stock_name: str, shares_to_sell: float, curMarket) -> None:
        original_sell(stock_name, shares_to_sell, curMarket)
        price = curMarket.stocks[stock_name]
        fee = shares_to_sell * price * curMarket.transaction_fee
        tracker.record_sell(stock_name, shares_to_sell, price, fee)
    
    portfolio.buy = tracked_buy
    portfolio.sell = tracked_sell


def run_simulation(contestant_module, price_data: Dict[str, List[float]]) -> Tuple[float, List[float], TransactionTracker, object, object]:
    """
    Run the trading simulation with contestant's strategy.
    
    Args:
        contestant_module: Loaded contestant module
        price_data: Historical price data
        
    Returns:
        Tuple of (final_value, daily_values, transaction_tracker, portfolio, market)
    """
    # Initialize objects from contestant's module
    market = contestant_module.Market()
    portfolio = contestant_module.Portfolio()
    context = contestant_module.Context()
    update_portfolio = contestant_module.update_portfolio
    
    # Set Day 0 prices BEFORE starting simulation
    update_market(market, 0, price_data)
    
    # Validate starting conditions
    starting_value = portfolio.evaluate(market)
    if abs(starting_value - 100000.0) > 0.01:
        print(f"WARNING: Starting portfolio value is ${starting_value:,.2f}, expected $100,000.00")
    
    # Initialize transaction tracker
    tracker = TransactionTracker()
    wrap_portfolio_methods(portfolio, tracker)
    
    # Track portfolio value each day
    daily_values = [starting_value]
    
    print("\nRunning simulation...")
    
    try:
        with time_limit(60):  # 60 second timeout
            for day in range(252):
                # Contestant's strategy makes trades with current day's prices
                update_portfolio(market, portfolio, context)
                
                # Update market prices to next day (or keep final prices on last day)
                if day < 251:
                    update_market(market, day + 1, price_data)
                
                # Record portfolio value after market update
                daily_value = portfolio.evaluate(market)
                daily_values.append(daily_value)
                
                # Sanity check - portfolio shouldn't be negative
                if daily_value < 0:
                    print(f"WARNING: Portfolio value went negative on day {day+1}: ${daily_value:,.2f}")
                
                # Progress indicator
                if (day + 1) % 50 == 0:
                    print(f"  Day {day + 1}/252 complete...")
    
    except TimeoutException:
        print("ERROR: Simulation timed out (exceeded 60 seconds)")
        print("This usually indicates an infinite loop in the strategy")
        return None, None, None, None, None
    except Exception as e:
        print(f"ERROR during simulation: {e}")
        traceback.print_exc()
        return None, None, None, None, None
    
    final_value = portfolio.evaluate(market)
    print("✓ Simulation completed successfully\n")
    
    return final_value, daily_values, tracker, portfolio, market


def generate_report(contestant_file: str, portfolio, market, final_value: float, 
                   daily_values: List[float], tracker: TransactionTracker) -> None:
    """
    Generate comprehensive performance report.
    
    Args:
        contestant_file: Name of contestant's file
        portfolio: Final portfolio state
        market: Final market state
        final_value: Final portfolio value
        daily_values: List of portfolio values over time
        tracker: Transaction tracker with trade history
    """
    print("=" * 70)
    print(f"GRADING REPORT: {contestant_file}")
    print("=" * 70)
    
    # Portfolio breakdown
    print("\nFINAL PORTFOLIO BREAKDOWN:")
    print(f"  Cash: ${portfolio.cash:,.2f}")
    print("\n  Stock Holdings:")
    
    total_stock_value = 0.0
    for stock_name, shares in portfolio.shares.items():
        stock_value = shares * market.stocks[stock_name]
        total_stock_value += stock_value
        if shares > 0:
            print(f"    {stock_name}: {shares:.2f} shares @ ${market.stocks[stock_name]:.2f} = ${stock_value:,.2f}")
        else:
            print(f"    {stock_name}: {shares:.2f} shares (no position)")
    
    print(f"\n  Total Stock Value: ${total_stock_value:,.2f}")
    print(f"  Total Portfolio Value: ${final_value:,.2f}")
    
    # Performance metrics
    starting_value = daily_values[0]
    total_return = final_value - starting_value
    percent_return = (total_return / starting_value) * 100
    
    print(f"\nPERFORMANCE METRICS:")
    print(f"  Starting Value: ${starting_value:,.2f}")
    print(f"  Final Value: ${final_value:,.2f}")
    print(f"  Total Return: ${total_return:,.2f}")
    print(f"  Percent Return: {percent_return:.2f}%")
    
    # Transaction summary
    print(tracker.get_summary())
    
    # Validation checks
    print("VALIDATION CHECKS:")
    if abs(starting_value - 100000.0) < 0.01:
        print("  ✓ Started with correct initial capital ($100,000)")
    else:
        print(f"  ✗ Started with ${starting_value:,.2f} instead of $100,000")
    
    if final_value >= 0:
        print("  ✓ Portfolio value remained positive")
    else:
        print("  ✗ Portfolio value went negative!")
    
    if portfolio.cash >= -0.01:  # Allow small floating point errors
        print("  ✓ Cash balance valid")
    else:
        print(f"  ✗ Negative cash balance: ${portfolio.cash:,.2f}")
    
    print("\n" + "=" * 70)


def save_daily_values(contestant_file: str, daily_values: List[float]) -> None:
    """
    Save day-by-day portfolio values to CSV for analysis.
    
    Args:
        contestant_file: Name of contestant's file
        daily_values: List of portfolio values over time
    """
    output_file = contestant_file.replace('.py', '_daily_values.csv')
    
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Day', 'Portfolio_Value'])
            for day, value in enumerate(daily_values):
                writer.writerow([day, f"{value:.2f}"])
        
        print(f"Daily portfolio values saved to: {output_file}")
    except Exception as e:
        print(f"Warning: Could not save daily values: {e}")


def main():
    """Main grading function"""
    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python grading_script.py <contestant_file.py> <price_data.csv>")
        print("\nExample:")
        print("  python grading_script.py john_doe_submission.py historical_prices.csv")
        sys.exit(1)
    
    contestant_file = sys.argv[1]
    price_data_file = sys.argv[2]
    
    print("=" * 70)
    print("FINANCIAL MODELING COMPETITION - GRADING SCRIPT")
    print("=" * 70)
    print(f"\nContestant File: {contestant_file}")
    print(f"Price Data File: {price_data_file}\n")
    
    # Load price data
    price_data = load_price_data(price_data_file)
    
    # Load contestant's module
    print(f"Loading contestant code...")
    contestant_module = load_contestant_module(contestant_file)
    print(f"✓ Successfully loaded contestant module\n")
    
    # Run simulation
    result = run_simulation(contestant_module, price_data)
    
    if result[0] is None:
        print("\n❌ GRADING FAILED - Simulation did not complete successfully")
        sys.exit(1)
    
    final_value, daily_values, tracker, portfolio, market = result
    
    # Generate report
    generate_report(
        contestant_file,
        portfolio,
        market,
        final_value,
        daily_values,
        tracker
    )
    
    # Save daily values for visualization
    save_daily_values(contestant_file, daily_values)
    
    print(f"\nFINAL SCORE: ${final_value:,.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
