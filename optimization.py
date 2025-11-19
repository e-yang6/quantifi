import itertools
import json
from typing import Dict, List

import pandas as pd

import UTEFA_QuantiFi_Backtesting_Script_testing as backtest
import UTEFA_QuantiFi_Contestant_Template_testing as strat
from UTEFA_QuantiFi_Contestant_Template_testing import (
    Market,
    Portfolio,
    Context,
    PARAMS,
    update_portfolio,
)

# -------------------------------------------------
# Load price data ONCE, reuse for all simulations
# -------------------------------------------------

PRICE_CSV = "UTEFA_QuantiFi_Contestant_Dataset.csv"
price_data = backtest.load_price_data(PRICE_CSV)  # mainly as a validation step

# Benchmark from your notes
BENCH_RETURN = 0.2272          # 22.72 %
BENCH_SHARPE = 2.275
BENCH_FEES = 1111.75


# -------------------------------------------------
# Parameter grid (balanced search)
# -------------------------------------------------

param_grid = {
   
    "ma_type": ["EMA"],
    
    # Aggressive enough to capture returns, but not overfit
    "fast_ma_window": [5, 9, 12],
    "slow_ma_window": [40, 50],

    # Slightly slower momentum to reduce noise
    "momentum_window": [10, 15],
    "proportional_momentum": [False],

    # 4 vs 5 stocks is critical; this decides diversification
    "num_stocks": [4, 5],

    # trailing stop now has RANGE
    "use_trailing_stop": [True],
    "trailing_stop_pct": [0.02, 0.03, 0.05],

    # Vol/volume minimal, but not removed
    "vol_lookback": [20],
    "low_vol_threshold": [0.01],
    "high_vol_threshold": [0.03],
    "volume_lookback": [40],

    # Macro unchanged
    "macro_lower": [0.95],
    "macro_upper": [1.05],

    # Trade count controls turnover
    "max_trades": [15, 20]

    "rsi_window": [5, 7],
    "rsi_min": [40],
    "rsi_max": [60],
}


def expand_grid(grid: Dict[str, list]):
    keys = list(grid.keys())
    for values in itertools.product(*grid.values()):
        yield dict(zip(keys, values))


# -------------------------------------------------
# Core backtest wrapper
# -------------------------------------------------

def run_backtest(params: Dict) -> Dict:
    """
    Run one full 252-day backtest for a given parameter set.
    Uses the contestant template strategy directly.
    """

    # Inject params into global PARAMS used by Context
    PARAMS.clear()
    PARAMS.update(params)

    ctx = Context()
    port = Portfolio()
    mkt = Market()

    # Read prices from the same CSV the contest uses
    df = pd.read_csv(PRICE_CSV)

    # Simple daily loop (0..251)
    for i in range(len(df)):
        # update market prices for this day
        for s in ["Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E"]:
            mkt.stocks[s] = df.loc[i, s]

        # let the strategy act
        update_portfolio(mkt, port, ctx)

    final_value = port.evaluate(mkt)
    total_return = final_value / 100000.0 - 1.0

    # Very rough Sharpe proxy (you can refine later if needed)
    # Here we use "contest style": annualized Sharpe under assumed 10 % vol
    sharpe = total_return / 0.10 if 0.10 > 0 else 0.0

    result = {
        "params": params,
        "final_value": final_value,
        "total_return": total_return,
        "sharpe": sharpe,
        "trades": ctx.trade_count,
        "total_fees": ctx.total_fees,
    }
    return result


# -------------------------------------------------
# Scoring function vs benchmark
# -------------------------------------------------

def score_strategy(res: Dict) -> float:
    """
    Higher is better.
    Reward Sharpe and alpha vs benchmark, penalize excess fees.
    """

    # If strategy completely blows up, nuke its score
    if res["total_return"] < -0.5 or res["sharpe"] < -0.5:
        return -1e9

    excess_ret = res["total_return"] - BENCH_RETURN
    sharpe_gain = res["sharpe"] - BENCH_SHARPE
    fee_ratio = res["total_fees"] / BENCH_FEES if BENCH_FEES > 0 else 1.0

    # Weight Sharpe most, then alpha, then fee penalty
    score = 2.0 * sharpe_gain + 1.0 * excess_ret - 0.5 * (fee_ratio - 1.0)
    return score


# -------------------------------------------------
# Grid search loop
# -------------------------------------------------

def run_grid_search():
    all_results = []

    for params in expand_grid(param_grid):
        print(f"Testing params: {params}")

        res = run_backtest(params)
        score = score_strategy(res)

        entry = {
            "params": params,
            "score": score,
            "return": res["total_return"],
            "sharpe": res["sharpe"],
            "fees": res["total_fees"],
            "trades": res["trades"],     # FIXED: was res['n_trades']
        }
        all_results.append(entry)

    # Sort by score
    all_results_sorted = sorted(all_results, key=lambda x: x["score"], reverse=True)

    # Save full table
    df = pd.DataFrame(all_results_sorted)
    df.to_csv("grid_results.csv", index=False)

    # Save top few for quick inspection
    top_k = all_results_sorted[:10]
    with open("grid_top10.json", "w") as f:
        json.dump(top_k, f, indent=2)

    print("\nDONE. Results saved to grid_results.csv and grid_top10.json")
    print("Top 3 summary:")
    for i, r in enumerate(top_k[:3], start=1):
        print(
            f"{i}. Sharpe {r['sharpe']:.3f}, "
            f"Ret {r['return']:.3f}, "
            f"Fees {r['fees']:.2f}, "
            f"Trades {r['trades']}, "
            f"Params {r['params']}"
        )

    return all_results_sorted


if __name__ == "__main__":
    # Run the grid search once
    all_results_sorted = run_grid_search()

    # Also save a second CSV focused on raw metrics for quick pandas work
    df = pd.DataFrame(all_results_sorted)
    df.to_csv("optimization_results.csv", index=False)

    # Best by Sharpe
    best = df.sort_values(by="sharpe", ascending=False).iloc[0]
    print("\nBEST BY SHARPE:")
    print(best)
