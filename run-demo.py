"""Download data (first run only) and compare VaR/ES across methods."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from varengine import (download_prices, historical_var_es, load_prices,
                       log_returns, monte_carlo_var_es, parametric_var_es,
                       portfolio_returns)
from varengine.data import DEFAULT_CACHE

if not DEFAULT_CACHE.exists():
    print("Downloading prices...")
    download_prices()

prices = load_prices()
rets = log_returns(prices)
weights = [1 / rets.shape[1]] * rets.shape[1]  # equal weight to start
port = portfolio_returns(rets, weights)

print(f"{rets.index[0].date()} to {rets.index[-1].date()}, "
      f"{len(rets)} daily observations, assets: {list(rets.columns)}\n")

rows = []
for alpha in (0.95, 0.99):
    methods = {
        "Historical": historical_var_es(port, alpha),
        "Parametric (normal)": parametric_var_es(port, alpha),
        "Monte Carlo (normal)": monte_carlo_var_es(rets, weights, alpha, dist="normal"),
        "Monte Carlo (t, df=5)": monte_carlo_var_es(rets, weights, alpha, dist="t", df=5),
    }
    for name, (var, es) in methods.items():
        rows.append({"alpha": alpha, "method": name,
                     "VaR (%)": 100 * var, "ES (%)": 100 * es})

print(pd.DataFrame(rows).round(3).to_string(index=False))