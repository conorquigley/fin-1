#Downloads finacial Data and saves it to a .csv file in the data folder.

from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_TICKERS = ["SPY", "EFA", "EEM", "AGG", "GLD"] # S&P 500, stocks from outside NA, Emerging market stocks, US bond market, Gold.
DEFAULT_START = "2015-01-01"
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "prices.csv" # walks back two folders, enters data and creats prices.csv .

def download_prices(tickers=DEFAULT_TICKERS, start=DEFAULT_START, end=None, # end none means today as an end.
                    path=DEFAULT_CACHE) -> pd.DataFrame:

    #Downloads adjusted close prices from Yahoo Finance and saves to CSV in the data folder.
    ## Committing the CSV keeps the project reproducible even if the source gets updated at a later date.

    import yfinance as yf  # importing within the function.
 
    raw = yf.download(list(tickers), start=start, end=end,
                      auto_adjust=True, progress=False)
    prices = raw["Close"][list(tickers)].dropna(how="any")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path)
    return prices
 
 
def load_prices(path=DEFAULT_CACHE) -> pd.DataFrame:
    """Loads prices from data/prices.csv (run download_prices first)."""
    return pd.read_csv(path, index_col=0, parse_dates=True)
 
 
def log_returns(prices: pd.DataFrame) -> pd.DataFrame: # log changes are additive, so a 10% increase one day followed by a 10% decrease adds to 0, which is where the stock started.
    """Daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()