"""VaR and Expected Shortfall models.
 
Conventions
-----------
* Returns are 1-day simple/log returns; `weights` sum to 1.
* `alpha` is the confidence level (e.g. 0.99).
* VaR and ES are reported as POSITIVE numbers representing a loss,
  as a fraction of portfolio value (0.02 = a 2% loss).
"""
import numpy as np
from scipy import stats
 
 
def portfolio_returns(returns, weights) -> np.ndarray:
    #Portfolio return series for fixed weights
    return np.asarray(returns) @ np.asarray(weights)
 
 
def historical_var_es(port_returns, alpha=0.99):
    #Historical simulation: empirical quantile and tail mean. No assumptions so doesn't miss anything in the history. Can't predict future issues because of this. Can be noisy when dealing with 1%.
    r = np.asarray(port_returns)
    var = -np.quantile(r, 1 - alpha)
    tail = r[r <= -var]
    es = -tail.mean() if tail.size else var #tail.mean averages all the days the where the loss was in the worst 1%
    return var, es
 
 
def parametric_var_es(port_returns, alpha=0.99):
    #Variance-covariance method assuming normal portfolio returns. But Real Markets aren't normal
    r = np.asarray(port_returns)
    mu, sigma = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(alpha)
    var = z * sigma - mu
    es = sigma * stats.norm.pdf(z) / (1 - alpha) - mu
    return var, es
 
# Simulate Days going forward based on the statistic profile of past events.
# Simulate a large amount of days, and from there apply the same historical VaR and ES methods from above.
# Basically create a new data set to draw from base on Sims
 
def simulate_returns(returns, n_sims=100_000, dist="normal", df=5, seed=42):
    #Simulate correlated asset returns via Cholesky decomposition. None of the stocks are independent of each other, so we try to link them.
 
    #dist="normal": multivariate normal with the sample mean and covariance.
    #dist="t":      multivariate Student-t with `df` degrees of freedom,
    #               scaled so the covariance still matches the sample
    #               covariance (fatter tails, same variance).
    
    r = np.asarray(returns)
    mu = r.mean(axis=0)
    cov = np.cov(r, rowvar=False) # covariance matrix of the 5 assests
    L = np.linalg.cholesky(cov) # Cholesky decomposition
 
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_sims, r.shape[1])) # random noise to stimulate the simulation
    x = z @ L.T
 
    if dist == "t": # opens the option for a student t dist, basically heavier tails. So more likely bad days i.e. more days with high loss. (or the opposite of course)
        if df <= 2:
            raise ValueError("df must be > 2 for finite variance")
        w = rng.chisquare(df, size=(n_sims, 1)) # random multiplier drawn from chi-squared fist.
        x = x * np.sqrt((df - 2) / w)
    elif dist != "normal":
        raise ValueError("dist must be 'normal' or 't'")
 
    return mu + x #adds daily average back to the noise, doesn't matter much over 1 day but larger sims itll have an effect.
 
# the actual simulation based off the profile built in the previous function.
def monte_carlo_var_es(returns, weights, alpha=0.99, n_sims=100_000, 
                       dist="normal", df=5, seed=42):
    #Monte Carlo VaR/ES from simulated correlated asset returns
    sims = simulate_returns(returns, n_sims=n_sims, dist=dist, df=df, seed=seed) # 
    return historical_var_es(sims @ np.asarray(weights), alpha)