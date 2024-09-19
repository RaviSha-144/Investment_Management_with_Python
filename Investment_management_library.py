import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def get_prices(ticker, start_date, end_date):
    # prices = web.DataReader(ticker, 'yahoo', start_date, end_date)['Adj Close']
    return pd.DataFrame()

def returns(prices):
    '''
    Calculate the return of the prices
    
    Parameters
    ----------
    prices : pandas.Series
        The prices of the asset
        
    Returns 
    -------
    pandas.Series
        The return of the asset
    '''
    return prices/prices.shift(1) - 1

def pct_returns(prices):
    '''
    Calculate the percentage return of the prices
    
    Parameters
    ----------
    prices : pandas.Series
        The prices of the asset
        
    Returns 
    -------
    pandas.Series
        The percentage return of the asset
    '''
    # percent_returns = prices.iloc[1:]/prices.iloc[:-1].values - 1
    # percent_returns = prices/prices.shift(1) - 1 # This is the same as pct_change
    return prices.pct_change()

def compound_returns(returns):
    '''
    Calculate the compound return of the prices
    
    Parameters
    ----------
    returns : pandas.Series
        The returns of the asset
        
    Returns 
    -------
    pandas.Series
        The compound return of the asset
    '''
    return np.prod(returns + 1) - 1

def annualized_returns(returns, periods_per_year: int = 12):
    '''
    Calculate the annualized return of the prices
    
    Parameters
    ----------
    returns : pandas.Series
        The returns of the asset
    periods_per_year : int
        The number of periods per year
        
    Returns 
    -------
    pandas.Series
        The annualized return of the asset
    '''
    return (np.prod(returns + 1)) ** (periods_per_year/len(returns)) - 1

def annualized_volatility(returns, periods_per_year: int = 12):
    '''
    Calculate the annualized volatility of the prices
    
    Parameters
    ----------
    returns : pandas.Series
        The returns of the asset
    periods_per_year : int
        The number of periods per year
        
    Returns 
    -------
    pandas.Series
        The annualized volatility of the asset
    '''
    return np.std(returns) * np.sqrt(periods_per_year)

def sharpe_ratio(returns, risk_free_rate: float = 0.03, periods_per_year: int = 12):
    '''
    Calculate the Sharpe ratio of the prices
    
    Parameters
    ----------
    returns : pandas.Series
        The returns of the asset
    risk_free_rate : float
        The risk-free rate
    periods_per_year : int
        The number of periods per year
        
    Returns 
    -------
    pandas.Series
        The Sharpe ratio of the asset
    '''
    return (annualized_returns(returns, periods_per_year) - risk_free_rate) / annualized_volatility(returns, periods_per_year)

def drawdown(return_series: pd.Series):
    '''
    Takes a time series of asset returns, returns a DataFrame with columns for
    the wealth index, 
    the previous peaks, and 
    the percentage drawdown
    
    Parameters
    ----------
    return_series : pd.Series
        The asset return series
        
    Returns
    -------
    pd.DataFrame
        A DataFrame with columns for the wealth index, the previous peaks, and the percentage drawdown    
    '''
    wealth_index = (1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})
    

def get_max_drawdown(return_series: pd.Series):
    '''
    Takes a time series of asset returns, returns the maximum drawdown.
    
    Parameters
    ----------
    return_series : pd.Series
        The asset return series
        
    Returns
    -------
    float
        The maximum drawdown
    '''
    drawdowns = drawdown(return_series)
    return drawdowns["Drawdown"].min()

def get_max_drawdown_period(return_series: pd.Series):
    '''
    Takes a time series of asset returns, returns the maximum drawdown period.
    
    Parameters
    ----------
    return_series : pd.Series
        The asset return series
        
    Returns
    -------
    pd.DataFrame
        The maximum drawdown period
    '''
    drawdowns = drawdown(return_series)
    max_drawdown = drawdowns["Drawdown"].min()
    max_drawdown_period = drawdowns[drawdowns["Drawdown"] == max_drawdown]
    return max_drawdown_period

def get_ffme_returns(columns: list):
    '''
    Load the Fama-French dataset for the returns of the top and bottom deciles of the market
    
    Parameters
    ----------
    columns : list
        The columns to load
        
    Returns
    -------
    pandas.DataFrame
        The returns of the specified columns of the market
    '''
    mequity_monthly = pd.read_csv(r"D:\UvA VU\Investment Management with Python and ML\notebooks_and_codem01_v02\data\Portfolios_Formed_on_ME_monthly_EW.csv",
                      header=0, index_col=0, na_values=-99.99)
    returns = mequity_monthly[['Lo 10', 'Hi 10']]
    returns.columns = columns
    returns = returns/100
    returns.index = pd.to_datetime(returns.index, format="%Y%m").to_period('M')
    return returns

def get_hfi_returns():
    '''
    Load the Hedge Fund Index dataset for the returns of the hedge funds
    
    Returns
    -------
    pandas.DataFrame
        The returns of the hedge funds
    '''
    hfi = pd.read_csv(r"D:\UvA VU\Investment Management with Python and ML\notebooks_and_codem01_v02\data\edhec-hedgefundindices.csv",
                      header=0, index_col=0, parse_dates=True, dayfirst=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

def get_ind_returns():
    '''
    Load the Industry dataset for the returns of the industries
    
    Returns
    -------
    pandas.DataFrame
        The returns of the industries
    '''
    ind = pd.read_csv(r"D:\UvA VU\Investment Management with Python and ML\notebooks_and_codem01_v02\data\ind30_m_vw_rets.csv",
                      header=0, index_col=0, parse_dates=True)/100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def skewness(r: pd.Series):
    '''
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    
    Parameters
    ----------
    r : pd.Series
        Series of returns
        
    Returns
    -------
    float
        Skewness of the returns
    '''
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r: pd.Series):
    '''
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    
    Kurtosis is a measure of the "fatness" of the tails of a distribution
    KURTOSIS = 3 for a normal distribution
    Positive kurtosis means fatter tails
    
    Parameters
    ----------
    r : pd.Series
        Series of returns
        
    Returns
    -------
    float
        Kurtosis of the returns
    '''
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal_jb(r: pd.Series, level: float = 0.01):
    '''
    Applies the Jarque-Bera test to determine if a Series is normally distributed
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    
    Parameters
    ----------
    r : pd.Series
        Series of returns
    level : float
    
    Returns
    -------
    bool
        True if the hypothesis of normality is accepted, False otherwise
    '''
    statistic, p_value = stats.jarque_bera(r)
    return p_value > level

def is_normal_shapiro(r: pd.Series, level: float = 0.01):
    '''
    Applies the Shapiro-Wilk test to determine if a Series is normally distributed
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    
    Parameters
    ----------
    r : pd.Series
        Series of returns
    level : float
        The significance level of the test
        
    Returns
    -------
    bool
        True if the hypothesis of normality is accepted, False otherwise
    '''
    statistic, p_value = stats.shapiro(r)
    return p_value > level

def semideviation(r: pd.Series):
    '''
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame
    
    Parameters
    ----------
    r : pd.Series
        Series of returns
        
    Returns
    -------
    float
        Semideviation of the returns
    '''
    is_negative = r < 0
    return r[is_negative].std(ddof=0)

def var_historic(r, level: float = 5):
    '''
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    
    Parameters
    ----------
    r : pd.Series
        Series of returns
    level : float
        The VaR level
        
    Returns
    -------
    float
        Historic VaR at the specified level
    '''
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level) # Apply the function to each column
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level) # Returns the number such that "level" percent of the returns fall below that number
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
    
def var_gaussian(r, level: float = 5):
    '''
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    If the input is a DataFrame, returns a DataFrame with the corresponding VaRs
    
    Parameters
    ----------
    r : pd.Series
        Series of returns
    level : float
        The VaR level
        
    Returns
    -------
    float
        Gaussian VaR at the specified level
    '''
    # Compute the Z score assuming it was Gaussian
    z = stats.norm.ppf(level/100)
    return -(r.mean() + z*r.std(ddof=0))

def var_cornish_fisher(r, level: float = 5):
    '''
    Returns the Parametric Cornish-Fisher VaR of a Series or DataFrame
    If the input is a DataFrame, returns a DataFrame with the corresponding VaRs
    
    Cornish-Fisher modification of VaR based on observed skewness and kurtosis
    
    Parameters
    ----------
    r : pd.Series
        Series of returns
    level : float
        The VaR level
        
    Returns
    -------
    float
        Cornish-Fisher VaR at the specified level
    '''
    # Compute the Z score assuming it was Gaussian
    z = stats.norm.ppf(level/100)
    # Modify the Z score based on observed skewness and kurtosis
    s = skewness(r)
    k = kurtosis(r)
    z = (z + 
         (z**2 - 1)*s/6 +
         (z**3 -3*z)*(k-3)/24 -
         (2*z**3 - 5*z)*(s**2)/36
        )
    return -(r.mean() + z*r.std(ddof=0))

def cvar_historic(r, level: float = 5):
    '''
    Computes the Conditional VaR of Series or DataFrame
    
    Parameters
    ----------
    r : pd.Series
        Series of returns
    level : float
        The VaR level
        
    Returns
    -------
    float
        Historic CVaR at the specified level
    '''
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    elif isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
    
def cvar_gaussian(r, level: float = 5):
    '''
    Computes the Conditional VaR of Series or DataFrame
    
    Parameters
    ----------
    r : pd.Series
        Series of returns
    level : float
        The VaR level
        
    Returns
    -------
    float
        Gaussian CVaR at the specified level
    '''
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_gaussian, level=level)
    elif isinstance(r, pd.Series):
        is_beyond = r <= -var_gaussian(r, level=level)
        return -r[is_beyond].mean()
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
    
def cvar_cornish_fisher(r, level: float = 5):
    '''
    Computes the Conditional VaR of Series or DataFrame
    
    Parameters
    ----------
    r : pd.Series
        Series of returns
    level : float
        The VaR level
        
    Returns
    -------
    float
        Cornish-Fisher CVaR at the specified level
    '''
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_cornish_fisher, level=level)
    elif isinstance(r, pd.Series):
        is_beyond = r <= -var_cornish_fisher(r, level=level)
        return -r[is_beyond].mean()
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
    
def portfolio_return(weights, exp_returns):
    '''
    Computes the return on a portfolio from constituent returns and weights
    
    Parameters
    ----------
    weights : np.array
        Array of weights of each asset in the portfolio
    exp_returns : pd.DataFrame
        DataFrame of returns of each asset in the portfolio
        
    Returns
    -------
    float
        Return of the portfolio
    '''
    return weights.T @ exp_returns

def portfolio_vol(weights, covmat):
    '''
    Computes the volatility on a portfolio from a covariance matrix and weights
    
    Parameters
    ----------
    weights : np.array
        Array of weights of each asset in the portfolio
    covmat : pd.DataFrame
        Covariance matrix of the assets in the portfolio
        
    Returns
    -------
    float
        Volatility of the portfolio
    '''
    # covmat = exp_returns.cov()
    return (weights.T @ covmat @ weights)**0.5

def plot_2A_effrontier(n_points, exp_rets, cov):
    '''
    Plots the 2-asset efficient frontier
    
    Parameters
    ----------
    n_points : int
        Number of points to plot
    exp_rets : np.array
        Array of expected returns
    cov : pd.DataFrame
        Covariance matrix of the assets
    '''
    if exp_rets.shape[0] != 2 or exp_rets.shape[0] != 2:
        raise ValueError("plot_ef can only plot 2-asset frontiers")
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, exp_rets) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    return ef.plot.line(x="Volatility", y="Returns", style='.-')
    
def minimize_vol(target_return, exp_rets, cov):
    '''
    Returns the weights of the portfolio that has the minimum volatility
    
    Parameters
    ----------
    target_return : float
        The target return of the portfolio
    exp_rets : np.array
        Array of expected returns
    cov : pd.DataFrame
        Covariance matrix of the assets
        
    Returns
    -------
    np.array
        Weights of the portfolio that has the minimum volatility
    '''
    n = exp_rets.shape[0]
    init_guess = np.repeat(1/n, n) # Start with equal weights
    bounds = ((0.0, 1.0),) * n # Bounds for the weights (excluding leverage (>1) and shorting (<1))
    # Constraint 1: Our portfolio return is equal to the specified target return
    return_is_target = { 
        'type': 'eq', # The constraints are equalities
        'args': (exp_rets,), # Additional arguments are passed to the function
        'fun': lambda weights, exp_rets: target_return - portfolio_return(weights, exp_rets) # The function that is 'type' compared with zero
    }
    # Constraint 2: The sum of the weights is 1
    weights_sum_to_1 = {
        'type': 'eq', # The constraints are equalities
        'fun': lambda weights: np.sum(weights) - 1 # The function that is 'type' compared with zero
    }
    # Minimize the volatility using the Sequential Least Squares Quadratic Programming (SLSQP) method
    results = minimize(portfolio_vol, init_guess, args=(cov,), method='SLSQP', constraints=(return_is_target, weights_sum_to_1), bounds=bounds)
    return results.x # x is the solution, i.e., the weights that have the minimum volatility

def optimal_weights(n_points, exp_rets, cov):
    '''
    Returns the weights of the portfolio that has the minimum volatility for each target return
    
    Parameters
    ----------
    n_points : int
        Number of points to plot
    exp_rets : np.array
        Array of expected returns
    cov : pd.DataFrame
        Covariance matrix of the assets
        
    Returns
    -------
    pd.DataFrame
        Weights of the portfolio that has the minimum volatility for each target return
    '''
    target_returns = np.linspace(exp_rets.min(), exp_rets.max(), n_points)
    weights = [minimize_vol(target_return, exp_rets, cov) for target_return in target_returns]
    return weights 
    
def plot_nA_effrontier(n_points, exp_rets, cov):
    '''
    Plots the n-asset efficient frontier
    
    Parameters
    ----------
    n_points : int
        Number of points to plot
    exp_rets : np.array
        Array of expected returns
    cov : pd.DataFrame
        Covariance matrix of the assets
    '''
    weights = optimal_weights(n_points, exp_rets, cov)
    rets = [portfolio_return(w, exp_rets) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    return ef.plot.line(x="Volatility", y="Returns", style='.-')
        
    
def msr_portfolio(exp_rets, cov, risk_free_rate):
    '''
    Returns the weights of the portfolio that has the maximum Sharpe ratio
    
    Parameters
    ----------
    exp_rets : np.array
        Array of expected returns
    cov : pd.DataFrame
        Covariance matrix of the assets
    risk_free_rate : float
        The risk-free rate
        
    Returns
    -------
    np.array
        Weights of the portfolio that has the maximum Sharpe ratio
    '''
    n = exp_rets.shape[0]
    init_guess = np.repeat(1/n, n) 
    bounds = ((0.0, 1.0),) * n # Excluding leverage (>1) and shorting (<1)
    weights_sum_to_1 = {
        'type': 'eq', 
        'fun': lambda weights: np.sum(weights) - 1 
    }
    def neg_sharpe(weights, exp_rets, cov, risk_free_rate):
        return -((portfolio_return(weights, exp_rets) - risk_free_rate) / portfolio_vol(weights, cov))
    results = minimize(neg_sharpe, init_guess, args=(exp_rets, cov, risk_free_rate), method='SLSQP', constraints=(weights_sum_to_1), bounds=bounds)
    return results.x 

def plot_msr_portfolio(n_points, exp_rets, cov, risk_free_rate):
    '''
    Plots the maximum Sharpe ratio portfolio
    
    Parameters
    ----------
    n_points : int
        Number of points to plot
    exp_rets : np.array
        Array of expected returns
    cov : pd.DataFrame
        Covariance matrix of the assets
    risk_free_rate : float
        The risk-free rate
    '''
    ax = plot_nA_effrontier(n_points, exp_rets, cov)
    w_msr = msr_portfolio(exp_rets, cov, risk_free_rate)
    ret_msr = portfolio_return(w_msr, exp_rets)
    vol_msr = portfolio_vol(w_msr, cov)
    # Add Capital Market Line
    cml_x = [0, vol_msr]
    cml_y = [risk_free_rate, ret_msr]
    ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed')
    ax.set_xlim(left=0)
    ax.legend(["Efficient Frontier", "Capital Market Line"])
    return ax

def gmv_portfolio(cov):
    '''
    Returns the weights of the Global Minimum Variance (Volatility) portfolio.
    GMV portfolio is the MSR portfolio with a risk-free rate of 0
    
    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix of the assets
        
    Returns
    -------
    np.array
        Weights of the Global Minimum Volatility portfolio
    '''
    n = cov.shape[0]
    return msr_portfolio(np.repeat(1, n), cov, 0.0) # If all returns are equal, the optimizer can only minimize volatility

def plot_msr_ew_gmv(n_points, exp_rets, cov, risk_free_rate):
    '''
    Plots the maximum Sharpe ratio portfolio
    
    Parameters
    ----------
    n_points : int
        Number of points to plot
    exp_rets : np.array
        Array of expected returns
    cov : pd.DataFrame
        Covariance matrix of the assets
    risk_free_rate : float
        The risk-free rate
    '''
    ax = plot_nA_effrontier(n_points, exp_rets, cov)
    w_msr = msr_portfolio(exp_rets, cov, risk_free_rate)
    ret_msr = portfolio_return(w_msr, exp_rets)
    vol_msr = portfolio_vol(w_msr, cov)
    # Add Capital Market Line
    cml_x = [0, vol_msr]
    cml_y = [risk_free_rate, ret_msr]
    ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed')
    ax.set_xlim(left=0)
    # Add Equal Weighted Portfolio
    n = exp_rets.shape[0]
    w_ew = np.repeat(1/n, n)
    ret_ew = portfolio_return(w_ew, exp_rets)
    vol_ew = portfolio_vol(w_ew, cov)
    ax.plot([vol_ew], [ret_ew], color='goldenrod', marker='o', markersize=10)
    # Add Global Minimum Volatility Portfolio
    w_gmv = gmv_portfolio(cov)
    ret_gmv = portfolio_return(w_gmv, exp_rets)
    vol_gmv = portfolio_vol(w_gmv, cov)
    ax.plot([vol_gmv], [ret_gmv], color='midnightblue', marker='o', markersize=10)
    ax.legend(["Efficient Frontier", "Capital Market Line", "Equal Weighted Portfolio", "Global Minimum Variance Portfolio"])
    return ax

def get_ind_size():
    '''
    Load the Industry dataset for the size of the industries
    
    Returns
    -------
    pandas.DataFrame
        The size of the industries
    '''
    ind = pd.read_csv(r"D:\UvA VU\Investment Management with Python and ML\notebooks_and_codem01_v02\data\ind30_m_size.csv",
                      header=0, index_col=0, parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_nfirms():
    '''
    Load the Industry dataset for the number of firms in the industries
    
    Returns
    -------
    pandas.DataFrame
        The number of firms in the industries
    '''
    ind = pd.read_csv(r"D:\UvA VU\Investment Management with Python and ML\notebooks_and_codem01_v02\data\ind30_m_nfirms.csv",
                      header=0, index_col=0, parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def run_cppi(risky_r, safe_r = None, m = 3, start = 1000, floor = 0.8, risk_free_rate = 0.03, drawdown = None):
    '''
    Run a backtest of the CPPI strategy, given the risky asset returns, the safe asset returns, the multiplier, the starting value, the floor, and the risk-free rate
    
    Parameters
    ----------
    risky_r : pd.Series
        The returns of the risky asset
    safe_r : pd.Series
        The returns of the safe asset
    m : int
        The multiplier
    start : float
        The starting value
    floor : float
        The floor
    risk_free_rate : float
        The risk-free rate
        
    Returns
    -------
    pd.DataFrame
        The backtest of the CPPI strategy
    '''
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = start
    
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=["Risky"])
        
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = risk_free_rate/12
        
    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    
    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown)
        cushion = (account_value - floor_value) / account_value
        risky_w = m * cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1 - risky_w
        risky_alloc = account_value * risky_w
        safe_alloc = account_value * safe_w
        # Update the account value for this time step
        account_value = risky_alloc * (1 + risky_r.iloc[step]) + safe_alloc * (1 + safe_r.iloc[step])
        # Update the peak
        peak = np.maximum(peak, account_value)
        # Update the floor value
        floor_value = peak * floor
        # Save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        
    risky_wealth = start * (1 + risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r": risky_r,
        "safe_r": safe_r
    }
    return backtest_result

def summary_stats(r, risk_free_rate = 0.03):
    '''
    Return a DataFrame that contains aggregated summary statistics for the returns in the columns of r
    
    Parameters
    ----------
    r : pd.DataFrame
        DataFrame of returns
    risk_free_rate : float
        The risk-free rate
        
    Returns
    -------
    pd.DataFrame
        DataFrame that contains aggregated summary statistics for the returns in the columns of r
    '''
    ann_r = r.aggregate(annualized_returns)
    ann_vol = r.aggregate(annualized_volatility)
    ann_sr = r.aggregate(sharpe_ratio, risk_free_rate=risk_free_rate)
    dd = r.aggregate(get_max_drawdown)
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_cornish_fisher, level=5)
    hist_cvar5 = r.aggregate(cvar_historic, level=5)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5
    })
    
def gbm(n_years = 10, n_scenarios = 1000, mu = 0.07, sigma = 0.15, steps_per_year = 12, start = 100.0):
    '''
    Evolution of a stock price using a Geometric Brownian Motion model
    
    Parameters
    ----------
    n_years : int
        The number of years to simulate
    n_scenarios : int
        The number of scenarios to simulate
    mu : float
        The annualized drift of the stock price
    sigma : float
        The annualized volatility of the stock price
    steps_per_year : int
        The number of steps per year
    start : float
        The starting stock price
        
    Returns
    -------
    pd.DataFrame
        DataFrame that contains the simulated stock price
    '''
    dt = 1/steps_per_year
    n_steps = int(n_years * steps_per_year)
    # Generate the normal random numbers, avoiding loops for efficiency 
    shock = np.random.normal(loc=1+mu*dt, scale=sigma*np.sqrt(dt), size=(n_steps + 1, n_scenarios))
    rets_plus1 = pd.DataFrame(shock)
    rets_plus1.iloc[0] = 1 # Set the initial value to 1
    prices = start * (rets_plus1).cumprod()
    return prices

def show_gbm(n_scenarios, mu, sigma):
    '''
    Show the evolution of a stock price using a Geometric Brownian Motion model
    
    Parameters
    ----------
    n_scenarios : int
        The number of scenarios to simulate
    mu : float
        The annualized drift of the stock price
    sigma : float
        The annualized volatility of the stock price
    '''
    s_0 = 100
    prices = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, start=s_0)
    ax = prices.plot(legend=False, color="indianred", alpha=0.5, linewidth=2, figsize=(12, 6))
    ax.axhline(y=s_0, ls=":", color="black")
    ax.set_ylim(top=400)
    ax.plot(0, s_0, marker='o', color='darkred', alpha=0.2)
    return ax

def show_cppi(n_scenarios = 50, mu = 0.07, sigma = 0.15, m = 3, floor = 0., risk_free_rate = 0.03, y_max = 100):
    '''
    Show the Monte Carlo evolution of a CPPI strategy
    
    Parameters
    ----------
    n_scenarios : int
        The number of scenarios to simulate
    mu : float
        The annualized drift of the stock price
    sigma : float
        The annualized volatility of the stock price
    m : int
        The multiplier
    floor : float
        The floor
    risk_free_rate : float
        The risk-free rate
    y_max : float
        The maximum value of the y-axis
        
    Returns
    -------
    pd.DataFrame
        DataFrame that contains the Monte Carlo evolution of a CPPI strategy
    '''
    start = 100
    risky_r = pd.DataFrame(gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, start=start))
    cppi_result = run_cppi(risky_r=pd.DataFrame(risky_r), m=m, start=start, floor=floor, risk_free_rate=risk_free_rate)
    wealth = cppi_result["Wealth"].dropna()
    # risky_wealth = cppi_result["Risky Wealth"]
    # terminal wealth stats
    y_max = wealth.values.max() * y_max/100
    # if np.isnan(y_max) or np.isinf(y_max):
    #     y_max = 100
    terminal_wealth = wealth.iloc[-1]
    
    tw_mean = terminal_wealth.mean()
    tw_median = terminal_wealth.median()
    failure_mask = np.less(terminal_wealth, start*floor)
    n_failures = failure_mask.sum()
    p_fail = n_failures/n_scenarios
    
    e_shortfall = np.dot(terminal_wealth-start*floor, failure_mask)/n_failures if n_failures > 0 else 0.0
    
    fig, (wealth_ax, hist_ax) = plt.subplots(nrows=1, ncols=2, sharey = True, gridspec_kw={'width_ratios': [3,2]}, figsize=(24, 9))
    plt.subplots_adjust(wspace=0.0)
    
    wealth.plot(ax=wealth_ax, legend=False, alpha=0.3, color="indianred", linewidth=2)
    wealth_ax.axhline(y=start, ls=":", color="black")
    wealth_ax.axhline(y=start*floor, ls="--", color="red")
    wealth_ax.set_ylim(top=y_max)
    
    terminal_wealth.plot.hist(ax=hist_ax, bins=50, ec='w', fc='indianred', orientation='horizontal')
    hist_ax.axhline(y=start, ls=":", color="black")
    hist_ax.axhline(y=tw_mean, ls=":", color="blue")
    hist_ax.axhline(y=tw_median, ls=":", color="purple")
    hist_ax.annotate(f"Mean: ${int(tw_mean)}", xy=(0.7, 0.9), xycoords='axes fraction', fontsize=24)
    hist_ax.annotate(f"Median: ${int(tw_median)}", xy=(0.7, 0.85), xycoords='axes fraction', fontsize=24)
    if (floor > 0.01):
        hist_ax.axhline(y=start*floor, ls="--", color="red", linewidth=3)
        hist_ax.annotate(f"Violations: {n_failures} ({p_fail*100:2.2f}%)\nE(shortfall)=${e_shortfall:2.2f}", xy=(0.7, 0.7), xycoords='axes fraction', fontsize=24)
    # return cppi_result

def discount(t, r):
    '''
    Compute the price of a pure discount bond that pays a dollar at time t where r is the per-period interest rate
    
    Parameters
    ----------
    t : int
        The time
    r : float
        The per-period interest rate
        
    Returns
    -------
    float
        The price of a pure discount bond
    '''
    return 1/(1+r)**t

# def discount(t, r):
#     """
#     Compute the price of a pure discount bond that pays a dollar at time period t
#     and r is the per-period interest rate
#     returns a |t| x |r| Series or DataFrame
#     r can be a float, Series or DataFrame
#     returns a DataFrame indexed by t
    
#     Parameters
#     ----------
#     t : int, array-like
#         The time
#     r : float, array-like
#         The per-period interest rate
        
#     Returns
#     -------
#     DataFrame
#         The price of a pure discount bond
#     """
#     discounts = pd.DataFrame([(r+1)**-i for i in t])
#     discounts.index = t
#     return discounts

def present_value(liabilities, r):
    '''
    Compute the present value of a sequence of liabilities
    
    Parameters
    ----------
    liabilities : pd.Series
        The time-indexed liabilities
    r : float
        The per-period interest rate
        
    Returns
    -------
    float
        The present value of the liabilities
    '''
    dates = liabilities.index
    discounts = discount(dates, r)
    return (discounts * liabilities).sum()

# def present_value(cash_flows, r):
#     '''
#     Compute the present value of a sequence of cash flows
    
#     Parameters
#     ----------
#     cash_flows : pd.Series
#         The time-indexed cash_flows
#     r : float
#         The per-period interest rate
        
#     Returns
#     -------
#     pd.Series
#         The present value of the cash_flows
#     '''
#     dates = cash_flows.index
#     discounts = discount(dates, r)
#     return (discounts * cash_flows).sum()

def funding_ratio(assets, liabilities, r):
    '''
    Compute the funding ratio of some assets given liabilities and a per-period interest rate
    
    Parameters
    ----------
    assets : pd.Series
        The time-indexed assets
    liabilities : pd.Series
        The time-indexed liabilities
    r : float
        The per-period interest rate
        
    Returns
    -------
    float
        The funding ratio of the assets
    '''
    return present_value(assets, r)/present_value(liabilities, r)

def inst_to_ann(r):
    '''
    Converts short rate to an annualized rate
    
    Parameters
    ----------
    r : float
        The short rate
        
    Returns
    -------
    float
        The annualized rate
    '''
    return np.expm1(r) 

def ann_to_inst(r):
    '''
    Converts annualized rate to a short rate
    
    Parameters
    ----------
    r : float
        The annualized rate
        
    Returns
    -------
    float
        The short rate
    '''
    return np.log1p(r)

def cir(n_years = 10, n_scenarios = 1, a = 0.05, b = 0.03, sigma = 0.05, steps_per_year = 12, r_0 = None):
    '''
    Generate random interest rate evolution using the CIR model
    
    Parameters
    ----------
    n_years : int
        The number of years to simulate
    n_scenarios : int
        The number of scenarios to simulate
    a : float
        The mean-reversion level
    b : float
        The mean-reversion speed
    sigma : float
        The volatility
    steps_per_year : int
        The number of steps per year
    r_0 : float
        The initial rate
        
    Returns
    -------
    pd.DataFrame
        DataFrame that contains the simulated interest rate evolution
    '''
    if r_0 is None:
        r_0 = b
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    n_steps = int(n_years * steps_per_year)
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(n_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0
    for step in range(1, n_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = np.abs(r_t + d_r_t)
    return pd.DataFrame(data=inst_to_ann(rates), index=range(n_steps))

def show_cir(n_scenarios=5, a = 0.05, b = 0.03, sigma = 0.05, r_0 = 0.03):
    '''
    Show the evolution of interest rates using the CIR model
    
    Parameters
    ----------
    n_scenarios : int
        The number of scenarios to simulate
    a : float
        The mean-reversion level
    b : float
        The mean-reversion speed
    sigma : float
        The volatility
    r_0 : float
        The initial rate
    '''
    cir(r_0=r_0, a=a, b=b, sigma=sigma, n_scenarios=n_scenarios).plot(legend=False, figsize=(12, 6))

def cir_prices(n_years = 10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None: r_0 = b 
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1 # because n_years might be a float
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    ## For Price Generation
    h = np.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    ####

    def price(ttm, r):
        _A = ((2*h*np.exp((h+a)*ttm/2))/(2*h+(h+a)*(np.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(np.exp(h*ttm)-1))/(2*h + (h+a)*(np.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    ####
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years-step*dt, rates[step])

    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    ### for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates, prices

def show_cir_prices(n_scenarios=5, a=0.05, b=0.03, sigma=0.05, r_0=0.03):
    """
    Show the CIR bond prices given the parameters
    """
    cir_prices(r_0=r_0, a=a, b=b, sigma=sigma, n_scenarios=n_scenarios)[1].plot(legend=False, figsize=(12, 6))
   
def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    '''
    Generates the sequence of cash flows generated by a bond, assuming the bond pays a fixed coupon rate
    
    Parameters
    ----------
    maturity : int
        The number of years until the bond matures
    principal : float
        The principal of the bond
    coupon_rate : float
        The coupon rate of the bond
    coupons_per_year : int
        The number of coupon payments per year
        
    Returns
    -------
    pd.Series
        Series that contains the sequence of cash flows generated by a bond
    '''
    n_coupons = int(maturity * coupons_per_year)
    coupon_amt = principal * coupon_rate / coupons_per_year
    coupon_times = np.arange(1, n_coupons+1)
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows

def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    '''
    Computes the price of a bond that pays periodic coupons
    
    Parameters
    ----------
    maturity : int
        The number of years until the bond matures
    principal : float
        The principal of the bond
    coupon_rate : float
        The coupon rate of the bond
    coupons_per_year : int
        The number of coupon payments per year
    discount_rate : float
        The discount rate
        
    Returns
    -------
    float
        The price of the bond
    '''
    cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
    return present_value(cash_flows, discount_rate/coupons_per_year)

def macaulay_duration(cash_flows, discount_rate):
    """
    Computes the Macaulay Duration of a sequence of cash flows based on weighted-average of duration with respect to present value of cash flows
    
    Parameters
    ----------
    cash_flows : pd.Series
        The time-indexed cash flows
    discount_rate : float
        The per-period discount rate
        
    Returns
    -------
    float
        The Macaulay Duration
    """
    discounted_flows = discount(cash_flows.index, discount_rate) * cash_flows
    weights = discounted_flows/discounted_flows.sum()
    return np.average(cash_flows.index, weights=weights)

def match_durations(cf_t, cf_s, cf_l, discount_rate):
    """
    Returns the weight W in cf_s and (1-W) in cf_l that, when combined, have the same duration as cf_t
    
    Parameters
    ----------
    cf_t : pd.Series
        The time-indexed cash flows of the target bond
    cf_s : pd.Series
        The time-indexed cash flows of the short bond
    cf_l : pd.Series
        The time-indexed cash flows of the long bond
    discount_rate : float
        The per-period discount rate
        
    Returns
    -------
    float
        The weight W
    """
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_l - d_t) / (d_l - d_s)

def btmix(r1, r2, allocator, **kwargs):
    '''
    Runs a backtest of a simple dynamic allocation strategy between two sets of returns
    
    Parameters
    ----------
    r1 : pd.Series
        The returns of the first asset
    r2 : pd.Series
        The returns of the second asset
    allocator : function
        The allocator function that determines the weights of the two assets
    **kwargs:
        Additional keyword arguments for the allocator function
    
    Returns
    -------
    pd.DataFrame:
        The backtest results
    '''
    if not r1.index.equals(r2.index):
        raise ValueError("Indices do not match")
    weights = allocator(r1, r2, **kwargs)
    if not weights.index.equals(r1.index):
        weights = weights.reindex(r1.index).ffill()
    r_mix = weights[0] * r1 + (1 - weights[0]) * r2
    return pd.DataFrame({
        "Wealth": (1 + r_mix).cumprod(),
        "R1": (1 + r1).cumprod(),
        "R2": (1 + r2).cumprod()
    })
    
def fixedmix_allocator(r1, r2, w1, **kwargs):
    '''
    Determine the weights of the two assets in a fixed mix strategy
    
    Parameters
    ----------
    r1 : pd.Series
        The returns of the first asset
    r2 : pd.Series
        The returns of the second asset
    w1 : float
        The weight of the first asset
        
    Returns
    -------
    pd.DataFrame
        The weights of the two assets
    '''
    return pd.DataFrame(data = w1, index=r1.index, columns=r1.columns)

def terminal_values(rets):
    '''
    Returns the terminal values of a set of returns
    
    Parameters
    ----------
    rets : pd.DataFrame
        DataFrame of returns
        
    Returns
    -------
    pd.Series
        Series of terminal values
    '''
    return (rets+1).prod()

def terminal_stats(rets, floor = 0.8, cap = np.inf, name = "Stats"):
    '''
    Returns the summary statistics of the terminal values of a set of returns
    
    Parameters
    ----------
    
    rets : pd.DataFrame
        DataFrame of returns
    floor : float
        The floor
    cap : float
        The cap
    name : str
        The name of the statistics
        
    Returns
    -------
    pd.DataFrame
        DataFrame of the summary statistics of the terminal values
    '''
    terminal_wealth = terminal_values(rets)
    breach = terminal_wealth < floor
    reach = terminal_wealth > cap
    p_breach = breach.mean()
    p_reach = reach.mean()
    e_shortfall = (floor - terminal_wealth).clip(lower=0).mean()
    e_excess = (terminal_wealth - cap).clip(lower=0).mean()
    var_floor = np.percentile(terminal_wealth, 100*floor)
    var_cap = np.percentile(terminal_wealth, 100*cap)
    return pd.DataFrame({
        "Mean": terminal_wealth.mean(),
        "Median": terminal_wealth.median(),
        "Min": terminal_wealth.min(),
        "Max": terminal_wealth.max(),
        "P_breach": p_breach,
        "P_reach": p_reach,
        "E(shortfall)": e_shortfall,
        "E(excess)": e_excess,
        "Var(floor)": var_floor,
        "Var(cap)": var_cap
    }, index=[name])
    
def glidepath_allocator(r1, r2, start_glide=1, end_glide=0):
    '''
    Determine the weights of the two assets in a target-date-fund style glidepath strategy
    
    Parameters
    ----------
    r1 : pd.Series
        The returns of the first asset
    r2 : pd.Series
        The returns of the second asset
    start_glide : float
        The starting weight of the first asset
    end_glide : float
        The ending weight of the first asset
        
    Returns
    -------
    pd.DataFrame
        The weights of the two assets
    '''
    n_points = r1.shape[0]
    weights = pd.Series(index=r1.index)
    n_points = len(weights)
    n_half = n_points // 2
    weights.iloc[:n_half] = start_glide
    weights.iloc[n_half:] = end_glide
    return pd.DataFrame(data = weights, index=r1.index, columns=r1.columns)

def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    '''
    Allocate between PSP and GHP with the goal to provide exposure to the PSP without violating the floor
    
    Parameters
    ----------
    psp_r : pd.Series
        The returns of the PSP
    ghp_r : pd.Series
        The returns of the GHP
    floor : float
        The floor
    zc_prices : pd.DataFrame
        The prices of the zero-coupon bond
    m : int
        The multiplier
        
    Returns
    -------
    pd.DataFrame
        The weights of the PSP and GHP
    '''
    if zc_prices.shape != psp_r.shape:
        raise ValueError("PSP and ZC Prices must have the same shape")
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc(step) # (1+ghp_r.iloc[step])/1.03
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1)
        ghp_w = 1 - psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    return w_history

def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    '''
    Allocate between PSP and GHP with the goal to provide exposure to the PSP without violating the floor
    
    Parameters
    ----------
    psp_r : pd.Series
        The returns of the PSP
    ghp_r : pd.Series
        The returns of the GHP
    maxdd : float
        The maximum drawdown
    m : int
        The multiplier
        
    Returns
    -------
    pd.DataFrame
        The weights of the PSP and GHP
    '''
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = (1-maxdd)*np.maximum.accumulate(account_value)
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1)
        ghp_w = 1 - psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # compute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        # apply the drawdown constraint
        peak = np.maximum.accumulate(account_value)
        floor_value = peak*(1-maxdd)
        account_value = np.minimum(account_value, floor_value) # back to the floor if drawdown is too high
        peak_value = np.maximum(peak_value, account_value)
        w_history.iloc[step] = psp_w
    return w_history

def simulate_cppi(risky_r, safe_r, m, start, floor=0.8, risk_free_rate=0.03, drawdown=None):
    '''
    Run a backtest of the CPPI strategy, given the risky asset returns, the safe asset returns, the multiplier, the starting value, the floor, and the risk-free rate
    
    Parameters
    ----------
    risky_r : pd.Series
        The returns of the risky asset
    safe_r : pd.Series
        The returns of the safe asset
    m : int
        The multiplier
    start : float
        The starting value
    floor : float
        The floor
    risk_free_rate : float
        The risk-free rate
    drawdown : float
        The maximum drawdown
        
    Returns
    -------
    pd.DataFrame
        The backtest of the CPPI strategy
    '''
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = start
    
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=["Risky"])
        
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = risk_free_rate/12
        
    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    
    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown)
        cushion = (account_value - floor_value) / account_value
        risky_w = m * cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1 - risky_w
        risky_alloc = account_value * risky_w
        safe_alloc = account_value * safe_w
        # Update the account value for this time step
        account_value = risky_alloc * (1 + risky_r.iloc[step]) + safe_alloc * (1 + safe_r.iloc[step])
        # Update the peak
        peak = np.maximum(peak, account_value)
        # Update the floor value
        floor_value = peak * floor
        # Save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        
    risky_wealth = start * (1 + risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r": risky_r,
        "safe_r": safe_r
    }
    return backtest_result

