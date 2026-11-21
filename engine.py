import numpy as np


def Run_Monte_Carlo_Simulation(days, assets, number_of_portfolios, risk_free_rate):
    #input assets, number of portfolios, risk-free rate and number of trading days
    prices = np.random.random((days, assets))

    #log returns calculation
    log_returns = np.log(prices[1:] / prices[:-1])
    mean_returns = np.mean(log_returns, axis=0)
    cov_matrix = np.cov(log_returns,rowvar=False) * days

    # Simulate random portfolio weights
    weights = np.random.random((number_of_portfolios, assets))
    weights /= np.sum(weights, axis=1, keepdims=True)  # Normalize weights to sum to 1

    # Calculate portfolio returns and volatility
    portfolio_returns = np.sum(weights * mean_returns, axis=1) * days
    portfolio_volatility = np.sqrt(np.sum(weights @ cov_matrix * weights, axis=1)) * np.sqrt(days)

    # Calculate Sharpe ratio
    sharpe_ratio = (portfolio_returns - risk_free_rate) / portfolio_volatility

    # Find the index of the maximum Sharpe ratio
    max_sharpe_idx = np.argmax(sharpe_ratio)

    # Calculate risk metrics (5th percentile)
    Risk_metrics = np.percentile(portfolio_returns, 5)

    return portfolio_returns, portfolio_volatility, sharpe_ratio, Risk_metrics, max_sharpe_idx

def return_weights(assets, number_of_portfolios):
    weights = np.random.random((number_of_portfolios, assets))
    weights /= np.sum(weights, axis=1, keepdims=True)  # Normalize weights to sum to 1
    return weights
