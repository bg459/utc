import numpy as np
import pandas as pd
from scipy.optimize import minimize, NonlinearConstraint

#########################################################################
## Change this code to take in all asset price data and predictions    ##
## for one day and allocate your portfolio accordingly.                ##
#########################################################################

returns = np.zeros((9, 0))
prices = np.zeros((9,0))

test_params = {
    'lookback': 2, 
    'EMA_eps': 0.14, 
    'alpha_w': [1, 0], 
    'pca_count': 1, 
    'var_penalty': 2, 
    'var_cap': 0.01, 
    'subtr': True, 
    'vol_look': 21, 
    'vol_power': 1.5,
}

def pca(vec, comp=-1):
    """Perform PCA, output components + explained variance %"""
    c_vec = vec - np.mean(vec, axis=0)
    cov = np.cov(c_vec) / vec.shape[0]
    v, w = np.linalg.eig(cov)
    idx = v.argsort()[::-1] # Sort descending and get sorted indices
    v = v[idx]
    w = w[:,idx]
    return w[:, :comp], np.sum(np.square(v[:comp]))/np.sum(np.square(v))

# Generate alphas from features and hyperparameters
# Considered methods include EMA on a lookback period
def getAlphas(rets, pred_rets, EMA_eps=0.01, alpha_w=[0.5, 0.5], mom_rev=1, **params):
    """
    Returns an array of alphas, same size as input rets.
    pred_rets should be a singular array of size # assets.
    Also returns cov of initial alphas = pred. returns
    """
    n = rets.shape[1]
    per_weights = np.power(1.000 - EMA_eps, np.arange(n)[::-1])
    EMA_rets = rets * np.expand_dims(per_weights, 0)
    # perform EMA on given rets
    self_alphas = EMA_rets.sum(axis=1)
    weighted_alphas = self_alphas*alpha_w[0] + pred_rets*alpha_w[1]
    return weighted_alphas*mom_rev, np.cov(rets)
    
# Generate market betas using PCA on alphas
# We use these to try to control the market neutrality of the opt
def getBetas(rets, pca_count=1, **params):
    """
    Returns an array of betas with size = # assets in universe
    """
    # standardize returns
    stan_rets = rets - np.mean(rets, axis=1, keepdims=True)
    stan_rets = stan_rets / np.std(stan_rets, axis=1, keepdims=True)
    betas, expl_var = pca(stan_rets, pca_count)
    betas = betas / np.sum(np.abs(betas))
    return betas

# Define utility function to optimize for weights
def util_func(w, alphas, cov, var_penalty, subtr=True, **params):
    ER = np.sum(w*alphas)
    var_factor = w.T @ cov @ w
    
    # change this to / to test other expression
    if subtr:
        return ER - (var_penalty*var_factor)
    else:
        return ER / var_factor

# Solve optimization problem from features and hyperparameters
def solveWeights(rets, pred_rets, var_penalty=1, var_cap=0.001, **params):
    """
    Returns an array of weights with size = # assets in universe.
    Also returns expected variance
    """
    m = rets.shape[0]
    alphas, cov_mat = getAlphas(rets, pred_rets, **params)
    betas = getBetas(rets, **params)
    
    # define optimization problem using scipy
    func = lambda x: util_func(x, alphas, cov_mat, var_penalty, **params) # utility
    init_x = [0.5/m for i in range(m)]
    bnds = ((0, 1),) * m # nonnegative elementwise bounds on weights
    portfolio_var = lambda x: x.T @ cov_mat @ x # portfolio variance
    cum_weights = lambda x: np.sum(np.abs(x))
    cum_betas = lambda x: np.sum(x*betas)
    cons = (
        NonlinearConstraint(cum_betas, 0, 0), # market neutrality
        NonlinearConstraint(cum_weights, 0, 1, keep_feasible=True), # weight allocation
        NonlinearConstraint(portfolio_var, 0, var_cap), # hard variance cap
    )
    res = minimize(func, init_x, bounds=bnds, constraints=cons)
    
    weights = res.x
    return weights

def allocate_portfolio(asset_prices, asset_price_predictions_1,
                       asset_price_predictions_2,
                       asset_price_predictions_3):
    # returns a vector (n_assets,) with the weights for the portfolio
    
    # This simple strategy equally weights all assets every period
    # (called a 1/n strategy).
    global returns
    global prices

    new_prices = np.array(asset_prices).reshape((9,1))

    prices = np.concatenate([prices, new_prices], axis=1)

    if prices.shape[1] > 1:
        new_returns = (prices[:,-1] - prices[:,-2]) / prices[:,-2]
        new_returns = new_returns.reshape(9,1)

        returns = np.concatenate([returns, new_returns], axis=1)

    # n_assets = len(asset_prices)
    # weights = np.repeat(1 / n_assets, n_assets)
    # return weights

    if returns.shape[1] <= 1:
        n_assets = len(asset_prices)
        weights = np.repeat(1 / n_assets, n_assets)
        return weights
    else:
        rets = returns[:, -test_params['lookback']:]
        pred_rets = returns[:,-1]
        
        act_look = min(returns.shape[1], test_params['vol_look'])
        vol_rets = returns[:, -act_look:]
        vol = np.std(vol_rets, axis=1)
        inv_vol = 1 / vol ** test_params['vol_power']
        inv_vol = inv_vol / np.sum(inv_vol) * returns.shape[0]
        
        test_params['mom_rev'] = inv_vol
        
        # solve
        weights = solveWeights(rets, pred_rets, **test_params)
        # assert np.sum(weights) <= 1
        # weights /= max(np.sum(np.abs(weights)), 1)
        weights /= np.sum(np.abs(weights))
        return weights
        
