import numpy as np
from statsmodels.regression.linear_model import OLS, OLSResults
import statsmodels.api as sm
from typing import List
import pandas as pd


def distributed_qr_regression(xparts: List[np.ndarray], yparts: List[np.ndarray]):
    """
    Perform distributed linear regression using QR decomposition.
    
    Parameters
    ----------
    xparts : list of ndarray
        List of design matrices from each client
    yparts : list of ndarray
        List of target vectors from each client
        
    Returns
    -------
    beta : ndarray
        Aggregated parameter estimates
    cov_params : ndarray
        Aggregated variance-covariance matrix
    """
    n_clients = len(xparts)
    if n_clients != len(yparts):
        raise ValueError("Number of X and y parts must match")
    
    # Store local results
    local_betas = []
    local_covs = []
    sample_sizes = []
    
    # Fit local models on each client
    for i in range(n_clients):
        X = xparts[i]
        y = yparts[i]
        sample_sizes.append(X.shape[0])
        
        # QR decomposition
        Q, R = np.linalg.qr(X)
        
        # Calculate parameters
        effects = np.dot(Q.T, y)
        beta = np.linalg.solve(R, effects)
        local_betas.append(beta)
        
        # Calculate covariance matrix
        normalized_cov = np.linalg.inv(np.dot(R.T, R))
        local_covs.append(normalized_cov)
    
    # Aggregate parameters based on sample sizes
    total_samples = sum(sample_sizes)
    weights = [size / total_samples for size in sample_sizes]
    
    # Weighted average of beta coefficients
    beta_agg = np.zeros_like(local_betas[0])
    for i in range(n_clients):
        beta_agg += weights[i] * local_betas[i]
    
    # Weighted average of covariance matrices
    cov_agg = np.zeros_like(local_covs[0])
    for i in range(n_clients):
        cov_agg += weights[i] * local_covs[i]
    
    return beta_agg, cov_agg


def eval_fed_reg_linear(X_train_imps: List[pd.DataFrame], y_trains: List[pd.Series]):
    
    np.random.seed(seed=233423)
    X_train_imps = [
        pd.concat(
            [
                pd.DataFrame(np.ones((X_train_imp.shape[0], 1), dtype=np.float64), columns=['const']), 
                X_train_imp.copy().reset_index(drop=True)
            ], 
            axis=1
        ).copy() 
        for X_train_imp in X_train_imps
    ]

    beta, cov_params = distributed_qr_regression(
        [item.values for item in X_train_imps],  [item.values for item in y_trains]
    )
    X = pd.concat(X_train_imps, axis=0)
    y = pd.concat(y_trains, axis=0)
    
    X_with_const = sm.add_constant(X)
    ols_model = sm.OLS(y, X_with_const)
    ols_results = ols_model.fit()

    return OLSResults(
        ols_model, beta, cov_params, cov_type=ols_results.cov_type, cov_kwds=ols_results.cov_kwds, use_t=ols_results.use_t
    )

