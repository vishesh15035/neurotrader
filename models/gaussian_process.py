"""
Gaussian Process Regression — Uncertainty Quantification
Full Bayesian inference: posterior mean + variance
Kernels: RBF, Matern, Periodic, Composite
"""
import numpy as np
from scipy.linalg import cholesky, solve_triangular

class GPKernel:
    """Composite kernel: RBF + Periodic + Linear"""
    def __init__(self, length_scale=10.0, period=5.0,
                 noise=0.01, alpha=1.0, beta=0.1):
        self.ls     = length_scale
        self.period = period
        self.noise  = noise
        self.alpha  = alpha
        self.beta   = beta

    def rbf(self, X1, X2):
        """Squared Exponential kernel"""
        d = X1[:,None] - X2[None,:]
        return np.exp(-0.5 * d**2 / self.ls**2)

    def periodic(self, X1, X2):
        """Periodic kernel for weekly/monthly cycles"""
        d = np.abs(X1[:,None] - X2[None,:])
        return np.exp(-2*np.sin(np.pi*d/self.period)**2 / self.ls**2)

    def linear(self, X1, X2):
        """Linear trend kernel"""
        return self.beta * X1[:,None] * X2[None,:]

    def __call__(self, X1, X2, noise=False):
        K = self.alpha * self.rbf(X1,X2) + 0.3*self.periodic(X1,X2) + 0.1*self.linear(X1,X2)
        if noise: K += self.noise**2 * np.eye(len(X1))
        return K

class GaussianProcessPredictor:
    """
    GP Regression for price uncertainty quantification
    Gives: predicted price + 95% confidence interval
    """
    def __init__(self, kernel: GPKernel = None):
        self.kernel  = kernel or GPKernel()
        self.X_train = None
        self.y_train = None
        self.L       = None
        self.alpha_  = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """O(n^3) training via Cholesky decomposition"""
        self.X_train = X
        self.y_train = y
        K      = self.kernel(X, X, noise=True)
        self.L = cholesky(K, lower=True)
        self.alpha_ = solve_triangular(
            self.L.T, solve_triangular(self.L, y, lower=True), lower=False)
        print(f"[GP] Fitted on {len(X)} points. Cholesky decomposition complete.")

    def predict(self, X_test: np.ndarray) -> dict:
        """O(n^2) prediction with uncertainty"""
        K_star  = self.kernel(X_test, self.X_train)
        K_ss    = self.kernel(X_test, X_test, noise=False)
        # Posterior mean
        mu      = K_star @ self.alpha_
        # Posterior variance
        V       = solve_triangular(self.L, K_star.T, lower=True)
        cov     = K_ss - V.T @ V
        std     = np.sqrt(np.maximum(np.diag(cov), 0))
        return {
            "mean":     mu.tolist(),
            "std":      std.tolist(),
            "upper_95": (mu + 1.96*std).tolist(),
            "lower_95": (mu - 1.96*std).tolist(),
            "upper_68": (mu + std).tolist(),
            "lower_68": (mu - std).tolist(),
        }

    def fit_predict_prices(self, prices: np.ndarray, horizon: int = 10) -> dict:
        """Fit on price returns, predict future with uncertainty"""
        ret    = np.diff(prices)/prices[:-1]
        n      = len(ret)
        X      = np.arange(n, dtype=float)
        # Normalize
        mu_r   = ret.mean(); std_r = ret.std()+1e-8
        y      = (ret - mu_r) / std_r
        # Subsample for speed (GP is O(n^3))
        step   = max(1, n//50)
        X_s    = X[::step]; y_s = y[::step]
        self.fit(X_s, y_s)
        # Predict future
        X_fut  = np.arange(n, n+horizon, dtype=float)
        pred   = self.predict(X_fut)
        # Denormalize
        mean_r = np.array(pred["mean"])*std_r + mu_r
        std_r_ = np.array(pred["std"])*std_r
        # Price forecast
        price_forecast = [prices[-1]]
        for r in mean_r:
            price_forecast.append(price_forecast[-1]*(1+r))
        price_upper = [prices[-1]]
        price_lower = [prices[-1]]
        for i,(r,s) in enumerate(zip(mean_r, std_r_)):
            price_upper.append(price_upper[-1]*(1+r+1.96*s))
            price_lower.append(price_lower[-1]*(1+r-1.96*s))
        direction = "UP" if mean_r.mean()>0 else "DOWN"
        return {
            "horizon":        horizon,
            "predicted_rets": [round(float(r)*100,4) for r in mean_r],
            "uncertainty":    [round(float(s)*100,4) for s in std_r_],
            "price_forecast": [round(p,2) for p in price_forecast],
            "price_upper_95": [round(p,2) for p in price_upper],
            "price_lower_95": [round(p,2) for p in price_lower],
            "direction":      direction,
            "avg_uncertainty":round(float(std_r_.mean()*100),4),
            "signal":         "BUY" if direction=="UP" and std_r_.mean()<0.02 else
                              "SELL" if direction=="DOWN" and std_r_.mean()<0.02 else "HOLD",
            "confidence":     round(float(max(0,1-std_r_.mean()*10)),4),
        }
