"""
GARCH(1,1) + EGARCH + GJR-GARCH + Heston Stochastic Volatility
Pure NumPy/SciPy — full MLE estimation
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ── GARCH(1,1) ───────────────────────────────────────────
class GARCH:
    """
    GARCH(1,1): sigma^2_t = omega + alpha*eps^2_{t-1} + beta*sigma^2_{t-1}
    MLE estimation via L-BFGS-B
    Persistence = alpha + beta (should be < 1)
    """
    def __init__(self):
        self.omega = 1e-6
        self.alpha = 0.05
        self.beta  = 0.90
        self.fitted= False

    def _log_likelihood(self, params, returns):
        omega, alpha, beta = params
        if omega<=0 or alpha<=0 or beta<=0 or alpha+beta>=1:
            return 1e10
        T      = len(returns)
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(returns)
        for t in range(1, T):
            sigma2[t] = omega + alpha*returns[t-1]**2 + beta*sigma2[t-1]
            if sigma2[t] <= 0: return 1e10
        ll = -0.5 * np.sum(np.log(2*np.pi*sigma2) + returns**2/sigma2)
        return -ll

    def fit(self, returns: np.ndarray) -> dict:
        ret = returns - returns.mean()
        x0  = [1e-6, 0.05, 0.90]
        bounds = [(1e-8,0.1),(1e-4,0.5),(0.1,0.999)]
        res = minimize(self._log_likelihood, x0, args=(ret,),
                       method='L-BFGS-B', bounds=bounds)
        self.omega, self.alpha, self.beta = res.x
        self.returns = ret
        # Compute fitted variance series
        T = len(ret)
        self.sigma2 = np.zeros(T)
        self.sigma2[0] = np.var(ret)
        for t in range(1,T):
            self.sigma2[t] = self.omega + self.alpha*ret[t-1]**2 + self.beta*self.sigma2[t-1]
        self.fitted = True
        persistence = self.alpha + self.beta
        # Forecast 1-step ahead
        sigma2_next = self.omega + self.alpha*ret[-1]**2 + self.beta*self.sigma2[-1]
        # Long-run variance
        lr_var = self.omega / (1 - persistence + 1e-10)
        return {
            "model":       "GARCH(1,1)",
            "omega":       round(float(self.omega),8),
            "alpha":       round(float(self.alpha),6),
            "beta":        round(float(self.beta),6),
            "persistence": round(float(persistence),6),
            "current_vol": round(float(np.sqrt(self.sigma2[-1]*252))*100,4),
            "forecast_vol":round(float(np.sqrt(sigma2_next*252))*100,4),
            "longrun_vol": round(float(np.sqrt(lr_var*252))*100,4),
            "log_lik":     round(float(-res.fun),4),
            "converged":   bool(res.success),
            "vol_regime":  "HIGH" if np.sqrt(sigma2_next*252)>0.30 else
                           "LOW"  if np.sqrt(sigma2_next*252)<0.15 else "NORMAL",
            "signal":      "SELL" if np.sqrt(sigma2_next*252)>0.35 else "HOLD"
        }

    def forecast(self, h: int = 10) -> list:
        """Multi-step variance forecast: E[sigma^2_{t+h}]"""
        persistence = self.alpha + self.beta
        lr_var      = self.omega / (1-persistence+1e-10)
        sigma2_1    = self.omega + self.alpha*self.returns[-1]**2 + self.beta*self.sigma2[-1]
        forecasts   = [sigma2_1]
        for i in range(1, h):
            forecasts.append(lr_var + persistence**i * (sigma2_1 - lr_var))
        return [round(float(np.sqrt(f*252))*100,4) for f in forecasts]


class EGARCH:
    """
    EGARCH(1,1): log(sigma^2_t) = omega + alpha*|z_{t-1}| + gamma*z_{t-1} + beta*log(sigma^2_{t-1})
    Captures leverage effect: gamma < 0 means negative returns increase vol more
    No positivity constraints needed (log transform)
    """
    def __init__(self):
        self.params = None

    def _log_likelihood(self, params, returns):
        omega, alpha, gamma, beta = params
        if abs(beta) >= 1: return 1e10
        T      = len(returns)
        log_s2 = np.zeros(T)
        log_s2[0] = np.log(np.var(returns)+1e-10)
        for t in range(1, T):
            z          = returns[t-1] / (np.exp(0.5*log_s2[t-1])+1e-10)
            log_s2[t]  = omega + alpha*(abs(z)-np.sqrt(2/np.pi)) + gamma*z + beta*log_s2[t-1]
        sigma2 = np.exp(log_s2)
        ll     = -0.5*np.sum(np.log(2*np.pi*sigma2) + returns**2/sigma2)
        return -ll if np.isfinite(ll) else 1e10

    def fit(self, returns: np.ndarray) -> dict:
        ret = returns - returns.mean()
        x0  = [-0.1, 0.1, -0.05, 0.85]
        res = minimize(self._log_likelihood, x0, args=(ret,), method='Nelder-Mead',
                       options={'maxiter':2000,'xatol':1e-6})
        omega, alpha, gamma, beta = res.x
        self.params = res.x
        # Compute series
        T      = len(ret)
        log_s2 = np.zeros(T)
        log_s2[0] = np.log(np.var(ret)+1e-10)
        for t in range(1,T):
            z         = ret[t-1]/(np.exp(0.5*log_s2[t-1])+1e-10)
            log_s2[t] = omega + alpha*(abs(z)-np.sqrt(2/np.pi)) + gamma*z + beta*log_s2[t-1]
        sigma2 = np.exp(log_s2)
        # Forecast
        z_last  = ret[-1]/(np.sqrt(sigma2[-1])+1e-10)
        ls2_next= omega + alpha*(abs(z_last)-np.sqrt(2/np.pi)) + gamma*z_last + beta*log_s2[-1]
        return {
            "model":         "EGARCH(1,1)",
            "omega":         round(float(omega),6),
            "alpha":         round(float(alpha),6),
            "gamma":         round(float(gamma),6),
            "beta":          round(float(beta),6),
            "leverage_effect": bool(gamma < 0),
            "current_vol":   round(float(np.sqrt(sigma2[-1]*252))*100,4),
            "forecast_vol":  round(float(np.sqrt(np.exp(ls2_next)*252))*100,4),
            "asymmetry":     "Negative returns amplify vol more" if gamma<0 else "Symmetric",
            "signal":        "SELL" if np.sqrt(np.exp(ls2_next)*252)>0.35 else "HOLD"
        }


class GJR_GARCH:
    """
    GJR-GARCH(1,1): sigma^2_t = omega + (alpha + gamma*I_{t-1})*eps^2_{t-1} + beta*sigma^2_{t-1}
    I_{t-1}=1 if eps_{t-1}<0 (bad news has extra impact)
    """
    def fit(self, returns: np.ndarray) -> dict:
        ret = returns - returns.mean()
        def neg_ll(params):
            omega, alpha, gamma, beta = params
            if omega<=0 or alpha<0 or beta<0 or alpha+0.5*gamma+beta>=1: return 1e10
            T = len(ret)
            s2 = np.zeros(T); s2[0] = np.var(ret)
            for t in range(1,T):
                I     = 1.0 if ret[t-1]<0 else 0.0
                s2[t] = omega + (alpha+gamma*I)*ret[t-1]**2 + beta*s2[t-1]
                if s2[t]<=0: return 1e10
            ll = -0.5*np.sum(np.log(2*np.pi*s2)+ret**2/s2)
            return -ll if np.isfinite(ll) else 1e10
        res = minimize(neg_ll,[1e-6,0.03,0.05,0.88],method='Nelder-Mead')
        omega,alpha,gamma,beta = res.x
        T = len(ret); s2=np.zeros(T); s2[0]=np.var(ret)
        for t in range(1,T):
            I=1.0 if ret[t-1]<0 else 0.0
            s2[t]=omega+(alpha+gamma*I)*ret[t-1]**2+beta*s2[t-1]
        I_last = 1.0 if ret[-1]<0 else 0.0
        s2_next = omega+(alpha+gamma*I_last)*ret[-1]**2+beta*s2[-1]
        return {
            "model":       "GJR-GARCH(1,1)",
            "alpha":       round(float(alpha),6),
            "gamma":       round(float(gamma),6),
            "beta":        round(float(beta),6),
            "bad_news_effect": round(float(alpha+gamma),6),
            "good_news_effect":round(float(alpha),6),
            "current_vol": round(float(np.sqrt(s2[-1]*252))*100,4),
            "forecast_vol":round(float(np.sqrt(s2_next*252))*100,4),
            "signal":      "SELL" if np.sqrt(s2_next*252)>0.35 else "HOLD"
        }


class HestonModel:
    """
    Heston Stochastic Volatility Model
    dS = mu*S dt + sqrt(v)*S dW1
    dv = kappa*(theta-v) dt + xi*sqrt(v) dW2
    corr(dW1,dW2) = rho
    Characteristic function for option pricing
    Monte Carlo simulation for path generation
    """
    def __init__(self):
        self.kappa = 2.0    # mean reversion speed
        self.theta = 0.04   # long-run variance
        self.xi    = 0.3    # vol of vol
        self.rho   = -0.7   # leverage correlation
        self.v0    = 0.04   # initial variance

    def calibrate(self, returns: np.ndarray) -> dict:
        """Calibrate Heston params from return series"""
        ret    = returns
        # Method of moments for variance process
        var    = ret**2
        self.v0    = float(np.var(ret))
        self.theta = float(np.var(ret))
        # Estimate kappa from autocorrelation of variance
        acf1   = float(np.corrcoef(var[:-1], var[1:])[0,1])
        self.kappa = max(0.1, -np.log(abs(acf1)+1e-10))
        # Vol of vol from variance of variance
        self.xi    = float(np.std(var) / (np.sqrt(self.theta)+1e-10))
        # Leverage from return-variance correlation
        self.rho   = float(np.corrcoef(ret[1:], var[:-1])[0,1])
        self.rho   = np.clip(self.rho, -0.99, 0.99)
        return self

    def simulate(self, S0: float, T: float = 1.0,
                 n_paths: int = 5000, n_steps: int = 252) -> dict:
        """Euler-Maruyama discretization of Heston SDE"""
        dt   = T / n_steps
        S    = np.full(n_paths, S0, dtype=float)
        v    = np.full(n_paths, self.v0, dtype=float)
        S_paths = [S.copy()]
        v_paths = [v.copy()]
        for _ in range(n_steps):
            Z1 = np.random.standard_normal(n_paths)
            Z2 = np.random.standard_normal(n_paths)
            W1 = Z1
            W2 = self.rho*Z1 + np.sqrt(1-self.rho**2)*Z2
            v_new = np.abs(v + self.kappa*(self.theta-v)*dt +
                          self.xi*np.sqrt(np.maximum(v,0)*dt)*W2)
            S_new = S * np.exp((0-0.5*v)*dt + np.sqrt(np.maximum(v,0)*dt)*W1)
            v, S  = v_new, S_new
            S_paths.append(S.copy())
            v_paths.append(v.copy())
        final    = S_paths[-1]
        final_v  = v_paths[-1]
        ret_dist = (final - S0) / S0
        return {
            "model":         "Heston SV",
            "kappa":         round(float(self.kappa),4),
            "theta":         round(float(self.theta),6),
            "xi":            round(float(self.xi),4),
            "rho":           round(float(self.rho),4),
            "v0":            round(float(self.v0),6),
            "S0":            S0,
            "final_mean":    round(float(final.mean()),2),
            "final_median":  round(float(np.median(final)),2),
            "implied_vol":   round(float(np.sqrt(final_v.mean()*252))*100,4),
            "VaR_95":        round(float(np.percentile(ret_dist,5))*100,4),
            "CVaR_95":       round(float(ret_dist[ret_dist<np.percentile(ret_dist,5)].mean())*100,4),
            "p_up10":        round(float((final>S0*1.1).mean())*100,2),
            "p_down10":      round(float((final<S0*0.9).mean())*100,2),
            "signal":        "BUY" if final.mean()>S0*1.02 else
                             "SELL" if final.mean()<S0*0.98 else "HOLD",
            "confidence":    round(float(min(abs(final.mean()-S0)/S0*20,0.95)),4)
        }
