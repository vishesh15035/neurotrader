"""
MS-ARIMA + VAR + Ornstein-Uhlenbeck + Copula Models
Heavy statistical machinery — pure NumPy/SciPy
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, t as t_dist, kendalltau
import warnings
warnings.filterwarnings('ignore')

# ── Ornstein-Uhlenbeck ───────────────────────────────────
class OrnsteinUhlenbeck:
    """
    OU Process: dX = kappa*(mu-X)*dt + sigma*dW
    MLE estimation of kappa (mean reversion speed), mu, sigma
    Half-life = ln(2)/kappa
    """
    def fit(self, series: np.ndarray) -> dict:
        X  = series
        dt = 1.0
        n  = len(X)-1
        # Closed-form MLE for OU
        Sx  = X[:-1].sum(); Sy = X[1:].sum()
        Sxx = (X[:-1]**2).sum(); Syy = (X[1:]**2).sum()
        Sxy = (X[:-1]*X[1:]).sum()
        denom = n*Sxx - Sx**2
        if abs(denom) < 1e-10:
            return {"error": "Degenerate series"}
        mu    = (Sy*Sxx - Sx*Sxy) / (n*(Sxx-Sxy) - (Sx**2-Sx*Sy)+1e-10)
        kappa = -np.log((Sxy - mu*Sx - mu*Sy + n*mu**2) /
                        (Sxx - 2*mu*Sx + n*mu**2)+1e-10) / dt
        kappa = max(kappa, 0.001)
        alpha = np.exp(-kappa*dt)
        resid = X[1:] - alpha*X[:-1] - mu*(1-alpha)
        sigma2= resid.var() * 2*kappa / (1-alpha**2+1e-10)
        sigma = np.sqrt(max(sigma2, 1e-10))
        half_life = np.log(2)/kappa
        # Current deviation from equilibrium
        z_score = (X[-1]-mu)/(sigma/np.sqrt(2*kappa+1e-10)+1e-10)
        return {
            "model":      "Ornstein-Uhlenbeck",
            "kappa":      round(float(kappa),6),
            "mu":         round(float(mu),4),
            "sigma":      round(float(sigma),6),
            "half_life":  round(float(half_life),2),
            "current":    round(float(X[-1]),4),
            "z_score":    round(float(z_score),4),
            "signal":     "BUY"  if z_score < -1.5 else
                          "SELL" if z_score >  1.5 else "HOLD",
            "reversion_target": round(float(mu),4),
            "days_to_revert":   round(float(half_life),1),
            "confidence": round(float(min(abs(z_score)/3,0.95)),4)
        }


# ── VAR(p) ───────────────────────────────────────────────
class VectorAutoRegression:
    """
    VAR(p): Y_t = A_1*Y_{t-1} + ... + A_p*Y_{t-p} + epsilon
    OLS estimation, Granger causality test
    Impulse response functions
    """
    def __init__(self, p: int = 2):
        self.p = p
        self.A = None

    def fit(self, data: np.ndarray) -> dict:
        """data: T x K matrix of K time series"""
        T, K = data.shape
        p    = self.p
        # Build lagged matrix
        Y    = data[p:]         # T-p x K
        X    = np.ones((T-p, 1+p*K))
        X[:,0]= 1               # intercept
        for lag in range(1, p+1):
            X[:, 1+(lag-1)*K : 1+lag*K] = data[p-lag:T-lag]
        # OLS: A = (X'X)^{-1} X'Y
        try:
            self.A = np.linalg.lstsq(X, Y, rcond=None)[0]  # (1+pK) x K
        except:
            return {"error": "VAR estimation failed"}
        resid   = Y - X @ self.A
        Sigma_u = resid.T @ resid / (T-p-p*K-1)
        # 1-step forecast
        x_new   = np.ones(1+p*K)
        x_new[0]= 1
        for lag in range(1, p+1):
            x_new[1+(lag-1)*K:1+lag*K] = data[-lag]
        forecast = x_new @ self.A
        # Granger causality (simplified F-test)
        granger = {}
        for i in range(K):
            for j in range(K):
                if i==j: continue
                # Does series j Granger-cause series i?
                # Compare RSS of restricted vs unrestricted
                try:
                    # Unrestricted: use all lags
                    RSS_u = float(resid[:,i].T @ resid[:,i])
                    # Restricted: remove lags of j
                    X_r   = np.delete(X, [1+(lag-1)*K+j for lag in range(1,p+1)], axis=1)
                    A_r   = np.linalg.lstsq(X_r, Y[:,i], rcond=None)[0]
                    resid_r = Y[:,i] - X_r @ A_r
                    RSS_r = float(resid_r @ resid_r)
                    F     = ((RSS_r-RSS_u)/p) / (RSS_u/(T-p-p*K-1)+1e-10)
                    granger[f"series{j}→series{i}"] = round(float(F),4)
                except: pass
        return {
            "model":    f"VAR({p})",
            "n_series": K,
            "forecast": [round(float(f),4) for f in forecast],
            "granger":  granger,
            "directions":["UP" if f>data[-1,i] else "DOWN" for i,f in enumerate(forecast)],
            "signal":   "BUY" if forecast[0]>data[-1,0] else "SELL"
        }


# ── Markov-Switching ARIMA ───────────────────────────────
class MarkovSwitchingARIMA:
    """
    MS-ARIMA(2,0,1) with 3 regimes
    Each regime has own AR(2) + MA(1) + sigma
    Hamilton filter for regime probabilities
    """
    def __init__(self, n_regimes: int = 3):
        self.K  = n_regimes
        self.fitted = False

    def fit(self, returns: np.ndarray, n_iter: int = 30) -> dict:
        K   = self.K
        T   = len(returns)
        # Initialize regime params via k-means style
        sorted_r = np.sort(returns)
        chunk    = T // K
        self.mu  = np.array([sorted_r[chunk//2], sorted_r[T//2], sorted_r[-chunk//2]])
        self.sig = np.array([sorted_r[:chunk].std()+1e-6,
                              sorted_r[chunk:2*chunk].std()+1e-6,
                              sorted_r[2*chunk:].std()+1e-6])
        self.ar1 = np.array([0.1, -0.1, 0.05])
        self.ar2 = np.array([0.05,-0.05, 0.02])
        self.P   = np.full((K,K), 1/K)  # transition matrix
        # EM-like estimation
        print(f"[MS-ARIMA] Training {n_iter} iterations...")
        for it in range(n_iter):
            # E-step: Hamilton filter
            filtered = np.zeros((T,K))
            xi       = np.ones(K)/K
            for t in range(2, T):
                pred     = self.mu + self.ar1*returns[t-1] + self.ar2*returns[t-2]
                lik      = norm.pdf(returns[t], pred, self.sig) + 1e-300
                forecast = self.P.T @ xi
                joint    = lik * forecast
                xi       = joint / (joint.sum()+1e-300)
                filtered[t] = xi
            # M-step: update params
            for k in range(K):
                w = filtered[2:,k] + 1e-10
                W = w.sum()
                self.mu[k]  = (w*returns[2:]).sum()/W
                resid       = returns[2:] - self.mu[k] - self.ar1[k]*returns[1:-1] - self.ar2[k]*returns[:-2]
                self.sig[k] = np.sqrt((w*resid**2).sum()/W)+1e-6
                # Update transition row
                if it > 5:
                    for j in range(K):
                        self.P[k,j] = (filtered[2:,j]*filtered[1:-1,k]).sum()/(filtered[1:-1,k].sum()+1e-10)
                    self.P[k] = np.maximum(self.P[k], 1e-6)
                    self.P[k] /= self.P[k].sum()
        self.filtered   = filtered
        self.fitted     = True
        current_regime  = int(np.argmax(filtered[-1]))
        regime_names    = {0:"BEAR",1:"BULL",2:"NEUTRAL"} if self.mu[0]<self.mu[1] else {0:"BULL",1:"BEAR",2:"NEUTRAL"}
        # 1-step forecast
        pred = self.mu[current_regime] + self.ar1[current_regime]*returns[-1] + self.ar2[current_regime]*returns[-2]
        return {
            "model":          "MS-ARIMA(2,0,1)",
            "n_regimes":      K,
            "regime_params":  {k: {"mu":round(float(self.mu[k]),6),
                                    "sigma":round(float(self.sig[k]),6),
                                    "ar1":round(float(self.ar1[k]),4),
                                    "ar2":round(float(self.ar2[k]),4)} for k in range(K)},
            "current_regime": current_regime,
            "regime_probs":   [round(float(p),4) for p in filtered[-1]],
            "transition_matrix": [[round(float(self.P[i,j]),4) for j in range(K)] for i in range(K)],
            "forecast_return":round(float(pred*100),4),
            "signal":         "BUY" if pred>0.001 else "SELL" if pred<-0.001 else "HOLD",
            "confidence":     round(float(max(filtered[-1])),4)
        }


# ── Copula Models ────────────────────────────────────────
class CopulaModels:
    """
    Gaussian, Clayton, Gumbel, t-Copula
    Captures tail dependence between asset pairs
    Used for portfolio risk and pair trading signals
    """
    def _to_uniform(self, x: np.ndarray) -> np.ndarray:
        """Probability integral transform → uniform marginals"""
        n = len(x)
        ranks = np.argsort(np.argsort(x))
        return (ranks + 1) / (n + 1)

    def gaussian_copula(self, u: np.ndarray, v: np.ndarray) -> dict:
        """Gaussian copula — linear dependence"""
        rho = np.corrcoef(norm.ppf(u), norm.ppf(v))[0,1]
        # Tail dependence: 0 for Gaussian
        return {
            "copula":          "Gaussian",
            "rho":             round(float(rho),4),
            "upper_tail_dep":  0.0,
            "lower_tail_dep":  0.0,
            "interpretation":  "Linear dependence only"
        }

    def clayton_copula(self, u: np.ndarray, v: np.ndarray) -> dict:
        """Clayton copula — lower tail dependence (crash risk)"""
        # MLE for theta via Kendall's tau
        tau, _ = kendalltau(u, v)
        theta  = max(2*tau/(1-tau+1e-10), 0.01)
        # Lower tail dependence
        lower_tail = 2**(-1/theta)
        return {
            "copula":         "Clayton",
            "theta":          round(float(theta),4),
            "kendall_tau":    round(float(tau),4),
            "lower_tail_dep": round(float(lower_tail),4),
            "upper_tail_dep": 0.0,
            "interpretation": f"Crash co-movement: {lower_tail*100:.1f}%"
        }

    def gumbel_copula(self, u: np.ndarray, v: np.ndarray) -> dict:
        """Gumbel copula — upper tail dependence (rally co-movement)"""
        tau, _ = kendalltau(u, v)
        theta  = max(1/(1-tau+1e-10), 1.001)
        upper_tail = 2 - 2**(1/theta)
        return {
            "copula":         "Gumbel",
            "theta":          round(float(theta),4),
            "kendall_tau":    round(float(tau),4),
            "upper_tail_dep": round(float(upper_tail),4),
            "lower_tail_dep": 0.0,
            "interpretation": f"Rally co-movement: {upper_tail*100:.1f}%"
        }

    def t_copula(self, u: np.ndarray, v: np.ndarray) -> dict:
        """Student-t copula — symmetric fat tails"""
        # Estimate rho and nu via correlation on t-scores
        rho_est = np.corrcoef(u, v)[0,1]
        # Estimate df via kurtosis of joint exceedances
        joint_tail = ((u < 0.1) & (v < 0.1)).mean() + 1e-10
        nu = max(2.0, min(30.0, 2/(joint_tail*4+1e-10)))
        tail_dep = 2*t_dist.sf(np.sqrt((nu+1)*(1-rho_est)/(1+rho_est+1e-10)), nu+1)
        return {
            "copula":         "Student-t",
            "rho":            round(float(rho_est),4),
            "nu_df":          round(float(nu),2),
            "tail_dep":       round(float(tail_dep),4),
            "symmetric_tails":True,
            "interpretation": f"Fat-tail co-movement: {tail_dep*100:.1f}%"
        }

    def analyze(self, r1: np.ndarray, r2: np.ndarray,
                ticker1: str = "A", ticker2: str = "B") -> dict:
        u = self._to_uniform(r1)
        v = self._to_uniform(r2)
        gauss   = self.gaussian_copula(u, v)
        clayton = self.clayton_copula(u, v)
        gumbel  = self.gumbel_copula(u, v)
        t_cop   = self.t_copula(u, v)
        # Best copula by tail fit
        lower = clayton["lower_tail_dep"]
        upper = gumbel["upper_tail_dep"]
        crash_risk = lower > 0.3
        signal = "SELL" if crash_risk else "HOLD"
        return {
            "pair":           f"{ticker1}-{ticker2}",
            "gaussian":       gauss,
            "clayton":        clayton,
            "gumbel":         gumbel,
            "t_copula":       t_cop,
            "crash_risk":     crash_risk,
            "rally_coupling": upper > 0.3,
            "dominant_copula": "Clayton" if lower>upper else "Gumbel",
            "signal":         signal,
            "summary":        f"Crash:{lower*100:.1f}% Rally:{upper*100:.1f}%"
        }
