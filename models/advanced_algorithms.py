"""
Monte Carlo, Fourier, Wavelet, Cointegration, Black-Litterman
Heavy mathematical models — pure NumPy/SciPy
"""
import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import norm
import pywt

# ── Monte Carlo ─────────────────────────────────────────
class MonteCarloEngine:
    """
    Geometric Brownian Motion + Jump Diffusion
    10,000 price paths → VaR, CVaR, probability distributions
    """
    def __init__(self, n_paths: int = 10000, n_steps: int = 252):
        self.n_paths = n_paths
        self.n_steps = n_steps

    def simulate_gbm(self, S0: float, mu: float, sigma: float,
                     T: float = 1.0) -> np.ndarray:
        """Geometric Brownian Motion: dS = μS dt + σS dW"""
        dt      = T / self.n_steps
        Z       = np.random.standard_normal((self.n_paths, self.n_steps))
        log_ret = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
        paths   = S0 * np.exp(np.cumsum(log_ret, axis=1))
        return np.column_stack([np.full(self.n_paths, S0), paths])

    def simulate_jump_diffusion(self, S0: float, mu: float, sigma: float,
                                 lam: float = 0.1, jump_mu: float = -0.05,
                                 jump_sig: float = 0.1) -> np.ndarray:
        """Merton Jump Diffusion: adds Poisson jumps to GBM"""
        dt      = 1.0 / self.n_steps
        Z       = np.random.standard_normal((self.n_paths, self.n_steps))
        J_count = np.random.poisson(lam*dt, (self.n_paths, self.n_steps))
        J_size  = np.random.normal(jump_mu, jump_sig, (self.n_paths, self.n_steps))
        log_ret = (mu - 0.5*sigma**2 - lam*jump_mu)*dt + sigma*np.sqrt(dt)*Z + J_count*J_size
        paths   = S0 * np.exp(np.cumsum(log_ret, axis=1))
        return np.column_stack([np.full(self.n_paths, S0), paths])

    def analyze(self, S0: float, returns: np.ndarray, T_days: int = 30) -> dict:
        mu    = returns.mean() * 252
        sigma = returns.std()  * np.sqrt(252)
        T     = T_days / 252

        # Both GBM and Jump Diffusion
        gbm_paths  = self.simulate_gbm(S0, mu, sigma, T)
        jd_paths   = self.simulate_jump_diffusion(S0, mu, sigma)

        final_gbm  = gbm_paths[:, -1]
        final_jd   = jd_paths[:, min(T_days, self.n_steps-1)]

        # Risk metrics
        ret_gbm    = (final_gbm - S0) / S0
        confidence = [0.95, 0.99]
        var_95     = float(np.percentile(ret_gbm, 5))
        var_99     = float(np.percentile(ret_gbm, 1))
        cvar_95    = float(ret_gbm[ret_gbm <= var_95].mean())

        # Probability analysis
        p_up10     = float((final_gbm > S0*1.10).mean())
        p_down10   = float((final_gbm < S0*0.90).mean())
        p_up5      = float((final_gbm > S0*1.05).mean())

        return {
            "S0":            S0,
            "mu_annual":     round(mu*100,2),
            "sigma_annual":  round(sigma*100,2),
            "n_paths":       self.n_paths,
            "horizon_days":  T_days,
            "gbm": {
                "median":    round(float(np.median(final_gbm)),2),
                "mean":      round(float(final_gbm.mean()),2),
                "p5":        round(float(np.percentile(final_gbm,5)),2),
                "p95":       round(float(np.percentile(final_gbm,95)),2),
                "p25":       round(float(np.percentile(final_gbm,25)),2),
                "p75":       round(float(np.percentile(final_gbm,75)),2),
            },
            "risk": {
                "VaR_95":    round(var_95*100,2),
                "VaR_99":    round(var_99*100,2),
                "CVaR_95":   round(cvar_95*100,2),
                "p_up_10pct":  round(p_up10*100,1),
                "p_down_10pct":round(p_down10*100,1),
                "p_up_5pct":   round(p_up5*100,1),
            },
            "signal": "BUY" if p_up10 > 0.35 else "SELL" if p_down10 > 0.35 else "HOLD"
        }

# ── Fourier Transform ────────────────────────────────────
class FourierAnalyzer:
    """
    FFT-based cycle detection in price series
    Identifies dominant cycles (weekly, monthly, quarterly)
    """
    def analyze(self, prices: np.ndarray) -> dict:
        n     = len(prices)
        # Detrend
        trend = np.polyfit(np.arange(n), prices, 1)
        detrended = prices - np.polyval(trend, np.arange(n))
        # Hanning window to reduce spectral leakage
        windowed  = detrended * np.hanning(n)
        # FFT
        fft_vals  = np.fft.rfft(windowed)
        freqs     = np.fft.rfftfreq(n, d=1.0)  # daily
        power     = np.abs(fft_vals)**2
        # Find dominant cycles (skip DC component)
        top_idx   = np.argsort(power[1:])[-5:][::-1] + 1
        cycles    = []
        for idx in top_idx:
            period = round(1.0/freqs[idx], 1) if freqs[idx] > 0 else 0
            cycles.append({
                "period_days": period,
                "power":       round(float(power[idx]),2),
                "phase_deg":   round(float(np.angle(fft_vals[idx], deg=True)),1)
            })

        # Reconstruct dominant signal
        n_components = 5
        recon_fft    = np.zeros_like(fft_vals)
        for idx in top_idx[:n_components]:
            recon_fft[idx] = fft_vals[idx]
        reconstructed = np.fft.irfft(recon_fft, n=n)

        # Current phase of dominant cycle
        dom_idx   = top_idx[0]
        dom_phase = np.angle(fft_vals[dom_idx])
        phase_pct = (dom_phase + np.pi) / (2*np.pi)

        return {
            "dominant_cycles":  cycles,
            "dominant_period":  cycles[0]["period_days"] if cycles else 0,
            "phase_position":   round(phase_pct, 4),
            "cycle_signal":     "BUY" if phase_pct < 0.25 else "SELL" if phase_pct > 0.75 else "HOLD",
            "spectral_entropy": round(float(-(power/power.sum()+1e-300) @ np.log(power/power.sum()+1e-300)),4),
            "reconstructed":    reconstructed.tolist()[-10:]
        }

# ── Wavelet Transform ────────────────────────────────────
class WaveletAnalyzer:
    """
    Discrete Wavelet Transform for multi-scale decomposition
    Separates trend, cycle, and noise components
    """
    def analyze(self, prices: np.ndarray, wavelet: str = "db4", level: int = 4) -> dict:
        ret  = np.diff(prices) / prices[:-1]
        # DWT decomposition
        coeffs = pywt.wavedec(ret, wavelet, level=level)
        # Reconstruct components
        components = {}
        for i in range(len(coeffs)):
            c  = [np.zeros_like(c) for c in coeffs]
            c[i] = coeffs[i]
            rec = pywt.waverec(c, wavelet)[:len(ret)]
            name = f"level_{i}" if i > 0 else "approx"
            components[name] = {
                "energy": round(float((rec**2).sum()),8),
                "std":    round(float(rec.std()),8),
                "last":   round(float(rec[-1]),8) if len(rec)>0 else 0
            }

        # Denoise: threshold detail coefficients
        threshold = np.sqrt(2*np.log(len(ret))) * np.median(np.abs(coeffs[-1]))/0.6745
        denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, "soft") for c in coeffs[1:]]
        denoised = pywt.waverec(denoised_coeffs, wavelet)[:len(ret)]

        # Signal from denoised trend
        trend_dir = "UP" if denoised[-1] > denoised[-5] else "DOWN"
        return {
            "wavelet":        wavelet,
            "levels":         level,
            "components":     components,
            "threshold":      round(float(threshold),8),
            "trend_direction":trend_dir,
            "denoised_last5": [round(float(x),6) for x in denoised[-5:]],
            "signal":         "BUY" if trend_dir=="UP" and denoised[-1]>0 else
                              "SELL" if trend_dir=="DOWN" and denoised[-1]<0 else "HOLD",
            "noise_ratio":    round(float(components.get("level_1",{}).get("energy",0) /
                              (sum(c["energy"] for c in components.values())+1e-300)),4)
        }

# ── Cointegration + ECM ──────────────────────────────────
class CointegrationECM:
    """
    Engle-Granger cointegration + Error Correction Model
    Find pairs, estimate spread, generate mean-reversion signals
    """
    def test_cointegration(self, y1: np.ndarray, y2: np.ndarray) -> dict:
        from scipy.stats import pearsonr
        # OLS regression: y1 = beta*y2 + alpha + epsilon
        X    = np.column_stack([np.ones(len(y2)), y2])
        beta = np.linalg.lstsq(X, y1, rcond=None)[0]
        spread = y1 - beta[0] - beta[1]*y2
        # ADF test (simplified)
        d_spread  = np.diff(spread)
        spread_lag= spread[:-1]
        X_adf     = np.column_stack([np.ones(len(d_spread)), spread_lag])
        coef      = np.linalg.lstsq(X_adf, d_spread, rcond=None)[0]
        residuals = d_spread - X_adf @ coef
        rho       = coef[1]
        se        = np.sqrt((residuals**2).sum()/(len(d_spread)-2) /
                    ((spread_lag**2).sum()-(spread_lag.sum())**2/len(spread_lag)))
        adf_stat  = rho / (se+1e-10)
        # Critical values (approximate)
        is_cointegrated = adf_stat < -3.34  # 5% level
        corr, _   = pearsonr(y1, y2)
        return {
            "beta":             round(float(beta[1]),4),
            "alpha":            round(float(beta[0]),4),
            "adf_statistic":    round(float(adf_stat),4),
            "cointegrated":     bool(is_cointegrated),
            "correlation":      round(float(corr),4),
            "spread_mean":      round(float(spread.mean()),4),
            "spread_std":       round(float(spread.std()),4),
            "half_life":        self._half_life(spread),
        }

    def _half_life(self, spread: np.ndarray) -> float:
        d  = np.diff(spread)
        X  = np.column_stack([np.ones(len(d)), spread[:-1]])
        b  = np.linalg.lstsq(X, d, rcond=None)[0]
        hl = -np.log(2) / b[1] if b[1] < 0 else float('inf')
        return round(float(hl),2)

    def ecm_signal(self, y1: np.ndarray, y2: np.ndarray) -> dict:
        """Error Correction Model signal generation"""
        coint = self.test_cointegration(y1, y2)
        spread = y1 - coint["alpha"] - coint["beta"]*y2
        zscore = (spread[-1] - coint["spread_mean"]) / (coint["spread_std"]+1e-8)
        # ECM: d_spread = gamma*(spread-mean) + noise
        ecm_speed = (spread[-1] - spread[-2]) / (coint["spread_mean"] - spread[-2]+1e-8)
        signal = ("SELL_Y1_BUY_Y2" if zscore >  2.0 else
                  "BUY_Y1_SELL_Y2" if zscore < -2.0 else
                  "CLOSE" if abs(zscore) < 0.5 else "HOLD")
        return {
            **coint,
            "current_spread":   round(float(spread[-1]),4),
            "zscore":           round(float(zscore),4),
            "ecm_speed":        round(float(ecm_speed),4),
            "signal":           signal,
            "confidence":       round(min(abs(float(zscore))/3, 1.0),4),
        }

# ── Black-Litterman ──────────────────────────────────────
class BlackLittermanOptimizer:
    """
    Black-Litterman model: combines market equilibrium with investor views
    Produces optimal portfolio weights with uncertainty
    """
    def __init__(self, tau: float = 0.05, risk_aversion: float = 3.0):
        self.tau   = tau
        self.delta = risk_aversion

    def optimize(self, returns: np.ndarray, tickers: list,
                 views: dict = None) -> dict:
        """
        returns: T x N matrix of asset returns
        views: {ticker: expected_return} investor views
        """
        T, N = returns.shape
        # Sample moments
        mu_hist  = returns.mean(axis=0)
        Sigma    = returns.T @ returns / T  # covariance
        # Market cap weights (equal if not provided)
        w_mkt    = np.ones(N) / N
        # Equilibrium returns: Pi = delta * Sigma * w_mkt
        Pi       = self.delta * Sigma @ w_mkt
        # Prior: mu ~ N(Pi, tau*Sigma)
        if not views:
            mu_bl = Pi
            Sigma_bl = (1 + self.tau) * Sigma
        else:
            # Build view matrix P and view vector Q
            view_tickers = [t for t in tickers if t in views]
            k  = len(view_tickers)
            P  = np.zeros((k, N))
            Q  = np.zeros(k)
            for i, t in enumerate(view_tickers):
                idx    = tickers.index(t)
                P[i,idx]= 1.0
                Q[i]   = views[t]
            # Omega: view uncertainty (proportional to tau*P*Sigma*P')
            Omega = np.diag(np.diag(self.tau * P @ Sigma @ P.T))
            # BL posterior
            M1   = np.linalg.inv(self.tau * Sigma)
            M2   = P.T @ np.linalg.inv(Omega) @ P
            M3   = M1 @ Pi + P.T @ np.linalg.inv(Omega) @ Q
            Sigma_bl_inv = M1 + M2
            Sigma_bl = np.linalg.inv(Sigma_bl_inv)
            mu_bl    = Sigma_bl @ M3

        # Optimal weights: w* = (delta*Sigma)^{-1} * mu_bl
        w_opt = np.linalg.solve(self.delta * Sigma, mu_bl)
        # Normalize to long-only
        w_opt = np.maximum(w_opt, 0)
        w_opt = w_opt / (w_opt.sum() + 1e-8)
        # Portfolio metrics
        port_ret = float(w_opt @ mu_bl) * 252
        port_vol = float(np.sqrt(w_opt @ Sigma @ w_opt)) * np.sqrt(252)
        sharpe   = port_ret / (port_vol + 1e-8)
        return {
            "weights":        {t: round(float(w),4) for t,w in zip(tickers,w_opt)},
            "equilibrium_ret":{t: round(float(p)*252*100,2) for t,p in zip(tickers,Pi)},
            "bl_expected_ret":{t: round(float(m)*252*100,2) for t,m in zip(tickers,mu_bl)},
            "portfolio_return":round(port_ret*100,2),
            "portfolio_vol":   round(port_vol*100,2),
            "sharpe_ratio":    round(sharpe,4),
            "diversification": round(float(1/(w_opt**2).sum()),2),
        }
