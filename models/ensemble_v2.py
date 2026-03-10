"""
Advanced Ensemble:
- Walk-Forward Cross-Validation
- Bayesian Model Averaging (BMA)
- Stacking Meta-Learner
- Sharpe-Weighted Dynamic Signal Weighting
"""
import numpy as np
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')

class WalkForwardCV:
    """
    Walk-Forward Cross-Validation
    Expanding window: train on [0..t], test on [t..t+step]
    No look-ahead bias — correct for financial time series
    """
    def __init__(self, min_train: int = 120, step: int = 20):
        self.min_train = min_train
        self.step      = step

    def evaluate(self, returns: np.ndarray, model_fn,
                 model_name: str = "model") -> dict:
        """
        model_fn(train_returns) -> predicted_return (float)
        Returns OOS metrics: hit rate, Sharpe, MAE
        """
        T          = len(returns)
        oos_preds  = []
        oos_actual = []
        windows    = []
        t = self.min_train
        while t + self.step <= T:
            train  = returns[:t]
            actual = returns[t:t+self.step].mean()
            try:
                pred = model_fn(train)
            except:
                pred = 0.0
            oos_preds.append(pred)
            oos_actual.append(actual)
            windows.append(t)
            t += self.step
        if not oos_preds:
            return {"error": "insufficient data"}
        preds  = np.array(oos_preds)
        actual = np.array(oos_actual)
        # Hit rate: did we predict direction correctly?
        hit_rate = float((np.sign(preds) == np.sign(actual)).mean())
        # IC: information coefficient (rank correlation)
        from scipy.stats import spearmanr
        ic, _ = spearmanr(preds, actual)
        # Strategy returns: go long if pred>0
        strat_ret  = np.sign(preds) * actual
        sharpe     = float(strat_ret.mean()/(strat_ret.std()+1e-10)*np.sqrt(252/self.step))
        mae        = float(np.abs(preds-actual).mean())
        return {
            "model":        model_name,
            "n_windows":    len(windows),
            "hit_rate":     round(hit_rate,4),
            "IC":           round(float(ic),4),
            "oos_sharpe":   round(sharpe,4),
            "mae":          round(mae,6),
            "skill":        hit_rate > 0.52,
        }


class BayesianModelAveraging:
    """
    BMA: weighted average of model predictions
    Weights = P(M_k | data) ∝ P(data | M_k) * P(M_k)
    Approximated via BIC: log P(data|M_k) ≈ -0.5*BIC_k
    """
    def __init__(self):
        self.weights = None
        self.model_names = None

    def compute_weights(self, model_predictions: dict,
                        actual_returns: np.ndarray) -> dict:
        """
        model_predictions: {name: array of OOS predictions}
        Returns posterior weights for each model
        """
        names = list(model_predictions.keys())
        K     = len(names)
        T     = len(actual_returns)
        log_liks = []
        for name in names:
            preds  = np.array(model_predictions[name])
            n      = min(len(preds), T)
            resid  = actual_returns[-n:] - preds[-n:]
            sigma2 = resid.var() + 1e-10
            ll     = -0.5*n*(np.log(2*np.pi*sigma2) + 1)
            # BIC penalty: -0.5 * k * log(n) where k=1 param
            bic    = ll - 0.5*np.log(n)
            log_liks.append(bic)
        log_liks = np.array(log_liks)
        # Normalize via log-sum-exp for numerical stability
        log_liks -= log_liks.max()
        weights  = np.exp(log_liks)
        weights /= weights.sum() + 1e-10
        self.weights     = weights
        self.model_names = names
        return {name: round(float(w),4) for name,w in zip(names,weights)}

    def predict(self, current_predictions: dict) -> dict:
        """Weighted average of current model predictions"""
        if self.weights is None:
            w = np.ones(len(current_predictions))/len(current_predictions)
            names = list(current_predictions.keys())
        else:
            names = self.model_names
            w     = self.weights
        total = 0.0
        for i,name in enumerate(names):
            if name in current_predictions:
                total += w[i] * current_predictions[name]
        return {
            "bma_prediction": round(float(total),6),
            "signal":         "BUY" if total>0.001 else "SELL" if total<-0.001 else "HOLD",
            "weights":        {n:round(float(ww),4) for n,ww in zip(names,w)},
            "confidence":     round(float(min(abs(total)*100,0.95)),4)
        }


class StackingMetaLearner:
    """
    Stacking: train ridge regression on OOS model predictions
    Meta-learner learns optimal combination weights
    Corrects for model correlation and overfitting
    """
    def __init__(self, alpha: float = 0.01):
        self.alpha  = alpha  # ridge penalty
        self.weights= None
        self.bias   = 0.0

    def fit(self, model_preds: np.ndarray, actual: np.ndarray) -> dict:
        """
        model_preds: T x K matrix of K model predictions
        actual: T-vector of actual returns
        Ridge regression: min ||y - X*w||^2 + alpha*||w||^2
        Closed form: w = (X'X + alpha*I)^{-1} X'y
        """
        T, K  = model_preds.shape
        X     = np.column_stack([np.ones(T), model_preds])
        A     = X.T @ X + self.alpha * np.eye(K+1)
        b     = X.T @ actual
        try:
            w = np.linalg.solve(A, b)
        except:
            w = np.zeros(K+1)
        self.bias    = w[0]
        self.weights = w[1:]
        preds    = X @ w
        resid    = actual - preds
        r2       = 1 - resid.var()/(actual.var()+1e-10)
        return {
            "model_weights": [round(float(ww),4) for ww in self.weights],
            "bias":          round(float(self.bias),6),
            "R2":            round(float(r2),4),
            "alpha_ridge":   self.alpha,
        }

    def predict(self, model_preds: np.ndarray) -> float:
        if self.weights is None: return 0.0
        return float(self.bias + self.weights @ model_preds)


class SharpeWeightedEnsemble:
    """
    Dynamic weights based on rolling Sharpe ratio
    Models with better recent Sharpe get more weight
    Rebalances every window_size periods
    """
    def __init__(self, window: int = 60, min_sharpe: float = -2.0):
        self.window    = window
        self.min_sharpe= min_sharpe

    def compute_weights(self, model_returns: dict) -> dict:
        """
        model_returns: {model_name: array of strategy returns}
        Returns Sharpe-proportional weights
        """
        sharpes = {}
        for name, rets in model_returns.items():
            r = np.array(rets[-self.window:])
            if len(r) < 10:
                sharpes[name] = 0.0
                continue
            s = r.mean()/(r.std()+1e-10) * np.sqrt(252)
            sharpes[name] = max(float(s), self.min_sharpe)
        # Softmax of Sharpe ratios → weights
        names  = list(sharpes.keys())
        vals   = np.array([sharpes[n] for n in names])
        # Shift so min is 0 before softmax
        vals   = vals - vals.min()
        weights= softmax(vals)
        return {
            "sharpes":  {n:round(float(sharpes[n]),4) for n in names},
            "weights":  {n:round(float(w),4) for n,w in zip(names,weights)},
            "best_model": names[int(np.argmax(weights))],
            "worst_model":names[int(np.argmin(weights))],
        }

    def ensemble_signal(self, model_signals: dict,
                        model_returns: dict) -> dict:
        """Combine signals weighted by rolling Sharpe"""
        weight_res = self.compute_weights(model_returns)
        weights    = weight_res["weights"]
        SCORE      = {"BUY":1,"STRONG BUY":2,"HOLD":0,"SELL":-1}
        total = 0.0
        for name, sig in model_signals.items():
            w     = weights.get(name, 1/len(model_signals))
            total += SCORE.get(sig, 0) * w
        signal = "BUY" if total>0.3 else "SELL" if total<-0.3 else "HOLD"
        return {
            "ensemble_score": round(float(total),4),
            "signal":         signal,
            "weights":        weights,
            "sharpes":        weight_res["sharpes"],
            "best_model":     weight_res["best_model"],
            "confidence":     round(float(min(abs(total),1.0)),4)
        }
