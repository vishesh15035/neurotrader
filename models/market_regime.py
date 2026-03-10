"""
Hidden Markov Model — Market Regime Detection
3 regimes: Bull (high return, low vol), Bear (neg return, high vol), Sideways
Baum-Welch EM algorithm for parameter estimation
Viterbi for most likely state sequence
"""
import numpy as np
from scipy.stats import norm

class HiddenMarkovModel:
    """
    Gaussian HMM with Baum-Welch training
    States: 0=Bull, 1=Bear, 2=Sideways
    Observations: daily returns
    """
    REGIME_NAMES = {0: "BULL", 1: "BEAR", 2: "SIDEWAYS"}
    REGIME_COLORS= {0: "🟢", 1: "🔴", 2: "🟡"}

    def __init__(self, n_states: int = 3):
        self.K   = n_states
        # Transition matrix A[i,j] = P(s_t=j | s_{t-1}=i)
        self.A   = np.full((n_states,n_states), 1/n_states)
        # Initial state distribution
        self.pi  = np.full(n_states, 1/n_states)
        # Emission: Gaussian params per state
        self.mu  = np.array([ 0.001, -0.002,  0.0])
        self.sig = np.array([ 0.008,  0.020, 0.005])
        self.fitted = False

    def _emission(self, obs: np.ndarray) -> np.ndarray:
        """B[t,k] = P(o_t | state=k)"""
        T   = len(obs)
        B   = np.zeros((T, self.K))
        for k in range(self.K):
            B[:,k] = norm.pdf(obs, self.mu[k], self.sig[k])
        return np.clip(B, 1e-300, None)

    def _forward(self, obs: np.ndarray, B: np.ndarray):
        """Forward algorithm — alpha[t,k] = P(o_1...o_t, s_t=k)"""
        T     = len(obs)
        alpha = np.zeros((T, self.K))
        alpha[0] = self.pi * B[0]
        alpha[0] /= alpha[0].sum() + 1e-300
        scales   = [alpha[0].sum()]
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ self.A) * B[t]
            s         = alpha[t].sum() + 1e-300
            alpha[t] /= s
            scales.append(s)
        return alpha, np.array(scales)

    def _backward(self, obs: np.ndarray, B: np.ndarray, scales: np.ndarray):
        """Backward algorithm — beta[t,k] = P(o_{t+1}...o_T | s_t=k)"""
        T    = len(obs)
        beta = np.zeros((T, self.K))
        beta[-1] = 1.0
        for t in range(T-2, -1, -1):
            beta[t]  = self.A @ (B[t+1] * beta[t+1])
            beta[t] /= scales[t+1] + 1e-300
        return beta

    def fit(self, returns: np.ndarray, n_iter: int = 50) -> dict:
        """Baum-Welch EM algorithm"""
        # Initialize with k-means style
        sorted_r = np.sort(returns)
        n        = len(sorted_r)
        self.mu  = np.array([sorted_r[int(n*0.75)],
                              sorted_r[int(n*0.25)],
                              sorted_r[int(n*0.50)]])
        self.sig = np.array([returns.std()*0.5, returns.std()*1.5, returns.std()*0.3])

        print(f"[HMM] Baum-Welch training {n_iter} iterations on {len(returns)} observations...")
        log_liks = []

        for iteration in range(n_iter):
            B     = self._emission(returns)
            alpha, scales = self._forward(returns, B)
            beta  = self._backward(returns, B, scales)
            T     = len(returns)

            # E-step: compute gamma and xi
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            xi = np.zeros((T-1, self.K, self.K))
            for t in range(T-1):
                xi[t] = alpha[t:t+1].T * self.A * B[t+1] * beta[t+1]
                xi[t] /= xi[t].sum() + 1e-300

            # M-step: update parameters
            self.pi  = gamma[0] / gamma[0].sum()
            self.A   = xi.sum(0) / (xi.sum(0).sum(1, keepdims=True) + 1e-300)
            for k in range(self.K):
                w        = gamma[:,k] + 1e-300
                self.mu[k]  = (w * returns).sum() / w.sum()
                self.sig[k] = np.sqrt((w*(returns-self.mu[k])**2).sum()/w.sum()) + 1e-6

            log_lik = np.log(scales+1e-300).sum()
            log_liks.append(log_lik)
            if iteration > 5 and abs(log_liks[-1]-log_liks[-2]) < 1e-6:
                print(f"[HMM] Converged at iteration {iteration}")
                break

        self.fitted    = True
        self.log_liks  = log_liks
        # Decode with Viterbi
        states = self.viterbi(returns)
        regime_counts = {self.REGIME_NAMES[k]: int((states==k).sum()) for k in range(self.K)}
        return {
            "regimes":       {k: {"mu":round(float(self.mu[k]),6),
                                   "sigma":round(float(self.sig[k]),6)}
                              for k in range(self.K)},
            "regime_names":  self.REGIME_NAMES,
            "regime_counts": regime_counts,
            "final_log_lik": round(float(log_liks[-1]),4),
            "convergence":   len(log_liks),
        }

    def viterbi(self, obs: np.ndarray) -> np.ndarray:
        """Viterbi algorithm — O(T*K^2) most likely state sequence"""
        T     = len(obs)
        B     = self._emission(obs)
        delta = np.zeros((T, self.K))
        psi   = np.zeros((T, self.K), dtype=int)
        delta[0] = np.log(self.pi + 1e-300) + np.log(B[0] + 1e-300)
        log_A    = np.log(self.A + 1e-300)
        for t in range(1, T):
            for k in range(self.K):
                trans    = delta[t-1] + log_A[:,k]
                psi[t,k] = np.argmax(trans)
                delta[t,k]= trans[psi[t,k]] + np.log(B[t,k]+1e-300)
        # Backtrack
        states      = np.zeros(T, dtype=int)
        states[-1]  = np.argmax(delta[-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states

    def predict_regime(self, returns: np.ndarray) -> dict:
        if not self.fitted:
            return {"error": "Model not fitted"}
        B          = self._emission(returns)
        alpha, _   = self._forward(returns, B)
        probs      = alpha[-1]
        probs     /= probs.sum() + 1e-300
        current    = int(np.argmax(probs))
        states     = self.viterbi(returns)
        # Regime duration
        current_run = 1
        for i in range(len(states)-2, -1, -1):
            if states[i] == states[-1]: current_run += 1
            else: break
        return {
            "current_regime":      self.REGIME_NAMES[current],
            "regime_icon":         self.REGIME_COLORS[current],
            "regime_probs":        {self.REGIME_NAMES[k]: round(float(probs[k]),4) for k in range(self.K)},
            "regime_duration_days":current_run,
            "transition_probs":    {self.REGIME_NAMES[i]: {self.REGIME_NAMES[j]: round(float(self.A[i,j]),4)
                                    for j in range(self.K)} for i in range(self.K)},
            "signal":              "BUY" if current==0 else "SELL" if current==1 else "HOLD",
            "bull_mu":             round(float(self.mu[0])*100,4),
            "bear_mu":             round(float(self.mu[1])*100,4),
        }
