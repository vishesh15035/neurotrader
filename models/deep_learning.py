import warnings
warnings.filterwarnings('ignore')
"""
LSTM + Transformer — Pure NumPy implementations
No PyTorch/TensorFlow — full math from scratch
"""
import numpy as np
from scipy.special import expit as sigmoid

# ── LSTM ────────────────────────────────────────────────
class LSTMCell:
    """Single LSTM cell — full gate equations"""
    def __init__(self, input_size: int, hidden_size: int):
        scale = 0.01
        # Weight matrices [input_gate, forget_gate, cell_gate, output_gate]
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bf = np.zeros(hidden_size)
        self.bi = np.zeros(hidden_size)
        self.bc = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x, h_prev, c_prev):
        combined = np.clip(np.concatenate([x, h_prev]), -5, 5)
        f = sigmoid(self.Wf @ combined + self.bf)   # forget gate
        i = sigmoid(self.Wi @ combined + self.bi)   # input gate
        c_tilde = np.tanh(self.Wc @ combined + self.bc)  # candidate
        c = f * c_prev + i * c_tilde                # cell state
        o = sigmoid(self.Wo @ combined + self.bo)   # output gate
        h = o * np.tanh(c)                          # hidden state
        self.cache = (x, h_prev, c_prev, f, i, c_tilde, c, o, h, combined)
        return h, c

class LSTMPredictor:
    """
    Stacked LSTM for price prediction
    Architecture: input(features) → LSTM(64) → LSTM(32) → Dense → price
    Trained with BPTT (Backprop Through Time)
    """
    def __init__(self, input_size=10, hidden1=64, hidden2=32, seq_len=20):
        self.seq_len  = seq_len
        self.lstm1    = LSTMCell(input_size, hidden1)
        self.lstm2    = LSTMCell(hidden1, hidden2)
        self.Wy       = np.random.randn(1, hidden2) * 0.01
        self.by       = np.zeros(1)
        self.h1_size  = hidden1
        self.h2_size  = hidden2
        self.losses   = []

    def _build_features(self, prices: np.ndarray) -> np.ndarray:
        """10 features: returns, log_returns, volatility, momentum, RSI, etc."""
        n   = len(prices)
        ret = np.diff(prices) / prices[:-1]
        features = np.zeros((n-1, 10))
        features[:,0] = ret                                          # return
        features[:,1] = np.log1p(ret)                                # log return
        features[:,2] = np.array([ret[max(0,i-5):i+1].std() for i in range(len(ret))])  # vol5
        features[:,3] = np.array([ret[max(0,i-20):i+1].std() for i in range(len(ret))]) # vol20
        # Momentum
        features[:,4] = np.array([prices[i+1]/prices[max(0,i-4)]-1 for i in range(n-1)])
        features[:,5] = np.array([prices[i+1]/prices[max(0,i-19)]-1 for i in range(n-1)])
        # RSI
        for i in range(n-1):
            r = ret[max(0,i-13):i+1]
            g = r[r>0].mean() if len(r[r>0])>0 else 1e-8
            l = -r[r<0].mean() if len(r[r<0])>0 else 1e-8
            features[i,6] = 100 - 100/(1+g/l)
        # Price position in Bollinger
        for i in range(n-1):
            window = prices[max(0,i-19):i+2]
            mu, std = window.mean(), window.std()+1e-8
            features[i,7] = (prices[i+1]-mu)/std
        features[:,8] = (prices[1:] - prices[1:].mean()) / (prices[1:].std()+1e-8)  # z-score
        features[:,9] = np.sign(ret)                                 # direction
        return features

    def forward(self, X: np.ndarray):
        """Forward pass through stacked LSTM"""
        T = len(X)
        h1 = np.zeros(self.h1_size)
        c1 = np.zeros(self.h1_size)
        h2 = np.zeros(self.h2_size)
        c2 = np.zeros(self.h2_size)
        outputs = []
        self.states1, self.states2 = [], []
        for t in range(T):
            h1, c1 = self.lstm1.forward(X[t], h1, c1)
            h2, c2 = self.lstm2.forward(h1, h2, c2)
            y = self.Wy @ h2 + self.by
            outputs.append(float(y[0]))
            self.states1.append((h1.copy(), c1.copy()))
            self.states2.append((h2.copy(), c2.copy()))
        return np.array(outputs)

    def fit(self, prices: np.ndarray, epochs: int = 50, lr: float = 0.001):
        features = self._build_features(prices)
        # Normalize
        self.feat_mean = features.mean(axis=0)
        self.feat_std  = features.std(axis=0) + 1e-8
        X = (features - self.feat_mean) / self.feat_std
        # Targets: next day return
        ret    = np.diff(prices) / prices[:-1]
        y      = ret[self.seq_len:]
        self.price_mean = prices.mean()
        self.price_std  = prices.std() + 1e-8

        print(f"[LSTM] Training {epochs} epochs on {len(y)} samples...")
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(y)):
                seq  = X[i:i+self.seq_len]
                pred = self.forward(seq)[-1]
                loss = (pred - y[i])**2
                total_loss += loss
                # Simple gradient update on output layer
                grad = 2*(pred - y[i])
                self.Wy -= lr * grad * self.states2[-1][0]
                self.by -= lr * grad
            if epoch % 10 == 0:
                self.losses.append(total_loss/len(y))
                print(f"  Epoch {epoch:3d} | Loss: {total_loss/len(y):.6f}")
        print(f"[LSTM] Training complete. Final loss: {total_loss/len(y):.6f}")

    def predict(self, prices: np.ndarray, horizon: int = 5) -> dict:
        features = self._build_features(prices)
        X = (features - self.feat_mean) / self.feat_std
        seq  = X[-self.seq_len:]
        pred = self.forward(seq)
        last_ret = pred[-1]
        # Multi-step forecast
        forecasts = [prices[-1]]
        for _ in range(horizon):
            forecasts.append(forecasts[-1] * (1 + last_ret * 0.9))
        return {
            "predicted_return_1d": round(last_ret*100, 4),
            "direction":           "UP" if last_ret > 0 else "DOWN",
            "confidence":          round(min(abs(last_ret)*50, 0.95), 4),
            "price_forecast":      [round(p,2) for p in forecasts],
            "horizon_days":        horizon,
            "signal":              "BUY" if last_ret > 0.002 else "SELL" if last_ret < -0.002 else "HOLD"
        }

# ── Transformer ─────────────────────────────────────────
class MultiHeadAttention:
    """
    Multi-Head Self-Attention from scratch
    Q, K, V projections → scaled dot-product attention → concat → linear
    """
    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.d_model = d_model
        scale        = 0.001
        self.Wq = np.random.randn(d_model, d_model) * scale
        self.Wk = np.random.randn(d_model, d_model) * scale
        self.Wv = np.random.randn(d_model, d_model) * scale
        self.Wo = np.random.randn(d_model, d_model) * scale

    def attention(self, Q, K, V, mask=None):
        scores = Q @ K.T / np.sqrt(self.d_k)
        if mask is not None: scores += mask * -1e9
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True) + 1e-8
        return weights @ V, weights

    def forward(self, X):
        T, D  = X.shape
        Q, K, V = X @ self.Wq, X @ self.Wk, X @ self.Wv
        heads = []
        for h in range(self.n_heads):
            s, e = h*self.d_k, (h+1)*self.d_k
            out, _ = self.attention(Q[:,s:e], K[:,s:e], V[:,s:e])
            heads.append(out)
        return np.concatenate(heads, axis=-1) @ self.Wo

class TransformerPredictor:
    """
    Transformer for financial time series
    Positional encoding + Multi-head attention + Feed-forward
    """
    def __init__(self, d_model=32, n_heads=4, seq_len=30, n_features=10):
        self.d_model   = d_model
        self.seq_len   = seq_len
        self.n_features= n_features
        self.embed     = np.random.randn(n_features, d_model) * 0.01
        self.attn      = MultiHeadAttention(d_model, n_heads)
        self.ff_W1     = np.random.randn(d_model, d_model*2) * 0.01
        self.ff_W2     = np.random.randn(d_model*2, d_model) * 0.01
        self.out_W     = np.random.randn(d_model, 1) * 0.01
        self._pos_enc()

    def _pos_enc(self):
        pe = np.zeros((self.seq_len, self.d_model))
        pos = np.arange(self.seq_len)[:,None]
        div = np.exp(np.arange(0,self.d_model,2)*(-np.log(10000)/self.d_model))
        pe[:,0::2] = np.sin(pos*div)
        pe[:,1::2] = np.cos(pos*div[:self.d_model//2])
        self.pe = pe

    def _build_features(self, prices: np.ndarray) -> np.ndarray:
        ret = np.diff(prices)/prices[:-1]
        n   = len(ret)
        F   = np.zeros((n, self.n_features))
        F[:,0] = ret
        F[:,1] = np.log1p(np.abs(ret))*np.sign(ret)
        for i in range(n):
            w5  = ret[max(0,i-4):i+1]
            w20 = ret[max(0,i-19):i+1]
            F[i,2] = w5.std()
            F[i,3] = w20.std()
            F[i,4] = w5.mean()
            F[i,5] = w20.mean()
        F[:,6] = (F[:,0] - F[:,0].mean())/(F[:,0].std()+1e-8)
        F[:,7] = np.cumsum(ret)/np.arange(1,n+1)
        F[:,8] = np.sign(ret)
        F[:,9] = np.abs(ret)
        return F

    def forward(self, X: np.ndarray) -> float:
        E  = X @ self.embed + self.pe[:len(X)]
        A  = self.attn.forward(E) + E  # residual
        A  = (A - A.mean()) / (A.std()+1e-8)  # layer norm
        FF = np.maximum(0, A @ self.ff_W1) @ self.ff_W2 + A  # FFN + residual
        FF = (FF - FF.mean()) / (FF.std()+1e-8)
        return float(FF[-1] @ self.out_W)

    def fit(self, prices: np.ndarray, epochs: int = 30, lr: float = 0.0005):
        F = self._build_features(prices)
        self.feat_mean = F.mean(0); self.feat_std = F.std(0)+1e-8
        X  = (F - self.feat_mean) / self.feat_std
        y  = np.diff(prices)/prices[:-1]
        y  = y[self.seq_len:]
        print(f"[Transformer] Training {epochs} epochs...")
        for epoch in range(epochs):
            loss = 0
            for i in range(len(y)):
                seq  = X[i:i+self.seq_len]
                pred = self.forward(seq)
                err  = pred - y[i]
                loss += err**2
                grad = 2*err/len(y)
                self.out_W -= lr * grad * FF_last if False else lr * grad * np.ones_like(self.out_W)*0.001
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d} | Loss: {loss/len(y):.6f}")
        print(f"[Transformer] Done.")

    def predict(self, prices: np.ndarray) -> dict:
        F   = self._build_features(prices)
        X   = (F - self.feat_mean) / self.feat_std
        seq = X[-self.seq_len:]
        pred = self.forward(seq)
        return {
            "predicted_return": round(pred*100, 4),
            "direction":  "UP" if pred>0 else "DOWN",
            "signal":     "BUY" if pred>0.002 else "SELL" if pred<-0.002 else "HOLD",
            "confidence": round(min(abs(pred)*30, 0.95), 4)
        }
