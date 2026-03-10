"""
Graph Neural Network — Stock Correlation Graph
Nodes: stocks, Edges: correlation strength
Message passing → node embeddings → signal aggregation
"""
import numpy as np

class StockGraphNode:
    def __init__(self, ticker: str, features: np.ndarray):
        self.ticker   = ticker
        self.features = features  # [return, vol, momentum, rsi, ...]
        self.embedding= features.copy()
        self.neighbors= []

class StockGNN:
    """
    Graph Neural Network for stock market
    - Builds correlation graph from return data
    - Message passing: aggregate neighbor information
    - Produces graph-aware signals
    """
    def __init__(self, n_layers: int = 3, threshold: float = 0.6):
        self.n_layers  = n_layers
        self.threshold = threshold
        self.nodes     = {}
        self.adj       = None

    def build_graph(self, returns: dict) -> dict:
        """Build correlation graph from return series"""
        tickers = list(returns.keys())
        N       = len(tickers)
        # Correlation matrix
        ret_matrix = np.column_stack([returns[t] for t in tickers])
        # Normalize
        ret_norm   = (ret_matrix - ret_matrix.mean(0)) / (ret_matrix.std(0)+1e-8)
        corr       = ret_norm.T @ ret_norm / len(ret_norm)
        # Adjacency: threshold correlation
        self.adj   = (np.abs(corr) > self.threshold).astype(float)
        np.fill_diagonal(self.adj, 0)
        # Graph stats
        degrees    = self.adj.sum(1)
        # Find communities (simple: connected components via BFS)
        visited    = set()
        communities= []
        for start in range(N):
            if start in visited: continue
            community = []
            queue     = [start]
            while queue:
                node = queue.pop(0)
                if node in visited: continue
                visited.add(node)
                community.append(tickers[node])
                neighbors = np.where(self.adj[node]>0)[0]
                queue.extend(n for n in neighbors if n not in visited)
            if community: communities.append(community)

        return {
            "n_nodes":      N,
            "n_edges":      int(self.adj.sum()//2),
            "density":      round(float(self.adj.sum()/(N*(N-1)+1e-8)),4),
            "avg_degree":   round(float(degrees.mean()),2),
            "max_degree":   int(degrees.max()),
            "hub_stock":    tickers[int(degrees.argmax())],
            "communities":  communities,
            "corr_matrix":  {tickers[i]: {tickers[j]: round(float(corr[i,j]),4)
                              for j in range(N)} for i in range(N)},
            "threshold":    self.threshold,
        }

    def message_passing(self, node_features: np.ndarray,
                        n_steps: int = 3) -> np.ndarray:
        """
        Graph convolution: h^(l+1) = ReLU(D^{-1} A h^(l) W^(l))
        Aggregates neighbor information iteratively
        """
        H    = node_features.copy()
        N, F = H.shape
        if self.adj is None or self.adj.shape[0] != N:
            return H
        # Degree matrix for normalization
        D_inv = np.diag(1.0 / (self.adj.sum(1) + 1e-8))
        # Random weight matrices (in practice: trained)
        np.random.seed(42)
        for l in range(n_steps):
            W  = np.random.randn(F, F) * np.sqrt(2.0/F)
            # Aggregate: A_norm * H * W
            H_agg = D_inv @ self.adj @ H @ W
            # Residual + ReLU
            H = np.maximum(0, H + H_agg)
            # Layer norm
            H = (H - H.mean(0)) / (H.std(0)+1e-8)
        return H

    def analyze(self, prices: dict) -> dict:
        """Full GNN analysis pipeline"""
        tickers = list(prices.keys())
        N       = len(tickers)
        # Compute returns
        returns = {}
        for t, p in prices.items():
            arr = np.array(p)
            returns[t] = np.diff(arr)/arr[:-1] if len(arr)>1 else np.array([0.0])
        # Build graph
        graph   = self.build_graph(returns)
        # Node features: [mean_ret, vol, momentum, skew, kurt]
        min_len = min(len(r) for r in returns.values())
        features= np.zeros((N,5))
        for i, t in enumerate(tickers):
            r = returns[t][-min_len:]
            features[i,0] = r.mean()*252
            features[i,1] = r.std()*np.sqrt(252)
            features[i,2] = (r[-1]/r[0]-1) if len(r)>1 else 0
            from scipy.stats import skew, kurtosis
            features[i,3] = skew(r) if len(r)>3 else 0
            features[i,4] = kurtosis(r) if len(r)>3 else 0
        # Message passing
        embeddings = self.message_passing(features, self.n_layers)
        # Signal: positive embedding[0] (return dimension) = bullish
        signals = {}
        for i,t in enumerate(tickers):
            score = float(embeddings[i,0])
            signals[t] = {
                "gnn_score": round(score,4),
                "signal":    "BUY" if score>0.5 else "SELL" if score<-0.5 else "HOLD",
                "degree":    int(self.adj[i].sum()) if self.adj is not None else 0,
            }
        return {**graph, "signals": signals, "n_layers": self.n_layers}
