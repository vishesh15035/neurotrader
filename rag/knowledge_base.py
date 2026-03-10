"""
RAG Engine — ChromaDB vector store
Stores trading strategies, market patterns, financial concepts
Agent searches this before making decisions
"""
import chromadb
from chromadb.utils import embedding_functions
import json
from pathlib import Path

TRADING_KNOWLEDGE = [
    {"id":"k001","text":"Momentum strategy: Buy stocks with strong recent performance (12-1 month returns). Works because trends persist due to investor underreaction. Best in trending markets. Signal: 20-day MA > 50-day MA.","tags":["momentum","trend","signal"]},
    {"id":"k002","text":"Mean reversion strategy: Buy oversold stocks (RSI < 30) and sell overbought (RSI > 70). Based on the principle that prices revert to mean. Works best in ranging markets with low volatility.","tags":["mean_reversion","RSI","oscillator"]},
    {"id":"k003","text":"Sharpe ratio above 1.0 is considered good. Above 2.0 is excellent. Formula: (Return - RiskFree) / Volatility. Annualize by multiplying daily Sharpe by sqrt(252).","tags":["risk","sharpe","metrics"]},
    {"id":"k004","text":"MACD crossover: When MACD line crosses above signal line = bullish. When crosses below = bearish. Best for trend confirmation not generation. Lagging indicator.","tags":["MACD","crossover","trend"]},
    {"id":"k005","text":"Bollinger Bands: Price touching lower band = oversold, upper band = overbought. Band squeeze predicts volatility breakout. 2 standard deviations from 20-day MA.","tags":["bollinger","volatility","bands"]},
    {"id":"k006","text":"Position sizing: Kelly Criterion f = (bp - q) / b where b = odds, p = win prob, q = loss prob. Never risk more than 2% of portfolio on single trade.","tags":["position_sizing","risk","kelly"]},
    {"id":"k007","text":"Max drawdown measures largest peak-to-trough decline. Good systems have max drawdown < 20%. Calmar ratio = Annual Return / Max Drawdown, good if > 1.0.","tags":["drawdown","risk","calmar"]},
    {"id":"k008","text":"Pairs trading: Find cointegrated pairs (ADF test p < 0.05). Trade the spread. Enter when z-score > 2.0, exit at 0.5, stop-loss at 3.5. Market neutral strategy.","tags":["pairs","cointegration","stat_arb"]},
    {"id":"k009","text":"Fama-French 5 factors: Market risk (Mkt-RF), Size (SMB), Value (HML), Profitability (RMW), Investment (CMA). Alpha = excess return unexplained by these factors.","tags":["factor_model","fama_french","alpha"]},
    {"id":"k010","text":"VIX above 30 = high fear, market likely oversold. VIX below 15 = complacency. Contrarian: buy when VIX spikes. VIX mean reverts around 20.","tags":["VIX","volatility","sentiment","fear"]},
    {"id":"k011","text":"Earnings surprise effect: Stocks beat estimates → gap up and continue higher for days (Post Earnings Announcement Drift PEAD). Buy the dip after earnings beat on pullback.","tags":["earnings","PEAD","catalyst"]},
    {"id":"k012","text":"Fed rate impact: Rate hikes → growth stocks fall (high duration), financials benefit, bonds fall. Rate cuts → opposite. Watch Fed Funds Futures for expectations.","tags":["fed","rates","macro"]},
    {"id":"k013","text":"Technical support and resistance: Round numbers act as psychological levels. Previous highs/lows = strong levels. Volume at price confirms level strength.","tags":["support","resistance","technical"]},
    {"id":"k014","text":"Volume confirms price moves. High volume breakout = valid. Low volume breakout = suspect. On-Balance Volume (OBV) divergence predicts reversals.","tags":["volume","breakout","confirmation"]},
    {"id":"k015","text":"Risk-reward ratio: Never take a trade with R:R below 2:1. If risking $100, target must be $200 minimum. Professional traders target 3:1 or better.","tags":["risk_reward","trade_management","professional"]},
    {"id":"k016","text":"Sector rotation: Economy expansion → Tech/Discretionary lead. Peak → Energy/Materials. Contraction → Defensive (Utilities/Healthcare). Recovery → Financials/Industrials.","tags":["sector","rotation","macro","cycle"]},
    {"id":"k017","text":"Short selling signals: RSI > 75, price > 2 std above 20MA, negative earnings revision, deteriorating fundamentals, high short interest already covering.","tags":["short","sell","overbought"]},
    {"id":"k018","text":"Golden cross (50MA crosses above 200MA) = strong buy signal, long-term trend change. Death cross (50MA below 200MA) = bearish. These signals have 65-70% historical accuracy.","tags":["golden_cross","death_cross","moving_average"]},
    {"id":"k019","text":"Options: Call options rise when stock rises (positive delta). Put options rise when stock falls (negative delta). Gamma highest near expiry. Theta decay accelerates last 30 days.","tags":["options","greeks","delta","theta"]},
    {"id":"k020","text":"Black-Scholes model: C = S*N(d1) - K*e^(-rT)*N(d2). Implied volatility derived by inverting BS equation. IV > HV means options expensive, consider selling premium.","tags":["black_scholes","options","implied_volatility"]},
]

class RAGKnowledgeBase:
    def __init__(self, persist_dir: str = "data/chromadb"):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.ef     = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="trading_knowledge",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )
        self._populate()

    def _populate(self):
        if self.collection.count() >= len(TRADING_KNOWLEDGE):
            print(f"[RAG] Knowledge base ready: {self.collection.count()} documents")
            return
        print(f"[RAG] Loading {len(TRADING_KNOWLEDGE)} knowledge documents...")
        self.collection.upsert(
            ids       = [k["id"] for k in TRADING_KNOWLEDGE],
            documents = [k["text"] for k in TRADING_KNOWLEDGE],
            metadatas = [{"tags": json.dumps(k["tags"])} for k in TRADING_KNOWLEDGE]
        )
        print(f"[RAG] Knowledge base ready: {self.collection.count()} documents")

    def search(self, query: str, n_results: int = 3) -> list:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        output  = []
        for i, doc in enumerate(results["documents"][0]):
            output.append({
                "text":     doc,
                "distance": round(results["distances"][0][i], 4),
                "tags":     json.loads(results["metadatas"][0][i].get("tags","[]"))
            })
        return output

    def add_insight(self, insight_id: str, text: str, tags: list):
        self.collection.upsert(
            ids=[insight_id], documents=[text],
            metadatas=[{"tags": json.dumps(tags)}]
        )

    def count(self) -> int:
        return self.collection.count()
