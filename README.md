c
# NeuroTrader v4.0 📈

> Agentic AI quantitative trading system with 19 advanced models, Bloomberg-style dashboard, and real-time market analysis.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Models](https://img.shields.io/badge/Models-19-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

##  Architecture
```
19 Models across 5 layers:
├── Layer 1 — Deep Learning      LSTM · Transformer · Gaussian Process
├── Layer 2 — State Estimation   Kalman Filter · Particle Filter · HMM
├── Layer 3 — Volatility         GARCH · EGARCH · Heston SV
├── Layer 4 — Time Series        Ornstein-Uhlenbeck · MS-ARIMA · Copula
└── Layer 5 — Signal Sources     Options Flow · Cross-Asset · Microstructure · Sentiment
         Ensemble               BMA · Stacking · Walk-Forward CV · Sharpe Weighting
```

##  Results

| Metric | Value |
|--------|-------|
| Total Models | 19 |
| Analysis Time | ~16s (AAPL) |
| Monte Carlo Paths | 10,000 |
| Particle Filter | 1,000 particles |
| GARCH Persistence | 0.95 (AAPL) |
| Copula Crash Risk | 52.6% (AAPL vs SPY) |
| OU Half-Life | 7.93 days |

##  Quick Start
```bash
# Clone
git clone https://github.com/vishesh15035/neurotrader.git
cd neurotrader

# Install dependencies
pip install -r requirements.txt

# Terminal 1 — API Server + Dashboard
python3 api/server.py

# Terminal 2 — CLI
python3 main.py
```

Open dashboard: `http://127.0.0.1:5000/dashboard`

##  CLI Commands
```
quant AAPL      — Run all 19 models (~16s)
analyze AAPL    — Ollama AI ReAct agent analysis
earnings AAPL   — earnings brief
portfolio       — portfolio builder
scan            — Scan AAPL MSFT NVDA GOOGL SPY
q               — Quit
```

## 📊 Dashboard

Bloomberg-style dark UI with 4 tabs:
- **Overview** — Price chart, GARCH vol, risk metrics, HMM regime radar
- **Models** — All 19 signals grouped by layer, vote breakdown
- **Volatility** — GARCH/Heston/Copula/OU deep stats
- **Monte Carlo** — 10k path simulation, VaR/CVaR

Supports **US stocks** (USD) and **Indian NSE stocks** (INR).

##  Tech Stack

| Component | Technology |
|-----------|-----------|
| Quant Engine | Pure NumPy/SciPy (no sklearn) |
| AI Agent | Ollama llama3.1 (local) |
| RAG | ChromaDB |
| Memory | SQLite + in-memory deque |
| Dashboard | Vanilla HTML/CSS/Canvas |
| API | Flask + Flask-CORS |
| Data | yfinance |

##  Installation
bash
pip install yfinance numpy scipy pandas flask flask-cors \
            chromadb statsmodels PyWavelets python-dotenv arch


Requires [Ollama](https://ollama.ai) for AI agent features:
```bash
ollama pull llama3.1
```

##  Author

**Vishesh Brahmbhatt**  
B.Tech Electronics & Instrumentation, Nirma University  

GitHub: [@vishesh15035](https://github.com/vishesh15035)

##  License

MIT License — see [LICENSE](LICENSE)
