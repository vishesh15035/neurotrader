"""
Trading Tools — Claude calls these via function calling
Each tool is a real function the agent can invoke
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# ── Tool Definitions (sent to Claude API) ──────────────
TOOLS = [
    {
        "name": "get_price_data",
        "description": "Fetch real-time and historical price data for a stock ticker. Returns OHLCV data, current price, 52-week high/low, and recent returns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol e.g. AAPL, SPY, NVDA"},
                "period": {"type": "string", "description": "Period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y", "default": "3mo"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_fundamentals",
        "description": "Get fundamental data: P/E ratio, market cap, revenue, EPS, debt/equity, profit margins, analyst recommendations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "compute_technical_signals",
        "description": "Compute technical indicators: RSI, MACD, Bollinger Bands, EMA, momentum, volume analysis. Returns buy/sell/hold signal with strength.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "indicators": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["RSI", "MACD", "BOLLINGER", "EMA", "MOMENTUM", "VOLUME"]},
                    "description": "List of indicators to compute"
                }
            },
            "required": ["ticker", "indicators"]
        }
    },
    {
        "name": "fetch_market_news",
        "description": "Fetch recent news headlines and sentiment for a ticker or market topic. Returns headlines with sentiment scores.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker or topic e.g. AAPL or 'fed rates'"},
                "limit": {"type": "integer", "description": "Number of news items", "default": 5}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "run_backtest",
        "description": "Run a quick backtest of a trading strategy on historical data. Returns Sharpe ratio, total return, max drawdown, win rate.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "strategy": {"type": "string", "enum": ["momentum", "mean_reversion", "rsi", "macd_crossover"], "description": "Strategy to backtest"},
                "period": {"type": "string", "default": "1y"}
            },
            "required": ["ticker", "strategy"]
        }
    },
    {
        "name": "get_portfolio_status",
        "description": "Get current simulated portfolio: holdings, cash, total value, P&L, allocation percentages.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "execute_trade",
        "description": "Execute a simulated trade: buy or sell a stock. Updates portfolio and logs the decision with reasoning.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker":   {"type": "string"},
                "action":   {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                "quantity": {"type": "integer", "description": "Number of shares"},
                "reasoning":{"type": "string", "description": "Agent's reasoning for this trade"}
            },
            "required": ["ticker", "action", "quantity", "reasoning"]
        }
    },
    {
        "name": "search_knowledge_base",
        "description": "Search the RAG knowledge base for relevant trading strategies, market patterns, or financial concepts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query about trading strategies or market concepts"},
                "n_results": {"type": "integer", "default": 3}
            },
            "required": ["query"]
        }
    }
]

# ── Tool Implementations ────────────────────────────────

def get_price_data(ticker: str, period: str = "3mo") -> dict:
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period=period)
        if hist.empty:
            return {"error": f"No data for {ticker}"}
        current = float(hist['Close'].iloc[-1])
        prev    = float(hist['Close'].iloc[-2])
        ret_1d  = (current - prev) / prev * 100
        ret_1mo = (current - float(hist['Close'].iloc[-22])) / float(hist['Close'].iloc[-22]) * 100 if len(hist) >= 22 else 0
        return {
            "ticker":        ticker,
            "current_price": round(current, 2),
            "change_1d_pct": round(ret_1d, 2),
            "change_1mo_pct":round(ret_1mo, 2),
            "high_52w":      round(float(hist['High'].max()), 2),
            "low_52w":       round(float(hist['Low'].min()), 2),
            "avg_volume":    int(hist['Volume'].mean()),
            "data_points":   len(hist),
            "period":        period
        }
    except Exception as e:
        return {"error": str(e)}

def get_fundamentals(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        return {
            "ticker":          ticker,
            "market_cap":      info.get("marketCap", "N/A"),
            "pe_ratio":        info.get("trailingPE", "N/A"),
            "forward_pe":      info.get("forwardPE", "N/A"),
            "eps":             info.get("trailingEps", "N/A"),
            "revenue":         info.get("totalRevenue", "N/A"),
            "profit_margin":   info.get("profitMargins", "N/A"),
            "debt_to_equity":  info.get("debtToEquity", "N/A"),
            "roe":             info.get("returnOnEquity", "N/A"),
            "analyst_rating":  info.get("recommendationKey", "N/A"),
            "target_price":    info.get("targetMeanPrice", "N/A"),
            "sector":          info.get("sector", "N/A"),
            "beta":            info.get("beta", "N/A"),
        }
    except Exception as e:
        return {"error": str(e)}

def compute_technical_signals(ticker: str, indicators: list) -> dict:
    try:
        hist = yf.Ticker(ticker).history(period="6mo")
        if hist.empty:
            return {"error": f"No data for {ticker}"}
        close  = hist['Close']
        volume = hist['Volume']
        result = {"ticker": ticker, "signals": {}, "overall": "HOLD", "strength": 0}
        scores = []

        if "RSI" in indicators:
            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss
            rsi   = float(100 - 100/(1+rs.iloc[-1]))
            signal = "BUY" if rsi < 30 else "SELL" if rsi > 70 else "HOLD"
            scores.append(1 if signal=="BUY" else -1 if signal=="SELL" else 0)
            result["signals"]["RSI"] = {"value": round(rsi,2), "signal": signal}

        if "MACD" in indicators:
            ema12  = close.ewm(span=12).mean()
            ema26  = close.ewm(span=26).mean()
            macd   = ema12 - ema26
            signal_line = macd.ewm(span=9).mean()
            cross  = "BUY" if float(macd.iloc[-1]) > float(signal_line.iloc[-1]) else "SELL"
            scores.append(1 if cross=="BUY" else -1)
            result["signals"]["MACD"] = {"macd": round(float(macd.iloc[-1]),4), "signal": cross}

        if "BOLLINGER" in indicators:
            sma  = close.rolling(20).mean()
            std  = close.rolling(20).std()
            ub   = sma + 2*std
            lb   = sma - 2*std
            cur  = float(close.iloc[-1])
            sig  = "BUY" if cur < float(lb.iloc[-1]) else "SELL" if cur > float(ub.iloc[-1]) else "HOLD"
            scores.append(1 if sig=="BUY" else -1 if sig=="SELL" else 0)
            result["signals"]["BOLLINGER"] = {
                "upper": round(float(ub.iloc[-1]),2),
                "lower": round(float(lb.iloc[-1]),2),
                "signal": sig
            }

        if "MOMENTUM" in indicators:
            mom = float(close.iloc[-1]) / float(close.iloc[-20]) - 1
            sig = "BUY" if mom > 0.05 else "SELL" if mom < -0.05 else "HOLD"
            scores.append(1 if sig=="BUY" else -1 if sig=="SELL" else 0)
            result["signals"]["MOMENTUM"] = {"value": round(mom*100,2), "signal": sig}

        if "VOLUME" in indicators:
            avg_vol = float(volume.rolling(20).mean().iloc[-1])
            cur_vol = float(volume.iloc[-1])
            ratio   = cur_vol / avg_vol
            sig     = "HIGH" if ratio > 1.5 else "LOW" if ratio < 0.5 else "NORMAL"
            result["signals"]["VOLUME"] = {"ratio": round(ratio,2), "signal": sig}

        if scores:
            avg = np.mean(scores)
            result["overall"]  = "BUY" if avg > 0.3 else "SELL" if avg < -0.3 else "HOLD"
            result["strength"] = round(abs(avg)*100, 1)
        return result
    except Exception as e:
        return {"error": str(e)}

def fetch_market_news(ticker: str, limit: int = 5) -> dict:
    try:
        stock = yf.Ticker(ticker)
        news  = stock.news[:limit] if stock.news else []
        items = []
        for n in news:
            items.append({
                "title":     n.get("title", ""),
                "publisher": n.get("publisher", ""),
                "sentiment": _simple_sentiment(n.get("title", "")),
            })
        return {"ticker": ticker, "news_count": len(items), "items": items}
    except Exception as e:
        return {"ticker": ticker, "news_count": 0, "items": [], "note": "News unavailable"}

def _simple_sentiment(text: str) -> str:
    text = text.lower()
    pos = ["beat", "surge", "rally", "profit", "growth", "strong", "record", "up", "gain", "bullish"]
    neg = ["miss", "drop", "fall", "loss", "weak", "cut", "down", "bear", "concern", "risk"]
    ps  = sum(1 for w in pos if w in text)
    ns  = sum(1 for w in neg if w in text)
    return "POSITIVE" if ps > ns else "NEGATIVE" if ns > ps else "NEUTRAL"

def run_backtest(ticker: str, strategy: str, period: str = "1y") -> dict:
    try:
        hist  = yf.Ticker(ticker).history(period=period)
        close = hist['Close']
        if strategy == "momentum":
            signal = (close.rolling(20).mean() > close.rolling(50).mean()).astype(int)
        elif strategy == "mean_reversion":
            z = (close - close.rolling(20).mean()) / close.rolling(20).std()
            signal = (z < -1).astype(int)
        elif strategy == "rsi":
            d = close.diff()
            rsi = 100 - 100/(1 + d.clip(lower=0).rolling(14).mean()/(-d.clip(upper=0)).rolling(14).mean())
            signal = (rsi < 35).astype(int)
        else:  # macd_crossover
            signal = ((close.ewm(span=12).mean() - close.ewm(span=26).mean()) > 0).astype(int)

        returns  = close.pct_change().fillna(0)
        strat_r  = (returns * signal.shift(1).fillna(0))
        cum_r    = float((1 + strat_r).prod() - 1)
        sharpe   = float(strat_r.mean() / strat_r.std() * np.sqrt(252)) if strat_r.std() > 0 else 0
        cum_eq   = (1 + strat_r).cumprod()
        drawdown = float((cum_eq / cum_eq.cummax() - 1).min())
        trades   = int(signal.diff().abs().sum())
        wins     = int((strat_r > 0).sum())
        total    = int((strat_r != 0).sum())
        return {
            "ticker":       ticker,
            "strategy":     strategy,
            "total_return": round(cum_r*100, 2),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(drawdown*100, 2),
            "win_rate":     round(wins/total*100, 1) if total > 0 else 0,
            "num_trades":   trades,
            "period":       period
        }
    except Exception as e:
        return {"error": str(e)}

# Global simulated portfolio
PORTFOLIO = {
    "cash": 100000.0,
    "holdings": {},
    "trades": [],
    "initial_value": 100000.0
}

def get_portfolio_status() -> dict:
    holdings_value = 0
    holdings_detail = {}
    for ticker, qty in PORTFOLIO["holdings"].items():
        try:
            price = float(yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1])
            val   = price * qty
            holdings_value += val
            holdings_detail[ticker] = {"qty": qty, "price": round(price,2), "value": round(val,2)}
        except:
            pass
    total  = PORTFOLIO["cash"] + holdings_value
    pnl    = total - PORTFOLIO["initial_value"]
    return {
        "cash":            round(PORTFOLIO["cash"], 2),
        "holdings":        holdings_detail,
        "holdings_value":  round(holdings_value, 2),
        "total_value":     round(total, 2),
        "pnl":             round(pnl, 2),
        "pnl_pct":         round(pnl/PORTFOLIO["initial_value"]*100, 2),
        "num_trades":      len(PORTFOLIO["trades"])
    }

def execute_trade(ticker: str, action: str, quantity: int, reasoning: str) -> dict:
    try:
        price = float(yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1])
        cost  = price * quantity
        if action == "BUY":
            if cost > PORTFOLIO["cash"]:
                return {"status": "REJECTED", "reason": f"Insufficient cash: need ${cost:.0f}, have ${PORTFOLIO['cash']:.0f}"}
            PORTFOLIO["cash"] -= cost
            PORTFOLIO["holdings"][ticker] = PORTFOLIO["holdings"].get(ticker, 0) + quantity
        elif action == "SELL":
            held = PORTFOLIO["holdings"].get(ticker, 0)
            if quantity > held:
                return {"status": "REJECTED", "reason": f"Insufficient shares: have {held}, want to sell {quantity}"}
            PORTFOLIO["cash"] += cost
            PORTFOLIO["holdings"][ticker] = held - quantity
            if PORTFOLIO["holdings"][ticker] == 0:
                del PORTFOLIO["holdings"][ticker]
        trade = {
            "timestamp": datetime.now().isoformat(),
            "ticker":    ticker,
            "action":    action,
            "quantity":  quantity,
            "price":     round(price, 2),
            "value":     round(cost, 2),
            "reasoning": reasoning
        }
        PORTFOLIO["trades"].append(trade)
        return {
            "status":    "EXECUTED",
            "ticker":    ticker,
            "action":    action,
            "quantity":  quantity,
            "price":     round(price, 2),
            "total":     round(cost, 2),
            "cash_remaining": round(PORTFOLIO["cash"], 2)
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}

def search_knowledge_base(query: str, n_results: int = 3) -> dict:
    # Will be overridden by RAG engine
    return {"query": query, "results": [], "note": "RAG not initialized"}

# Tool dispatcher
TOOL_MAP = {
    "get_price_data":          get_price_data,
    "get_fundamentals":        get_fundamentals,
    "compute_technical_signals": compute_technical_signals,
    "fetch_market_news":       fetch_market_news,
    "run_backtest":            run_backtest,
    "get_portfolio_status":    get_portfolio_status,
    "execute_trade":           execute_trade,
    "search_knowledge_base":   search_knowledge_base,
}

def dispatch(tool_name: str, tool_input: dict) -> str:
    fn = TOOL_MAP.get(tool_name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    result = fn(**tool_input)
    return json.dumps(result, default=str)
