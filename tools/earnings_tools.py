"""
Earnings + Portfolio Tools — Real market data
"""
import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

def get_earnings_history(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        # Earnings history
        earnings = stock.earnings_dates
        hist     = stock.history(period="2y")
        info     = stock.info
        quarterly = stock.quarterly_financials

        # Build last 4 quarters
        quarters = []
        if earnings is not None and not earnings.empty:
            recent = earnings.head(8)
            for date, row in recent.iterrows():
                eps_est = row.get("EPS Estimate", None)
                eps_act = row.get("Reported EPS", None)
                surprise = row.get("Surprise(%)", None)
                if eps_est is not None and eps_act is not None:
                    quarters.append({
                        "date":     str(date)[:10],
                        "eps_est":  round(float(eps_est),2) if eps_est else None,
                        "eps_act":  round(float(eps_act),2) if eps_act else None,
                        "surprise": round(float(surprise),1) if surprise else None,
                        "beat":     float(eps_act) > float(eps_est) if (eps_est and eps_act) else None
                    })

        # Stock reactions after earnings
        reactions = []
        for q in quarters[:4]:
            try:
                date = pd.Timestamp(q["date"])
                if date in hist.index or True:
                    idx = hist.index.searchsorted(date)
                    if idx < len(hist)-1:
                        before = float(hist['Close'].iloc[max(0,idx-1)])
                        after  = float(hist['Close'].iloc[min(len(hist)-1,idx+1)])
                        reaction = (after - before) / before * 100
                        reactions.append(round(reaction, 2))
            except: pass

        return {
            "ticker":             ticker,
            "company":            info.get("longName",""),
            "sector":             info.get("sector",""),
            "quarters":           quarters[:4],
            "avg_surprise":       round(np.mean([q["surprise"] for q in quarters if q.get("surprise")]),2) if quarters else 0,
            "beat_rate":          f"{sum(1 for q in quarters if q.get('beat'))/max(len(quarters),1)*100:.0f}%",
            "post_earnings_moves": reactions,
            "current_price":      info.get("currentPrice", info.get("regularMarketPrice",0)),
            "market_cap":         info.get("marketCap",0),
            "pe_ratio":           info.get("trailingPE",0),
            "forward_pe":         info.get("forwardPE",0),
            "eps_fwd":            info.get("forwardEps",0),
            "revenue_growth":     info.get("revenueGrowth",0),
            "analyst_target":     info.get("targetMeanPrice",0),
            "recommendation":     info.get("recommendationKey",""),
            "num_analysts":       info.get("numberOfAnalystOpinions",0),
        }
    except Exception as e:
        return {"error": str(e)}

def get_segment_data(ticker: str) -> dict:
    try:
        stock    = yf.Ticker(ticker)
        info     = stock.info
        fin      = stock.quarterly_financials
        rev_cols = [c for c in fin.columns[:4]] if fin is not None and not fin.empty else []
        revenues = {}
        if fin is not None and not fin.empty and "Total Revenue" in fin.index:
            for col in rev_cols:
                revenues[str(col)[:10]] = int(fin.loc["Total Revenue", col]) if not pd.isna(fin.loc["Total Revenue", col]) else 0

        return {
            "ticker":          ticker,
            "company":         info.get("longName",""),
            "revenues":        revenues,
            "gross_margin":    info.get("grossMargins",0),
            "operating_margin":info.get("operatingMargins",0),
            "profit_margin":   info.get("profitMargins",0),
            "revenue_growth":  info.get("revenueGrowth",0),
            "earnings_growth": info.get("earningsGrowth",0),
            "total_revenue":   info.get("totalRevenue",0),
            "free_cashflow":   info.get("freeCashflow",0),
            "debt_equity":     info.get("debtToEquity",0),
        }
    except Exception as e:
        return {"error": str(e)}

def get_options_implied_move(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get("currentPrice", stock.info.get("regularMarketPrice", 0))
        exps  = stock.options
        if not exps:
            return {"ticker": ticker, "implied_move_pct": None, "note": "No options data"}

        # Use nearest expiration
        nearest = exps[0]
        chain   = stock.option_chain(nearest)
        calls   = chain.calls
        puts    = chain.puts

        # ATM straddle approximation
        atm_calls = calls[abs(calls['strike'] - price) < price*0.02]
        atm_puts  = puts[abs(puts['strike']  - price) < price*0.02]

        if atm_calls.empty or atm_puts.empty:
            atm_calls = calls.iloc[(calls['strike']-price).abs().argsort()[:1]]
            atm_puts  = puts.iloc[(puts['strike']-price).abs().argsort()[:1]]

        straddle_price = float(atm_calls['lastPrice'].iloc[0]) + float(atm_puts['lastPrice'].iloc[0])
        implied_move   = straddle_price / price * 100

        return {
            "ticker":           ticker,
            "current_price":    round(price, 2),
            "expiration":       nearest,
            "straddle_cost":    round(straddle_price, 2),
            "implied_move_pct": round(implied_move, 2),
            "implied_move_usd": round(straddle_price, 2),
            "bull_target":      round(price + straddle_price, 2),
            "bear_target":      round(price - straddle_price, 2),
        }
    except Exception as e:
        return {"ticker": ticker, "implied_move_pct": None, "error": str(e)}

def build_portfolio(age: int, income: int, savings: int,
                    risk: str, goal: str, monthly_invest: int = 0) -> dict:
    """BlackRock-style portfolio construction"""
    # Risk profiles
    profiles = {
        "aggressive": {"stocks":0.85,"bonds":0.10,"alternatives":0.05,"exp_return":(10,14),"max_dd":-35},
        "moderate":   {"stocks":0.65,"bonds":0.25,"alternatives":0.10,"exp_return":(7,10), "max_dd":-25},
        "conservative":{"stocks":0.40,"bonds":0.45,"alternatives":0.15,"exp_return":(4,7),  "max_dd":-15},
    }
    profile = profiles.get(risk.lower(), profiles["moderate"])
    years_to_retire = max(65 - age, 1)

    # Core ETF recommendations
    etfs = {
        "US_Large_Cap":    {"ticker":"VTI",  "name":"Vanguard Total Market",      "alloc": profile["stocks"]*0.40, "expense":0.03},
        "International":   {"ticker":"VXUS", "name":"Vanguard Total Intl",         "alloc": profile["stocks"]*0.20, "expense":0.07},
        "US_Growth":       {"ticker":"QQQ",  "name":"Invesco Nasdaq 100",          "alloc": profile["stocks"]*0.15, "expense":0.20},
        "Dividend":        {"ticker":"VYM",  "name":"Vanguard High Dividend",      "alloc": profile["stocks"]*0.15, "expense":0.06},
        "Small_Cap":       {"ticker":"VB",   "name":"Vanguard Small Cap",          "alloc": profile["stocks"]*0.10, "expense":0.05},
        "US_Bonds":        {"ticker":"BND",  "name":"Vanguard Total Bond",         "alloc": profile["bonds"]*0.60,  "expense":0.03},
        "Intl_Bonds":      {"ticker":"BNDX", "name":"Vanguard Intl Bond",          "alloc": profile["bonds"]*0.20,  "expense":0.07},
        "TIPS":            {"ticker":"VTIP", "name":"Vanguard Short-Term TIPS",    "alloc": profile["bonds"]*0.20,  "expense":0.04},
        "Real_Estate":     {"ticker":"VNQ",  "name":"Vanguard Real Estate",        "alloc": profile["alternatives"]*0.40,"expense":0.12},
        "Commodities":     {"ticker":"PDBC", "name":"Invesco Commodities",         "alloc": profile["alternatives"]*0.30,"expense":0.59},
        "Gold":            {"ticker":"GLD",  "name":"SPDR Gold Shares",            "alloc": profile["alternatives"]*0.30,"expense":0.40},
    }

    total_alloc = sum(e["alloc"] for e in etfs.values())
    for e in etfs.values():
        e["alloc"] = round(e["alloc"]/total_alloc*100, 1)
        e["monthly_invest"] = round(monthly_invest * e["alloc"]/100, 0) if monthly_invest else 0

    # Projections
    exp_ret = np.mean(profile["exp_return"]) / 100
    future_value = savings * (1+exp_ret)**years_to_retire
    if monthly_invest:
        future_value += monthly_invest * 12 * ((1+exp_ret)**years_to_retire - 1) / exp_ret

    return {
        "profile":          risk,
        "allocation":       profile,
        "etfs":             etfs,
        "expected_return":  profile["exp_return"],
        "max_drawdown":     profile["max_dd"],
        "years_to_retire":  years_to_retire,
        "projected_value":  round(future_value),
        "monthly_invest":   monthly_invest,
        "rebalance_trigger":"5% drift from target",
        "rebalance_schedule":"Quarterly",
        "benchmark":        "60/40 (AOM ETF)" if risk=="moderate" else "S&P 500 (SPY)",
        "tax_strategy":     "Max 401k/IRA first, then taxable. Hold ETFs >1yr for LTCG.",
        "dca_plan":         f"Invest ${monthly_invest}/mo on 1st of each month across allocations"
    }
