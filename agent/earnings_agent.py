"""
EarningsAgent — Brief-style pre-earnings research brief
PortfolioAgent — Build portfolio-style portfolio builder
Both use Ollama llama3.1 for AI reasoning
"""
import requests, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.earnings_tools import (get_earnings_history, get_segment_data,
                                   get_options_implied_move, build_portfolio)
from tools.trading_tools  import get_price_data, compute_technical_signals, get_fundamentals

class EarningsAgent:
    def __init__(self, model: str = "llama3.1"):
        self.model    = model
        self.base_url = "http://localhost:11434"

    def _call_ollama(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={"model":self.model,
                  "messages":[{"role":"user","content":prompt}],
                  "stream":False},
            timeout=180
        ).json()
        return resp.get("message",{}).get("content","")

    def analyze(self, ticker: str, earnings_date: str = "upcoming") -> str:
        print(f"\n[EarningsAgent] Fetching data for {ticker}...")

        # Fetch all real data
        earnings  = get_earnings_history(ticker)
        segments  = get_segment_data(ticker)
        options   = get_options_implied_move(ticker)
        price     = get_price_data(ticker, "6mo")
        signals   = compute_technical_signals(ticker, ["RSI","MACD","BOLLINGER","MOMENTUM"])
        funds     = get_fundamentals(ticker)

        print(f"[EarningsAgent] Data fetched. Generating Brief-style brief...")

        prompt = f"""You are a senior equity research analyst.
Write a complete pre-earnings research brief for institutional investors.

REAL MARKET DATA:
Company: {earnings.get('company',ticker)} ({ticker})
Sector: {earnings.get('sector','')}
Current Price: ${price.get('current_price',0)}
Market Cap: ${earnings.get('market_cap',0):,}

EARNINGS HISTORY (Last 4 quarters):
{json.dumps(earnings.get('quarters',[]), indent=2)}
Beat Rate: {earnings.get('beat_rate','N/A')}
Avg Surprise: {earnings.get('avg_surprise','N/A')}%
Post-Earnings Stock Moves: {earnings.get('post_earnings_moves',[])}

CONSENSUS ESTIMATES:
Forward EPS: ${earnings.get('eps_fwd',0)}
Forward P/E: {earnings.get('forward_pe',0)}x
Analyst Target: ${earnings.get('analyst_target',0)}
Recommendation: {earnings.get('recommendation','').upper()}
# Analysts: {earnings.get('num_analysts',0)}

FINANCIALS:
Revenue Growth: {segments.get('revenue_growth',0)*100:.1f}%
Gross Margin: {segments.get('gross_margin',0)*100:.1f}%
Operating Margin: {segments.get('operating_margin',0)*100:.1f}%
Net Margin: {segments.get('profit_margin',0)*100:.1f}%
Free Cash Flow: ${segments.get('free_cashflow',0):,}
Quarterly Revenues: {json.dumps(segments.get('revenues',{}), indent=2)}

TECHNICAL SIGNALS:
RSI: {signals.get('signals',{}).get('RSI',{}).get('value','N/A')} ({signals.get('signals',{}).get('RSI',{}).get('signal','N/A')})
MACD: {signals.get('signals',{}).get('MACD',{}).get('signal','N/A')}
Bollinger: {signals.get('signals',{}).get('BOLLINGER',{}).get('signal','N/A')}
Momentum: {signals.get('signals',{}).get('MOMENTUM',{}).get('value','N/A')}%
Overall Technical: {signals.get('overall','N/A')} (Strength: {signals.get('strength',0)}%)

OPTIONS MARKET:
Implied Move: ±{options.get('implied_move_pct','N/A')}%
Straddle Cost: ${options.get('straddle_cost','N/A')}
Bull Target: ${options.get('bull_target','N/A')}
Bear Target: ${options.get('bear_target','N/A')}
Next Expiration: {options.get('expiration','N/A')}

Write the following sections:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 DECISION SUMMARY (top, 3 lines max)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. EARNINGS BEAT/MISS HISTORY (last 4 quarters table)
2. CONSENSUS ESTIMATES (EPS, Revenue for upcoming quarter)
3. KEY METRICS WALL STREET IS WATCHING
4. SEGMENT REVENUE BREAKDOWN & TRENDS
5. MANAGEMENT GUIDANCE SUMMARY (based on known data)
6. OPTIONS IMPLIED MOVE ANALYSIS
7. HISTORICAL STOCK REACTIONS AFTER EARNINGS
8. BULL CASE (scenario + price target + % upside)
9. BEAR CASE (scenario + price target + % downside)
10. RECOMMENDED PLAY: Buy before / Sell before / Wait
    With specific reasoning and risk/reward ratio

Format professionally. Use numbers everywhere. Be specific."""

        brief = self._call_ollama(prompt)

        print("\n" + "="*65)
        print(f"  Brief-Style Pre-Earnings Brief: {ticker}")
        print("="*65)
        print(brief)
        print("="*65)

        # Save to file
        fname = f"data/{ticker}_earnings_brief.txt"
        os.makedirs("data", exist_ok=True)
        with open(fname,"w") as f:
            f.write(f"PRE-EARNINGS BRIEF: {ticker}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(brief)
        print(f"\n[Saved] {fname}")
        return brief


class PortfolioAgent:
    def __init__(self, model: str = "llama3.1"):
        self.model    = model
        self.base_url = "http://localhost:11434"

    def _call_ollama(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={"model":self.model,
                  "messages":[{"role":"user","content":prompt}],
                  "stream":False},
            timeout=300
        ).json()
        return resp.get("message",{}).get("content","")

    def build(self, age: int, income: int, savings: int,
              risk: str, goal: str, monthly: int = 0,
              account_type: str = "taxable") -> str:

        print(f"\n[PortfolioAgent] Building portfolio-style portfolio...")
        portfolio = build_portfolio(age, income, savings, risk, goal, monthly)

        prompt = f"""You are a senior portfolio strategist managing $500M+ for institutional clients.
Build a complete investment portfolio for this client.

CLIENT PROFILE:
Age: {age} | Income: ${income:,}/yr | Savings: ${savings:,}
Risk Tolerance: {risk.upper()} | Goal: {goal}
Monthly Investment: ${monthly:,} | Account Type: {account_type}
Years to Retirement: {portfolio['years_to_retire']}

QUANTITATIVE ANALYSIS (already computed):
Risk Profile: {json.dumps(portfolio['allocation'], indent=2)}
Expected Return: {portfolio['expected_return'][0]}-{portfolio['expected_return'][1]}% annually
Max Drawdown: {portfolio['max_drawdown']}% in bad year
Projected Portfolio Value in {portfolio['years_to_retire']} years: ${portfolio['projected_value']:,}

ETF ALLOCATION:
{json.dumps({k: {'ticker':v['ticker'],'name':v['name'],'allocation':f"{v['alloc']}%",'monthly':f"${v['monthly_invest']}",'expense':f"{v['expense']}%"} for k,v in portfolio['etfs'].items()}, indent=2)}

Write a complete professional investment policy document:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 PORTFOLIO SUMMARY (top — key numbers only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ASSET ALLOCATION (exact % with pie chart description)
   Stocks {portfolio['allocation']['stocks']*100:.0f}% | Bonds {portfolio['allocation']['bonds']*100:.0f}% | Alternatives {portfolio['allocation']['alternatives']*100:.0f}%

2. SPECIFIC ETF RECOMMENDATIONS (ticker, name, %, why)
   Label each as CORE (>10%) or SATELLITE (<10%)

3. EXPECTED PERFORMANCE
   - Annual return range: {portfolio['expected_return'][0]}-{portfolio['expected_return'][1]}%
   - Max drawdown: {portfolio['max_drawdown']}%
   - Projected value in {portfolio['years_to_retire']} yrs: ${portfolio['projected_value']:,}

4. DOLLAR COST AVERAGING PLAN
   ${monthly}/month breakdown by ETF

5. REBALANCING RULES
   Schedule + trigger thresholds

6. TAX EFFICIENCY STRATEGY
   For {account_type} account

7. BENCHMARK
   How to measure performance

8. INVESTMENT POLICY STATEMENT (1 page)
   Rules the client commits to follow

Use real numbers everywhere. Be specific and institutional-grade."""

        doc = self._call_ollama(prompt)

        print("\n" + "="*65)
        print("  Portfolio-Style Investment Portfolio")
        print("="*65)
        print(doc)
        print("="*65)

        # Save
        fname = f"data/portfolio_{age}_{risk}.txt"
        os.makedirs("data", exist_ok=True)
        with open(fname,"w") as f:
            f.write(f"INVESTMENT PORTFOLIO — {risk.upper()} RISK\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(doc)
        print(f"\n[Saved] {fname}")
        return doc
