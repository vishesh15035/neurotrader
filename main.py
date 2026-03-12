import os, sys
from agent.ollama_agent   import OllamaReActAgent as OllamaAgent, patch_rag
from agent.earnings_agent import EarningsAgent, PortfolioAgent
from rag.knowledge_base   import RAGKnowledgeBase
from memory.memory_system import LongTermMemory
from models.quant_engine  import QuantEngine

class NeuroTrader:
    def __init__(self):
        self.rag       = RAGKnowledgeBase("data/chromadb")
        self.ltm       = LongTermMemory("data/memory.db")
        self.agent     = OllamaAgent("llama3.1")
        self.earnings  = EarningsAgent("llama3.1")
        self.portfolio = PortfolioAgent("llama3.1")
        self.quant     = QuantEngine()
        patch_rag(self.rag)

    def analyze(self, ticker):
        task = f"get_price_data {ticker}, compute_technical_signals RSI MACD BOLLINGER MOMENTUM, get_fundamentals {ticker}, run_backtest momentum 1y, give BUY/SELL/HOLD"
        result = self.agent.run(task)
        print(f"\n  {ticker}: {result.decision} | Conf={result.confidence*100:.0f}% | {result.latency_ms:.0f}ms")
        self.ltm.log_decision(ticker=ticker, action=result.decision,
            reasoning=result.reasoning[:200], price=0, confidence=result.confidence)

    def scan(self):
        for t in ["AAPL","MSFT","NVDA","GOOGL","SPY"]:
            self.analyze(t)

def main():
    nt = NeuroTrader()
    print("""
╔══════════════════════════════════════════════════════════╗
║   NeuroTrader Pro v3.0 — Agentic Quant Trading System   ║
║   10 Models: LSTM · Transformer · Kalman · Particle     ║
║   HMM · Monte Carlo · Fourier · Wavelet · GP · GNN      ║
║   + Ollama ReAct · RAG · Memory · Earnings · Portfolio   ║
╚══════════════════════════════════════════════════════════╝

Commands:
  quant <TICKER>    — Run ALL 10 advanced models
  analyze <TICKER>  — Ollama AI ReAct analysis
  earnings <TICKER> — Brief-style earnings brief
  portfolio         — Build portfolio-style portfolio builder
  scan              — Scan AAPL MSFT NVDA GOOGL SPY
  q                 — quit
""")
    while True:
        try:
            cmd   = input("NeuroTrader> ").strip()
            if not cmd: continue
            if cmd.lower() in ["q","quit","exit"]: break
            parts = cmd.split(maxsplit=1)
            mode  = parts[0].lower()
            arg   = parts[1].strip().upper() if len(parts)>1 else ""
            if   mode=="quant"    and arg: nt.quant.run(arg)
            elif mode=="analyze"  and arg: nt.analyze(arg)
            elif mode=="earnings" and arg: nt.earnings.analyze(arg)
            elif mode=="scan":             nt.scan()
            elif mode=="portfolio":
                print("\nPortfolio Builder:")
                age     = int(input("  Age: "))
                income  = int(input("  Annual income ($): "))
                savings = int(input("  Current savings ($): "))
                risk    = input("  Risk (aggressive/moderate/conservative): ").strip()
                goal    = input("  Goal: ").strip()
                monthly = int(input("  Monthly investment ($): "))
                account = input("  Account (401k/IRA/taxable): ").strip()
                nt.portfolio.build(age, income, savings, risk, goal, monthly, account)
            else:
                print("Commands: quant AAPL | analyze AAPL | earnings AAPL | portfolio | scan")
        except KeyboardInterrupt:
            print("\nExiting..."); break
        except Exception as e:
            print(f"Error: {e}")
            import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
