"""
NeuroTrader — Core ReAct Agent
Reason → Act → Observe loop using Claude API
Full agentic loop with tool use, RAG, and memory
"""
import anthropic
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.trading_tools import TOOLS, dispatch, search_knowledge_base
from memory.memory_system import ShortTermMemory, LongTermMemory
from rag.knowledge_base   import RAGKnowledgeBase

SYSTEM_PROMPT = """You are NeuroTrader, an elite autonomous trading agent powered by advanced AI.

Your decision framework:
1. ALWAYS search the knowledge base first for relevant strategies
2. ALWAYS fetch current price data before any analysis
3. Compute technical signals to understand market conditions
4. Check fundamentals for medium/long-term decisions
5. Review portfolio status before executing trades
6. Execute trades only when conviction is HIGH (multiple confirming signals)
7. Log your reasoning clearly for every decision

Your personality:
- Methodical and data-driven — never guess
- Risk-aware — never risk more than 5% of portfolio on one trade
- Patient — HOLD is often the best action
- Transparent — always explain your reasoning step by step

Available tools: get_price_data, get_fundamentals, compute_technical_signals,
fetch_market_news, run_backtest, get_portfolio_status, execute_trade, search_knowledge_base

ReAct loop: Think → Use tool → Observe result → Think again → Decide
"""

class NeuroTrader:
    def __init__(self, api_key: str = None):
        self.client   = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.stm      = ShortTermMemory(max_size=20)
        self.ltm      = LongTermMemory("data/memory.db")
        self.rag      = RAGKnowledgeBase("data/chromadb")
        self.model    = "claude-sonnet-4-5"
        self.step     = 0
        self._patch_rag()

    def _patch_rag(self):
        """Override search_knowledge_base tool with real RAG"""
        import tools.trading_tools as tt
        rag = self.rag
        def real_search(query: str, n_results: int = 3) -> dict:
            results = rag.search(query, n_results)
            return {"query": query, "results": results, "total_docs": rag.count()}
        tt.TOOL_MAP["search_knowledge_base"] = real_search

    def run(self, task: str, max_steps: int = 10) -> str:
        """
        Main ReAct loop:
        1. Add task to short-term memory
        2. Send to Claude with tools
        3. Execute tool calls
        4. Feed results back
        5. Repeat until final answer
        """
        print(f"\n{'='*60}")
        print(f"  NeuroTrader — Task: {task}")
        print(f"{'='*60}\n")

        # Add past context from long-term memory
        past = self.ltm.get_past_decisions(limit=3)
        context = ""
        if past:
            context = "\nRecent decisions from memory:\n"
            for d in past:
                context += f"  [{d['timestamp'][:10]}] {d['action']} {d['ticker']} @ ${d['price']} — {d['reasoning'][:80]}\n"

        self.stm.add("user", task + context)
        messages = self.stm.get_context()

        self.step = 0
        while self.step < max_steps:
            self.step += 1
            print(f"[Step {self.step}] Calling Claude...")

            response = self.client.messages.create(
                model      = self.model,
                max_tokens = 4096,
                system     = SYSTEM_PROMPT,
                tools      = TOOLS,
                messages   = messages
            )

            # Extract text reasoning
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\n[Claude Reasoning]\n{block.text}\n")
                    self.stm.add("assistant", block.text)

            # Check stop condition
            if response.stop_reason == "end_turn":
                final = next((b.text for b in response.content if hasattr(b,"text")), "Done.")
                print(f"\n{'='*60}")
                print(f"  FINAL ANSWER")
                print(f"{'='*60}")
                print(final)
                return final

            # Execute tool calls
            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name  = block.name
                        tool_input = block.input
                        print(f"[Tool] {tool_name}({json.dumps(tool_input)[:80]}...)")

                        result_str = dispatch(tool_name, tool_input)
                        result_obj = json.loads(result_str)

                        # Pretty print result
                        print(f"[Result] {json.dumps(result_obj, default=str)[:200]}\n")

                        # Log trade decisions to long-term memory
                        if tool_name == "execute_trade" and result_obj.get("status") == "EXECUTED":
                            self.ltm.log_decision(
                                ticker     = tool_input["ticker"],
                                action     = tool_input["action"],
                                reasoning  = tool_input["reasoning"],
                                price      = result_obj.get("price", 0),
                                confidence = 0.7
                            )
                            # Add insight to RAG
                            self.rag.add_insight(
                                f"trade_{self.step}_{tool_input['ticker']}",
                                f"Executed {tool_input['action']} {tool_input['ticker']}: {tool_input['reasoning']}",
                                ["trade", "decision", tool_input['ticker']]
                            )

                        tool_results.append({
                            "type":        "tool_result",
                            "tool_use_id": block.id,
                            "content":     result_str
                        })

                # Add assistant response + tool results to message history
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user",      "content": tool_results})
                self.stm.add("assistant", f"Used tools: {[b.name for b in response.content if b.type=='tool_use']}")

        return "Max steps reached."

    def analyze(self, ticker: str) -> str:
        return self.run(f"Perform a complete analysis of {ticker}. Check price data, technical signals, fundamentals, recent news, and run a backtest. Then give a clear BUY/SELL/HOLD recommendation with confidence level and reasoning.")

    def trade(self, ticker: str) -> str:
        return self.run(f"Analyze {ticker} and make a trading decision. Search knowledge base for relevant strategies, analyze all signals, check portfolio status, then execute a trade if conviction is high enough. Start with $1000 max position size.")

    def portfolio_review(self) -> str:
        return self.run("Review the entire portfolio. Check current holdings, compute P&L, analyze each position's current signals, and recommend any adjustments needed.")

    def market_scan(self, tickers: list) -> str:
        tickers_str = ", ".join(tickers)
        return self.run(f"Scan these tickers for the best opportunity: {tickers_str}. Quickly analyze each one and rank them by opportunity. Then execute a trade on the best one if signals are strong.")
