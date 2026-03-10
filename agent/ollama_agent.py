"""
Ollama Agent — Manual ReAct loop
Since llama3.1 doesn't reliably call tools,
we use ReAct: parse JSON from text → execute → feed back
"""
import json, re, requests, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.trading_tools import dispatch
from dataclasses import dataclass

@dataclass
class ModelResult:
    model:      str
    decision:   str
    confidence: float
    reasoning:  str
    signals:    dict
    latency_ms: float
    steps:      int

REACT_PROMPT = """You are NeuroTrader, an AI trading agent. You must analyze stocks using tools.

To use a tool, output EXACTLY this format (nothing else on that line):
ACTION: tool_name({"param": "value"})

Available tools:
- get_price_data({"ticker": "AAPL", "period": "3mo"})
- compute_technical_signals({"ticker": "AAPL", "indicators": ["RSI","MACD","BOLLINGER","MOMENTUM"]})
- get_fundamentals({"ticker": "AAPL"})
- run_backtest({"ticker": "AAPL", "strategy": "momentum", "period": "1y"})
- fetch_market_news({"ticker": "AAPL", "limit": 3})
- search_knowledge_base({"query": "momentum strategy", "n_results": 2})
- get_portfolio_status({})

After getting tool results, continue reasoning. At the end output:
DECISION: BUY/SELL/HOLD
CONFIDENCE: 75%
REASON: brief explanation

Now analyze the following task:
"""

def extract_action(text: str):
    """Extract ACTION: tool(args) from model output"""
    pattern = r'ACTION:\s*(\w+)\s*\((\{.*?\})\)'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        name, args_str = matches[0]
        try:
            args = json.loads(args_str)
            return name, args
        except:
            pass
    return None, None

def extract_decision(text: str):
    decision   = "HOLD"
    confidence = 0.5
    d_match = re.search(r'DECISION:\s*(BUY|SELL|HOLD|STRONG BUY)', text, re.IGNORECASE)
    c_match = re.search(r'CONFIDENCE:\s*(\d+)%', text, re.IGNORECASE)
    if d_match: decision   = d_match.group(1).upper()
    if c_match: confidence = int(c_match.group(1)) / 100
    else:
        # fallback parse
        t = text.upper()
        if "STRONG BUY" in t: decision = "STRONG BUY"
        elif "BUY"  in t:     decision = "BUY"
        elif "SELL" in t:     decision = "SELL"
        nums = re.findall(r'(\d+)\s*%', text)
        if nums:
            valid = [int(n) for n in nums if 50 <= int(n) <= 100]
            if valid: confidence = max(valid)/100
    return decision, confidence

class OllamaReActAgent:
    def __init__(self, model: str = "llama3.1"):
        self.model    = model
        self.base_url = "http://localhost:11434"

    def _call(self, messages: list) -> str:
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=180
        ).json()
        return resp.get("message", {}).get("content", "")

    def run(self, task: str, max_steps: int = 8) -> ModelResult:
        t0       = time.time()
        messages = [{"role":"user", "content": REACT_PROMPT + task}]
        steps    = 0
        all_text = []
        used_tools = set()

        print(f"\n[Ollama ReAct] Starting {self.model}...")

        while steps < max_steps:
            steps += 1
            response = self._call(messages)
            all_text.append(response)
            print(f"\n[Step {steps}] {response[:300]}")

            # Extract and execute tool call
            tool_name, tool_args = extract_action(response)

            # Stop if final decision reached or no more tools
            if not tool_name or tool_name in used_tools:
                if "DECISION:" in response or steps >= 4:
                    break

            if tool_name and tool_name not in used_tools:
                used_tools.add(tool_name)
                print(f"[Tool] {tool_name}({str(tool_args)[:80]})")
                result = dispatch(tool_name, tool_args or {})
                result_obj = json.loads(result)
                print(f"[Result] {str(result_obj)[:200]}")

                # Feed result back
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user",
                    "content": f"Tool result for {tool_name}:\n{json.dumps(result_obj, default=str)[:500]}\n\nContinue analysis. Use another ACTION or give final DECISION."})
            else:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user",
                    "content": "Continue. Use more tools or give final DECISION: BUY/SELL/HOLD, CONFIDENCE: X%"})

        full_text          = "\n".join(all_text)
        decision, confidence = extract_decision(full_text)
        return ModelResult(
            model      = f"Ollama-{self.model}",
            decision   = decision,
            confidence = confidence,
            reasoning  = full_text[:1500],
            signals    = {},
            latency_ms = round((time.time()-t0)*1000, 1),
            steps      = steps
        )

def patch_rag(rag):
    """Patch the RAG search tool with real ChromaDB"""
    import tools.trading_tools as tt
    def real_search(query: str, n_results: int = 2) -> dict:
        return {"query": query, "results": rag.search(query, n_results)}
    tt.TOOL_MAP["search_knowledge_base"] = real_search
