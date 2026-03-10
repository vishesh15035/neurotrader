"""
DualAgent — Gemini vs Ollama comparison engine
Both run same task → Ensemble predictor → Final decision
"""
from google import genai
from google.genai import types
import json, os, sys, requests, time, threading, re
from dataclasses import dataclass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.trading_tools import dispatch
from memory.memory_system import ShortTermMemory, LongTermMemory
from rag.knowledge_base   import RAGKnowledgeBase

SYSTEM_PROMPT = """You are NeuroTrader, an elite autonomous AI trading agent.
Steps:
1. Search knowledge base for relevant strategies
2. Get current price data
3. Compute technical signals (RSI, MACD, BOLLINGER, MOMENTUM)
4. Check fundamentals
5. Give BUY/SELL/HOLD with confidence % and target price
Be concise and data-driven."""

TOOLS_SCHEMA = [
    {"name":"get_price_data","description":"Fetch real-time price data","parameters":{"type":"object","properties":{"ticker":{"type":"string"},"period":{"type":"string","default":"3mo"}},"required":["ticker"]}},
    {"name":"get_fundamentals","description":"Get P/E, market cap, EPS, analyst rating","parameters":{"type":"object","properties":{"ticker":{"type":"string"}},"required":["ticker"]}},
    {"name":"compute_technical_signals","description":"Compute RSI,MACD,BOLLINGER,MOMENTUM signals","parameters":{"type":"object","properties":{"ticker":{"type":"string"},"indicators":{"type":"array","items":{"type":"string","enum":["RSI","MACD","BOLLINGER","MOMENTUM","VOLUME"]}}},"required":["ticker","indicators"]}},
    {"name":"run_backtest","description":"Backtest strategy, returns Sharpe, return, drawdown","parameters":{"type":"object","properties":{"ticker":{"type":"string"},"strategy":{"type":"string","enum":["momentum","mean_reversion","rsi","macd_crossover"]},"period":{"type":"string","default":"1y"}},"required":["ticker","strategy"]}},
    {"name":"fetch_market_news","description":"Get recent news with sentiment","parameters":{"type":"object","properties":{"ticker":{"type":"string"},"limit":{"type":"integer","default":3}},"required":["ticker"]}},
    {"name":"search_knowledge_base","description":"Search RAG for trading strategies","parameters":{"type":"object","properties":{"query":{"type":"string"},"n_results":{"type":"integer","default":2}},"required":["query"]}},
    {"name":"get_portfolio_status","description":"Get portfolio holdings and P&L","parameters":{"type":"object","properties":{},"required":[]}},
    {"name":"execute_trade","description":"Execute BUY/SELL trade","parameters":{"type":"object","properties":{"ticker":{"type":"string"},"action":{"type":"string","enum":["BUY","SELL","HOLD"]},"quantity":{"type":"integer"},"reasoning":{"type":"string"}},"required":["ticker","action","quantity","reasoning"]}}
]

@dataclass
class ModelResult:
    model:      str
    decision:   str
    confidence: float
    reasoning:  str
    signals:    dict
    latency_ms: float
    steps:      int

def parse_decision(text: str):
    text_upper = text.upper()
    decision   = "HOLD"
    if "STRONG BUY" in text_upper: decision = "STRONG BUY"
    elif "BUY"  in text_upper:     decision = "BUY"
    elif "SELL" in text_upper:     decision = "SELL"
    conf    = 0.5
    matches = re.findall(r'(\d+)\s*%', text)
    if matches:
        nums = [int(m) for m in matches if 50 <= int(m) <= 100]
        if nums: conf = max(nums) / 100
    return decision, conf

# ── Gemini Agent ────────────────────────────────────────
class GeminiAgent:
    def __init__(self, api_key: str, rag):
        self.client = genai.Client(api_key=api_key)
        self.rag    = rag
        self._patch_rag()
        self.tools  = [types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name        = t["name"],
                description = t["description"],
                parameters  = types.Schema(
                    type       = types.Type.OBJECT,
                    properties = {
                        k: types.Schema(
                            type        = types.Type.STRING if v.get("type") in ["string","array"]
                                          else types.Type.INTEGER,
                            description = v.get("description",""),
                            enum        = v.get("enum",[]) if v.get("enum") else [],
                        )
                        for k,v in t["parameters"]["properties"].items()
                    },
                    required = t["parameters"].get("required",[])
                )
            ) for t in TOOLS_SCHEMA
        ])]

    def _patch_rag(self):
        import tools.trading_tools as tt
        rag = self.rag
        def real_search(query: str, n_results: int = 2) -> dict:
            return {"query": query, "results": rag.search(query, n_results)}
        tt.TOOL_MAP["search_knowledge_base"] = real_search

    def run(self, task: str, max_steps: int = 10) -> ModelResult:
        t0       = time.time()
        messages = [types.Content(role="user", parts=[types.Part(text=task)])]
        steps    = 0
        reasoning_parts = []

        while steps < max_steps:
            steps += 1
            resp = self.client.models.generate_content(
                model    = "gemini-1.5-flash",
                contents = messages,
                config   = types.GenerateContentConfig(
                    system_instruction = SYSTEM_PROMPT,
                    tools              = self.tools,
                )
            )

            fn_calls = []
            for part in resp.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    reasoning_parts.append(part.text)
                if hasattr(part, "function_call") and part.function_call:
                    fn_calls.append(part.function_call)

            messages.append(resp.candidates[0].content)

            if not fn_calls:
                break

            tool_parts = []
            for fc in fn_calls:
                result = dispatch(fc.name, dict(fc.args))
                tool_parts.append(types.Part(
                    function_response=types.FunctionResponse(
                        name     = fc.name,
                        response = {"result": json.loads(result)}
                    )
                ))
            messages.append(types.Content(role="user", parts=tool_parts))

        full_text          = " ".join(reasoning_parts)
        decision, conf     = parse_decision(full_text)
        return ModelResult("Gemini-2.0-Flash", decision, conf,
                           full_text[:1000], {}, round((time.time()-t0)*1000,1), steps)

# ── Ollama Agent ────────────────────────────────────────
class OllamaAgent:
    def __init__(self, model: str = "llama3.1"):
        self.model    = model
        self.base_url = "http://localhost:11434"

    def run(self, task: str, max_steps: int = 10) -> ModelResult:
        t0       = time.time()
        messages = [
            {"role":"system",  "content": SYSTEM_PROMPT},
            {"role":"user",    "content": task}
        ]
        tools = [{"type":"function","function":{
            "name":        t["name"],
            "description": t["description"],
            "parameters":  t["parameters"]
        }} for t in TOOLS_SCHEMA]
        steps           = 0
        reasoning_parts = []

        while steps < max_steps:
            steps += 1
            try:
                resp = requests.post(f"{self.base_url}/api/chat",
                    json={"model":self.model,"messages":messages,"tools":tools,"stream":False},
                    timeout=180).json()
            except Exception as e:
                return ModelResult(f"Ollama-{self.model}","ERROR",0.0,str(e),{},0.0,steps)

            msg        = resp.get("message",{})
            content    = msg.get("content","")
            tool_calls = msg.get("tool_calls",[])
            if content: reasoning_parts.append(content)
            if not tool_calls: break

            messages.append({"role":"assistant","content":content,"tool_calls":tool_calls})
            for tc in tool_calls:
                fn   = tc.get("function",{})
                name = fn.get("name","")
                args = fn.get("arguments",{})
                if isinstance(args,str):
                    try: args=json.loads(args)
                    except: args={}
                result = dispatch(name, args)
                messages.append({"role":"tool","content":result,"name":name})

        full_text      = " ".join(reasoning_parts)
        decision, conf = parse_decision(full_text)
        return ModelResult(f"Ollama-{self.model}", decision, conf,
                           full_text[:1000], {}, round((time.time()-t0)*1000,1), steps)

# ── Ensemble Predictor ──────────────────────────────────
class EnsemblePredictor:
    SCORE = {"STRONG BUY":2,"BUY":1,"HOLD":0,"SELL":-1,"ERROR":0}

    def predict(self, g: ModelResult, o: ModelResult) -> dict:
        gs = self.SCORE.get(g.decision,0)
        os_ = self.SCORE.get(o.decision,0)
        total = g.confidence + o.confidence or 1
        gw, ow = g.confidence/total, o.confidence/total
        score  = gs*gw + os_*ow
        if   score >  1.2: final = "STRONG BUY"
        elif score >  0.4: final = "BUY"
        elif score < -0.4: final = "SELL"
        else:              final = "HOLD"
        conf      = (g.confidence*gw + o.confidence*ow)
        agreement = g.decision == o.decision
        rec = (f"✅ HIGH CONVICTION {final} — both models agree ({conf*100:.0f}%)" if agreement and conf>0.75
               else f"⚠️  Models DISAGREE — wait for clearer signal" if not agreement
               else f"📊 MODERATE {final} — size carefully")
        return {"final_decision":final,"ensemble_score":round(score,3),
                "confidence":round(conf*100,1),"agreement":agreement,
                "gemini_vote":g.decision,"ollama_vote":o.decision,
                "gemini_conf":round(g.confidence*100,1),"ollama_conf":round(o.confidence*100,1),
                "gemini_latency":g.latency_ms,"ollama_latency":o.latency_ms,
                "recommendation":rec}

# ── DualAgent ───────────────────────────────────────────
class DualAgent:
    def __init__(self):
        key = os.environ.get("GEMINI_API_KEY")
        if not key: raise ValueError("Set GEMINI_API_KEY")
        self.rag      = RAGKnowledgeBase("data/chromadb")
        self.ltm      = LongTermMemory("data/memory.db")
        self.gemini   = GeminiAgent(key, self.rag)
        self.ollama   = OllamaAgent("llama3.1")
        self.ensemble = EnsemblePredictor()

    def compare(self, task: str, ticker: str = "") -> dict:
        print(f"\n{'='*65}")
        print(f"  DualAgent: {ticker or task[:40]}")
        print(f"{'='*65}")
        g_res, o_res = [None], [None]

        def run_g(): print("[Gemini] Starting..."); g_res[0]=self.gemini.run(task); print(f"[Gemini] Done {g_res[0].latency_ms}ms")
        def run_o(): print("[Ollama] Starting..."); o_res[0]=self.ollama.run(task); print(f"[Ollama] Done {o_res[0].latency_ms}ms")

        
        t2=threading.Thread(target=run_o)
        t1.start(); t2.start()
        t1.join();  t2.join()

        g, o  = g_res[0], o_res[0]
        pred  = self.ensemble.predict(g, o)

        print(f"""
╔══════════════════════════════════════════════════════════════╗
║              MODEL COMPARISON RESULTS                        ║
╠══════════════╦═══════════════════════╦══════════════════════╣
║  Metric      ║  Gemini-2.0-Flash     ║  Ollama-llama3.1     ║
╠══════════════╬═══════════════════════╬══════════════════════╣
║  Decision    ║  {g.decision:<21} ║  {o.decision:<20} ║
║  Confidence  ║  {g.confidence*100:>5.1f}%               ║  {o.confidence*100:>5.1f}%              ║
║  Latency     ║  {g.latency_ms:>7.0f}ms             ║  {o.latency_ms:>7.0f}ms            ║
║  Steps       ║  {g.steps:<21} ║  {o.steps:<20} ║
╠══════════════╩═══════════════════════╩══════════════════════╣
║  ENSEMBLE: {pred['final_decision']:<10} Score={pred['ensemble_score']:>6.3f} Conf={pred['confidence']:>5.1f}%     ║
╠══════════════════════════════════════════════════════════════╣
║  {pred['recommendation']:<62}║
╚══════════════════════════════════════════════════════════════╝
""")
        print("── Gemini Reasoning ──")
        print(g.reasoning[:400])
        print("\n── Ollama Reasoning ──")
        print(o.reasoning[:400])

        if ticker:
            self.ltm.log_decision(ticker=ticker, action=pred["final_decision"],
                reasoning=f"Gemini={g.decision} Ollama={o.decision}",
                price=0, confidence=pred["confidence"]/100)
        return pred

    def analyze(self, ticker: str) -> dict:
        return self.compare(f"""Analyze {ticker}:
1. search_knowledge_base for momentum strategies
2. get_price_data for {ticker}
3. compute_technical_signals RSI MACD BOLLINGER MOMENTUM
4. get_fundamentals for {ticker}
5. run_backtest momentum 1y
6. Final BUY/SELL/HOLD with confidence %""", ticker)

    def scan(self, tickers: list) -> dict:
        results = {}
        for t in tickers:
            results[t] = self.analyze(t)
            time.sleep(3)
        ranked = sorted(results.items(), key=lambda x: x[1]["ensemble_score"], reverse=True)
        print("\n── SCAN RANKING ──")
        for i,(t,r) in enumerate(ranked,1):
            print(f"  {i}. {t:<6} {r['final_decision']:<12} Score={r['ensemble_score']:>6.3f}")
        return results
