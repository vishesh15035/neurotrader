from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import threading, os, sys
import numpy as np
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

cache        = {}
cache_lock   = threading.Lock()
running_jobs = set()

DASHBOARD_DIR = "/Users/vishesh/Desktop/dbms/neurotrader/dashboard"

def get_basic_data(ticker):
    try:
        stock  = yf.Ticker(ticker)
        hist   = stock.history(period="6mo")
        info   = stock.info
        prices = hist['Close'].values
        ret    = np.diff(prices)/prices[:-1]
        price  = float(prices[-1])
        change = float(prices[-1]-prices[-2])
        pct    = float(change/prices[-2]*100)
        volumes = hist['Volume'].values.tolist()
        price_history = [{"t":i,"price":round(float(p),2),"day":str(hist.index[i])[:10]} for i,p in enumerate(prices)]
        vol_series = []
        v = float(np.var(ret))
        for r in ret:
            v = 1e-6 + 0.05*r**2 + 0.90*v
            vol_series.append(round(float(np.sqrt(v*252))*100, 2))
        gains  = np.where(ret>0, ret, 0)
        losses = np.where(ret<0, -ret, 0)
        ag = np.convolve(gains,  np.ones(14)/14, 'valid')
        al = np.convolve(losses, np.ones(14)/14, 'valid')
        rsi = float(100 - 100/(1 + ag[-1]/(al[-1]+1e-10)))
        return {
            "ticker":      ticker,
            "price":       round(price, 2),
            "change":      round(change, 2),
            "changePct":   round(pct, 2),
            "volume":      int(volumes[-1]) if volumes else 0,
            "pe":          round(float(info.get("trailingPE", 0) or 0), 2),
            "rsi":         round(rsi, 2),
            "priceHistory":price_history[-120:],
            "volSeries":   [{"t":i,"vol":v} for i,v in enumerate(vol_series[-120:])],
            "returns":     [round(float(r),6) for r in ret[-252:].tolist()],
            "status":      "basic"
        }
    except Exception as e:
        return {"error":str(e),"ticker":ticker,"status":"error"}

def run_quant_engine(ticker):
    global cache, running_jobs
    if ticker in running_jobs: return
    running_jobs.add(ticker)
    try:
        print(f"[API] Starting QuantEngine for {ticker}...")
        from models.quant_engine import QuantEngine
        engine = QuantEngine()
        result = engine.run(ticker)
        s = result["summary"]
        d = result["details"]
        hist   = yf.Ticker(ticker).history(period="2y")
        prices = hist['Close'].values
        ret    = np.diff(prices)/prices[:-1]
        S0, mu, sigma = float(prices[-1]), ret.mean()*252, ret.std()*np.sqrt(252)
        mc_paths = []
        for step in range(31):
            obj = {"day":step,
                   "mean":round(S0*np.exp(mu*step/252),2),
                   "p95": round(S0*np.exp(mu*step/252+1.645*sigma*np.sqrt(max(step,1)/252)),2),
                   "p5":  round(S0*np.exp(mu*step/252-1.645*sigma*np.sqrt(max(step,1)/252)),2)}
            for p in range(5):
                np.random.seed(p*7+step)
                obj[f"p{p}"] = round(S0*np.exp(mu*step/252+np.random.randn()*sigma*np.sqrt(max(step,1)/252)),2)
            mc_paths.append(obj)
        rp = d.get("hmm",{}).get("regime_probs",{})
        regime_probs = [{"name":k,"value":round(v*100,1)} for k,v in rp.items()] if rp else [
            {"name":"BULL","value":33},{"name":"BEAR","value":33},{"name":"SIDEWAYS","value":34}]
        opts  = d.get("options",{})
        micro = d.get("micro",{})
        sent  = d.get("sentiment",{})
        full  = {
            "ticker":      s["ticker"], "price":s["price"],
            "decision":    s["final_decision"], "score":s["raw_score"],
            "confidence":  s["confidence"], "votes":s["votes"],
            "signals":     s["signals"], "elapsed":s["elapsed_sec"],
            "totalModels": s["total_models"],
            "var95":       s.get("var95",0),
            "garchVol":    d.get("garch",{}).get("forecast_vol", s.get("garch_vol",0)),
            "garchPersistence": d.get("garch",{}).get("persistence",0),
            "hestonRho":   d.get("heston",{}).get("rho", s.get("heston_rho",0)),
            "ouHalfLife":  d.get("ou",{}).get("half_life", s.get("ou_halflife",0)),
            "ivSkew":      opts.get("iv_skew",0),
            "vwapDev":     micro.get("vwap_dev_pct",0),
            "pcRatio":     opts.get("pc_ratio_vol",0),
            "hmm":         s.get("hmm_regime","N/A"),
            "macro":       s.get("macro_regime","N/A"),
            "copula": {
                "crash":   d.get("copula",{}).get("clayton",{}).get("lower_tail_dep",0),
                "rally":   d.get("copula",{}).get("gumbel",{}).get("upper_tail_dep",0),
                "kendall": d.get("copula",{}).get("clayton",{}).get("kendall_tau",0),
            },
            "regimeProbabilities": regime_probs,
            "monteCarloPaths":     mc_paths,
            "mcRisk":              d.get("monte_carlo",{}).get("risk",{}),
            "layers": [
                {"name":"Deep Learning", "score":round(float(np.mean([{"BUY":1,"SELL":-1,"HOLD":0}.get(s["signals"].get(m,"HOLD"),0) for m in ["lstm","transformer","gp"]])),3)},
                {"name":"State Est.",    "score":round(float(np.mean([{"BUY":1,"SELL":-1,"HOLD":0}.get(s["signals"].get(m,"HOLD"),0) for m in ["kalman","particle","hmm"]])),3)},
                {"name":"Volatility",    "score":round(float(np.mean([{"BUY":1,"SELL":-1,"HOLD":0}.get(s["signals"].get(m,"HOLD"),0) for m in ["garch","egarch","heston"]])),3)},
                {"name":"Time Series",   "score":round(float(np.mean([{"BUY":1,"SELL":-1,"HOLD":0}.get(s["signals"].get(m,"HOLD"),0) for m in ["ou","ms_arima","copula"]])),3)},
                {"name":"Signal Src.",   "score":round(float(np.mean([{"BUY":1,"SELL":-1,"HOLD":0}.get(s["signals"].get(m,"HOLD"),0) for m in ["options","cross_asset","micro","sentiment"]])),3)},
                {"name":"Ensemble",      "score":round(s["raw_score"],3)},
            ],
            "sentiment": {
                "analystUpside":  sent.get("analyst_upside",0),
                "shortPct":       sent.get("short_pct",0),
                "rsiDivergence":  sent.get("rsi_divergence","NONE"),
                "compositeScore": sent.get("composite_score",0),
            },
            "status": "full"
        }
        with cache_lock:
            cache[ticker] = {**cache.get(ticker,{}), **full}
        print(f"[API] Done: {ticker} in {s['elapsed_sec']}s")
    except Exception as e:
        import traceback; traceback.print_exc()
    finally:
        running_jobs.discard(ticker)

@app.route("/")
@app.route("/dashboard")
def dashboard():
    return send_from_directory(DASHBOARD_DIR, "index.html")

@app.route("/api/health")
def health():
    return jsonify({"status":"ok","cached":list(cache.keys()),"running":list(running_jobs)})

@app.route("/api/ticker/<ticker>")
def get_ticker(ticker):
    ticker = ticker.upper()
    with cache_lock:
        cached = cache.get(ticker)
    if cached and cached.get("status")=="full":
        return jsonify(cached)
    if ticker not in running_jobs:
        threading.Thread(target=run_quant_engine, args=(ticker,), daemon=True).start()
    basic = get_basic_data(ticker)
    if cached:
        basic.update({k:v for k,v in cached.items() if k not in basic})
    return jsonify(basic)

@app.route("/api/ticker/<ticker>/refresh")
def refresh_ticker(ticker):
    ticker = ticker.upper()
    with cache_lock: cache.pop(ticker, None)
    threading.Thread(target=run_quant_engine, args=(ticker,), daemon=True).start()
    return jsonify({"status":"refresh_started","ticker":ticker})

@app.route("/api/scan")
def scan():
    tickers = request.args.get("tickers","AAPL,MSFT,NVDA,GOOGL,SPY").split(",")
    results = {}
    for t in tickers:
        t = t.upper().strip()
        with cache_lock: cached = cache.get(t)
        if cached:
            results[t] = {"ticker":t,"price":cached.get("price",0),
                          "changePct":cached.get("changePct",0),
                          "decision":cached.get("decision","N/A"),"status":cached.get("status")}
        else:
            results[t] = get_basic_data(t)
            if t not in running_jobs:
                threading.Thread(target=run_quant_engine, args=(t,), daemon=True).start()
    return jsonify(results)

if __name__ == "__main__":
    print("\n[NeuroTrader API v4.0] http://127.0.0.1:5000")
    print("[Dashboard] http://127.0.0.1:5000/dashboard\n")
    app.run(port=5000, threaded=True, debug=False)

# ── Indian Stock Helper ──────────────────────────────────
INDIAN_TICKERS = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","WIPRO.NS",
    "ICICIBANK.NS","KOTAKBANK.NS","SBIN.NS","BAJFINANCE.NS","BHARTIARTL.NS",
    "HINDUNILVR.NS","ASIANPAINT.NS","MARUTI.NS","TATAMOTORS.NS","HCLTECH.NS",
    "SUNPHARMA.NS","ONGC.NS","POWERGRID.NS","NTPC.NS","ADANIPORTS.NS",
    "ULTRACEMCO.NS","TITAN.NS","NESTLEIND.NS","TECHM.NS","DRREDDY.NS",
    "^NSEI","INDA","INDY"
]

@app.route("/api/scan/india")
def scan_india():
    results = {}
    for t in INDIAN_TICKERS[:10]:  # top 10 to avoid timeout
        with cache_lock:
            cached = cache.get(t)
        if cached:
            results[t] = {"ticker":t,"price":cached.get("price",0),
                          "changePct":cached.get("changePct",0),
                          "decision":cached.get("decision","N/A"),"status":cached.get("status")}
        else:
            results[t] = get_basic_data(t)
    return jsonify(results)

@app.route("/api/india/tickers")
def india_tickers():
    return jsonify({"tickers": INDIAN_TICKERS})
