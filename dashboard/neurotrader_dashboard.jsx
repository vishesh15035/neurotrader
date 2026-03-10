import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, Cell
} from "recharts";

const API = "http://localhost:5000/api";

const C = {
  bg:"#050508", panel:"#0a0a10", border:"#1a1a2e", border2:"#252540",
  green:"#00ff88", red:"#ff3366", yellow:"#ffcc00", blue:"#4488ff",
  purple:"#aa66ff", cyan:"#00ddff", text:"#e0e0f0", muted:"#606080", dim:"#303050",
};

const signalColor = s => s==="BUY"?C.green:s==="SELL"?C.red:C.yellow;
const decisionColor = d => (d==="BUY"||d==="STRONG BUY")?C.green:d==="SELL"?C.red:C.yellow;

// ── Hooks ──────────────────────────────────────────────
function useTickerData(ticker) {
  const [data, setData]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [fullReady, setFullReady] = useState(false);
  const pollRef = useRef(null);

  const fetchData = useCallback(async () => {
    try {
      const res  = await fetch(`${API}/ticker/${ticker}`);
      const json = await res.json();
      setData(json);
      setLoading(false);
      if (json.status === "full") {
        setFullReady(true);
        clearInterval(pollRef.current);
      }
    } catch(e) { console.error(e); }
  }, [ticker]);

  useEffect(() => {
    setLoading(true);
    setFullReady(false);
    setData(null);
    fetchData();
    // Poll every 3s until full result
    pollRef.current = setInterval(fetchData, 3000);
    return () => clearInterval(pollRef.current);
  }, [ticker, fetchData]);

  const refresh = async () => {
    setFullReady(false);
    await fetch(`${API}/ticker/${ticker}/refresh`);
    pollRef.current = setInterval(fetchData, 3000);
  };

  return { data, loading, fullReady, refresh };
}

// ── Components ─────────────────────────────────────────
function TickerTape({ scanData }) {
  const items = Object.values(scanData).map(d =>
    `${d.ticker} $${d.price} ${d.changePct>=0?"▲":"▼"}${Math.abs(d.changePct||0).toFixed(2)}%`
  );
  if (!items.length) return null;
  return (
    <div style={{ background:C.panel, borderBottom:`1px solid ${C.border}`,
      overflow:"hidden", height:28, display:"flex", alignItems:"center" }}>
      <div style={{ display:"flex", gap:48, animation:"ticker 25s linear infinite",
        whiteSpace:"nowrap", fontSize:11, fontFamily:"'JetBrains Mono',monospace" }}>
        {[...items,...items].map((t,i) => (
          <span key={i} style={{ color:t.includes("▼")?C.red:C.green }}>{t}</span>
        ))}
      </div>
    </div>
  );
}

function Spinner() {
  return (
    <div style={{ display:"flex", alignItems:"center", gap:10, color:C.muted,
      fontSize:11, fontFamily:"'JetBrains Mono',monospace" }}>
      <div style={{ width:8, height:8, borderRadius:"50%", background:C.cyan,
        animation:"pulse 0.8s infinite" }}/>
      RUNNING 19 MODELS...
    </div>
  );
}

function SectionHeader({ title, sub }) {
  return (
    <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:14 }}>
      <div style={{ width:3, height:18, background:C.cyan, borderRadius:2 }}/>
      <span style={{ fontSize:11, fontWeight:700, letterSpacing:2, color:C.cyan,
        fontFamily:"'JetBrains Mono',monospace" }}>{title}</span>
      {sub && <span style={{ fontSize:10, color:C.muted }}>{sub}</span>}
    </div>
  );
}

function StatCard({ label, value, sub, color }) {
  return (
    <div style={{ background:C.bg, border:`1px solid ${C.border}`,
      borderRadius:6, padding:"10px 14px" }}>
      <div style={{ fontSize:9, color:C.muted, letterSpacing:1.5,
        fontFamily:"'JetBrains Mono',monospace", marginBottom:4 }}>{label}</div>
      <div style={{ fontSize:18, fontWeight:700, color:color||C.text,
        fontFamily:"'JetBrains Mono',monospace" }}>{value??"-"}</div>
      {sub && <div style={{ fontSize:9, color:C.muted, marginTop:2 }}>{sub}</div>}
    </div>
  );
}

function SignalBadge({ signal }) {
  const col = signalColor(signal||"HOLD");
  return (
    <span style={{ background:col+"18", border:`1px solid ${col}44`, color:col,
      borderRadius:3, padding:"2px 6px", fontSize:10,
      fontFamily:"'JetBrains Mono',monospace", fontWeight:700, letterSpacing:1 }}>
      {signal||"HOLD"}
    </span>
  );
}

function VoteBar({ votes }) {
  if (!votes) return null;
  const total = (votes.BUY||0)+(votes.SELL||0)+(votes.HOLD||0)||1;
  return (
    <div>
      <div style={{ display:"flex", height:8, borderRadius:4, overflow:"hidden", marginBottom:8 }}>
        <div style={{ width:`${(votes.BUY||0)/total*100}%`, background:C.green }}/>
        <div style={{ width:`${(votes.HOLD||0)/total*100}%`, background:C.yellow }}/>
        <div style={{ width:`${(votes.SELL||0)/total*100}%`, background:C.red }}/>
      </div>
      <div style={{ display:"flex", justifyContent:"space-between", fontSize:11,
        fontFamily:"'JetBrains Mono',monospace" }}>
        <span style={{ color:C.green }}>▲ BUY {votes.BUY||0}</span>
        <span style={{ color:C.yellow }}>— HOLD {votes.HOLD||0}</span>
        <span style={{ color:C.red }}>▼ SELL {votes.SELL||0}</span>
      </div>
    </div>
  );
}

function PriceChart({ data, decision }) {
  if (!data?.priceHistory?.length) return null;
  const col = decisionColor(decision);
  const prices = data.priceHistory;
  const min = Math.min(...prices.map(d=>d.price));
  const max = Math.max(...prices.map(d=>d.price));
  return (
    <ResponsiveContainer width="100%" height={180}>
      <AreaChart data={prices} margin={{ top:5, right:5, bottom:0, left:0 }}>
        <defs>
          <linearGradient id="pg" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={col} stopOpacity={0.3}/>
            <stop offset="95%" stopColor={col} stopOpacity={0}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false}/>
        <XAxis dataKey="day" tick={false} axisLine={false} tickLine={false}/>
        <YAxis domain={[min*0.99,max*1.01]} tick={{ fill:C.muted, fontSize:9 }}
          axisLine={false} tickLine={false} width={50}
          tickFormatter={v=>`$${v.toFixed(0)}`}/>
        <Tooltip contentStyle={{ background:C.panel, border:`1px solid ${C.border2}`,
          borderRadius:4, fontSize:11, fontFamily:"'JetBrains Mono',monospace" }}
          formatter={v=>[`$${v}`,"Price"]}/>
        <Area type="monotone" dataKey="price" stroke={col} strokeWidth={1.5}
          fill="url(#pg)" dot={false} activeDot={{ r:3, fill:col }}/>
      </AreaChart>
    </ResponsiveContainer>
  );
}

function GarchChart({ data }) {
  if (!data?.volSeries?.length) return null;
  return (
    <ResponsiveContainer width="100%" height={120}>
      <AreaChart data={data.volSeries} margin={{ top:5, right:5, bottom:0, left:0 }}>
        <defs>
          <linearGradient id="vg" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={C.purple} stopOpacity={0.4}/>
            <stop offset="95%" stopColor={C.purple} stopOpacity={0}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false}/>
        <XAxis tick={false} axisLine={false} tickLine={false}/>
        <YAxis tick={{ fill:C.muted, fontSize:9 }} axisLine={false} tickLine={false}
          width={35} tickFormatter={v=>`${v}%`}/>
        <ReferenceLine y={30} stroke={C.red} strokeDasharray="4 4" strokeWidth={1}/>
        <Tooltip contentStyle={{ background:C.panel, border:`1px solid ${C.border2}`,
          fontSize:11 }} formatter={v=>[`${v}%`,"GARCH Vol"]}/>
        <Area type="monotone" dataKey="vol" stroke={C.purple} strokeWidth={1.5}
          fill="url(#vg)" dot={false}/>
      </AreaChart>
    </ResponsiveContainer>
  );
}

function MCChart({ paths, S0 }) {
  if (!paths?.length) return null;
  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={paths} margin={{ top:5, right:5, bottom:0, left:0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false}/>
        <XAxis tick={false} axisLine={false} tickLine={false}/>
        <YAxis tick={{ fill:C.muted, fontSize:9 }} axisLine={false} tickLine={false}
          width={50} tickFormatter={v=>`$${v.toFixed(0)}`}/>
        <Tooltip contentStyle={{ background:C.panel, border:`1px solid ${C.border2}`,
          fontSize:10 }} formatter={v=>[`$${v}`,""]}/>
        {[0,1,2,3,4].map(p=>(
          <Line key={p} type="monotone" dataKey={`p${p}`}
            stroke={["#ff336633","#ff883333","#ffcc0033","#44ff8833","#4488ff33"][p]}
            strokeWidth={1} dot={false}/>
        ))}
        <Line type="monotone" dataKey="p95" stroke={C.green} strokeWidth={1.5}
          dot={false} strokeDasharray="4 4"/>
        <Line type="monotone" dataKey="p5" stroke={C.red} strokeWidth={1.5}
          dot={false} strokeDasharray="4 4"/>
        <Line type="monotone" dataKey="mean" stroke={C.cyan} strokeWidth={2} dot={false}/>
        <ReferenceLine y={S0} stroke={C.muted} strokeDasharray="2 2"/>
      </LineChart>
    </ResponsiveContainer>
  );
}

function RegimeRadar({ data }) {
  if (!data?.length) return null;
  const radarData = data.map(d=>({ subject:d.name, A:d.value }));
  return (
    <ResponsiveContainer width="100%" height={160}>
      <RadarChart data={radarData}>
        <PolarGrid stroke={C.border}/>
        <PolarAngleAxis dataKey="subject"
          tick={{ fill:C.muted, fontSize:10, fontFamily:"'JetBrains Mono',monospace" }}/>
        <Radar dataKey="A" stroke={C.cyan} fill={C.cyan} fillOpacity={0.15} strokeWidth={1.5}/>
      </RadarChart>
    </ResponsiveContainer>
  );
}

function LayerChart({ layers }) {
  if (!layers?.length) return null;
  return (
    <ResponsiveContainer width="100%" height={160}>
      <BarChart data={layers} layout="vertical"
        margin={{ top:5, right:5, bottom:5, left:0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={C.border} horizontal={false}/>
        <XAxis type="number" domain={[-1,1]} tick={{ fill:C.muted, fontSize:9 }}
          axisLine={false} tickLine={false}/>
        <YAxis dataKey="name" type="category" width={82}
          tick={{ fill:C.muted, fontSize:9, fontFamily:"'JetBrains Mono',monospace" }}
          axisLine={false} tickLine={false}/>
        <ReferenceLine x={0} stroke={C.dim}/>
        <Tooltip contentStyle={{ background:C.panel, border:`1px solid ${C.border2}`,
          fontSize:11 }} formatter={v=>[v?.toFixed(4)||0,"Score"]}/>
        <Bar dataKey="score" radius={[0,3,3,0]}>
          {(layers||[]).map((l,i)=>(
            <Cell key={i} fill={(l.score||0)>0?C.green:C.red} fillOpacity={0.8}/>
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function ModelGrid({ signals }) {
  if (!signals) return <div style={{ color:C.muted, fontSize:11 }}>Loading signals...</div>;
  const groups = [
    { label:"DEEP LEARNING",   keys:["lstm","transformer","gp"] },
    { label:"STATE ESTIMATION",keys:["kalman","particle","hmm"] },
    { label:"VOLATILITY",      keys:["garch","egarch","heston"] },
    { label:"TIME SERIES",     keys:["ou","ms_arima","copula"] },
    { label:"SIGNAL SOURCES",  keys:["options","cross_asset","micro","sentiment"] },
    { label:"SPECTRAL",        keys:["fourier","wavelet","monte_carlo"] },
  ];
  const labels = {
    lstm:"LSTM",transformer:"Transformer",gp:"Gaussian Proc",
    kalman:"Kalman Filter",particle:"Particle (1k)",hmm:"HMM",
    garch:"GARCH(1,1)",egarch:"EGARCH",heston:"Heston SV",
    ou:"Ornstein-Uhl.",ms_arima:"MS-ARIMA",copula:"Copula",
    options:"Options Flow",cross_asset:"Cross-Asset",micro:"Microstructure",
    sentiment:"Sentiment",fourier:"Fourier FFT",wavelet:"Wavelet DWT",
    monte_carlo:"Monte Carlo",
  };
  return (
    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10 }}>
      {groups.map(g=>(
        <div key={g.label} style={{ background:C.bg, border:`1px solid ${C.border}`,
          borderRadius:6, padding:"10px 12px" }}>
          <div style={{ fontSize:9, color:C.muted, letterSpacing:1.5,
            fontFamily:"'JetBrains Mono',monospace", marginBottom:8 }}>{g.label}</div>
          <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
            {g.keys.map(k=>(
              <div key={k} style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                <span style={{ fontSize:11, color:C.text,
                  fontFamily:"'JetBrains Mono',monospace" }}>{labels[k]}</span>
                <SignalBadge signal={signals[k]}/>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Main Dashboard ─────────────────────────────────────
export default function NeuroTraderDashboard() {
  const [ticker, setTicker]   = useState("AAPL");
  const [input,  setInput]    = useState("");
  const [tab,    setTab]      = useState("overview");
  const [time,   setTime]     = useState(new Date());
  const [scanData,setScanData]= useState({});
  const { data, loading, fullReady, refresh } = useTickerData(ticker);

  useEffect(() => {
    const t = setInterval(()=>setTime(new Date()),1000);
    return ()=>clearInterval(t);
  }, []);

  // Scan on load
  useEffect(()=>{
    fetch(`${API}/scan`)
      .then(r=>r.json()).then(setScanData).catch(()=>{});
  },[]);

  const submit = () => {
    const t = input.trim().toUpperCase();
    if (t) { setTicker(t); setInput(""); setTab("overview"); }
  };

  const tabs = ["overview","models","volatility","montecarlo"];

  return (
    <div style={{ background:C.bg, minHeight:"100vh", color:C.text,
      fontFamily:"'JetBrains Mono',monospace" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        ::-webkit-scrollbar{width:4px}
        ::-webkit-scrollbar-track{background:${C.bg}}
        ::-webkit-scrollbar-thumb{background:${C.border2};border-radius:2px}
        @keyframes ticker{from{transform:translateX(0)}to{transform:translateX(-50%)}}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
        @keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
        @keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
      `}</style>

      {/* Header */}
      <div style={{ background:C.panel, borderBottom:`1px solid ${C.border}`,
        padding:"0 24px", height:52, display:"flex", alignItems:"center",
        justifyContent:"space-between" }}>
        <div style={{ display:"flex", alignItems:"center", gap:16 }}>
          <div style={{ display:"flex", alignItems:"center", gap:8 }}>
            <div style={{ width:8, height:8, borderRadius:"50%", background:C.green,
              animation:"pulse 2s infinite" }}/>
            <span style={{ fontSize:16, fontWeight:800, color:C.text,
              fontFamily:"'Syne',sans-serif", letterSpacing:1 }}>NEUROTRADER</span>
            <span style={{ fontSize:10, color:C.muted }}>v4.0</span>
          </div>
          <div style={{ width:1, height:20, background:C.border }}/>
          <span style={{ fontSize:10, color:C.muted }}>19 MODELS · LIVE DATA</span>
          {!fullReady && data && (
            <div style={{ display:"flex", alignItems:"center", gap:6,
              color:C.yellow, fontSize:10 }}>
              <div style={{ width:6, height:6, borderRadius:"50%",
                background:C.yellow, animation:"pulse 0.8s infinite" }}/>
              COMPUTING...
            </div>
          )}
          {fullReady && (
            <div style={{ color:C.green, fontSize:10 }}>● FULL ANALYSIS READY</div>
          )}
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:12 }}>
          {/* Search */}
          <div style={{ display:"flex", gap:0 }}>
            <input value={input} onChange={e=>setInput(e.target.value)}
              onKeyDown={e=>e.key==="Enter"&&submit()}
              placeholder="TICKER..." style={{
                background:C.bg, border:`1px solid ${C.border2}`, borderRight:"none",
                color:C.text, padding:"5px 12px", fontSize:11, outline:"none",
                fontFamily:"'JetBrains Mono',monospace", borderRadius:"4px 0 0 4px",
                width:100
              }}/>
            <button onClick={submit} style={{
              background:C.cyan+"22", border:`1px solid ${C.border2}`, color:C.cyan,
              padding:"5px 12px", cursor:"pointer", fontSize:11,
              fontFamily:"'JetBrains Mono',monospace", borderRadius:"0 4px 4px 0"
            }}>GO</button>
          </div>
          {["AAPL","NVDA","MSFT","GOOGL"].map(t=>(
            <button key={t} onClick={()=>{setTicker(t);setTab("overview");}} style={{
              background:ticker===t?C.border2:"transparent",
              border:`1px solid ${ticker===t?C.cyan:C.border}`,
              color:ticker===t?C.cyan:C.muted,
              borderRadius:4, padding:"4px 10px", cursor:"pointer",
              fontSize:11, fontFamily:"'JetBrains Mono',monospace"
            }}>{t}</button>
          ))}
          <button onClick={refresh} style={{
            background:"transparent", border:`1px solid ${C.border}`,
            color:C.muted, borderRadius:4, padding:"4px 10px", cursor:"pointer",
            fontSize:11, fontFamily:"'JetBrains Mono',monospace"
          }}>⟳ REFRESH</button>
          <div style={{ fontSize:11, color:C.muted }}>
            {time.toLocaleTimeString()} EST
          </div>
        </div>
      </div>

      <TickerTape scanData={scanData}/>

      {loading && !data ? (
        <div style={{ display:"flex", justifyContent:"center", alignItems:"center",
          height:"60vh", flexDirection:"column", gap:16 }}>
          <div style={{ width:32, height:32, border:`2px solid ${C.border}`,
            borderTop:`2px solid ${C.cyan}`, borderRadius:"50%",
            animation:"spin 0.8s linear infinite" }}/>
          <div style={{ color:C.muted, fontSize:12 }}>Fetching {ticker}...</div>
        </div>
      ) : data ? (
        <>
          {/* Hero */}
          <div style={{ padding:"20px 24px 0", display:"grid",
            gridTemplateColumns:"auto 1fr auto", gap:24, alignItems:"center" }}>
            <div>
              <div style={{ fontSize:11, color:C.muted, letterSpacing:2 }}>
                {data.ticker} · NASDAQ
              </div>
              <div style={{ fontSize:52, fontWeight:800, fontFamily:"'Syne',sans-serif",
                color:C.text, lineHeight:1.1 }}>${data.price}</div>
              <div style={{ fontSize:13,
                color:(data.changePct||0)<0?C.red:C.green, marginTop:2 }}>
                {(data.changePct||0)>=0?"▲":"▼"} ${Math.abs(data.change||0).toFixed(2)} ({Math.abs(data.changePct||0).toFixed(2)}%)
              </div>
              <div style={{ fontSize:10, color:C.muted, marginTop:4 }}>
                Vol: {((data.volume||0)/1e6).toFixed(1)}M · PE: {data.pe||"N/A"}
              </div>
            </div>
            <div style={{ height:80 }}>
              {data.priceHistory && (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={data.priceHistory.slice(-60)}
                    margin={{ top:5, right:0, bottom:0, left:0 }}>
                    <defs>
                      <linearGradient id="hg" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={decisionColor(data.decision)}
                          stopOpacity={0.3}/>
                        <stop offset="95%" stopColor={decisionColor(data.decision)}
                          stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <Area type="monotone" dataKey="price"
                      stroke={decisionColor(data.decision||"HOLD")}
                      strokeWidth={2} fill="url(#hg)" dot={false}/>
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>
            <div style={{ textAlign:"right" }}>
              <div style={{ fontSize:10, color:C.muted, letterSpacing:1.5, marginBottom:6 }}>
                ENSEMBLE DECISION
              </div>
              {data.decision ? (
                <div style={{ fontSize:36, fontWeight:800, fontFamily:"'Syne',sans-serif",
                  color:decisionColor(data.decision) }}>{data.decision}</div>
              ) : <Spinner/>}
              <div style={{ fontSize:11, color:C.muted, marginTop:4 }}>
                Score: {(data.score||0).toFixed(4)} · Conf: {(data.confidence||0).toFixed(1)}%
              </div>
              <div style={{ marginTop:8 }}>
                <VoteBar votes={data.votes}/>
              </div>
            </div>
          </div>

          {/* Tabs */}
          <div style={{ display:"flex", padding:"16px 24px 0",
            borderBottom:`1px solid ${C.border}`, marginTop:16 }}>
            {tabs.map(t=>(
              <button key={t} onClick={()=>setTab(t)} style={{
                background:"transparent",
                borderBottom:tab===t?`2px solid ${C.cyan}`:"2px solid transparent",
                border:"none",
                color:tab===t?C.cyan:C.muted,
                padding:"8px 20px", cursor:"pointer", fontSize:11,
                fontFamily:"'JetBrains Mono',monospace", letterSpacing:1.5,
                textTransform:"uppercase"
              }}>{t}</button>
            ))}
          </div>

          <div style={{ padding:"20px 24px 40px" }}>

            {/* OVERVIEW */}
            {tab==="overview" && (
              <div style={{ display:"grid",
                gridTemplateColumns:"1fr 1fr 300px", gap:16 }}>
                <div style={{ display:"flex", flexDirection:"column", gap:14 }}>
                  <div style={{ background:C.panel, border:`1px solid ${C.border}`,
                    borderRadius:6, padding:16 }}>
                    <SectionHeader title="PRICE HISTORY" sub="120 sessions"/>
                    <PriceChart data={data} decision={data.decision}/>
                  </div>
                  <div style={{ background:C.panel, border:`1px solid ${C.border}`,
                    borderRadius:6, padding:16 }}>
                    <SectionHeader title="GARCH CONDITIONAL VOLATILITY"/>
                    <GarchChart data={data}/>
                    <div style={{ fontSize:10, color:C.muted, marginTop:6 }}>
                      <span style={{ color:C.red }}>— 30% threshold</span>
                      {"  "}Forecast: {data.garchVol||"..."}%
                      {"  "}Persistence: {data.garchPersistence||"..."}
                    </div>
                  </div>
                </div>

                <div style={{ display:"flex", flexDirection:"column", gap:14 }}>
                  <div style={{ background:C.panel, border:`1px solid ${C.border}`,
                    borderRadius:6, padding:16 }}>
                    <SectionHeader title="RISK METRICS"/>
                    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
                      <StatCard label="VaR 95% (30d)"
                        value={data.var95?`${data.var95}%`:"..."} color={C.red}/>
                      <StatCard label="GARCH VOL"
                        value={data.garchVol?`${data.garchVol}%`:"..."} color={C.purple}/>
                      <StatCard label="IV SKEW"
                        value={data.ivSkew?.toFixed(4)||"..."}
                        color={(data.ivSkew||0)>0.1?C.red:C.green}/>
                      <StatCard label="VWAP DEV"
                        value={data.vwapDev?`${data.vwapDev}%`:"..."}
                        color={(data.vwapDev||0)<0?C.red:C.green}/>
                      <StatCard label="OU HALF-LIFE"
                        value={data.ouHalfLife?`${data.ouHalfLife}d`:"..."}
                        color={C.cyan}/>
                      <StatCard label="HESTON ρ"
                        value={data.hestonRho?.toFixed(4)||"..."} color={C.purple}/>
                      <StatCard label="CRASH RISK"
                        value={data.copula?.crash?`${(data.copula.crash*100).toFixed(1)}%`:"..."}
                        color={C.red}/>
                      <StatCard label="HMM REGIME"
                        value={data.hmm||"..."}
                        color={data.hmm==="BULL"?C.green:data.hmm==="BEAR"?C.red:C.yellow}/>
                    </div>
                  </div>
                  <div style={{ background:C.panel, border:`1px solid ${C.border}`,
                    borderRadius:6, padding:16 }}>
                    <SectionHeader title="LAYER SCORES"/>
                    <LayerChart layers={data.layers}/>
                  </div>
                </div>

                <div style={{ display:"flex", flexDirection:"column", gap:14 }}>
                  <div style={{ background:C.panel, border:`1px solid ${C.border}`,
                    borderRadius:6, padding:16 }}>
                    <SectionHeader title="HMM REGIME"/>
                    <RegimeRadar data={data.regimeProbabilities}/>
                    <div style={{ display:"flex", justifyContent:"space-around", marginTop:4 }}>
                      {(data.regimeProbabilities||[]).map(r=>(
                        <div key={r.name} style={{ textAlign:"center" }}>
                          <div style={{ fontSize:16, fontWeight:700,
                            color:r.name==="BULL"?C.green:r.name==="BEAR"?C.red:C.yellow }}>
                            {r.value}%</div>
                          <div style={{ fontSize:9, color:C.muted }}>{r.name}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
                    <StatCard label="MACRO"
                      value={data.macro||"..."} color={C.yellow}/>
                    <StatCard label="RSI"
                      value={data.rsi?.toFixed(1)||"..."}
                      color={(data.rsi||50)>70?C.red:(data.rsi||50)<30?C.green:C.yellow}/>
                    <StatCard label="PC RATIO"
                      value={data.pcRatio?.toFixed(3)||"..."} color={C.cyan}/>
                    <StatCard label="MODELS"
                      value={data.totalModels||19} sub="active" color={C.green}/>
                  </div>
                </div>
              </div>
            )}

            {/* MODELS */}
            {tab==="models" && (
              <div style={{ display:"grid", gridTemplateColumns:"1fr 300px", gap:16 }}>
                <div style={{ background:C.panel, border:`1px solid ${C.border}`,
                  borderRadius:6, padding:16 }}>
                  <SectionHeader title="ALL MODEL SIGNALS" sub="19 models · 5 layers"/>
                  <ModelGrid signals={data.signals}/>
                </div>
                <div style={{ display:"flex", flexDirection:"column", gap:14 }}>
                  <div style={{ background:C.panel, border:`1px solid ${C.border}`,
                    borderRadius:6, padding:16 }}>
                    <SectionHeader title="VOTE BREAKDOWN"/>
                    <VoteBar votes={data.votes}/>
                    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr",
                      gap:8, marginTop:12 }}>
                      {["BUY","HOLD","SELL"].map(v=>(
                        <div key={v} style={{ textAlign:"center",
                          background:signalColor(v)+"18",
                          border:`1px solid ${signalColor(v)}44`,
                          borderRadius:5, padding:12 }}>
                          <div style={{ fontSize:28, fontWeight:700,
                            color:signalColor(v) }}>{data.votes?.[v]||0}</div>
                          <div style={{ fontSize:10, color:C.muted }}>{v}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div style={{ background:C.panel, border:`1px solid ${C.border}`,
                    borderRadius:6, padding:16 }}>
                    <SectionHeader title="LAYER SCORES"/>
                    <LayerChart layers={data.layers}/>
                  </div>
                  <div style={{ background:C.panel, border:`1px solid ${C.border}`,
                    borderRadius:6, padding:16 }}>
                    <SectionHeader title="SENTIMENT"/>
                    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
                      <StatCard label="ANALYST ↑"
                        value={`${data.sentiment?.analystUpside?.toFixed(1)||"..."}%`}
                        color={C.green}/>
                      <StatCard label="SHORT %"
                        value={`${data.sentiment?.shortPct?.toFixed(1)||"..."}%`}
                        color={C.red}/>
                      <StatCard label="RSI DIV"
                        value={data.sentiment?.rsiDivergence||"..."} color={C.yellow}/>
                      <StatCard label="SCORE"
                        value={data.sentiment?.compositeScore||"..."}
                        color={(data.sentiment?.compositeScore||0)>0?C.green:C.red}/>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* VOLATILITY */}
            {tab==="volatility" && (
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16 }}>
                <div style={{ background:C.panel, border:`1px solid ${C.border}`,
                  borderRadius:6, padding:16 }}>
                  <SectionHeader title="GARCH(1,1) CONDITIONAL VOLATILITY"/>
                  <GarchChart data={data}/>
                  <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr",
                    gap:8, marginTop:14 }}>
                    <StatCard label="FORECAST VOL"
                      value={`${data.garchVol||"..."}%`} color={C.purple}/>
                    <StatCard label="PERSISTENCE"
                      value={data.garchPersistence?.toFixed(4)||"..."} color={C.cyan}/>
                    <StatCard label="REGIME"
                      value={(data.garchVol||0)>30?"HIGH":"NORMAL"}
                      color={(data.garchVol||0)>30?C.red:C.green}/>
                  </div>
                </div>
                <div style={{ background:C.panel, border:`1px solid ${C.border}`,
                  borderRadius:6, padding:16 }}>
                  <SectionHeader title="HESTON STOCHASTIC VOLATILITY"/>
                  <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10, marginTop:8 }}>
                    <StatCard label="HESTON ρ"
                      value={data.hestonRho?.toFixed(4)||"..."} color={C.red}/>
                    <StatCard label="KAPPA κ" value="calibrated" color={C.cyan}/>
                    <StatCard label="VaR 95%"
                      value={`${data.var95||"..."}%`} color={C.red}/>
                    <StatCard label="P(+10%)"
                      value={data.mcRisk?.p_up_10pct?`${data.mcRisk.p_up_10pct}%`:"..."}
                      color={C.green}/>
                  </div>
                </div>
                <div style={{ background:C.panel, border:`1px solid ${C.border}`,
                  borderRadius:6, padding:16 }}>
                  <SectionHeader title="COPULA TAIL DEPENDENCE"
                    sub={`${ticker} vs SPY`}/>
                  <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr 1fr",
                    gap:8, marginTop:8 }}>
                    <StatCard label="CRASH"
                      value={data.copula?.crash?(data.copula.crash*100).toFixed(1)+"%":"..."}
                      color={C.red}/>
                    <StatCard label="RALLY"
                      value={data.copula?.rally?(data.copula.rally*100).toFixed(1)+"%":"..."}
                      color={C.green}/>
                    <StatCard label="KENDALL τ"
                      value={data.copula?.kendall?.toFixed(3)||"..."} color={C.cyan}/>
                    <StatCard label="DOMINANT" value="Clayton" color={C.red}/>
                  </div>
                </div>
                <div style={{ background:C.panel, border:`1px solid ${C.border}`,
                  borderRadius:6, padding:16 }}>
                  <SectionHeader title="ORNSTEIN-UHLENBECK"/>
                  <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr",
                    gap:10, marginTop:8 }}>
                    <StatCard label="HALF-LIFE"
                      value={data.ouHalfLife?`${data.ouHalfLife}d`:"..."} color={C.cyan}/>
                    <StatCard label="SIGNAL"
                      value={data.signals?.ou||"..."} color={signalColor(data.signals?.ou||"HOLD")}/>
                  </div>
                </div>
              </div>
            )}

            {/* MONTE CARLO */}
            {tab==="montecarlo" && (
              <div style={{ display:"grid", gridTemplateColumns:"1fr 380px", gap:16 }}>
                <div style={{ background:C.panel, border:`1px solid ${C.border}`,
                  borderRadius:6, padding:16 }}>
                  <SectionHeader title="MONTE CARLO — 10,000 PATHS"
                    sub="GBM + Merton Jump Diffusion · 30-day horizon"/>
                  <MCChart paths={data.monteCarloPaths} S0={data.price}/>
                  <div style={{ fontSize:10, color:C.muted, marginTop:8,
                    display:"flex", gap:20 }}>
                    <span style={{ color:C.cyan }}>— Mean</span>
                    <span style={{ color:C.green, opacity:0.8 }}>-- P95</span>
                    <span style={{ color:C.red, opacity:0.8 }}>-- P5</span>
                  </div>
                </div>
                <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
                  <div style={{ background:C.panel, border:`1px solid ${C.border}`,
                    borderRadius:6, padding:16 }}>
                    <SectionHeader title="RISK DISTRIBUTION"/>
                    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
                      <StatCard label="VaR 95%"
                        value={data.var95?`${data.var95}%`:"..."} color={C.red}/>
                      <StatCard label="CVaR 95%"
                        value={data.mcRisk?.CVaR_95?`${data.mcRisk.CVaR_95}%`:"..."} color={C.red}/>
                      <StatCard label="P(+10%)"
                        value={data.mcRisk?.p_up_10pct?`${data.mcRisk.p_up_10pct}%`:"..."}
                        color={C.green}/>
                      <StatCard label="P(-10%)"
                        value={data.mcRisk?.p_down_10pct?`${data.mcRisk.p_down_10pct}%`:"..."}
                        color={C.yellow}/>
                    </div>
                  </div>
                  <div style={{ background:C.panel, border:`1px solid ${C.border}`,
                    borderRadius:6, padding:16 }}>
                    <SectionHeader title="GBM PARAMS"/>
                    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
                      <StatCard label="SIGMA σ"
                        value={`${data.garchVol||"..."}%`} color={C.purple}/>
                      <StatCard label="HESTON ρ"
                        value={data.hestonRho?.toFixed(4)||"..."} color={C.red}/>
                      <StatCard label="JUMP λ" value="0.10" color={C.yellow}/>
                      <StatCard label="HORIZON" value="30d" color={C.muted}/>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </>
      ) : null}
    </div>
  );
}
