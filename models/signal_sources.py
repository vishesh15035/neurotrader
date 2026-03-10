"""
Advanced Signal Sources:
- Options Flow (put/call ratio, IV skew, term structure)
- Cross-Asset Signals (VIX, yield curve, DXY, gold)
- Microstructure (bid-ask, VWAP, volume profile)
- Sentiment (news velocity + price divergence)
"""
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class OptionsFlowAnalyzer:
    """
    Options market intelligence:
    - Put/Call ratio (sentiment)
    - IV skew (tail risk pricing)
    - Term structure (contango/backwardation)
    - Max pain level
    """
    def analyze(self, ticker: str) -> dict:
        try:
            stock = yf.Ticker(ticker)
            price = stock.info.get("currentPrice",
                    stock.info.get("regularMarketPrice", 0))
            exps  = stock.options
            if not exps or len(exps) < 2:
                return {"ticker":ticker,"error":"No options data","signal":"HOLD"}
            # Near-term and next expiry
            exp1  = exps[0]
            exp2  = exps[1] if len(exps)>1 else exps[0]
            ch1   = stock.option_chain(exp1)
            ch2   = stock.option_chain(exp2)
            calls1, puts1 = ch1.calls, ch1.puts
            calls2, puts2 = ch2.calls, ch2.puts
            # Put/Call ratio by volume
            c_vol = calls1['volume'].fillna(0).sum()
            p_vol = puts1['volume'].fillna(0).sum()
            pc_ratio = float(p_vol / (c_vol + 1e-10))
            # Put/Call by OI
            c_oi  = calls1['openInterest'].fillna(0).sum()
            p_oi  = puts1['openInterest'].fillna(0).sum()
            pc_oi = float(p_oi / (c_oi + 1e-10))
            # IV skew: OTM put IV - OTM call IV (fear gauge)
            atm   = price
            otm_calls = calls1[calls1['strike'] > atm*1.05]
            otm_puts  = puts1[puts1['strike']  < atm*0.95]
            call_iv   = otm_calls['impliedVolatility'].mean() if not otm_calls.empty else 0
            put_iv    = otm_puts['impliedVolatility'].mean()  if not otm_puts.empty  else 0
            skew      = float(put_iv - call_iv)  # positive = fear
            # Max pain: strike where total options loss is minimized
            all_strikes = sorted(set(calls1['strike'].tolist() + puts1['strike'].tolist()))
            max_pain_val = float('inf')
            max_pain_strike = price
            for K in all_strikes:
                call_pain = float(calls1[calls1['strike']<=K]['openInterest'].fillna(0).sum() * max(price-K,0))
                put_pain  = float(puts1[puts1['strike']>=K]['openInterest'].fillna(0).sum()  * max(K-price,0))
                total     = call_pain + put_pain
                if total < max_pain_val:
                    max_pain_val    = total
                    max_pain_strike = K
            # Term structure: near vs far IV
            atm_calls1 = calls1.iloc[(calls1['strike']-price).abs().argsort()[:1]]
            atm_calls2 = calls2.iloc[(calls2['strike']-price).abs().argsort()[:1]]
            iv1 = float(atm_calls1['impliedVolatility'].iloc[0]) if not atm_calls1.empty else 0
            iv2 = float(atm_calls2['impliedVolatility'].iloc[0]) if not atm_calls2.empty else 0
            term_slope = iv2 - iv1  # negative = inverted (fear)
            # Signals
            bearish_flags = sum([
                pc_ratio > 1.2,    # heavy put buying
                skew > 0.05,       # fear premium
                term_slope < -0.02,# inverted term structure
                pc_oi > 1.5        # heavy put OI
            ])
            bullish_flags = sum([
                pc_ratio < 0.7,
                skew < -0.02,
                term_slope > 0.02,
                pc_oi < 0.8
            ])
            signal = "SELL" if bearish_flags>=2 else "BUY" if bullish_flags>=2 else "HOLD"
            return {
                "ticker":        ticker,
                "price":         round(price,2),
                "pc_ratio_vol":  round(pc_ratio,4),
                "pc_ratio_oi":   round(pc_oi,4),
                "iv_skew":       round(skew,4),
                "iv_near":       round(iv1,4),
                "iv_far":        round(iv2,4),
                "term_slope":    round(term_slope,4),
                "max_pain":      round(max_pain_strike,2),
                "max_pain_diff": round((max_pain_strike-price)/price*100,2),
                "bearish_flags": bearish_flags,
                "bullish_flags": bullish_flags,
                "signal":        signal,
                "fear_gauge":    "HIGH" if skew>0.05 else "LOW" if skew<-0.02 else "NORMAL"
            }
        except Exception as e:
            return {"ticker":ticker,"error":str(e),"signal":"HOLD"}


class CrossAssetAnalyzer:
    """
    Cross-asset regime signals:
    - VIX level and term structure
    - Yield curve (2y10y spread)
    - Dollar index (DXY proxy)
    - Gold as risk-off signal
    - Credit spreads proxy
    """
    def analyze(self, ticker: str) -> dict:
        try:
            # Fetch cross-asset data
            symbols = {"VIX":"^VIX","TNX":"^TNX","FVX":"^FVX",
                       "GOLD":"GLD","DXY":"UUP","HYG":"HYG","SPY":"SPY"}
            data = {}
            for name, sym in symbols.items():
                try:
                    h = yf.Ticker(sym).history(period="3mo")
                    if not h.empty:
                        data[name] = h['Close'].values
                except: pass
            signals = {}
            score   = 0
            # VIX signal
            if "VIX" in data and len(data["VIX"])>20:
                vix     = data["VIX"]
                vix_now = float(vix[-1])
                vix_ma20= float(vix[-20:].mean())
                vix_trend = "RISING" if vix[-1]>vix[-5] else "FALLING"
                vix_sig = "SELL" if vix_now>25 else "BUY" if vix_now<15 else "HOLD"
                signals["VIX"] = {
                    "level": round(vix_now,2),
                    "ma20":  round(vix_ma20,2),
                    "trend": vix_trend,
                    "signal":vix_sig,
                    "regime":"FEAR" if vix_now>25 else "GREED" if vix_now<15 else "NORMAL"
                }
                score += 1 if vix_sig=="BUY" else -1 if vix_sig=="SELL" else 0
            # Yield curve (2s10s)
            if "TNX" in data and "FVX" in data:
                tnx = float(data["TNX"][-1])  # 10Y
                fvx = float(data["FVX"][-1])  # 5Y (proxy for 2Y)
                spread = tnx - fvx
                signals["YIELD_CURVE"] = {
                    "10Y":    round(tnx,3),
                    "5Y":     round(fvx,3),
                    "spread": round(spread,3),
                    "inverted": spread < 0,
                    "signal": "SELL" if spread<0 else "BUY" if spread>1.5 else "HOLD"
                }
                score += 1 if spread>0.5 else -1 if spread<0 else 0
            # Gold (risk-off indicator)
            if "GOLD" in data and len(data["GOLD"])>20:
                gold     = data["GOLD"]
                gold_ret = float((gold[-1]-gold[-20])/gold[-20])
                gold_sig = "SELL" if gold_ret>0.05 else "BUY" if gold_ret<-0.03 else "HOLD"
                signals["GOLD"] = {
                    "return_1mo": round(gold_ret*100,2),
                    "signal":     gold_sig,
                    "interpretation": "Risk-off" if gold_ret>0.03 else "Risk-on"
                }
                score += -1 if gold_ret>0.05 else 1 if gold_ret<-0.03 else 0
            # HYG credit spreads proxy
            if "HYG" in data and "SPY" in data and len(data["HYG"])>20:
                hyg_ret = float((data["HYG"][-1]-data["HYG"][-20])/data["HYG"][-20])
                spy_ret = float((data["SPY"][-1]-data["SPY"][-20])/data["SPY"][-20])
                divergence = spy_ret - hyg_ret
                signals["CREDIT"] = {
                    "hyg_1mo":    round(hyg_ret*100,2),
                    "divergence": round(divergence*100,2),
                    "signal":     "SELL" if divergence>0.05 else "BUY" if divergence<-0.03 else "HOLD",
                    "interpretation": "Credit leading stocks down" if divergence>0.05 else "Normal"
                }
                score += -1 if divergence>0.05 else 1 if divergence<-0.03 else 0
            final_signal = "BUY" if score>=2 else "SELL" if score<=-2 else "HOLD"
            return {
                "ticker":      ticker,
                "cross_asset_score": score,
                "signals":     signals,
                "final_signal":final_signal,
                "confidence":  round(min(abs(score)/4, 0.95),4),
                "macro_regime":"RISK-ON" if score>=2 else "RISK-OFF" if score<=-2 else "NEUTRAL"
            }
        except Exception as e:
            return {"ticker":ticker,"error":str(e),"signal":"HOLD"}


class MicrostructureAnalyzer:
    """
    Market microstructure signals:
    - VWAP deviation
    - Volume profile (POC, value area)
    - Bid-ask spread proxy
    - Amihud illiquidity ratio
    - Price impact estimation
    """
    def analyze(self, ticker: str, period: str = "1mo") -> dict:
        try:
            stock = yf.Ticker(ticker)
            hist  = stock.history(period=period, interval="1d")
            if hist.empty:
                return {"error":"No data","signal":"HOLD"}
            prices = hist['Close'].values
            volumes= hist['Volume'].values
            highs  = hist['High'].values
            lows   = hist['Low'].values
            opens  = hist['Open'].values
            n      = len(prices)
            # VWAP
            vwap   = (prices*volumes).sum() / (volumes.sum()+1e-10)
            vwap_dev = float((prices[-1]-vwap)/vwap)
            # Volume-weighted momentum
            vol_norm= volumes / (volumes.mean()+1e-10)
            vw_ret  = np.sum(np.diff(prices)/prices[:-1] * vol_norm[1:]) / vol_norm[1:].sum()
            # Amihud illiquidity: |ret|/volume (in millions)
            ret     = np.abs(np.diff(prices)/prices[:-1])
            amihud  = float((ret / (volumes[1:]/1e6+1e-10)).mean())
            # Volume profile: POC (point of control)
            price_bins  = np.linspace(prices.min(), prices.max(), 20)
            vol_profile = np.zeros(19)
            for i in range(len(prices)):
                idx = min(int((prices[i]-prices.min())/(prices.max()-prices.min()+1e-10)*19), 18)
                vol_profile[idx] += volumes[i]
            poc_idx   = int(np.argmax(vol_profile))
            poc_price = float(price_bins[poc_idx])
            # Value area (70% of volume)
            total_vol = vol_profile.sum()
            sorted_idx= np.argsort(vol_profile)[::-1]
            va_vol    = 0
            va_indices= []
            for idx in sorted_idx:
                va_vol += vol_profile[idx]
                va_indices.append(idx)
                if va_vol >= 0.70*total_vol: break
            va_high = float(price_bins[max(va_indices)])
            va_low  = float(price_bins[min(va_indices)])
            # Average true range (proxy for spread)
            atr = float(np.mean([max(h-l, abs(h-c), abs(l-c))
                         for h,l,c in zip(highs[1:],lows[1:],prices[:-1])]))
            atr_pct = atr/prices[-1]*100
            # Microstructure signal
            above_vwap = prices[-1] > vwap
            above_poc  = prices[-1] > poc_price
            vol_surge  = volumes[-1] > volumes[-20:].mean()*1.5
            signal = "BUY"  if (above_vwap and above_poc and vol_surge) else \
                     "SELL" if (not above_vwap and not above_poc) else "HOLD"
            return {
                "ticker":      ticker,
                "price":       round(float(prices[-1]),2),
                "vwap":        round(float(vwap),2),
                "vwap_dev_pct":round(vwap_dev*100,4),
                "poc_price":   round(poc_price,2),
                "va_high":     round(va_high,2),
                "va_low":      round(va_low,2),
                "atr_pct":     round(atr_pct,4),
                "amihud":      round(amihud,8),
                "vol_surge":   bool(vol_surge),
                "above_vwap":  bool(above_vwap),
                "above_poc":   bool(above_poc),
                "vw_momentum": round(float(vw_ret*100),4),
                "signal":      signal,
                "confidence":  round(float(min(abs(vwap_dev)*10,0.95)),4)
            }
        except Exception as e:
            return {"error":str(e),"signal":"HOLD"}


class SentimentAnalyzer:
    """
    Quantitative sentiment from price action:
    - News velocity proxy (price gap frequency)
    - Put-call divergence from price trend
    - Momentum-price divergence (bearish/bullish divergence)
    - Short interest proxy from borrow cost
    """
    def analyze(self, ticker: str) -> dict:
        try:
            stock  = yf.Ticker(ticker)
            hist   = stock.history(period="3mo")
            info   = stock.info
            prices = hist['Close'].values
            volumes= hist['Volume'].values
            n      = len(prices)
            ret    = np.diff(prices)/prices[:-1]
            # Gap frequency (proxy for news velocity)
            opens  = hist['Open'].values
            gaps   = np.abs(opens[1:] - prices[:-1]) / prices[:-1]
            gap_freq = float((gaps > 0.01).mean())
            avg_gap  = float(gaps.mean()*100)
            # Momentum divergence (price vs volume momentum)
            price_mom  = float((prices[-1]-prices[-20])/prices[-20]) if n>20 else 0
            vol_mom    = float((volumes[-5:].mean()-volumes[-20:].mean())/volumes[-20:].mean()) if n>20 else 0
            divergence = price_mom - vol_mom*0.5
            # Short interest (from info)
            short_ratio = info.get("shortRatio", 0) or 0
            short_pct   = info.get("shortPercentOfFloat", 0) or 0
            # Analyst sentiment
            target   = info.get("targetMeanPrice",0) or 0
            curr     = info.get("currentPrice", info.get("regularMarketPrice",prices[-1]))
            upside   = (target-curr)/curr if curr>0 and target>0 else 0
            rec      = info.get("recommendationKey","hold")
            rec_score= {"strongbuy":2,"buy":1,"hold":0,"sell":-1,"strongsell":-2}.get(rec,0)
            # RSI divergence (price makes new high but RSI doesn't)
            def rsi(r, period=14):
                gains = np.where(r>0,r,0)
                losses= np.where(r<0,-r,0)
                ag = np.convolve(gains, np.ones(period)/period, 'valid')
                al = np.convolve(losses,np.ones(period)/period, 'valid')
                rs = ag/(al+1e-10)
                return 100-100/(1+rs)
            rsi_vals = rsi(ret)
            rsi_now  = float(rsi_vals[-1]) if len(rsi_vals)>0 else 50
            rsi_div  = "BEARISH" if prices[-1]>prices[-20] and rsi_now<rsi_vals[-20] else \
                       "BULLISH" if prices[-1]<prices[-20] and rsi_now>rsi_vals[-20] else "NONE"
            # Composite sentiment score
            score = 0
            score += 1 if upside>0.10 else -1 if upside<-0.05 else 0
            score += rec_score
            score += -1 if short_pct>0.15 else 1 if short_pct<0.03 else 0
            score += -1 if rsi_div=="BEARISH" else 1 if rsi_div=="BULLISH" else 0
            score += 1 if vol_mom>0.2 and price_mom>0 else -1 if vol_mom<-0.2 else 0
            signal = "BUY" if score>=2 else "SELL" if score<=-2 else "HOLD"
            return {
                "ticker":       ticker,
                "analyst_upside":round(float(upside*100),2),
                "analyst_rec":  rec,
                "rec_score":    rec_score,
                "short_pct":    round(float(short_pct*100),2) if short_pct else 0,
                "short_ratio":  round(float(short_ratio),2) if short_ratio else 0,
                "rsi_now":      round(rsi_now,2),
                "rsi_divergence":rsi_div,
                "gap_frequency":round(gap_freq,4),
                "price_momentum":round(float(price_mom*100),2),
                "vol_momentum": round(float(vol_mom*100),2),
                "composite_score":score,
                "signal":       signal,
                "confidence":   round(min(abs(score)/4,0.95),4)
            }
        except Exception as e:
            return {"error":str(e),"signal":"HOLD"}
