
import argparse, json, math, os, time, traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import pandas as pd, requests, yfinance as yf

DEFAULT_CONFIG = {
    "scan_interval_seconds": 300,
    "cooldown_minutes": 60,
    "min_confidence": 70,
    "risk_reward_min": 2.0,
    "telegram": {"enabled": False, "bot_token": "", "chat_id": ""},
    "heartbeat": {"enabled": True, "interval_minutes": 90},
    "watchlists": {
        "intraday": ["AAPL", "NVDA", "TSLA", "SPY", "QQQ", "AMZN", "EURUSD=X", "BTC-USD"],
        "swing": ["AAPL", "NVDA", "SPY", "QQQ", "AMZN", "GLD", "BTC-USD"],
        "position": ["AAPL", "MSFT", "SPY", "QQQ", "AMZN", "GLD", "BTC-USD"]
    },
    "intraday_filter": {
        "enabled": True, "min_atr_pct": 0.35, "max_distance_from_ema20_pct": 1.0,
        "max_distance_from_vwap_pct": 0.9, "max_bar_range_atr": 0.9,
        "us_session": {"start_utc": "13:35", "end_utc": "20:00"},
        "forex_session": {"start_utc": "06:00", "end_utc": "20:00"},
        "crypto_session": {"start_utc": "00:00", "end_utc": "23:59"}
    },
    "benchmark_filter": {"enabled": True, "equity": ["SPY", "QQQ"], "crypto": ["BTC-USD"], "forex": []},
    "logs": {"signals_path": "signals_log.csv", "watchlist_path": "watchlist_candidates.csv"}
}
def deep_merge(a,b):
    o=a.copy()
    for k,v in b.items():
        o[k]=deep_merge(o[k],v) if isinstance(v,dict) and isinstance(o.get(k),dict) else v
    return o
def load_config(path="config.json"):
    if os.path.exists(path):
        with open(path,"r",encoding="utf-8") as f: return deep_merge(DEFAULT_CONFIG,json.load(f))
    return DEFAULT_CONFIG
def ema(s,span): return s.ewm(span=span,adjust=False).mean()
def rsi(s,period=14):
    d=s.diff(); g=d.where(d>0,0.0); l=-d.where(d<0,0.0)
    ag=g.ewm(alpha=1/period,min_periods=period,adjust=False).mean()
    al=l.ewm(alpha=1/period,min_periods=period,adjust=False).mean()
    rs=ag/al.replace(0,math.nan); out=100-(100/(1+rs)); return out.fillna(50)
def macd(s,fast=12,slow=26,signal=9):
    m=ema(s,fast)-ema(s,slow); sig=ema(m,signal); return m,sig,m-sig
def atr(df,period=14):
    pc=df["Close"].shift(1)
    tr=pd.concat([(df["High"]-df["Low"]),(df["High"]-pc).abs(),(df["Low"]-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/period,min_periods=period,adjust=False).mean()
def vwap(df):
    typ=(df["High"]+df["Low"]+df["Close"])/3; vol=df["Volume"].fillna(0)
    return ((typ*vol).cumsum()/vol.cumsum().replace(0,math.nan)).fillna(df["Close"])
def download_ohlcv(symbol, interval, period):
    try:
        df=yf.download(tickers=symbol,interval=interval,period=period,auto_adjust=False,progress=False,threads=False,group_by="column",prepost=True)
        if df is None or df.empty: return None
        if isinstance(df.columns,pd.MultiIndex): df.columns=[c[0] for c in df.columns]
        need=["Open","High","Low","Close","Volume"]
        for c in need:
            if c not in df.columns:
                if c=="Volume": df[c]=0
                else: return None
        return df[need].dropna(subset=["Open","High","Low","Close"])
    except Exception:
        return None
def classify_asset(symbol):
    if symbol.endswith("-USD"): return "crypto"
    if symbol.endswith("=X"): return "forex"
    return "equity"
def parse_hhmm_utc(text): hh,mm=text.split(":"); return int(hh)*60+int(mm)
def now_utc_minutes():
    n=datetime.now(timezone.utc); return n.hour*60+n.minute
def in_session(symbol,cfg):
    if not cfg.get("enabled",True): return True
    session=cfg.get(f"{classify_asset(symbol)}_session")
    if not session: return True
    start,end=parse_hhmm_utc(session["start_utc"]),parse_hhmm_utc(session["end_utc"]); cur=now_utc_minutes()
    return start<=cur<=end if start<=end else (cur>=start or cur<=end)

@dataclass
class Signal:
    symbol:str; strategy:str; side:str; timeframe:str; price:float; entry:float; stop:float; target:float; risk_reward:float; confidence:int; reasons:List[str]; timestamp_utc:str; category:str="tradable"

class AlertManager:
    def __init__(self, config):
        self.cooldown_seconds=int(config["cooldown_minutes"])*60
        self.telegram_cfg=config["telegram"]; self.heartbeat_cfg=config.get("heartbeat",{}); self.sent_cache={}; self.meta_path=".heartbeat_state.json"
        self.signals_path=config.get("logs",{}).get("signals_path","signals_log.csv"); self.watchlist_path=config.get("logs",{}).get("watchlist_path","watchlist_candidates.csv")
        self._ensure_csv(self.signals_path); self._ensure_csv(self.watchlist_path)
    def _ensure_csv(self,path):
        if not os.path.exists(path):
            pd.DataFrame(columns=list(asdict(Signal("", "", "", "", 0,0,0,0,0,0,[], "", "")).keys())).to_csv(path,index=False)
    def should_send(self,sig):
        now_ts=time.time(); key=f"{sig.symbol}|{sig.strategy}|{sig.side}|{sig.timeframe}"; last=self.sent_cache.get(key,0)
        if now_ts-last>=self.cooldown_seconds: self.sent_cache[key]=now_ts; return True
        return False
    def persist(self,sig,watchlist=False):
        row=asdict(sig).copy(); row["reasons"]=" | ".join(sig.reasons); path=self.watchlist_path if watchlist else self.signals_path
        pd.DataFrame([row]).to_csv(path,mode="a",index=False,header=False)
    def _send_telegram(self,text):
        token=self.telegram_cfg.get("bot_token",""); chat_id=self.telegram_cfg.get("chat_id","")
        if not token or not chat_id: return
        try: requests.post(f"https://api.telegram.org/bot{token}/sendMessage",json={"chat_id":chat_id,"text":text},timeout=15)
        except Exception: pass
    @staticmethod
    def format_signal(sig, telegram=False):
        msg=(f"ALERTA {sig.side} | {sig.symbol} | {sig.strategy} | TF {sig.timeframe}\n"
             f"Precio: {sig.price:.4f}\nEntrada trigger: {sig.entry:.4f}\nStop: {sig.stop:.4f}\nObjetivo: {sig.target:.4f}\n"
             f"R/R: {sig.risk_reward:.2f}\nConfianza: {sig.confidence}/100\nMotivos: {'; '.join(sig.reasons)}\nUTC: {sig.timestamp_utc}")
        if not telegram: msg += "\nAviso: alerta cuantitativa. Confirma liquidez, spread, noticias y horario antes de operar."
        return msg
    def send_signal(self,sig):
        print(self.format_signal(sig,telegram=False)); print("-"*100); self.persist(sig,watchlist=(sig.category!="tradable"))
        if self.telegram_cfg.get("enabled") and sig.category=="tradable": self._send_telegram(self.format_signal(sig,telegram=True))
    def maybe_send_heartbeat(self, summary):
        if not self.heartbeat_cfg.get("enabled",False): return
        interval=int(self.heartbeat_cfg.get("interval_minutes",90))*60; last_ts=0.0
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path,"r",encoding="utf-8") as f: last_ts=float(json.load(f).get("last_heartbeat_ts",0))
            except Exception: last_ts=0.0
        now_ts=time.time()
        if now_ts-last_ts<interval: return
        top=summary.get("top_watchlist",[]); top_text=" | ".join([f"{s.symbol} {s.side} {s.confidence}/100" for s in top[:3]]) or "sin candidatas destacadas"
        msg=(f"🟢 Heartbeat inteligente\nUTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}\n"
             f"Activos analizados: {summary.get('assets_scanned',0)}\nSesgo mercado: {summary.get('market_bias','NEUTRAL')}\n"
             f"Volatilidad media intradía: {summary.get('avg_intraday_atr_pct',0):.2f}%\nSeñales operables: {summary.get('tradable_count',0)}\n"
             f"Candidatas watchlist: {summary.get('watchlist_count',0)}\nTop candidatas: {top_text}")
        print(msg)
        if self.telegram_cfg.get("enabled"): self._send_telegram(msg)
        with open(self.meta_path,"w",encoding="utf-8") as f: json.dump({"last_heartbeat_ts": now_ts},f)

class SignalEngine:
    def __init__(self,config):
        self.config=config; self.min_confidence=int(config["min_confidence"]); self.rr_min=float(config["risk_reward_min"])
        self.intraday_cfg=config.get("intraday_filter",{}); self.benchmark_cfg=config.get("benchmark_filter",{})
    def enrich(self,df,intraday=False):
        out=df.copy(); out["ema20"]=ema(out["Close"],20); out["ema50"]=ema(out["Close"],50); out["ema200"]=ema(out["Close"],200)
        out["rsi14"]=rsi(out["Close"],14); out["macd"],out["macd_signal"],out["macd_hist"]=macd(out["Close"]); out["atr14"]=atr(out,14)
        out["vol_ma20"]=out["Volume"].rolling(20).mean().replace(0,math.nan); out["rel_vol"]=(out["Volume"]/out["vol_ma20"]).replace([math.inf,-math.inf],math.nan).fillna(1)
        out["bar_range"]=out["High"]-out["Low"]; out["atr_pct"]=(out["atr14"]/out["Close"]*100).replace([math.inf,-math.inf],math.nan); out["vwap"]=vwap(out) if intraday else out["Close"]
        return out.dropna()
    def benchmark_regime(self,asset_class,side):
        if not self.benchmark_cfg.get("enabled",True): return True,"benchmark deshabilitado"
        symbols=self.benchmark_cfg.get(asset_class,[]); 
        if not symbols: return True,"sin benchmark aplicable"
        votes=[]; reasons=[]
        for sym in symbols:
            df=download_ohlcv(sym,"60m","30d")
            if df is None or len(df)<80: continue
            df=self.enrich(df,False); row=df.iloc[-1]; prev=df.iloc[-2]
            ok_long=row["Close"]>row["ema20"]>row["ema50"] and row["macd_hist"]>=prev["macd_hist"]
            ok_short=row["Close"]<row["ema20"]<row["ema50"] and row["macd_hist"]<=prev["macd_hist"]
            vote=ok_long if side=="LONG" else ok_short; votes.append(vote); reasons.append(f"{sym}:{'OK' if vote else 'NO'}")
        if not votes: return True,"benchmark sin datos"
        return sum(votes)>=math.ceil(len(votes)/2), f"benchmark {', '.join(reasons)}"
    def evaluate_intraday(self,symbol,df15,df60):
        if not in_session(symbol,self.intraday_cfg): return [],[],{"atr_pct":None}
        df15=self.enrich(df15,True); df60=self.enrich(df60,False)
        if len(df15)<220 or len(df60)<80: return [],[],{"atr_pct":None}
        row,prev,row60,prev60=df15.iloc[-1],df15.iloc[-2],df60.iloc[-1],df60.iloc[-2]
        tradable=[]; watchlist=[]; asset_class=classify_asset(symbol)
        min_atr=float(self.intraday_cfg.get("min_atr_pct",0.35)); max_ema=float(self.intraday_cfg.get("max_distance_from_ema20_pct",1.0)); max_vwap=float(self.intraday_cfg.get("max_distance_from_vwap_pct",0.9)); max_bar=float(self.intraday_cfg.get("max_bar_range_atr",0.9))
        atr_pct=float(row["atr_pct"]); dist_ema=abs((row["Close"]/row["ema20"]-1)*100); dist_vwap=abs((row["Close"]/row["vwap"]-1)*100); bar_range_atr=row["bar_range"]/max(row["atr14"],1e-9)
        # LONG
        reasons=[]; confidence=0
        if row60["ema20"]>row60["ema50"] and row60["macd_hist"]>=prev60["macd_hist"]: confidence+=22; reasons.append("1h alcista y MACD 1h acompaña")
        if row["ema20"]>row["ema50"]>row["ema200"]: confidence+=20; reasons.append("15m con medias alineadas al alza")
        if row["Close"]>row["vwap"] and prev["Low"]<=prev["ema20"]*1.003: confidence+=18; reasons.append("pullback a EMA20 y recuperación sobre VWAP")
        if 50<=row["rsi14"]<=67 and row["rsi14"]>prev["rsi14"]: confidence+=12; reasons.append("RSI fuerte sin sobreextensión")
        if row["macd_hist"]>prev["macd_hist"]: confidence+=10; reasons.append("momentum 15m mejorando")
        if row["rel_vol"]>=1.15: confidence+=8; reasons.append(f"volumen relativo {row['rel_vol']:.2f}x")
        if atr_pct>=min_atr: confidence+=6; reasons.append(f"ATR% suficiente {atr_pct:.2f}%")
        gated=[]
        gated.append(dist_ema<=max_ema); gated.append(dist_vwap<=max_vwap); gated.append(bar_range_atr<=max_bar)
        if dist_ema>max_ema: reasons.append(f"demasiado extendido de EMA20 ({dist_ema:.2f}%)")
        if dist_vwap>max_vwap: reasons.append(f"demasiado extendido de VWAP ({dist_vwap:.2f}%)")
        if bar_range_atr>max_bar: reasons.append(f"vela agotada ({bar_range_atr:.2f} ATR)")
        bench_ok, bench_reason=self.benchmark_regime(asset_class,"LONG"); reasons.append(bench_reason)
        entry=float(prev["High"]+0.05*row["atr14"]); stop=float(min(df15["Low"].iloc[-4:-1].min(),row["ema20"])-0.20*row["atr14"]); risk=max(entry-stop,1e-9); target=float(entry+2.2*risk); rr=(target-entry)/risk
        sig=Signal(symbol,"continuación intradía MTF","LONG","15m",float(row["Close"]),entry,stop,target,rr,min(confidence,99),reasons,datetime.now(timezone.utc).isoformat(),"watchlist")
        if confidence>=self.min_confidence and rr>=self.rr_min and all(gated) and bench_ok: sig.category="tradable"; tradable.append(sig)
        elif confidence>=max(55,self.min_confidence-12): watchlist.append(sig)
        # SHORT
        reasons=[]; confidence=0
        if row60["ema20"]<row60["ema50"] and row60["macd_hist"]<=prev60["macd_hist"]: confidence+=22; reasons.append("1h bajista y MACD 1h acompaña")
        if row["ema20"]<row["ema50"]<row["ema200"]: confidence+=20; reasons.append("15m con medias alineadas a la baja")
        if row["Close"]<row["vwap"] and prev["High"]>=prev["ema20"]*0.997: confidence+=18; reasons.append("pullback a EMA20 y rechazo bajo VWAP")
        if 33<=row["rsi14"]<=50 and row["rsi14"]<prev["rsi14"]: confidence+=12; reasons.append("RSI débil sin sobreextensión")
        if row["macd_hist"]<prev["macd_hist"]: confidence+=10; reasons.append("momentum 15m empeorando")
        if row["rel_vol"]>=1.15: confidence+=8; reasons.append(f"volumen relativo {row['rel_vol']:.2f}x")
        if atr_pct>=min_atr: confidence+=6; reasons.append(f"ATR% suficiente {atr_pct:.2f}%")
        gated=[]
        gated.append(dist_ema<=max_ema); gated.append(dist_vwap<=max_vwap); gated.append(bar_range_atr<=max_bar)
        if dist_ema>max_ema: reasons.append(f"demasiado extendido de EMA20 ({dist_ema:.2f}%)")
        if dist_vwap>max_vwap: reasons.append(f"demasiado extendido de VWAP ({dist_vwap:.2f}%)")
        if bar_range_atr>max_bar: reasons.append(f"vela agotada ({bar_range_atr:.2f} ATR)")
        bench_ok, bench_reason=self.benchmark_regime(asset_class,"SHORT"); reasons.append(bench_reason)
        entry=float(prev["Low"]-0.05*row["atr14"]); stop=float(max(df15["High"].iloc[-4:-1].max(),row["ema20"])+0.20*row["atr14"]); risk=max(stop-entry,1e-9); target=float(entry-2.2*risk); rr=(entry-target)/risk
        sig=Signal(symbol,"continuación intradía MTF","SHORT","15m",float(row["Close"]),entry,stop,target,rr,min(confidence,99),reasons,datetime.now(timezone.utc).isoformat(),"watchlist")
        if confidence>=self.min_confidence and rr>=self.rr_min and all(gated) and bench_ok: sig.category="tradable"; tradable.append(sig)
        elif confidence>=max(55,self.min_confidence-12): watchlist.append(sig)
        return tradable,watchlist,{"atr_pct":atr_pct}
    def evaluate_swing(self,symbol,df):
        df=self.enrich(df,False); 
        if len(df)<220: return []
        row,prev=df.iloc[-1],df.iloc[-2]; highs20=df["High"].rolling(20).max(); lows20=df["Low"].rolling(20).min(); out=[]
        reasons=[]; confidence=0
        if row["ema20"]>row["ema50"]: confidence+=20; reasons.append("tendencia positiva de corto/medio plazo")
        if row["Close"]>=highs20.iloc[-2]: confidence+=25; reasons.append("ruptura de máximo de 20 velas")
        if row["rel_vol"]>=1.5: confidence+=15; reasons.append(f"volumen relativo {row['rel_vol']:.2f}x")
        if row["rsi14"]>55: confidence+=10; reasons.append("RSI confirma fortaleza")
        if row["macd_hist"]>prev["macd_hist"]: confidence+=10; reasons.append("MACD acelera al alza")
        entry=float(row["Close"]); stop=float(entry-1.5*row["atr14"]); target=float(entry+3.0*row["atr14"]); rr=(target-entry)/max(entry-stop,1e-9)
        if confidence>=self.min_confidence and rr>=self.rr_min: out.append(Signal(symbol,"breakout intrasemana","LONG","1h",entry,entry,stop,target,rr,confidence,reasons,datetime.now(timezone.utc).isoformat()))
        reasons=[]; confidence=0
        if row["ema20"]<row["ema50"]: confidence+=20; reasons.append("tendencia negativa de corto/medio plazo")
        if row["Close"]<=lows20.iloc[-2]: confidence+=25; reasons.append("ruptura de mínimo de 20 velas")
        if row["rel_vol"]>=1.5: confidence+=15; reasons.append(f"volumen relativo {row['rel_vol']:.2f}x")
        if row["rsi14"]<45: confidence+=10; reasons.append("RSI confirma debilidad")
        if row["macd_hist"]<prev["macd_hist"]: confidence+=10; reasons.append("MACD acelera a la baja")
        entry=float(row["Close"]); stop=float(entry+1.5*row["atr14"]); target=float(entry-3.0*row["atr14"]); rr=(entry-target)/max(stop-entry,1e-9)
        if confidence>=self.min_confidence and rr>=self.rr_min: out.append(Signal(symbol,"breakdown intrasemana","SHORT","1h",entry,entry,stop,target,rr,confidence,reasons,datetime.now(timezone.utc).isoformat()))
        return out
    def evaluate_position(self,symbol,df):
        df=self.enrich(df,False)
        if len(df)<220: return []
        row,prev=df.iloc[-1],df.iloc[-2]; highs120=df["High"].rolling(120).max(); lows120=df["Low"].rolling(120).min(); out=[]
        reasons=[]; confidence=0
        if row["Close"]>row["ema200"]: confidence+=20; reasons.append("sesgo estructural alcista sobre EMA200")
        if row["ema20"]>row["ema50"]>row["ema200"]: confidence+=25; reasons.append("alineación alcista de medias")
        if row["Close"]>=highs120.iloc[-2]: confidence+=20; reasons.append("ruptura de máximo de 6 meses aprox.")
        if row["rsi14"]>55: confidence+=10; reasons.append("RSI acompaña")
        if row["macd_hist"]>prev["macd_hist"]: confidence+=10; reasons.append("momentum posicional mejorando")
        entry=float(row["Close"]); stop=float(entry-2.0*row["atr14"]); target=float(entry+4.0*row["atr14"]); rr=(target-entry)/max(entry-stop,1e-9)
        if confidence>=self.min_confidence and rr>=self.rr_min: out.append(Signal(symbol,"tendencia posicional","LONG","1d",entry,entry,stop,target,rr,confidence,reasons,datetime.now(timezone.utc).isoformat()))
        reasons=[]; confidence=0
        if row["Close"]<row["ema200"]: confidence+=20; reasons.append("sesgo estructural bajista bajo EMA200")
        if row["ema20"]<row["ema50"]<row["ema200"]: confidence+=25; reasons.append("alineación bajista de medias")
        if row["Close"]<=lows120.iloc[-2]: confidence+=20; reasons.append("ruptura de mínimo de 6 meses aprox.")
        if row["rsi14"]<45: confidence+=10; reasons.append("RSI acompaña debilidad")
        if row["macd_hist"]<prev["macd_hist"]: confidence+=10; reasons.append("momentum posicional empeorando")
        entry=float(row["Close"]); stop=float(entry+2.0*row["atr14"]); target=float(entry-4.0*row["atr14"]); rr=(entry-target)/max(stop-entry,1e-9)
        if confidence>=self.min_confidence and rr>=self.rr_min: out.append(Signal(symbol,"tendencia posicional","SHORT","1d",entry,entry,stop,target,rr,confidence,reasons,datetime.now(timezone.utc).isoformat()))
        return out

class Scanner:
    def __init__(self,config): self.config=config; self.engine=SignalEngine(config); self.alerts=AlertManager(config)
    def compute_market_bias(self,tradable,watchlist):
        longs=sum(1 for s in tradable if s.side=="LONG"); shorts=sum(1 for s in tradable if s.side=="SHORT")
        if longs-shorts>=2: return "ALCISTA"
        if shorts-longs>=2: return "BAJISTA"
        wl_longs=sum(1 for s in watchlist[:5] if s.side=="LONG"); wl_shorts=sum(1 for s in watchlist[:5] if s.side=="SHORT")
        if wl_longs>wl_shorts: return "LIGERAMENTE ALCISTA"
        if wl_shorts>wl_longs: return "LIGERAMENTE BAJISTA"
        return "NEUTRAL"
    def scan_once(self):
        tradable=[]; watchlist=[]; scanned=0; atr_values=[]
        for symbol in self.config["watchlists"].get("intraday",[]):
            df15=download_ohlcv(symbol,"15m","30d"); df60=download_ohlcv(symbol,"60m","60d")
            if df15 is not None and df60 is not None:
                scanned+=1; t,w,meta=self.engine.evaluate_intraday(symbol,df15,df60); tradable.extend(t); watchlist.extend(w)
                if meta.get("atr_pct") is not None: atr_values.append(meta["atr_pct"])
        for symbol in self.config["watchlists"].get("swing",[]):
            df=download_ohlcv(symbol,"60m","180d")
            if df is not None: scanned+=1; tradable.extend(self.engine.evaluate_swing(symbol,df))
        for symbol in self.config["watchlists"].get("position",[]):
            df=download_ohlcv(symbol,"1d","2y")
            if df is not None: scanned+=1; tradable.extend(self.engine.evaluate_position(symbol,df))
        tradable.sort(key=lambda s:(s.confidence,s.risk_reward),reverse=True); watchlist.sort(key=lambda s:(s.confidence,s.risk_reward),reverse=True)
        sent=0
        for sig in tradable:
            if self.alerts.should_send(sig): self.alerts.send_signal(sig); sent+=1
        for sig in watchlist[:5]: self.alerts.persist(sig,watchlist=True)
        avg_atr=sum(atr_values)/len(atr_values) if atr_values else 0
        summary={"assets_scanned":scanned,"market_bias":self.compute_market_bias(tradable,watchlist),"avg_intraday_atr_pct":avg_atr,"tradable_count":len(tradable),"watchlist_count":len(watchlist),"top_watchlist":watchlist[:3]}
        self.alerts.maybe_send_heartbeat(summary)
        if sent==0:
            print(f"{datetime.now().isoformat()} | Escaneo completado sin alertas operables nuevas. Candidatas watchlist: {len(watchlist)}")
            for sig in watchlist[:3]: print(f"WATCHLIST | {sig.symbol} | {sig.side} | {sig.strategy} | score {sig.confidence}/100 | R/R {sig.risk_reward:.2f}")
        else:
            print(f"{datetime.now().isoformat()} | Escaneo completado. Alertas enviadas: {sent}. Candidatas watchlist: {len(watchlist)}")
    def run_forever(self):
        interval=int(self.config["scan_interval_seconds"]); print(f"Iniciando scanner. Intervalo: {interval} s")
        while True:
            try: self.scan_once()
            except KeyboardInterrupt: print("Detenido por el usuario."); break
            except Exception: print("Error en el escaneo:"); traceback.print_exc()
            time.sleep(interval)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--once",action="store_true"); args=parser.parse_args()
    config=load_config(); scanner=Scanner(config); github=os.getenv("GITHUB_ACTIONS","").lower()=="true"
    if args.once or github: print("Iniciando scanner en modo una sola pasada"); scanner.scan_once()
    else: scanner.run_forever()
if __name__=="__main__": main()
