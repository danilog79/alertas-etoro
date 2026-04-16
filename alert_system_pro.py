import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf

DEFAULT_CONFIG = {
    "scan_interval_seconds": 300,
    "cooldown_minutes": 60,
    "min_confidence": 52,
    "risk_reward_min": 1.2,
    "top_candidates_in_heartbeat": 8,
    "heartbeat": {
        "enabled": True,
        "interval_minutes": 60,
    },
    "telegram": {
        "enabled": False,
        "bot_token": "",
        "chat_id": "",
    },
    "sources": {
        "fmp_enabled": False,
        "fmp_api_key": "",
        "fred_enabled": False,
        "fred_api_key": "",
        "etoro_enabled": False,
        "etoro_base_url": "https://api.etoro.com",
        "etoro_api_key": "",
        "etoro_user_key": "",
    },
    "intraday_filter": {
        "enabled": True,
        "min_atr_pct": 0.20,
        "max_distance_from_ema20_pct": 2.20,
        "max_distance_from_vwap_pct": 1.80,
        "max_trigger_bar_range_atr": 1.30,
        "us_session": {"start_utc": "13:35", "end_utc": "20:00"},
        "forex_session": {"start_utc": "06:00", "end_utc": "20:00"},
        "crypto_session": {"start_utc": "00:00", "end_utc": "23:59"},
    },
    "weights": {
        "regime": 20,
        "trend": 15,
        "setup": 20,
        "momentum": 10,
        "liquidity": 10,
        "space": 10,
        "event": 10,
        "etoro": 10,
    },
    "watchlists": {
        "benchmarks": ["SPY", "QQQ", "IWM", "DIA", "GLD", "TLT", "BTC-USD"],
        "intraday": [
            "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "NFLX", "AMD", "COIN",
            "SPY", "QQQ", "IWM", "DIA", "VTI", "ARKK", "XLE", "XLF", "XLK", "GLD", "SLV",
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "LINK-USD"
        ],
        "swing": [
            "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "NFLX", "AMD", "COIN",
            "JPM", "XOM", "LLY", "AVGO", "COST", "SPY", "QQQ", "IWM", "DIA", "VTI", "VT",
            "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLC", "XLRE",
            "GLD", "SLV", "TLT", "HYG", "LQD", "TIP", "EURUSD=X", "GBPUSD=X", "USDJPY=X",
            "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "LINK-USD", "XRP-USD", "DOGE-USD"
        ],
        "position": [
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "JPM", "XOM", "LLY", "AVGO", "COST",
            "SPY", "QQQ", "VTI", "VT", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLC", "XLRE",
            "GLD", "SLV", "TLT", "HYG", "LQD", "TIP", "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "LINK-USD"
        ],
        "etoro_featured_winners": ["GLD", "ARKK", "VXUS", "USDJPY=X"],
        "etoro_featured_losers": ["LINK-USD", "ETH-USD", "ADA-USD", "SOL-USD", "XLE"],
    },
}

FMP_BASE = "https://financialmodelingprep.com/stable"
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def deep_merge(a: Dict, b: Dict) -> Dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str = "config.json") -> Dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg = deep_merge(cfg, user_cfg)
    env_map = {
        ("telegram", "bot_token"): os.getenv("TELEGRAM_BOT_TOKEN", ""),
        ("telegram", "chat_id"): os.getenv("TELEGRAM_CHAT_ID", ""),
        ("sources", "fmp_api_key"): os.getenv("FMP_API_KEY", ""),
        ("sources", "fred_api_key"): os.getenv("FRED_API_KEY", ""),
        ("sources", "etoro_api_key"): os.getenv("ETORO_API_KEY", ""),
        ("sources", "etoro_user_key"): os.getenv("ETORO_USER_KEY", ""),
    }
    for path_tuple, val in env_map.items():
        if val:
            d = cfg
            for part in path_tuple[:-1]:
                d = d[part]
            d[path_tuple[-1]] = val
    return cfg


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, math.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].fillna(0)
    cum_tpv = (typical * vol).cumsum()
    cum_vol = vol.cumsum().replace(0, math.nan)
    return (cum_tpv / cum_vol).fillna(df["Close"])


def clean_download(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            if c == "Volume":
                df[c] = 0
            else:
                return None
    return df[needed].dropna(subset=["Open", "High", "Low", "Close"])


def download_ohlcv(symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(
            tickers=symbol,
            interval=interval,
            period=period,
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
            prepost=True,
        )
        return clean_download(df)
    except Exception:
        return None


def fetch_fmp_earnings(symbol: str, api_key: str) -> Optional[pd.Timestamp]:
    if not api_key:
        return None
    try:
        url = f"{FMP_BASE}/earnings-calendar-confirmed"
        params = {"symbol": symbol.replace("-USD", ""), "apikey": api_key}
        r = requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data:
            return None
        dt = pd.to_datetime(data[0].get("date"), utc=True, errors="coerce")
        return dt
    except Exception:
        return None


def fetch_fred_series(series_id: str, api_key: str) -> Optional[pd.DataFrame]:
    if not api_key:
        return None
    try:
        r = requests.get(FRED_BASE, params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
        }, timeout=12)
        if r.status_code != 200:
            return None
        obs = r.json().get("observations", [])
        if not obs:
            return None
        df = pd.DataFrame(obs)
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["date", "value"]].dropna()
    except Exception:
        return None


def try_etoro_recommendations(cfg: Dict) -> List[str]:
    src = cfg.get("sources", {})
    if not src.get("etoro_enabled"):
        return []
    if not src.get("etoro_api_key") or not src.get("etoro_user_key"):
        return []
    try:
        url = src.get("etoro_base_url", "https://api.etoro.com").rstrip("/") + "/api/market-recommendations"
        headers = {
            "x-api-key": src["etoro_api_key"],
            "x-user-key": src["etoro_user_key"],
        }
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200:
            return []
        data = r.json()
        out = []
        for item in data[:20]:
            symbol = item.get("symbol") or item.get("ticker") or item.get("displayName")
            if symbol:
                out.append(symbol)
        return out
    except Exception:
        return []




def detect_asset_type(symbol: str) -> str:
    if symbol.endswith("=X"):
        return "forex"
    if symbol.endswith("-USD"):
        return "crypto"
    return "equity"


def get_dynamic_movers(symbols: List[str], top_n: int = 10) -> Dict[str, List[str]]:
    ranked = []
    for symbol in list(dict.fromkeys(symbols)):
        df = download_ohlcv(symbol, "1d", "15d")
        if df is None or len(df) < 6:
            continue
        try:
            close = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2])
            week_prev = float(df["Close"].iloc[-6])
            day_chg = (close / prev - 1.0) * 100.0 if prev else 0.0
            week_chg = (close / week_prev - 1.0) * 100.0 if week_prev else 0.0
            vol_ma5 = float(df["Volume"].tail(6).iloc[:-1].mean()) if "Volume" in df.columns else 0.0
            rel_vol = float(df["Volume"].iloc[-1] / vol_ma5) if vol_ma5 else 1.0
            impulse = abs(day_chg) * 0.60 + abs(week_chg) * 0.25 + max(rel_vol - 1.0, 0) * 8.0
            ranked.append((symbol, day_chg, week_chg, rel_vol, impulse))
        except Exception:
            continue

    ranked.sort(key=lambda x: x[4], reverse=True)
    top = ranked[:top_n]
    winners = sorted([x for x in ranked if x[1] > 0], key=lambda x: (x[1], x[4]), reverse=True)[:top_n]
    losers = sorted([x for x in ranked if x[1] < 0], key=lambda x: (x[1], -x[4]))[:top_n]
    return {
        "dynamic_movers": [x[0] for x in top],
        "dynamic_winners": [x[0] for x in winners],
        "dynamic_losers": [x[0] for x in losers],
    }

@dataclass
class Signal:
    symbol: str
    strategy: str
    side: str
    timeframe: str
    price: float
    entry: float
    stop: float
    target: float
    risk_reward: float
    confidence: int
    score: float
    reasons: List[str]
    candidate_only: bool
    timestamp_utc: str


class SignalEngine:
    def __init__(self, config: Dict, regime: Dict):
        self.config = config
        self.min_confidence = config["min_confidence"]
        self.rr_min = config["risk_reward_min"]
        self.weights = config["weights"]
        self.regime = regime
        self.featured = set(config["watchlists"].get("etoro_featured_winners", []) + config["watchlists"].get("etoro_featured_losers", []))

    def enrich(self, df: pd.DataFrame, intraday: bool = False) -> pd.DataFrame:
        out = df.copy()
        out["ema20"] = ema(out["Close"], 20)
        out["ema50"] = ema(out["Close"], 50)
        out["ema200"] = ema(out["Close"], 200)
        out["rsi14"] = rsi(out["Close"], 14)
        out["macd"], out["macd_signal"], out["macd_hist"] = macd(out["Close"])
        out["atr14"] = atr(out, 14)
        out["vol_ma20"] = out["Volume"].rolling(20).mean().replace(0, math.nan)
        out["rel_vol"] = (out["Volume"] / out["vol_ma20"]).replace([math.inf, -math.inf], math.nan).fillna(1)
        out["vwap"] = vwap(out) if intraday else out["Close"]
        out["atr_pct"] = out["atr14"] / out["Close"] * 100
        out["daily_pct"] = out["Close"].pct_change() * 100
        out["bar_range"] = (out["High"] - out["Low"]) / out["Close"] * 100
        return out.dropna()

    def is_market_open_for_symbol(self, symbol: str) -> bool:
        filt = self.config.get("intraday_filter", {})
        if not filt.get("enabled", True):
            return True
        now = datetime.now(timezone.utc).time()
        if symbol.endswith("=X"):
            sess = filt["forex_session"]
        elif symbol.endswith("-USD"):
            sess = filt["crypto_session"]
        else:
            sess = filt["us_session"]
        start = datetime.strptime(sess["start_utc"], "%H:%M").time()
        end = datetime.strptime(sess["end_utc"], "%H:%M").time()
        return start <= now <= end

    def market_bias_score(self, side: str) -> Tuple[int, List[str]]:
        score = 0
        reasons = []
        if self.regime.get("tag") == "risk_on" and side == "LONG":
            score += self.weights["regime"]
            reasons.append("régimen risk-on acompaña")
        elif self.regime.get("tag") == "risk_off" and side == "SHORT":
            score += self.weights["regime"]
            reasons.append("régimen risk-off acompaña")
        elif self.regime.get("tag") == "mixed":
            score += self.weights["regime"] // 2
            reasons.append("régimen mixto: reducir exigencia táctica")
        return score, reasons

    def extra_symbol_bonus(self, symbol: str) -> Tuple[int, List[str]]:
        score = 0
        reasons = []
        if symbol in self.featured:
            score += self.weights["etoro"]
            reasons.append("activo aparece en destacados eToro")
        if symbol in self.regime.get("etoro_recommendations", []):
            score += self.weights["etoro"]
            reasons.append("activo aparece en recomendaciones eToro")
        if symbol in self.regime.get("dynamic_movers", []):
            score += self.weights["etoro"]
            reasons.append("activo aparece en movers dinámicos")
        if symbol in self.regime.get("dynamic_winners", []):
            score += self.weights["etoro"] // 2
            reasons.append("activo aparece en top ganadores dinámicos")
        if symbol in self.regime.get("dynamic_losers", []):
            score += self.weights["etoro"] // 2
            reasons.append("activo aparece en top perdedores dinámicos")
        return score, reasons

    def event_penalty(self, symbol: str) -> Tuple[int, List[str]]:
        score = self.weights["event"]
        reasons = []
        src = self.config.get("sources", {})
        if src.get("fmp_enabled") and src.get("fmp_api_key") and not symbol.endswith("=X") and not symbol.endswith("-USD"):
            earnings_dt = fetch_fmp_earnings(symbol, src["fmp_api_key"])
            if earnings_dt is not None:
                delta_h = abs((earnings_dt.to_pydatetime() - datetime.now(timezone.utc)).total_seconds()) / 3600.0
                if delta_h <= 48:
                    return 0, ["earnings cerca: penalización dura"]
        return score, reasons

    def evaluate_intraday(self, symbol: str, df15: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame) -> List[Signal]:
        if not self.is_market_open_for_symbol(symbol):
            return []
        df15 = self.enrich(df15, intraday=True)
        df1h = self.enrich(df1h, intraday=False)
        df4h = self.enrich(df4h, intraday=False)
        if df15 is None or df1h is None or df4h is None or df15.empty or df1h.empty or df4h.empty:
            return []
        if len(df15) < 60 or len(df1h) < 60 or len(df4h) < 30:
            return []

        row15 = df15.iloc[-1]
        prev15 = df15.iloc[-2]
        row1h = df1h.iloc[-1]
        row4h = df4h.iloc[-1]
        cfgf = self.config["intraday_filter"]
        results = []
        regime_tag = self.regime.get("tag", "mixed")

        for side in ["LONG", "SHORT"]:
            score = 0
            reasons: List[str] = []
            regime_score, regime_reasons = self.market_bias_score(side)
            score += regime_score
            reasons += regime_reasons

            if side == "LONG":
                cond_trend = bool(row4h["ema20"] > row4h["ema50"] and row1h["ema20"] > row1h["ema50"])
                cond_price = bool(row15["Close"] > row15["ema20"] or row15["Close"] > row15["vwap"])
                cond_rsi = bool(row15["rsi14"] > prev15["rsi14"] and 45 <= row15["rsi14"] <= 72)
                cond_macd = bool(row15["macd_hist"] > prev15["macd_hist"])
                entry = float(max(prev15["High"], row15["Close"]) * 1.0003)
                struct_stop = float(df15["Low"].tail(4).min())
                stop = min(struct_stop, float(entry - 1.4 * row15["atr14"]))
                target = float(entry + 1.9 * (entry - stop))
                candle_close_quality = (row15["Close"] - row15["Low"]) / max((row15["High"] - row15["Low"]), 1e-9)
                rev_cond = bool(regime_tag in ("mixed", "risk_on") and row15["Close"] < row15["vwap"] and row15["rsi14"] <= 42 and candle_close_quality >= 0.55)
                if rev_cond:
                    score += self.weights["setup"]
                    reasons.append("reversión intradía alcista")
                    entry = float(row15["Close"])
                    stop = float(min(df15["Low"].tail(3).min(), entry - 1.2 * row15["atr14"]))
                    target = float(max(row15["vwap"], entry + 1.5 * (entry - stop)))
            else:
                cond_trend = bool(row4h["ema20"] < row4h["ema50"] and row1h["ema20"] < row1h["ema50"])
                cond_price = bool(row15["Close"] < row15["ema20"] or row15["Close"] < row15["vwap"])
                cond_rsi = bool(row15["rsi14"] < prev15["rsi14"] and 28 <= row15["rsi14"] <= 55)
                cond_macd = bool(row15["macd_hist"] < prev15["macd_hist"])
                entry = float(min(prev15["Low"], row15["Close"]) * 0.9997)
                struct_stop = float(df15["High"].tail(4).max())
                stop = max(struct_stop, float(entry + 1.4 * row15["atr14"]))
                target = float(entry - 1.9 * (stop - entry))
                candle_close_quality = (row15["High"] - row15["Close"]) / max((row15["High"] - row15["Low"]), 1e-9)
                rev_cond = bool(regime_tag in ("mixed", "risk_off") and row15["Close"] > row15["vwap"] and row15["rsi14"] >= 58 and candle_close_quality >= 0.55)
                if rev_cond:
                    score += self.weights["setup"]
                    reasons.append("reversión intradía bajista")
                    entry = float(row15["Close"])
                    stop = float(max(df15["High"].tail(3).max(), entry + 1.2 * row15["atr14"]))
                    target = float(min(row15["vwap"], entry - 1.5 * (stop - entry)))

            conditions_met = sum([cond_trend, cond_price, cond_rsi, cond_macd])
            if cond_trend:
                score += self.weights["trend"]
                reasons.append("tendencia acompaña")
            if cond_price:
                score += self.weights["setup"] // 2
                reasons.append("precio acompaña EMA/VWAP")
            if cond_rsi:
                score += self.weights["momentum"] // 2
                reasons.append("RSI acompaña")
            if cond_macd:
                score += self.weights["momentum"] // 2
                reasons.append("MACD acompaña")

            atr_ok = bool(row15["atr_pct"] >= cfgf["min_atr_pct"])
            vol_ok = bool(row15["rel_vol"] >= 1.0)
            if atr_ok:
                score += self.weights["liquidity"] // 2
                reasons.append(f"ATR% suficiente ({row15['atr_pct']:.2f}%)")
            if vol_ok:
                score += self.weights["liquidity"] // 2
                reasons.append(f"volumen relativo {row15['rel_vol']:.2f}x")

            dist_ema20 = abs((row15["Close"] - row15["ema20"]) / row15["ema20"] * 100)
            dist_vwap = abs((row15["Close"] - row15["vwap"]) / row15["vwap"] * 100)
            too_extended = dist_ema20 > cfgf["max_distance_from_ema20_pct"] or dist_vwap > cfgf["max_distance_from_vwap_pct"]
            if not too_extended:
                score += self.weights["space"] // 2
                reasons.append("precio no está extendido")
            bar_ok = bool(row15["bar_range"] <= max(row15["atr_pct"] * cfgf["max_trigger_bar_range_atr"], 0.15))
            if bar_ok:
                score += self.weights["space"] // 2
                reasons.append("vela gatillo operable")
            if candle_close_quality >= 0.55:
                score += self.weights["setup"] // 2
                reasons.append("cierre útil en vela gatillo")

            event_score, event_reasons = self.event_penalty(symbol)
            score += event_score
            reasons += event_reasons
            extra_score, extra_reasons = self.extra_symbol_bonus(symbol)
            score += extra_score
            reasons += extra_reasons

            rr = ((target - entry) / max(entry - stop, 1e-9)) if side == "LONG" else ((entry - target) / max(stop - entry, 1e-9))
            has_reversal = any("reversión intradía" in r for r in reasons)
            candidate = (conditions_met >= 2 or has_reversal) and rr >= 1.05
            actionable = score >= self.min_confidence and rr >= self.rr_min and ((conditions_met >= 2 and not too_extended) or has_reversal)
            if candidate:
                strategy = "reversión intradía táctica" if has_reversal else "continuación intradía avanzada"
                results.append(Signal(
                    symbol=symbol,
                    strategy=strategy,
                    side=side,
                    timeframe="15m/1h/4h",
                    price=float(row15["Close"]),
                    entry=float(entry),
                    stop=float(stop),
                    target=float(target),
                    risk_reward=float(rr),
                    confidence=int(score),
                    score=float(score),
                    reasons=reasons,
                    candidate_only=not actionable,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                ))
        return results

    def evaluate_swing(self, symbol: str, dfd: pd.DataFrame, df4h: pd.DataFrame) -> List[Signal]:
        dfd = self.enrich(dfd, intraday=False)
        df4h = self.enrich(df4h, intraday=False)
        if dfd is None or df4h is None or dfd.empty or df4h.empty:
            return []
        if len(dfd) < 120 or len(df4h) < 60:
            return []
        rowd = dfd.iloc[-1]
        prevd = dfd.iloc[-2]
        row4h = df4h.iloc[-1]
        highs60 = dfd["High"].rolling(60).max()
        lows60 = dfd["Low"].rolling(60).min()
        results = []
        for side in ["LONG", "SHORT"]:
            score = 0
            reasons = []
            regime_score, regime_reasons = self.market_bias_score(side)
            score += regime_score
            reasons += regime_reasons

            if side == "LONG":
                cond_trend = bool(rowd["ema20"] > rowd["ema50"] and row4h["ema20"] > row4h["ema50"])
                cond_break = bool(rowd["Close"] >= highs60.iloc[-2] * 0.995)
                cond_momo = bool(rowd["rsi14"] > 50 and rowd["macd_hist"] >= prevd["macd_hist"])
                entry = float(rowd["Close"])
                stop = float(min(dfd["Low"].tail(6).min(), entry - 1.6 * rowd["atr14"]))
                target = float(entry + 2.1 * (entry - stop))
            else:
                cond_trend = bool(rowd["ema20"] < rowd["ema50"] and row4h["ema20"] < row4h["ema50"])
                cond_break = bool(rowd["Close"] <= lows60.iloc[-2] * 1.005)
                cond_momo = bool(rowd["rsi14"] < 50 and rowd["macd_hist"] <= prevd["macd_hist"])
                entry = float(rowd["Close"])
                stop = float(max(dfd["High"].tail(6).max(), entry + 1.6 * rowd["atr14"]))
                target = float(entry - 2.1 * (stop - entry))

            conditions_met = sum([cond_trend, cond_break, cond_momo])
            if cond_trend:
                score += self.weights["trend"] + 3
                reasons.append("sesgo swing acompaña")
            if cond_break:
                score += self.weights["setup"]
                reasons.append("cerca de ruptura relevante")
            if cond_momo:
                score += self.weights["momentum"]
                reasons.append("momentum swing acompaña")

            event_score, event_reasons = self.event_penalty(symbol)
            score += event_score
            reasons += event_reasons
            extra_score, extra_reasons = self.extra_symbol_bonus(symbol)
            score += extra_score
            reasons += extra_reasons

            rr = ((target - entry) / max(entry - stop, 1e-9)) if side == "LONG" else ((entry - target) / max(stop - entry, 1e-9))
            candidate = conditions_met >= 2 and rr >= 1.1
            actionable = score >= self.min_confidence and rr >= self.rr_min and conditions_met >= 2
            if candidate:
                results.append(Signal(symbol, "swing 1-8 semanas", side, "4h/1d", float(rowd["Close"]), entry, float(stop), float(target), float(rr), int(score), float(score), reasons, not actionable, datetime.now(timezone.utc).isoformat()))
        return results


class AlertManager:
    def __init__(self, config: Dict):
        self.cooldown_seconds = config["cooldown_minutes"] * 60
        self.telegram_cfg = config["telegram"]
        self.heartbeat_cfg = config.get("heartbeat", {})
        self.sent_cache: Dict[str, float] = {}
        self.last_heartbeat = 0.0
        self.log_path = "signals_log.csv"
        self.debug_path = "last_run_summary.json"
        if not os.path.exists(self.log_path):
            pd.DataFrame(columns=list(asdict(Signal("", "", "", "", 0, 0, 0, 0, 0, 0, 0, [], False, "")).keys())).to_csv(self.log_path, index=False)

    def should_send(self, sig: Signal) -> bool:
        key = f"{sig.symbol}|{sig.strategy}|{sig.side}|{sig.timeframe}|{int(sig.candidate_only)}"
        now_ts = time.time()
        last = self.sent_cache.get(key, 0)
        if now_ts - last >= self.cooldown_seconds:
            self.sent_cache[key] = now_ts
            return True
        return False

    def persist(self, sig: Signal):
        row = asdict(sig).copy()
        row["reasons"] = " | ".join(sig.reasons)
        pd.DataFrame([row]).to_csv(self.log_path, mode="a", index=False, header=False)

    def _send_telegram(self, text: str):
        if not self.telegram_cfg.get("enabled"):
            return
        token = self.telegram_cfg.get("bot_token", "")
        chat_id = self.telegram_cfg.get("chat_id", "")
        if not token or not chat_id:
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=12)
        except Exception:
            pass

    def send(self, sig: Signal):
        message = self.format_signal(sig)
        print(message)
        print("-" * 100)
        self.persist(sig)
        self._send_telegram(message)

    def save_summary(self, summary: Dict):
        with open(self.debug_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def maybe_send_heartbeat(self, summary: Dict):
        if not self.heartbeat_cfg.get("enabled"):
            return
        now = time.time()
        interval = self.heartbeat_cfg.get("interval_minutes", 60) * 60
        if now - self.last_heartbeat < interval:
            return
        msg = self.format_heartbeat(summary)
        print(msg)
        self._send_telegram(msg)
        self.last_heartbeat = now

    @staticmethod
        @staticmethod     def format_signal(sig: Signal) -> str:         if sig.score >= 65 and sig.risk_reward >= 1.5:             quality = "ALTA"         elif sig.score >= 60:             quality = "MEDIA"         else:             quality = "BAJA"          header = "🟡 CANDIDATA" if sig.candidate_only else "🚨 SEÑAL"          return (             f"{header} {sig.side} | {sig.symbol} | {sig.strategy} | TF {sig.timeframe}
             f"Calidad: {quality}
             f"Precio: {sig.price:.4f}
             f"Entrada: {sig.entry:.4f}
             f"Stop: {sig.stop:.4f}
             f"Objetivo: {sig.target:.4f}
             f"R/R: {sig.risk_reward:.2f}
             f"Confianza: {sig.confidence}/100
             f"Motivos: {'; '.join(sig.reasons[:8])}
             f"UTC: {sig.timestamp_utc}"         )
        header = "🟡 CANDIDATA" if sig.candidate_only else "🚨 SEÑAL"
        if sig.score >= 65 and sig.risk_reward >= 1.5:
            quality = "ALTA"
        elif sig.score >= 60 and sig.risk_reward >= 1.3:
            quality = "MEDIA"
        else:
            quality = "BAJA"
        return (
            f"{header} {sig.side} | {sig.symbol} | {sig.strategy} | TF {sig.timeframe}

            f"Calidad: {quality}

            f"Precio: {sig.price:.4f}

            f"Entrada: {sig.entry:.4f}

            f"Stop: {sig.stop:.4f}

            f"Objetivo: {sig.target:.4f}

            f"R/R: {sig.risk_reward:.2f}

            f"Confianza: {sig.confidence}/100

            f"Motivos: {'; '.join(sig.reasons[:8])}

            f"UTC: {sig.timestamp_utc}"
        )

    @staticmethod
    def format_heartbeat(summary: Dict) -> str:
        candidates = summary.get("top_candidates", [])
        lines = [
            "🟢 Heartbeat inteligente",
            f"UTC: {summary.get('timestamp_utc')}",
            f"Régimen: {summary.get('regime_tag')} | Bias: {summary.get('market_bias')}",
            f"Volatilidad media intradía: {summary.get('avg_intraday_atr_pct', 0):.2f}%",
            f"Activos analizados: {summary.get('scanned_symbols', 0)} | Señales: {summary.get('signals_sent', 0)} | Candidatas: {summary.get('candidate_count', 0)}",
        ]
        if summary.get("featured_context"):
            lines.append(f"Destacados eToro: {summary['featured_context']}")
        if candidates:
            lines.append("Top oportunidades no confirmadas:")
            for c in candidates[:5]:
                lines.append(f"- {c['symbol']} {c['side']} score {c['score']:.0f} RR {c['risk_reward']:.2f}")
        return "\n".join(lines)


class Scanner:
    def __init__(self, config: Dict):
        self.config = config
        self.alerts = AlertManager(config)
        self.regime = self.build_regime()
        self.engine = SignalEngine(config, self.regime)

    def build_regime(self) -> Dict:
        regime = {"tag": "mixed", "market_bias": "neutral", "etoro_recommendations": []}
        spy = download_ohlcv("SPY", "1d", "1y")
        qqq = download_ohlcv("QQQ", "1d", "1y")
        gld = download_ohlcv("GLD", "1d", "1y")
        btc = download_ohlcv("BTC-USD", "1d", "1y")
        bullish = 0
        bearish = 0
        for df in [spy, qqq, btc]:
            if df is None or len(df) < 220:
                continue
            e = ema(df["Close"], 20)
            e50 = ema(df["Close"], 50)
            e200 = ema(df["Close"], 200)
            if df["Close"].iloc[-1] > e.iloc[-1] > e50.iloc[-1] > e200.iloc[-1]:
                bullish += 1
            elif df["Close"].iloc[-1] < e.iloc[-1] < e50.iloc[-1] < e200.iloc[-1]:
                bearish += 1
        if bullish >= 2:
            regime["tag"] = "risk_on"
            regime["market_bias"] = "favorece largos"
        elif bearish >= 2:
            regime["tag"] = "risk_off"
            regime["market_bias"] = "favorece cortos / defensivos"
        else:
            regime["tag"] = "mixed"
            regime["market_bias"] = "mercado mixto"
        regime["gld_trend"] = None
        if gld is not None and len(gld) >= 200:
            regime["gld_trend"] = "up" if gld["Close"].iloc[-1] > ema(gld["Close"], 50).iloc[-1] else "down"
        regime["etoro_recommendations"] = try_etoro_recommendations(self.config)
        mover_seed = (
            self.config["watchlists"].get("intraday", [])
            + self.config["watchlists"].get("swing", [])
            + self.config["watchlists"].get("etoro_featured_winners", [])
            + self.config["watchlists"].get("etoro_featured_losers", [])
            + regime["etoro_recommendations"]
        )
        regime.update(get_dynamic_movers(mover_seed, top_n=8))
        return regime

    def analyze_symbol_intraday(self, symbol: str) -> List[Signal]:
        df15 = download_ohlcv(symbol, "15m", "30d")
        df1h = download_ohlcv(symbol, "60m", "180d")
        df4h = download_ohlcv(symbol, "1h", "730d")
        if df15 is None or df1h is None or df4h is None:
            return []
        # resample 4h from 1h
        try:
            temp = df4h.copy()
            temp.index = pd.to_datetime(temp.index)
            df4h_res = temp.resample("4H").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
            return self.engine.evaluate_intraday(symbol, df15, df1h, df4h_res)
        except Exception:
            return []

    def analyze_symbol_swing(self, symbol: str) -> List[Signal]:
        dfd = download_ohlcv(symbol, "1d", "2y")
        df4h_raw = download_ohlcv(symbol, "1h", "730d")
        if dfd is None or df4h_raw is None:
            return []
        try:
            temp = df4h_raw.copy()
            temp.index = pd.to_datetime(temp.index)
            df4h = temp.resample("4H").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
            return self.engine.evaluate_swing(symbol, dfd, df4h)
        except Exception:
            return []

    def scan_once(self):
        actionable: List[Signal] = []
        candidates: List[Signal] = []
        all_signals: List[Signal] = []
        scanned = 0
        intraday_atr_pcts = []
        universe_intraday = list(dict.fromkeys(
            self.config["watchlists"].get("intraday", [])
            + self.config["watchlists"].get("etoro_featured_winners", [])
            + self.config["watchlists"].get("etoro_featured_losers", [])
            + self.regime.get("etoro_recommendations", [])
            + self.regime.get("dynamic_movers", [])
            + self.regime.get("dynamic_winners", [])
            + self.regime.get("dynamic_losers", [])
        ))
        universe_swing = list(dict.fromkeys(
            self.config["watchlists"].get("swing", [])
            + self.config["watchlists"].get("etoro_featured_winners", [])
            + self.config["watchlists"].get("etoro_featured_losers", [])
            + self.regime.get("dynamic_movers", [])
        ))

        seen = set()
        for symbol in universe_intraday:
            if symbol in seen:
                continue
            seen.add(symbol)
            scanned += 1
            try:
                sigs = self.analyze_symbol_intraday(symbol)
                for s in sigs:
                    all_signals.append(s)
                    if s.candidate_only:
                        candidates.append(s)
                    else:
                        actionable.append(s)
                df15 = download_ohlcv(symbol, "15m", "5d")
                if df15 is not None and len(df15) > 30:
                    edf = self.engine.enrich(df15, intraday=True)
                    if edf is not None and not edf.empty and "atr_pct" in edf.columns:
                        val = edf.iloc[-1].get("atr_pct")
                        if val is not None and pd.notna(val):
                            intraday_atr_pcts.append(float(val))
            except Exception:
                traceback.print_exc()

        for symbol in universe_swing:
            if symbol in seen:
                continue
            seen.add(symbol)
            scanned += 1
            try:
                sigs = self.analyze_symbol_swing(symbol)
                for s in sigs:
                    all_signals.append(s)
                    if s.candidate_only:
                        candidates.append(s)
                    else:
                        actionable.append(s)
            except Exception:
                traceback.print_exc()

        actionable.sort(key=lambda s: (s.score, s.risk_reward), reverse=True)
        candidates.sort(key=lambda s: (s.score, s.risk_reward), reverse=True)
        all_signals.sort(key=lambda s: (s.score, s.risk_reward, 0 if s.candidate_only else 1), reverse=True)

        sent = 0
        for sig in actionable[:12]:
            if self.alerts.should_send(sig):
                self.alerts.send(sig)
                sent += 1

        top_intraday = [s for s in all_signals if "intradía" in s.strategy][:3]
        top_swing = [s for s in all_signals if "swing" in s.strategy][:3]
        if top_intraday or top_swing:
            lines = ["🔥 TOP OPORTUNIDADES ACTUALES"]
            if top_intraday:
                lines.append("")
                lines.append("📊 INTRADÍA:")
                for i, s in enumerate(top_intraday, 1):
                    lines.append(f"{i}. {s.symbol} {s.side} | score {int(s.score)} | RR {s.risk_reward:.2f}")
            if top_swing:
                lines.append("")
                lines.append("📈 SWING:")
                for i, s in enumerate(top_swing, 1):
                    lines.append(f"{i}. {s.symbol} {s.side} | score {int(s.score)} | RR {s.risk_reward:.2f}")
            top_msg = "
".join(lines)
            print(top_msg)
            self.alerts._send_telegram(top_msg)

        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "regime_tag": self.regime.get("tag"),
            "market_bias": self.regime.get("market_bias"),
            "avg_intraday_atr_pct": round(sum(intraday_atr_pcts) / max(len(intraday_atr_pcts), 1), 3),
            "scanned_symbols": scanned,
            "signals_sent": sent,
            "candidate_count": len(candidates),
            "all_signal_count": len(all_signals),
            "featured_context": (
                f"ganadores={','.join(self.config['watchlists'].get('etoro_featured_winners', [])[:5])} | "
                f"perdedores={','.join(self.config['watchlists'].get('etoro_featured_losers', [])[:5])} | "
                f"dinámicos={','.join(self.regime.get('dynamic_movers', [])[:5])}"
            ),
            "top_candidates": [
                {
                    "symbol": c.symbol,
                    "side": c.side,
                    "score": c.score,
                    "risk_reward": c.risk_reward,
                } for c in all_signals[: self.config.get("top_candidates_in_heartbeat", 8)]
            ],
        }
        self.alerts.save_summary(summary)
        self.alerts.maybe_send_heartbeat(summary)

        print(f"{datetime.now().isoformat()} | Señales: {sent} | Total oportunidades: {len(all_signals)}")

    def run_forever(self):
        interval = int(self.config["scan_interval_seconds"])
        print(f"Iniciando scanner PRO. Intervalo: {interval} s")
        while True:
            try:
                self.scan_once()
            except KeyboardInterrupt:
                print("Detenido por el usuario.")
                break
            except Exception:
                print("Error en el escaneo:")
                traceback.print_exc()
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = load_config()
    scanner = Scanner(config)
    if args.once or os.getenv("GITHUB_ACTIONS", "").lower() == "true":
        scanner.scan_once()
    else:
        scanner.run_forever()


if __name__ == "__main__":
    main()
