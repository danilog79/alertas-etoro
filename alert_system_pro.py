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
    "min_confidence": 70,
    "risk_reward_min": 2.0,
    "top_candidates_in_heartbeat": 5,
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
        "min_atr_pct": 0.35,
        "max_distance_from_ema20_pct": 1.2,
        "max_distance_from_vwap_pct": 1.0,
        "max_trigger_bar_range_atr": 0.8,
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


def get_dynamic_market_movers(symbols: List[str], top_n: int = 12) -> Dict[str, List[str]]:
    """
    Reconstruye una versión dinámica de "destacados" a partir de movimiento diario
    y actividad reciente. Evita depender de una lista estática.
    """
    rows = []
    seen = set()

    for symbol in symbols:
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)

        try:
            df = download_ohlcv(symbol, "1d", "15d")
            if df is None or len(df) < 6:
                continue

            close = pd.to_numeric(df["Close"], errors="coerce")
            vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)

            if close.isna().all():
                continue

            daily_change_pct = float((close.iloc[-1] / close.iloc[-2] - 1.0) * 100.0) if len(close) >= 2 else 0.0
            weekly_change_pct = float((close.iloc[-1] / close.iloc[-6] - 1.0) * 100.0) if len(close) >= 6 else daily_change_pct

            vol_ma5 = float(vol.tail(6).head(5).mean()) if len(vol) >= 6 else float(vol.mean())
            rel_vol = float(vol.iloc[-1] / vol_ma5) if vol_ma5 and not math.isnan(vol_ma5) else 1.0

            move_score = abs(daily_change_pct) + 0.35 * abs(weekly_change_pct) + max(0.0, rel_vol - 1.0) * 4.0
            rows.append({
                "symbol": symbol,
                "daily_change_pct": daily_change_pct,
                "weekly_change_pct": weekly_change_pct,
                "rel_vol": rel_vol,
                "move_score": move_score,
            })
        except Exception:
            continue

    if not rows:
        return {"combined": [], "winners": [], "losers": []}

    rdf = pd.DataFrame(rows).sort_values(["move_score", "daily_change_pct"], ascending=[False, False])

    winners = rdf.sort_values(["daily_change_pct", "move_score"], ascending=[False, False])["symbol"].head(top_n).tolist()
    losers = rdf.sort_values(["daily_change_pct", "move_score"], ascending=[True, False])["symbol"].head(top_n).tolist()
    combined = rdf["symbol"].head(top_n).tolist()

    return {
        "combined": list(dict.fromkeys(combined)),
        "winners": list(dict.fromkeys(winners)),
        "losers": list(dict.fromkeys(losers)),
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
        etoro_weight = self.weights["etoro"]

        if symbol in self.featured:
            score += etoro_weight
            reasons.append("activo aparece en destacados eToro estáticos")

        if symbol in self.regime.get("dynamic_movers", []):
            score += etoro_weight + 3
            reasons.append("activo aparece en movers dinámicos actuales")

        if symbol in self.regime.get("dynamic_winners", []):
            score += 2
            reasons.append("activo en top ganadores dinámicos")

        if symbol in self.regime.get("dynamic_losers", []):
            score += 2
            reasons.append("activo en top perdedores dinámicos")

        if symbol in self.regime.get("etoro_recommendations", []):
            score += etoro_weight
            reasons.append("activo aparece en recomendaciones eToro")

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
        if len(df15) < 220 or len(df1h) < 220 or len(df4h) < 120:
            return []
        row15 = df15.iloc[-1]
        prev15 = df15.iloc[-2]
        row1h = df1h.iloc[-1]
        row4h = df4h.iloc[-1]
        cfgf = self.config["intraday_filter"]
        results = []

        for side in ["LONG", "SHORT"]:
            score = 0
            reasons: List[str] = []
            regime_score, regime_reasons = self.market_bias_score(side)
            score += regime_score
            reasons += regime_reasons

            # trend multi-timeframe
            if side == "LONG":
                if row4h["ema20"] > row4h["ema50"] and row1h["ema20"] > row1h["ema50"]:
                    score += self.weights["trend"]
                    reasons.append("tendencia 4h/1h alcista")
                if row15["Close"] > row15["ema20"] and row15["Close"] > row15["vwap"]:
                    score += self.weights["setup"] // 2
                    reasons.append("precio 15m sobre EMA20 y VWAP")
                if row15["rsi14"] > prev15["rsi14"] and 50 <= row15["rsi14"] <= 70:
                    score += self.weights["momentum"]
                    reasons.append("RSI 15m mejora en zona útil")
                if row15["macd_hist"] > prev15["macd_hist"]:
                    score += self.weights["momentum"] // 2
                    reasons.append("MACD 15m acelera")
                entry = float(prev15["High"] * 1.0005)
                struct_stop = float(df15["Low"].tail(3).min())
                stop = min(struct_stop, float(entry - 1.6 * row15["atr14"]))
                target = float(entry + 2.4 * (entry - stop))
                candle_close_quality = (row15["Close"] - row15["Low"]) / max((row15["High"] - row15["Low"]), 1e-9)
            else:
                if row4h["ema20"] < row4h["ema50"] and row1h["ema20"] < row1h["ema50"]:
                    score += self.weights["trend"]
                    reasons.append("tendencia 4h/1h bajista")
                if row15["Close"] < row15["ema20"] and row15["Close"] < row15["vwap"]:
                    score += self.weights["setup"] // 2
                    reasons.append("precio 15m bajo EMA20 y VWAP")
                if row15["rsi14"] < prev15["rsi14"] and 30 <= row15["rsi14"] <= 50:
                    score += self.weights["momentum"]
                    reasons.append("RSI 15m debilita en zona útil")
                if row15["macd_hist"] < prev15["macd_hist"]:
                    score += self.weights["momentum"] // 2
                    reasons.append("MACD 15m cae")
                entry = float(prev15["Low"] * 0.9995)
                struct_stop = float(df15["High"].tail(3).max())
                stop = max(struct_stop, float(entry + 1.6 * row15["atr14"]))
                target = float(entry - 2.4 * (stop - entry))
                candle_close_quality = (row15["High"] - row15["Close"]) / max((row15["High"] - row15["Low"]), 1e-9)

            # liquidity/volatility filters
            if row15["atr_pct"] >= cfgf["min_atr_pct"]:
                score += self.weights["liquidity"] // 2
                reasons.append(f"ATR% suficiente ({row15['atr_pct']:.2f}%)")
            if row15["rel_vol"] >= 1.2:
                score += self.weights["liquidity"] // 2
                reasons.append(f"volumen relativo {row15['rel_vol']:.2f}x")

            dist_ema20 = abs((row15["Close"] - row15["ema20"]) / row15["ema20"] * 100)
            dist_vwap = abs((row15["Close"] - row15["vwap"]) / row15["vwap"] * 100)
            too_extended = dist_ema20 > cfgf["max_distance_from_ema20_pct"] or dist_vwap > cfgf["max_distance_from_vwap_pct"]
            if not too_extended:
                score += self.weights["space"] // 2
                reasons.append("precio no está extendido")
            if row15["bar_range"] <= row15["atr_pct"] * cfgf["max_trigger_bar_range_atr"]:
                score += self.weights["space"] // 2
                reasons.append("vela gatillo no está excesivamente larga")
            if candle_close_quality >= 0.65:
                score += self.weights["setup"] // 2
                reasons.append("cierre fuerte en la vela gatillo")

            event_score, event_reasons = self.event_penalty(symbol)
            score += event_score
            reasons += event_reasons
            extra_score, extra_reasons = self.extra_symbol_bonus(symbol)
            score += extra_score
            reasons += extra_reasons

            rr = ((target - entry) / max(entry - stop, 1e-9)) if side == "LONG" else ((entry - target) / max(stop - entry, 1e-9))
            candidate = score >= max(55, self.min_confidence - 10) and rr >= 1.4
            actionable = score >= self.min_confidence and rr >= self.rr_min and not too_extended
            if candidate:
                results.append(Signal(
                    symbol=symbol,
                    strategy="continuación intradía avanzada",
                    side=side,
                    timeframe="15m/1h/4h",
                    price=float(row15["Close"]),
                    entry=entry,
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
        if len(dfd) < 220 or len(df4h) < 120:
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
                if rowd["ema20"] > rowd["ema50"] > rowd["ema200"] and row4h["ema20"] > row4h["ema50"]:
                    score += self.weights["trend"] + 5
                    reasons.append("sesgo swing alcista diario + 4h")
                if rowd["Close"] >= highs60.iloc[-2]:
                    score += self.weights["setup"]
                    reasons.append("rompe máximo de 60 días")
                if rowd["rsi14"] > 55 and rowd["macd_hist"] > prevd["macd_hist"]:
                    score += self.weights["momentum"]
                    reasons.append("momentum diario acompaña")
                entry = float(rowd["Close"])
                stop = float(min(dfd["Low"].tail(5).min(), entry - 1.8 * rowd["atr14"]))
                target = float(entry + 2.8 * (entry - stop))
            else:
                if rowd["ema20"] < rowd["ema50"] < rowd["ema200"] and row4h["ema20"] < row4h["ema50"]:
                    score += self.weights["trend"] + 5
                    reasons.append("sesgo swing bajista diario + 4h")
                if rowd["Close"] <= lows60.iloc[-2]:
                    score += self.weights["setup"]
                    reasons.append("rompe mínimo de 60 días")
                if rowd["rsi14"] < 45 and rowd["macd_hist"] < prevd["macd_hist"]:
                    score += self.weights["momentum"]
                    reasons.append("momentum diario acompaña")
                entry = float(rowd["Close"])
                stop = float(max(dfd["High"].tail(5).max(), entry + 1.8 * rowd["atr14"]))
                target = float(entry - 2.8 * (stop - entry))
            event_score, event_reasons = self.event_penalty(symbol)
            score += event_score
            reasons += event_reasons
            extra_score, extra_reasons = self.extra_symbol_bonus(symbol)
            score += extra_score
            reasons += extra_reasons
            rr = ((target - entry) / max(entry - stop, 1e-9)) if side == "LONG" else ((entry - target) / max(stop - entry, 1e-9))
            candidate = score >= max(58, self.min_confidence - 10) and rr >= 1.6
            actionable = score >= self.min_confidence and rr >= self.rr_min
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
    def format_signal(sig: Signal) -> str:
        header = "🟡 CANDIDATA" if sig.candidate_only else "🚨 SEÑAL"
        return (
            f"{header} {sig.side} | {sig.symbol} | {sig.strategy} | TF {sig.timeframe}\n"
            f"Precio: {sig.price:.4f}\n"
            f"Entrada: {sig.entry:.4f}\n"
            f"Stop: {sig.stop:.4f}\n"
            f"Objetivo: {sig.target:.4f}\n"
            f"R/R: {sig.risk_reward:.2f}\n"
            f"Confianza: {sig.confidence}/100\n"
            f"Motivos: {'; '.join(sig.reasons[:8])}\n"
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
        regime = {
            "tag": "mixed",
            "market_bias": "neutral",
            "etoro_recommendations": [],
            "dynamic_movers": [],
            "dynamic_winners": [],
            "dynamic_losers": [],
        }
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

        dynamic_universe = list(dict.fromkeys(
            self.config["watchlists"].get("intraday", []) +
            self.config["watchlists"].get("swing", []) +
            self.config["watchlists"].get("etoro_featured_winners", []) +
            self.config["watchlists"].get("etoro_featured_losers", []) +
            regime["etoro_recommendations"]
        ))
        movers = get_dynamic_market_movers(dynamic_universe, top_n=12)
        regime["dynamic_movers"] = movers.get("combined", [])
        regime["dynamic_winners"] = movers.get("winners", [])
        regime["dynamic_losers"] = movers.get("losers", [])

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
        scanned = 0
        intraday_atr_pcts = []
        universe_intraday = list(dict.fromkeys(
            self.config["watchlists"].get("intraday", []) +
            self.config["watchlists"].get("etoro_featured_winners", []) +
            self.config["watchlists"].get("etoro_featured_losers", []) +
            self.regime.get("etoro_recommendations", []) +
            self.regime.get("dynamic_movers", []) +
            self.regime.get("dynamic_winners", []) +
            self.regime.get("dynamic_losers", [])
        ))
        universe_swing = list(dict.fromkeys(
            self.config["watchlists"].get("swing", []) +
            self.config["watchlists"].get("etoro_featured_winners", []) +
            self.config["watchlists"].get("etoro_featured_losers", []) +
            self.regime.get("dynamic_movers", []) +
            self.regime.get("dynamic_winners", []) +
            self.regime.get("dynamic_losers", [])
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
                    if s.candidate_only:
                        candidates.append(s)
                    else:
                        actionable.append(s)
                df15 = download_ohlcv(symbol, "15m", "5d")
                if df15 is not None and len(df15) > 30:
                    edf = self.engine.enrich(df15, intraday=True)
                    intraday_atr_pcts.append(float(edf.iloc[-1]["atr_pct"]))
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
                    if s.candidate_only:
                        candidates.append(s)
                    else:
                        actionable.append(s)
            except Exception:
                traceback.print_exc()

        actionable.sort(key=lambda s: (s.score, s.risk_reward), reverse=True)
        candidates.sort(key=lambda s: (s.score, s.risk_reward), reverse=True)

        sent = 0
        for sig in actionable[:12]:
            if self.alerts.should_send(sig):
                self.alerts.send(sig)
                sent += 1

        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "regime_tag": self.regime.get("tag"),
            "market_bias": self.regime.get("market_bias"),
            "avg_intraday_atr_pct": round(sum(intraday_atr_pcts) / max(len(intraday_atr_pcts), 1), 3),
            "scanned_symbols": scanned,
            "signals_sent": sent,
            "candidate_count": len(candidates),
            "featured_context": (
                f"estáticos_ganadores={','.join(self.config['watchlists'].get('etoro_featured_winners', [])[:4])} | "
                f"estáticos_perdedores={','.join(self.config['watchlists'].get('etoro_featured_losers', [])[:4])} | "
                f"dinámicos={','.join(self.regime.get('dynamic_movers', [])[:6])}"
            ),
            "top_candidates": [
                {
                    "symbol": c.symbol,
                    "side": c.side,
                    "score": c.score,
                    "risk_reward": c.risk_reward,
                } for c in candidates[: self.config.get("top_candidates_in_heartbeat", 5)]
            ],
        }
        self.alerts.save_summary(summary)
        self.alerts.maybe_send_heartbeat(summary)

        if sent == 0:
            print(f"{datetime.now().isoformat()} | Escaneo completado sin señales operables nuevas. Candidatas: {len(candidates)}")
        else:
            print(f"{datetime.now().isoformat()} | Escaneo completado. Señales enviadas: {sent}. Candidatas: {len(candidates)}")

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
