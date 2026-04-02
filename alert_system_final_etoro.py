import argparse
import json
import math
import os
import time
import traceback
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf


DEFAULT_CONFIG = {
    "scan_interval_seconds": 300,
    "cooldown_minutes": 60,
    "min_confidence": 70,
    "risk_reward_min": 2.0,
    "telegram": {
        "enabled": False,
        "bot_token": "",
        "chat_id": ""
    },
    "heartbeat": {
        "enabled": True,
        "interval_minutes": 60,
        "send_only_when_no_signals": False
    },
    "intraday_filter": {
        "enabled": True,
        "min_atr_pct": 0.35,
        "max_distance_from_ema20_pct": 1.2,
        "max_distance_from_vwap_pct": 1.0,
        "max_trigger_candle_atr_multiple": 0.8,
        "us_session": {"start_utc": "13:35", "end_utc": "20:00"},
        "forex_session": {"start_utc": "06:00", "end_utc": "20:00"},
        "crypto_session": {"start_utc": "00:00", "end_utc": "23:59"}
    },
    "etoro_featured": {
        "enabled": True,
        "winners": ["GLD", "ARKK", "VXUS", "USDJPY=X"],
        "losers": ["LINK-USD", "ETH-USD", "ADA-USD", "SOL-USD", "XLE"],
        "score_boost": 10
    },
    "watchlists": {
        "intraday": [
            "SPY", "QQQ", "DIA", "IWM", "VTI", "GLD", "SLV", "ARKK", "VXUS", "XLE", "XLF", "XLK",
            "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "NFLX", "AMD", "COIN",
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
            "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "LINK-USD"
        ],
        "swing": [
            "SPY", "QQQ", "DIA", "IWM", "VTI", "GLD", "SLV", "ARKK", "VXUS", "XLE", "XLF", "XLK",
            "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "NFLX", "AMD", "COIN",
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
            "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "LINK-USD"
        ],
        "position": [
            "SPY", "QQQ", "DIA", "IWM", "VTI", "GLD", "SLV", "ARKK", "VXUS", "XLE", "XLF", "XLK",
            "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "NFLX", "AMD", "COIN",
            "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "LINK-USD"
        ]
    },
    "benchmarks": {
        "us": ["SPY", "QQQ"],
        "crypto": ["BTC-USD", "ETH-USD"],
        "forex": ["USDJPY=X", "EURUSD=X"]
    }
}


# -----------------------------
# Config loading
# -----------------------------
def deep_merge(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for k, v in override.items():
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
    return cfg


# -----------------------------
# Indicator functions
# -----------------------------
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


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
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
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].fillna(0)
    cum_tpv = (typical * vol).cumsum()
    cum_vol = vol.cumsum().replace(0, math.nan)
    return (cum_tpv / cum_vol).fillna(df["Close"])


# -----------------------------
# Helpers
# -----------------------------
def is_crypto_symbol(symbol: str) -> bool:
    return symbol.endswith("-USD")


def is_forex_symbol(symbol: str) -> bool:
    return symbol.endswith("=X")


def classify_symbol(symbol: str) -> str:
    if is_crypto_symbol(symbol):
        return "crypto"
    if is_forex_symbol(symbol):
        return "forex"
    return "us"


def pct_distance(a: float, b: float) -> float:
    if not b:
        return 0.0
    return abs((a / b) - 1.0) * 100.0


def _minutes_hhmm(value: str) -> int:
    h, m = value.split(":")
    return int(h) * 60 + int(m)


def _inside_window(now_min: int, start_min: int, end_min: int) -> bool:
    if start_min <= end_min:
        return start_min <= now_min <= end_min
    return now_min >= start_min or now_min <= end_min


def intraday_time_allowed(symbol: str, cfg: Dict, now_utc: Optional[datetime] = None) -> Tuple[bool, str]:
    filt = cfg.get("intraday_filter", {})
    if not filt.get("enabled", True):
        return True, "filtro horario desactivado"

    now = now_utc or datetime.now(timezone.utc)
    if is_forex_symbol(symbol) and now.weekday() >= 5:
        return False, "forex cerrado en fin de semana"

    kind = classify_symbol(symbol)
    if kind == "crypto":
        session = filt.get("crypto_session", {"start_utc": "00:00", "end_utc": "23:59"})
    elif kind == "forex":
        session = filt.get("forex_session", {"start_utc": "06:00", "end_utc": "20:00"})
    else:
        session = filt.get("us_session", {"start_utc": "13:35", "end_utc": "20:00"})

    now_min = now.hour * 60 + now.minute
    start = _minutes_hhmm(session["start_utc"])
    end = _minutes_hhmm(session["end_utc"])
    if not _inside_window(now_min, start, end):
        return False, f"fuera de ventana {session['start_utc']}-{session['end_utc']} UTC"
    return True, f"ventana activa {session['start_utc']}-{session['end_utc']} UTC"


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# -----------------------------
# Data access
# -----------------------------
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
        df = df[needed].dropna(subset=["Open", "High", "Low", "Close"])
        return df
    except Exception:
        return None


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
    reasons: List[str]
    timestamp_utc: str
    category: str = "operable"
    etoro_featured: bool = False


@dataclass
class HeartbeatSnapshot:
    timestamp_utc: str
    analyzed: int = 0
    regime_us: str = "NEUTRAL"
    regime_crypto: str = "NEUTRAL"
    regime_forex: str = "NEUTRAL"
    avg_intraday_atr_pct: float = 0.0
    featured_active: List[str] = field(default_factory=list)
    candidates: List[Signal] = field(default_factory=list)


class SignalEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.min_confidence = int(config["min_confidence"])
        self.rr_min = float(config["risk_reward_min"])
        self.intraday_filter_cfg = config.get("intraday_filter", {})
        etoro = config.get("etoro_featured", {})
        self.featured_winners = set(etoro.get("winners", []))
        self.featured_losers = set(etoro.get("losers", []))
        self.featured_all = self.featured_winners | self.featured_losers
        self.featured_boost = int(etoro.get("score_boost", 10)) if etoro.get("enabled", True) else 0
        self.market_regime: Dict[str, str] = {"us": "NEUTRAL", "crypto": "NEUTRAL", "forex": "NEUTRAL"}

    def set_market_regime(self, regime: Dict[str, str]):
        self.market_regime = regime

    def enrich(self, df: pd.DataFrame, intraday: bool = False) -> pd.DataFrame:
        out = df.copy()
        out["ema20"] = ema(out["Close"], 20)
        out["ema50"] = ema(out["Close"], 50)
        out["ema200"] = ema(out["Close"], 200)
        out["rsi14"] = rsi(out["Close"], 14)
        out["macd"], out["macd_signal"], out["macd_hist"] = macd(out["Close"])
        out["atr14"] = atr(out, 14)
        out["atr_pct"] = (out["atr14"] / out["Close"]).replace([math.inf, -math.inf], math.nan) * 100
        out["vol_ma20"] = out["Volume"].rolling(20).mean().replace(0, math.nan)
        out["rel_vol"] = (out["Volume"] / out["vol_ma20"]).replace([math.inf, -math.inf], math.nan).fillna(1)
        out["vwap"] = vwap(out) if intraday else out["Close"]
        out["bar_range"] = out["High"] - out["Low"]
        out["bar_range_atr"] = (out["bar_range"] / out["atr14"]).replace([math.inf, -math.inf], math.nan)
        return out.dropna()

    def _featured_info(self, symbol: str) -> Tuple[bool, Optional[str]]:
        if symbol in self.featured_winners:
            return True, "winner"
        if symbol in self.featured_losers:
            return True, "loser"
        return False, None

    def _apply_featured_score(self, symbol: str, confidence: int, reasons: List[str], side: str) -> int:
        featured, bucket = self._featured_info(symbol)
        if not featured:
            return confidence
        confidence += self.featured_boost
        if bucket == "winner":
            reasons.append("activo en destacados eToro: ganador del día")
            if side == "LONG":
                confidence += 3
        else:
            reasons.append("activo en destacados eToro: perdedor del día")
            if side == "SHORT":
                confidence += 3
        return confidence

    def _regime_bonus(self, symbol: str, side: str, reasons: List[str]) -> int:
        kind = classify_symbol(symbol)
        regime = self.market_regime.get(kind, "NEUTRAL")
        bonus = 0
        if regime == "RISK_ON" and side == "LONG":
            bonus += 10
            reasons.append(f"régimen {kind} favorable a largos")
        elif regime == "RISK_OFF" and side == "SHORT":
            bonus += 10
            reasons.append(f"régimen {kind} favorable a cortos")
        elif regime == "NEUTRAL":
            reasons.append(f"régimen {kind} neutral")
        else:
            bonus -= 8
            reasons.append(f"régimen {kind} no acompaña")
        return bonus

    def _candidate_or_operable(self, confidence: int, rr: float) -> str:
        if confidence >= self.min_confidence and rr >= self.rr_min:
            return "operable"
        if confidence >= max(self.min_confidence - 10, 55):
            return "watchlist"
        return "discard"

    def evaluate_intraday(self, symbol: str, df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> List[Signal]:
        df = self.enrich(df_15m, intraday=True)
        dfh = self.enrich(df_1h, intraday=False)
        if len(df) < 220 or len(dfh) < 120:
            return []

        row = df.iloc[-1]
        prev = df.iloc[-2]
        hrow = dfh.iloc[-1]
        hprev = dfh.iloc[-2]
        cfg = self.intraday_filter_cfg
        signals: List[Signal] = []
        allowed, reason = intraday_time_allowed(symbol, self.config)
        if not allowed:
            return []

        min_atr_pct = float(cfg.get("min_atr_pct", 0.35))
        max_dist_ema = float(cfg.get("max_distance_from_ema20_pct", 1.2))
        max_dist_vwap = float(cfg.get("max_distance_from_vwap_pct", 1.0))
        max_trigger_bar = float(cfg.get("max_trigger_candle_atr_multiple", 0.8))

        trigger_long = max(float(prev["High"]), float(row["Close"]))
        trigger_short = min(float(prev["Low"]), float(row["Close"]))
        stop_long = min(float(df["Low"].tail(3).min()), float(row["ema20"])) - 0.2 * float(row["atr14"])
        stop_short = max(float(df["High"].tail(3).max()), float(row["ema20"])) + 0.2 * float(row["atr14"])

        # LONG
        reasons = [reason]
        confidence = 0
        if hrow["ema20"] > hrow["ema50"] and hrow["Close"] > hrow["ema20"]:
            confidence += 20; reasons.append("1h en tendencia alcista")
        if hrow["macd_hist"] > hprev["macd_hist"]:
            confidence += 8; reasons.append("1h momentum mejorando")
        if row["ema20"] > row["ema50"] > row["ema200"]:
            confidence += 18; reasons.append("15m alineación alcista")
        if row["Close"] > row["vwap"]:
            confidence += 12; reasons.append("precio sobre VWAP")
        if row["Close"] >= row["ema20"] and prev["Close"] <= prev["ema20"] * 1.003:
            confidence += 10; reasons.append("pullback recuperó EMA20")
        if row["rsi14"] > prev["rsi14"] and 50 <= row["rsi14"] <= 67:
            confidence += 10; reasons.append("RSI acompaña sin sobreextensión")
        if row["macd_hist"] > prev["macd_hist"]:
            confidence += 7; reasons.append("MACD intradía acelera")
        if row["rel_vol"] >= 1.15:
            confidence += 8; reasons.append(f"volumen relativo {row['rel_vol']:.2f}x")
        if row["atr_pct"] >= min_atr_pct:
            confidence += 7; reasons.append(f"volatilidad suficiente ATR% {row['atr_pct']:.2f}")
        else:
            confidence -= 15; reasons.append("volatilidad intradía insuficiente")
        dist_ema = pct_distance(float(row["Close"]), float(row["ema20"]))
        dist_vwap = pct_distance(float(row["Close"]), float(row["vwap"]))
        if dist_ema <= max_dist_ema:
            confidence += 5; reasons.append("precio no está alejado de EMA20")
        else:
            confidence -= 12; reasons.append("precio demasiado extendido vs EMA20")
        if dist_vwap <= max_dist_vwap:
            confidence += 5; reasons.append("precio no está alejado de VWAP")
        else:
            confidence -= 10; reasons.append("precio demasiado extendido vs VWAP")
        if row["bar_range_atr"] <= max_trigger_bar:
            confidence += 5; reasons.append("vela gatillo no está agotada")
        else:
            confidence -= 10; reasons.append("vela gatillo demasiado larga")
        confidence += self._regime_bonus(symbol, "LONG", reasons)
        confidence = self._apply_featured_score(symbol, confidence, reasons, "LONG")
        entry = trigger_long
        risk = max(entry - stop_long, 1e-9)
        target = entry + 2.2 * risk
        rr = (target - entry) / risk
        category = self._candidate_or_operable(confidence, rr)
        if category != "discard":
            featured, _ = self._featured_info(symbol)
            signals.append(Signal(symbol, "continuación intradía avanzada", "LONG", "15m+1h", float(row["Close"]), float(entry), float(stop_long), float(target), float(rr), int(confidence), reasons, datetime.now(timezone.utc).isoformat(), category, featured))

        # SHORT
        reasons = [reason]
        confidence = 0
        if hrow["ema20"] < hrow["ema50"] and hrow["Close"] < hrow["ema20"]:
            confidence += 20; reasons.append("1h en tendencia bajista")
        if hrow["macd_hist"] < hprev["macd_hist"]:
            confidence += 8; reasons.append("1h momentum empeorando")
        if row["ema20"] < row["ema50"] < row["ema200"]:
            confidence += 18; reasons.append("15m alineación bajista")
        if row["Close"] < row["vwap"]:
            confidence += 12; reasons.append("precio bajo VWAP")
        if row["Close"] <= row["ema20"] and prev["Close"] >= prev["ema20"] * 0.997:
            confidence += 10; reasons.append("pullback perdió EMA20")
        if row["rsi14"] < prev["rsi14"] and 33 <= row["rsi14"] <= 50:
            confidence += 10; reasons.append("RSI acompaña debilidad")
        if row["macd_hist"] < prev["macd_hist"]:
            confidence += 7; reasons.append("MACD intradía empeora")
        if row["rel_vol"] >= 1.15:
            confidence += 8; reasons.append(f"volumen relativo {row['rel_vol']:.2f}x")
        if row["atr_pct"] >= min_atr_pct:
            confidence += 7; reasons.append(f"volatilidad suficiente ATR% {row['atr_pct']:.2f}")
        else:
            confidence -= 15; reasons.append("volatilidad intradía insuficiente")
        dist_ema = pct_distance(float(row["Close"]), float(row["ema20"]))
        dist_vwap = pct_distance(float(row["Close"]), float(row["vwap"]))
        if dist_ema <= max_dist_ema:
            confidence += 5; reasons.append("precio no está alejado de EMA20")
        else:
            confidence -= 12; reasons.append("precio demasiado extendido vs EMA20")
        if dist_vwap <= max_dist_vwap:
            confidence += 5; reasons.append("precio no está alejado de VWAP")
        else:
            confidence -= 10; reasons.append("precio demasiado extendido vs VWAP")
        if row["bar_range_atr"] <= max_trigger_bar:
            confidence += 5; reasons.append("vela gatillo no está agotada")
        else:
            confidence -= 10; reasons.append("vela gatillo demasiado larga")
        confidence += self._regime_bonus(symbol, "SHORT", reasons)
        confidence = self._apply_featured_score(symbol, confidence, reasons, "SHORT")
        entry = trigger_short
        risk = max(stop_short - entry, 1e-9)
        target = entry - 2.2 * risk
        rr = (entry - target) / risk
        category = self._candidate_or_operable(confidence, rr)
        if category != "discard":
            featured, _ = self._featured_info(symbol)
            signals.append(Signal(symbol, "continuación intradía avanzada", "SHORT", "15m+1h", float(row["Close"]), float(entry), float(stop_short), float(target), float(rr), int(confidence), reasons, datetime.now(timezone.utc).isoformat(), category, featured))

        return signals

    def evaluate_swing(self, symbol: str, df: pd.DataFrame) -> List[Signal]:
        df = self.enrich(df, intraday=False)
        if len(df) < 220:
            return []
        row = df.iloc[-1]
        prev = df.iloc[-2]
        highs20 = df["High"].rolling(20).max()
        lows20 = df["Low"].rolling(20).min()
        signals: List[Signal] = []

        reasons = []
        confidence = 0
        if row["ema20"] > row["ema50"]:
            confidence += 18; reasons.append("tendencia positiva")
        if row["Close"] >= highs20.iloc[-2]:
            confidence += 22; reasons.append("ruptura de máximo de 20 velas")
        if row["rel_vol"] >= 1.4:
            confidence += 10; reasons.append(f"volumen relativo {row['rel_vol']:.2f}x")
        if row["rsi14"] > 55:
            confidence += 8; reasons.append("RSI confirma fortaleza")
        if row["macd_hist"] > prev["macd_hist"]:
            confidence += 8; reasons.append("MACD mejora")
        confidence += self._regime_bonus(symbol, "LONG", reasons)
        confidence = self._apply_featured_score(symbol, confidence, reasons, "LONG")
        entry = float(row["Close"])
        stop = float(entry - 1.5 * row["atr14"])
        target = float(entry + 3.0 * row["atr14"])
        rr = (target - entry) / max(entry - stop, 1e-9)
        category = self._candidate_or_operable(confidence, rr)
        if category != "discard":
            featured, _ = self._featured_info(symbol)
            signals.append(Signal(symbol, "breakout intrasemana", "LONG", "1h", entry, entry, stop, target, rr, int(confidence), reasons, datetime.now(timezone.utc).isoformat(), category, featured))

        reasons = []
        confidence = 0
        if row["ema20"] < row["ema50"]:
            confidence += 18; reasons.append("tendencia negativa")
        if row["Close"] <= lows20.iloc[-2]:
            confidence += 22; reasons.append("ruptura de mínimo de 20 velas")
        if row["rel_vol"] >= 1.4:
            confidence += 10; reasons.append(f"volumen relativo {row['rel_vol']:.2f}x")
        if row["rsi14"] < 45:
            confidence += 8; reasons.append("RSI confirma debilidad")
        if row["macd_hist"] < prev["macd_hist"]:
            confidence += 8; reasons.append("MACD empeora")
        confidence += self._regime_bonus(symbol, "SHORT", reasons)
        confidence = self._apply_featured_score(symbol, confidence, reasons, "SHORT")
        entry = float(row["Close"])
        stop = float(entry + 1.5 * row["atr14"])
        target = float(entry - 3.0 * row["atr14"])
        rr = (entry - target) / max(stop - entry, 1e-9)
        category = self._candidate_or_operable(confidence, rr)
        if category != "discard":
            featured, _ = self._featured_info(symbol)
            signals.append(Signal(symbol, "breakdown intrasemana", "SHORT", "1h", entry, entry, stop, target, rr, int(confidence), reasons, datetime.now(timezone.utc).isoformat(), category, featured))
        return signals

    def evaluate_position(self, symbol: str, df: pd.DataFrame) -> List[Signal]:
        df = self.enrich(df, intraday=False)
        if len(df) < 220:
            return []
        row = df.iloc[-1]
        prev = df.iloc[-2]
        highs120 = df["High"].rolling(120).max()
        lows120 = df["Low"].rolling(120).min()
        signals: List[Signal] = []

        reasons = []
        confidence = 0
        if row["Close"] > row["ema200"]:
            confidence += 18; reasons.append("sesgo alcista sobre EMA200")
        if row["ema20"] > row["ema50"] > row["ema200"]:
            confidence += 20; reasons.append("alineación alcista de medias")
        if row["Close"] >= highs120.iloc[-2]:
            confidence += 18; reasons.append("ruptura de máximo de 6 meses")
        if row["rsi14"] > 55:
            confidence += 8; reasons.append("RSI acompaña")
        if row["macd_hist"] > prev["macd_hist"]:
            confidence += 8; reasons.append("momentum mejora")
        confidence += self._regime_bonus(symbol, "LONG", reasons)
        confidence = self._apply_featured_score(symbol, confidence, reasons, "LONG")
        entry = float(row["Close"])
        stop = float(entry - 2.0 * row["atr14"])
        target = float(entry + 4.0 * row["atr14"])
        rr = (target - entry) / max(entry - stop, 1e-9)
        category = self._candidate_or_operable(confidence, rr)
        if category != "discard":
            featured, _ = self._featured_info(symbol)
            signals.append(Signal(symbol, "tendencia posicional", "LONG", "1d", entry, entry, stop, target, rr, int(confidence), reasons, datetime.now(timezone.utc).isoformat(), category, featured))

        reasons = []
        confidence = 0
        if row["Close"] < row["ema200"]:
            confidence += 18; reasons.append("sesgo bajista bajo EMA200")
        if row["ema20"] < row["ema50"] < row["ema200"]:
            confidence += 20; reasons.append("alineación bajista de medias")
        if row["Close"] <= lows120.iloc[-2]:
            confidence += 18; reasons.append("ruptura de mínimo de 6 meses")
        if row["rsi14"] < 45:
            confidence += 8; reasons.append("RSI acompaña debilidad")
        if row["macd_hist"] < prev["macd_hist"]:
            confidence += 8; reasons.append("momentum empeora")
        confidence += self._regime_bonus(symbol, "SHORT", reasons)
        confidence = self._apply_featured_score(symbol, confidence, reasons, "SHORT")
        entry = float(row["Close"])
        stop = float(entry + 2.0 * row["atr14"])
        target = float(entry - 4.0 * row["atr14"])
        rr = (entry - target) / max(stop - entry, 1e-9)
        category = self._candidate_or_operable(confidence, rr)
        if category != "discard":
            featured, _ = self._featured_info(symbol)
            signals.append(Signal(symbol, "tendencia posicional", "SHORT", "1d", entry, entry, stop, target, rr, int(confidence), reasons, datetime.now(timezone.utc).isoformat(), category, featured))
        return signals


class AlertManager:
    def __init__(self, config: Dict):
        self.cooldown_seconds = int(config["cooldown_minutes"]) * 60
        self.telegram_cfg = config["telegram"]
        self.heartbeat_cfg = config.get("heartbeat", {})
        self.sent_cache: Dict[str, float] = {}
        self.last_heartbeat_ts = 0.0
        self.log_path = "signals_log.csv"
        self.candidate_log_path = "watchlist_candidates_log.csv"
        sig_keys = list(asdict(Signal("", "", "", "", 0, 0, 0, 0, 0, 0, [], "")).keys())
        for path in [self.log_path, self.candidate_log_path]:
            if not os.path.exists(path):
                pd.DataFrame(columns=sig_keys).to_csv(path, index=False)

    def should_send(self, sig: Signal) -> bool:
        now_ts = time.time()
        key = f"{sig.symbol}|{sig.strategy}|{sig.side}|{sig.timeframe}|{sig.category}"
        last = self.sent_cache.get(key, 0)
        if now_ts - last >= self.cooldown_seconds:
            self.sent_cache[key] = now_ts
            return True
        return False

    def persist(self, sig: Signal):
        row = asdict(sig).copy()
        row["reasons"] = " | ".join(sig.reasons)
        path = self.log_path if sig.category == "operable" else self.candidate_log_path
        pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=False)

    def send_signal(self, sig: Signal):
        message = self.format_signal(sig, for_telegram=False)
        print(message)
        print("-" * 100)
        self.persist(sig)
        if self.telegram_cfg.get("enabled"):
            self._send_telegram(self.format_signal(sig, for_telegram=True))

    def maybe_send_heartbeat(self, snapshot: HeartbeatSnapshot, operable_sent: int):
        if not self.heartbeat_cfg.get("enabled", False):
            return
        if self.heartbeat_cfg.get("send_only_when_no_signals", False) and operable_sent > 0:
            return
        interval = int(self.heartbeat_cfg.get("interval_minutes", 60)) * 60
        now = time.time()
        if now - self.last_heartbeat_ts < interval:
            return
        text = self.format_heartbeat(snapshot, operable_sent)
        print(text)
        if self.telegram_cfg.get("enabled"):
            self._send_telegram(text)
        self.last_heartbeat_ts = now

    def _send_telegram(self, text: str):
        token = self.telegram_cfg.get("bot_token", "")
        chat_id = self.telegram_cfg.get("chat_id", "")
        if not token or not chat_id:
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=12)
        except Exception:
            pass

    @staticmethod
    def format_signal(sig: Signal, for_telegram: bool = False) -> str:
        prefix = "🔥 ACTIVO DESTACADO ETORO\n" if sig.etoro_featured else ""
        category = "SEÑAL OPERABLE" if sig.category == "operable" else "CANDIDATA WATCHLIST"
        body = (
            f"{prefix}{category} | {sig.side} | {sig.symbol} | {sig.strategy} | TF {sig.timeframe}\n"
            f"Precio: {sig.price:.4f}\n"
            f"Entrada: {sig.entry:.4f}\n"
            f"Stop: {sig.stop:.4f}\n"
            f"Objetivo: {sig.target:.4f}\n"
            f"R/R: {sig.risk_reward:.2f}\n"
            f"Confianza: {sig.confidence}/100\n"
            f"Motivos: {'; '.join(sig.reasons[:8])}\n"
            f"UTC: {sig.timestamp_utc}"
        )
        if not for_telegram:
            return body + "\nAviso: esto es una alerta cuantitativa, no una orden automática."
        return body

    @staticmethod
    def format_heartbeat(snapshot: HeartbeatSnapshot, operable_sent: int) -> str:
        top = snapshot.candidates[:3]
        top_lines = []
        for sig in top:
            star = "🔥 " if sig.etoro_featured else ""
            top_lines.append(f"{star}{sig.symbol} {sig.side} score {sig.confidence} RR {sig.risk_reward:.2f}")
        featured = ", ".join(snapshot.featured_active[:8]) if snapshot.featured_active else "sin destacados activos"
        top_block = "\n".join(top_lines) if top_lines else "Sin candidatas relevantes en este ciclo"
        return (
            f"🟢 Heartbeat inteligente\n"
            f"UTC: {snapshot.timestamp_utc}\n"
            f"Activos analizados: {snapshot.analyzed}\n"
            f"Señales operables enviadas: {operable_sent}\n"
            f"Régimen US: {snapshot.regime_us}\n"
            f"Régimen Crypto: {snapshot.regime_crypto}\n"
            f"Régimen Forex: {snapshot.regime_forex}\n"
            f"Volatilidad intradía media ATR%: {snapshot.avg_intraday_atr_pct:.2f}\n"
            f"Destacados eToro activos: {featured}\n"
            f"Top oportunidades watchlist:\n{top_block}"
        )


class Scanner:
    def __init__(self, config: Dict):
        self.config = config
        self.engine = SignalEngine(config)
        self.alerts = AlertManager(config)
        self.watchlists = self._build_watchlists(config)

    def _build_watchlists(self, config: Dict) -> Dict[str, List[str]]:
        etoro = config.get("etoro_featured", {})
        featured = etoro.get("winners", []) + etoro.get("losers", []) if etoro.get("enabled", True) else []
        watchlists = {}
        for bucket, items in config.get("watchlists", {}).items():
            watchlists[bucket] = unique_preserve_order(list(items) + list(featured))
        return watchlists

    def _regime_from_df(self, df: Optional[pd.DataFrame]) -> str:
        if df is None or df.empty:
            return "NEUTRAL"
        df = self.engine.enrich(df, intraday=False)
        if len(df) < 80:
            return "NEUTRAL"
        row = df.iloc[-1]
        if row["Close"] > row["ema20"] > row["ema50"] and row["macd_hist"] > 0:
            return "RISK_ON"
        if row["Close"] < row["ema20"] < row["ema50"] and row["macd_hist"] < 0:
            return "RISK_OFF"
        return "NEUTRAL"

    def _compute_market_regime(self) -> Dict[str, str]:
        reg = {}
        for kind, symbols in self.config.get("benchmarks", {}).items():
            votes = []
            for symbol in symbols:
                df = download_ohlcv(symbol, interval="60m", period="90d")
                votes.append(self._regime_from_df(df))
            if votes.count("RISK_ON") > votes.count("RISK_OFF"):
                reg[kind] = "RISK_ON"
            elif votes.count("RISK_OFF") > votes.count("RISK_ON"):
                reg[kind] = "RISK_OFF"
            else:
                reg[kind] = "NEUTRAL"
        return reg

    def scan_once(self):
        regime = self._compute_market_regime()
        self.engine.set_market_regime(regime)
        all_signals: List[Signal] = []
        candidate_signals: List[Signal] = []
        intraday_atr_values: List[float] = []
        analyzed = 0
        featured_active: List[str] = []

        for symbol in self.watchlists.get("intraday", []):
            allowed, _ = intraday_time_allowed(symbol, self.config)
            if not allowed:
                continue
            df15 = download_ohlcv(symbol, interval="15m", period="30d")
            df1h = download_ohlcv(symbol, interval="60m", period="180d")
            if df15 is not None and df1h is not None:
                analyzed += 1
                enriched15 = self.engine.enrich(df15, intraday=True)
                if not enriched15.empty:
                    intraday_atr_values.append(float(enriched15.iloc[-1]["atr_pct"]))
                featured, _ = self.engine._featured_info(symbol)
                if featured:
                    featured_active.append(symbol)
                sigs = self.engine.evaluate_intraday(symbol, df15, df1h)
                for sig in sigs:
                    if sig.category == "operable":
                        all_signals.append(sig)
                    elif sig.category == "watchlist":
                        candidate_signals.append(sig)

        for symbol in self.watchlists.get("swing", []):
            df = download_ohlcv(symbol, interval="60m", period="180d")
            if df is not None:
                analyzed += 1
                featured, _ = self.engine._featured_info(symbol)
                if featured:
                    featured_active.append(symbol)
                sigs = self.engine.evaluate_swing(symbol, df)
                for sig in sigs:
                    if sig.category == "operable":
                        all_signals.append(sig)
                    elif sig.category == "watchlist":
                        candidate_signals.append(sig)

        for symbol in self.watchlists.get("position", []):
            df = download_ohlcv(symbol, interval="1d", period="2y")
            if df is not None:
                analyzed += 1
                featured, _ = self.engine._featured_info(symbol)
                if featured:
                    featured_active.append(symbol)
                sigs = self.engine.evaluate_position(symbol, df)
                for sig in sigs:
                    if sig.category == "operable":
                        all_signals.append(sig)
                    elif sig.category == "watchlist":
                        candidate_signals.append(sig)

        all_signals.sort(key=lambda s: (s.confidence, s.risk_reward, s.etoro_featured), reverse=True)
        candidate_signals.sort(key=lambda s: (s.confidence, s.risk_reward, s.etoro_featured), reverse=True)

        sent = 0
        for sig in all_signals:
            if self.alerts.should_send(sig):
                self.alerts.send_signal(sig)
                sent += 1

        snapshot = HeartbeatSnapshot(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            analyzed=analyzed,
            regime_us=regime.get("us", "NEUTRAL"),
            regime_crypto=regime.get("crypto", "NEUTRAL"),
            regime_forex=regime.get("forex", "NEUTRAL"),
            avg_intraday_atr_pct=sum(intraday_atr_values) / len(intraday_atr_values) if intraday_atr_values else 0.0,
            featured_active=unique_preserve_order(featured_active),
            candidates=candidate_signals[:10],
        )
        self.alerts.maybe_send_heartbeat(snapshot, sent)

        if sent == 0:
            print(f"{datetime.now().isoformat()} | Escaneo completado sin alertas operables nuevas. Candidatas: {len(candidate_signals)}")
        else:
            print(f"{datetime.now().isoformat()} | Escaneo completado. Alertas enviadas: {sent}. Candidatas: {len(candidate_signals)}")

    def run_forever(self):
        interval = int(self.config["scan_interval_seconds"])
        print(f"Iniciando scanner. Intervalo: {interval} s")
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


def is_running_in_github_actions() -> bool:
    return os.getenv("GITHUB_ACTIONS", "").lower() == "true"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scanner de alertas para mercado con contexto eToro.")
    parser.add_argument("--once", action="store_true", help="Ejecuta un solo escaneo y termina.")
    args = parser.parse_args()

    config = load_config()
    scanner = Scanner(config)
    if args.once or is_running_in_github_actions():
        scanner.scan_once()
    else:
        scanner.run_forever()
