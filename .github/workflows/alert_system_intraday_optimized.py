
import argparse
import json
import math
import os
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf


# -----------------------------
# Config loading
# -----------------------------
DEFAULT_CONFIG = {
    "scan_interval_seconds": 300,
    "cooldown_minutes": 60,
    "min_confidence": 60,
    "risk_reward_min": 1.8,
    "telegram": {
        "enabled": False,
        "bot_token": "",
        "chat_id": ""
    },
    "intraday_filter": {
        "enabled": True,
        "min_atr_pct": 0.35,
        "max_distance_from_ema20_pct": 1.2,
        "max_distance_from_vwap_pct": 1.0,
        "avoid_first_minutes": 5,
        "us_session": {"start_utc": "13:35", "end_utc": "20:00"},
        "forex_session": {"start_utc": "06:00", "end_utc": "20:00"},
        "crypto_session": {"start_utc": "00:00", "end_utc": "23:59"}
    },
    "watchlists": {
        "intraday": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "EURUSD=X", "GBPUSD=X", "BTC-USD", "ETH-USD"],
        "swing": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GLD", "SLV", "BTC-USD", "ETH-USD"],
        "position": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GLD", "SLV", "BTC-USD", "ETH-USD"]
    }
}


def load_config(path: str = "config.json") -> Dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg = DEFAULT_CONFIG.copy()
        for k, v in user_cfg.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                merged = cfg[k].copy()
                merged.update(v)
                cfg[k] = merged
            else:
                cfg[k] = v
        return cfg
    return DEFAULT_CONFIG


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


def _parse_hhmm(text: str) -> Tuple[int, int]:
    hour, minute = text.split(":")
    return int(hour), int(minute)


def _minutes_now_utc(now: datetime) -> int:
    return now.hour * 60 + now.minute


def _minutes_hhmm(value: str) -> int:
    h, m = _parse_hhmm(value)
    return h * 60 + m


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

    start = _minutes_hhmm(session["start_utc"])
    end = _minutes_hhmm(session["end_utc"])
    now_min = _minutes_now_utc(now)

    if not _inside_window(now_min, start, end):
        return False, f"fuera de ventana horaria {session['start_utc']}-{session['end_utc']} UTC"

    return True, f"ventana horaria activa {session['start_utc']}-{session['end_utc']} UTC"


def pct_distance(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return abs((a / b) - 1.0) * 100.0


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


# -----------------------------
# Signal definitions
# -----------------------------
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


class SignalEngine:
    def __init__(self, min_confidence: int, rr_min: float, intraday_filter_cfg: Optional[Dict] = None):
        self.min_confidence = min_confidence
        self.rr_min = rr_min
        self.intraday_filter_cfg = intraday_filter_cfg or {}

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
        if intraday:
            out["vwap"] = vwap(out)
        else:
            out["vwap"] = out["Close"]
        out["daily_pct"] = out["Close"].pct_change() * 100
        out["atr_pct"] = (out["atr14"] / out["Close"]).replace([math.inf, -math.inf], math.nan) * 100
        out["dist_ema20_pct"] = ((out["Close"] / out["ema20"]) - 1).abs() * 100
        out["dist_vwap_pct"] = ((out["Close"] / out["vwap"]) - 1).abs() * 100
        out["candle_body_pct"] = ((out["Close"] - out["Open"]).abs() / out["Close"]).replace([math.inf, -math.inf], math.nan) * 100
        out["upper_wick_pct"] = ((out["High"] - out[["Open", "Close"]].max(axis=1)) / out["Close"]).clip(lower=0) * 100
        out["lower_wick_pct"] = ((out[["Open", "Close"]].min(axis=1) - out["Low"]) / out["Close"]).clip(lower=0) * 100
        return out.dropna()

    def _intraday_market_filter(self, symbol: str, row: pd.Series) -> Tuple[bool, List[str], int]:
        reasons = []
        confidence_delta = 0

        allowed, window_reason = intraday_time_allowed(symbol, {"intraday_filter": self.intraday_filter_cfg})
        if not allowed:
            return False, [window_reason], 0
        reasons.append(window_reason)

        min_atr_pct = float(self.intraday_filter_cfg.get("min_atr_pct", 0.35))
        max_dist_ema20 = float(self.intraday_filter_cfg.get("max_distance_from_ema20_pct", 1.2))
        max_dist_vwap = float(self.intraday_filter_cfg.get("max_distance_from_vwap_pct", 1.0))

        if row["atr_pct"] < min_atr_pct:
            return False, [f"volatilidad insuficiente ATR% {row['atr_pct']:.2f} < {min_atr_pct:.2f}"], 0
        reasons.append(f"ATR% {row['atr_pct']:.2f}")

        if row["dist_ema20_pct"] > max_dist_ema20:
            return False, [f"precio demasiado extendido vs EMA20 ({row['dist_ema20_pct']:.2f}%)"], 0
        reasons.append(f"distancia EMA20 {row['dist_ema20_pct']:.2f}%")

        if row["dist_vwap_pct"] > max_dist_vwap:
            return False, [f"precio demasiado extendido vs VWAP ({row['dist_vwap_pct']:.2f}%)"], 0
        reasons.append(f"distancia VWAP {row['dist_vwap_pct']:.2f}%")

        if row["candle_body_pct"] > max(0.7, 1.2 * row["atr_pct"]):
            return False, [f"vela actual demasiado expandida ({row['candle_body_pct']:.2f}%)"], 0
        reasons.append("vela no está excesivamente extendida")

        confidence_delta += 8
        return True, reasons, confidence_delta

    def evaluate_intraday(self, symbol: str, df: pd.DataFrame) -> List[Signal]:
        df = self.enrich(df, intraday=True)
        if len(df) < 220:
            return []
        row = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        highs20 = df["High"].rolling(20).max()
        lows20 = df["Low"].rolling(20).min()
        signals = []

        allowed, filter_reasons, filter_boost = self._intraday_market_filter(symbol, row)
        if not allowed:
            return []

        # Long pullback continuation
        reasons = []
        confidence = filter_boost
        reasons.extend(filter_reasons)

        if row["ema20"] > row["ema50"] > row["ema200"]:
            confidence += 25; reasons.append("tendencia alcista alineada (EMA20>EMA50>EMA200)")
        if row["Close"] > row["vwap"]:
            confidence += 12; reasons.append("precio sobre VWAP")
        if row["Close"] >= row["ema20"] and prev["Close"] <= prev["ema20"] * 1.003:
            confidence += 12; reasons.append("pullback y recuperación sobre EMA20")
        if 50 <= row["rsi14"] <= 64 and row["rsi14"] > prev["rsi14"] >= prev2["rsi14"]:
            confidence += 14; reasons.append("RSI recuperando en zona operable")
        if row["macd_hist"] > prev["macd_hist"] > prev2["macd_hist"]:
            confidence += 12; reasons.append("momentum MACD mejora en 3 velas")
        if row["rel_vol"] >= 1.15:
            confidence += 8; reasons.append(f"volumen relativo {row['rel_vol']:.2f}x")
        if row["High"] <= highs20.iloc[-2] * 1.003:
            confidence += 6; reasons.append("aún no está demasiado lejos del rango reciente")
        if row["Close"] > row["Open"] and row["lower_wick_pct"] >= row["upper_wick_pct"]:
            confidence += 6; reasons.append("vela con rechazo inferior")

        entry = float(row["Close"])
        stop = float(min(row["ema20"], prev["Low"], row["Close"] - 1.15 * row["atr14"]))
        if stop >= entry:
            stop = float(entry - 1.15 * row["atr14"])
        target = float(entry + 2.1 * max(entry - stop, row["atr14"]))
        rr = (target - entry) / max(entry - stop, 1e-9)
        if confidence >= self.min_confidence and rr >= self.rr_min:
            signals.append(Signal(symbol, "continuación intradía filtrada", "LONG", "15m", entry, entry, stop, target, rr, confidence, reasons, datetime.now(timezone.utc).isoformat()))

        # Short breakdown continuation
        reasons = []
        confidence = filter_boost
        reasons.extend(filter_reasons)

        if row["ema20"] < row["ema50"] < row["ema200"]:
            confidence += 25; reasons.append("tendencia bajista alineada (EMA20<EMA50<EMA200)")
        if row["Close"] < row["vwap"]:
            confidence += 12; reasons.append("precio bajo VWAP")
        if row["Close"] <= row["ema20"] and prev["Close"] >= prev["ema20"] * 0.997:
            confidence += 12; reasons.append("pullback y pérdida de EMA20")
        if 36 <= row["rsi14"] <= 50 and row["rsi14"] < prev["rsi14"] <= prev2["rsi14"]:
            confidence += 14; reasons.append("RSI debilitándose en zona operable")
        if row["macd_hist"] < prev["macd_hist"] < prev2["macd_hist"]:
            confidence += 12; reasons.append("momentum MACD empeora en 3 velas")
        if row["rel_vol"] >= 1.15:
            confidence += 8; reasons.append(f"volumen relativo {row['rel_vol']:.2f}x")
        if row["Low"] >= lows20.iloc[-2] * 0.997:
            confidence += 6; reasons.append("aún no está demasiado lejos del rango reciente")
        if row["Close"] < row["Open"] and row["upper_wick_pct"] >= row["lower_wick_pct"]:
            confidence += 6; reasons.append("vela con rechazo superior")

        entry = float(row["Close"])
        stop = float(max(row["ema20"], prev["High"], row["Close"] + 1.15 * row["atr14"]))
        if stop <= entry:
            stop = float(entry + 1.15 * row["atr14"])
        target = float(entry - 2.1 * max(stop - entry, row["atr14"]))
        rr = (entry - target) / max(stop - entry, 1e-9)
        if confidence >= self.min_confidence and rr >= self.rr_min:
            signals.append(Signal(symbol, "continuación intradía filtrada", "SHORT", "15m", entry, entry, stop, target, rr, confidence, reasons, datetime.now(timezone.utc).isoformat()))

        # Spike / dip detector for opportunistic alerts
        signals.extend(self._evaluate_spike(symbol, df, timeframe="15m", use_intraday_filter=True))
        return signals

    def evaluate_swing(self, symbol: str, df: pd.DataFrame) -> List[Signal]:
        df = self.enrich(df, intraday=False)
        if len(df) < 220:
            return []
        row = df.iloc[-1]
        prev = df.iloc[-2]
        highs20 = df["High"].rolling(20).max()
        lows20 = df["Low"].rolling(20).min()
        signals = []

        # Breakout long
        reasons = []
        confidence = 0
        if row["ema20"] > row["ema50"]:
            confidence += 20; reasons.append("tendencia positiva de corto/medio plazo")
        if row["Close"] >= highs20.iloc[-2]:
            confidence += 25; reasons.append("ruptura de máximo de 20 velas")
        if row["rel_vol"] >= 1.5:
            confidence += 15; reasons.append(f"volumen relativo {row['rel_vol']:.2f}x")
        if row["rsi14"] > 55:
            confidence += 10; reasons.append("RSI confirma fortaleza")
        if row["macd_hist"] > prev["macd_hist"]:
            confidence += 10; reasons.append("MACD acelera al alza")
        entry = float(row["Close"])
        stop = float(entry - 1.5 * row["atr14"])
        target = float(entry + 3.0 * row["atr14"])
        rr = (target - entry) / max(entry - stop, 1e-9)
        if confidence >= self.min_confidence and rr >= self.rr_min:
            signals.append(Signal(symbol, "breakout intrasemana", "LONG", "1h", entry, entry, stop, target, rr, confidence, reasons, datetime.now(timezone.utc).isoformat()))

        # Breakdown short
        reasons = []
        confidence = 0
        if row["ema20"] < row["ema50"]:
            confidence += 20; reasons.append("tendencia negativa de corto/medio plazo")
        if row["Close"] <= lows20.iloc[-2]:
            confidence += 25; reasons.append("ruptura de mínimo de 20 velas")
        if row["rel_vol"] >= 1.5:
            confidence += 15; reasons.append(f"volumen relativo {row['rel_vol']:.2f}x")
        if row["rsi14"] < 45:
            confidence += 10; reasons.append("RSI confirma debilidad")
        if row["macd_hist"] < prev["macd_hist"]:
            confidence += 10; reasons.append("MACD acelera a la baja")
        entry = float(row["Close"])
        stop = float(entry + 1.5 * row["atr14"])
        target = float(entry - 3.0 * row["atr14"])
        rr = (entry - target) / max(stop - entry, 1e-9)
        if confidence >= self.min_confidence and rr >= self.rr_min:
            signals.append(Signal(symbol, "breakdown intrasemana", "SHORT", "1h", entry, entry, stop, target, rr, confidence, reasons, datetime.now(timezone.utc).isoformat()))

        signals.extend(self._evaluate_spike(symbol, df, timeframe="1h", use_intraday_filter=False))
        return signals

    def evaluate_position(self, symbol: str, df: pd.DataFrame) -> List[Signal]:
        df = self.enrich(df, intraday=False)
        if len(df) < 220:
            return []
        row = df.iloc[-1]
        prev = df.iloc[-2]
        highs120 = df["High"].rolling(120).max()
        lows120 = df["Low"].rolling(120).min()
        signals = []

        reasons = []
        confidence = 0
        if row["Close"] > row["ema200"]:
            confidence += 20; reasons.append("sesgo estructural alcista sobre EMA200")
        if row["ema20"] > row["ema50"] > row["ema200"]:
            confidence += 25; reasons.append("alineación alcista de medias")
        if row["Close"] >= highs120.iloc[-2]:
            confidence += 20; reasons.append("ruptura de máximo de 6 meses aprox.")
        if row["rsi14"] > 55:
            confidence += 10; reasons.append("RSI acompaña")
        if row["macd_hist"] > prev["macd_hist"]:
            confidence += 10; reasons.append("momentum mensual/posicional mejorando")
        entry = float(row["Close"])
        stop = float(entry - 2.0 * row["atr14"])
        target = float(entry + 4.0 * row["atr14"])
        rr = (target - entry) / max(entry - stop, 1e-9)
        if confidence >= self.min_confidence and rr >= self.rr_min:
            signals.append(Signal(symbol, "tendencia posicional", "LONG", "1d", entry, entry, stop, target, rr, confidence, reasons, datetime.now(timezone.utc).isoformat()))

        reasons = []
        confidence = 0
        if row["Close"] < row["ema200"]:
            confidence += 20; reasons.append("sesgo estructural bajista bajo EMA200")
        if row["ema20"] < row["ema50"] < row["ema200"]:
            confidence += 25; reasons.append("alineación bajista de medias")
        if row["Close"] <= lows120.iloc[-2]:
            confidence += 20; reasons.append("ruptura de mínimo de 6 meses aprox.")
        if row["rsi14"] < 45:
            confidence += 10; reasons.append("RSI acompaña debilidad")
        if row["macd_hist"] < prev["macd_hist"]:
            confidence += 10; reasons.append("momentum mensual/posicional empeorando")
        entry = float(row["Close"])
        stop = float(entry + 2.0 * row["atr14"])
        target = float(entry - 4.0 * row["atr14"])
        rr = (entry - target) / max(stop - entry, 1e-9)
        if confidence >= self.min_confidence and rr >= self.rr_min:
            signals.append(Signal(symbol, "tendencia posicional", "SHORT", "1d", entry, entry, stop, target, rr, confidence, reasons, datetime.now(timezone.utc).isoformat()))

        signals.extend(self._evaluate_spike(symbol, df, timeframe="1d", use_intraday_filter=False))
        return signals

    def _evaluate_spike(self, symbol: str, df: pd.DataFrame, timeframe: str, use_intraday_filter: bool) -> List[Signal]:
        row = df.iloc[-1]
        prev = df.iloc[-2]
        signals = []
        move_pct = ((row["Close"] / prev["Close"]) - 1) * 100
        atr_pct = (row["atr14"] / row["Close"]) * 100 if row["Close"] else 0

        if use_intraday_filter:
            allowed, _, _ = self._intraday_market_filter(symbol, row)
            if not allowed:
                return []

        # Oversold bounce candidate
        reasons = []
        confidence = 0
        if move_pct <= -max(2.5, 1.5 * atr_pct):
            confidence += 25; reasons.append(f"caída brusca de {move_pct:.2f}%")
        if row["rel_vol"] >= 1.8:
            confidence += 15; reasons.append(f"volumen extremo {row['rel_vol']:.2f}x")
        if row["rsi14"] < 32:
            confidence += 20; reasons.append("RSI en sobreventa")
        if row["Close"] < row["ema20"] and row["Close"] < row["ema50"]:
            confidence += 10; reasons.append("desviación fuerte bajo medias")
        if confidence >= self.min_confidence:
            entry = float(row["Close"])
            stop = float(entry - 1.0 * row["atr14"])
            target = float(entry + 2.0 * row["atr14"])
            rr = (target - entry) / max(entry - stop, 1e-9)
            if rr >= self.rr_min:
                signals.append(Signal(symbol, "caída brusca: posible rebote", "LONG", timeframe, entry, entry, stop, target, rr, confidence, reasons, datetime.now(timezone.utc).isoformat()))

        # Overextended momentum / breakout
        reasons = []
        confidence = 0
        if move_pct >= max(2.5, 1.5 * atr_pct):
            confidence += 25; reasons.append(f"subida brusca de {move_pct:.2f}%")
        if row["rel_vol"] >= 1.8:
            confidence += 15; reasons.append(f"volumen extremo {row['rel_vol']:.2f}x")
        if row["rsi14"] > 68:
            confidence += 20; reasons.append("RSI con momentum alto")
        if row["Close"] > row["ema20"] and row["ema20"] > row["ema50"]:
            confidence += 10; reasons.append("estructura alcista acompaña")
        if confidence >= self.min_confidence:
            entry = float(row["Close"])
            stop = float(entry - 1.1 * row["atr14"])
            target = float(entry + 2.2 * row["atr14"])
            rr = (target - entry) / max(entry - stop, 1e-9)
            if rr >= self.rr_min:
                signals.append(Signal(symbol, "subida brusca: continuación", "LONG", timeframe, entry, entry, stop, target, rr, confidence, reasons, datetime.now(timezone.utc).isoformat()))
        return signals


# -----------------------------
# Alert delivery / persistence
# -----------------------------
def signal_key(sig: Signal) -> str:
    return f"{sig.symbol}|{sig.strategy}|{sig.side}|{sig.timeframe}"


class AlertManager:
    def __init__(self, cooldown_minutes: int, telegram_cfg: Dict):
        self.cooldown_seconds = cooldown_minutes * 60
        self.telegram_cfg = telegram_cfg
        self.sent_cache: Dict[str, float] = {}
        self.log_path = "signals_log.csv"
        if not os.path.exists(self.log_path):
            pd.DataFrame(columns=list(asdict(Signal("", "", "", "", 0, 0, 0, 0, 0, 0, [], "")).keys())).to_csv(self.log_path, index=False)

    def should_send(self, sig: Signal) -> bool:
        now_ts = time.time()
        key = signal_key(sig)
        last = self.sent_cache.get(key, 0)
        if now_ts - last >= self.cooldown_seconds:
            self.sent_cache[key] = now_ts
            return True
        return False

    def persist(self, sig: Signal):
        row = asdict(sig).copy()
        row["reasons"] = " | ".join(sig.reasons)
        pd.DataFrame([row]).to_csv(self.log_path, mode="a", index=False, header=False)

    def send(self, sig: Signal):
        console_message = self.format_signal(sig, include_warning=True)
        telegram_message = self.format_signal(sig, include_warning=False)
        print(console_message)
        print("-" * 100)
        self.persist(sig)
        if self.telegram_cfg.get("enabled"):
            self._send_telegram(telegram_message)

    def _send_telegram(self, text: str):
        token = self.telegram_cfg.get("bot_token", "")
        chat_id = self.telegram_cfg.get("chat_id", "")
        if not token or not chat_id:
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
        except Exception:
            pass

    @staticmethod
    def format_signal(sig: Signal, include_warning: bool = True) -> str:
        base = (
            f"ALERTA {sig.side} | {sig.symbol} | {sig.strategy} | TF {sig.timeframe}\n"
            f"Precio: {sig.price:.4f}\n"
            f"Entrada: {sig.entry:.4f}\n"
            f"Stop: {sig.stop:.4f}\n"
            f"Objetivo: {sig.target:.4f}\n"
            f"R/R: {sig.risk_reward:.2f}\n"
            f"Confianza: {sig.confidence}/100\n"
            f"Motivos: {'; '.join(sig.reasons)}\n"
            f"UTC: {sig.timestamp_utc}"
        )
        if include_warning:
            base += "\nAviso: esto es una alerta cuantitativa, no una orden automática. Confirma liquidez, spread, noticias y horario en eToro antes de operar."
        return base


# -----------------------------
# Scanner orchestration
# -----------------------------
class Scanner:
    def __init__(self, config: Dict):
        self.config = config
        self.engine = SignalEngine(
            config["min_confidence"],
            config["risk_reward_min"],
            config.get("intraday_filter", {})
        )
        self.alerts = AlertManager(config["cooldown_minutes"], config["telegram"])

    def scan_once(self):
        all_signals: List[Signal] = []

        for symbol in self.config["watchlists"].get("intraday", []):
            allowed, reason = intraday_time_allowed(symbol, self.config)
            if not allowed:
                print(f"{datetime.now().isoformat()} | {symbol} omitido: {reason}")
                continue
            df = download_ohlcv(symbol, interval="15m", period="30d")
            if df is not None:
                all_signals.extend(self.engine.evaluate_intraday(symbol, df))

        for symbol in self.config["watchlists"].get("swing", []):
            df = download_ohlcv(symbol, interval="60m", period="180d")
            if df is not None:
                all_signals.extend(self.engine.evaluate_swing(symbol, df))

        for symbol in self.config["watchlists"].get("position", []):
            df = download_ohlcv(symbol, interval="1d", period="2y")
            if df is not None:
                all_signals.extend(self.engine.evaluate_position(symbol, df))

        all_signals.sort(key=lambda s: (s.confidence, s.risk_reward), reverse=True)
        sent = 0
        for sig in all_signals:
            if self.alerts.should_send(sig):
                self.alerts.send(sig)
                sent += 1
        if sent == 0:
            print(f"{datetime.now().isoformat()} | Escaneo completado sin alertas nuevas.")
        else:
            print(f"{datetime.now().isoformat()} | Escaneo completado. Alertas enviadas: {sent}")

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
    parser = argparse.ArgumentParser(description="Scanner de alertas para mercado.")
    parser.add_argument("--once", action="store_true", help="Ejecuta un solo escaneo y termina.")
    args = parser.parse_args()

    config = load_config()
    scanner = Scanner(config)

    if args.once or is_running_in_github_actions():
        scanner.scan_once()
    else:
        scanner.run_forever()
