
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


DEFAULT_CONFIG = {
    "scan_interval_seconds": 300,
    "cooldown_minutes": 60,
    "min_confidence": 65,
    "risk_reward_min": 2.0,
    "telegram": {
        "enabled": False,
        "bot_token": "",
        "chat_id": ""
    },
    "watchlists": {
        "intraday": ["AAPL", "NVDA", "TSLA", "SPY", "QQQ", "AMZN", "EURUSD=X", "BTC-USD"],
        "swing": ["AAPL", "NVDA", "SPY", "QQQ", "AMZN", "GLD", "BTC-USD"],
        "position": ["AAPL", "MSFT", "SPY", "QQQ", "AMZN", "GLD", "BTC-USD"]
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
    "heartbeat": {
        "enabled": True,
        "interval_minutes": 60,
        "include_market_summary": True,
        "include_top_candidates": True,
        "top_n": 5
    }
}


def deep_merge(base: Dict, override: Dict) -> Dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str = "config.json") -> Dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        return deep_merge(DEFAULT_CONFIG, user_cfg)
    return DEFAULT_CONFIG.copy()


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


def signal_key(sig: Signal) -> str:
    return f"{sig.symbol}|{sig.strategy}|{sig.side}|{sig.timeframe}"


def utc_hhmm_now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M")


def hhmm_to_minutes(value: str) -> int:
    h, m = value.split(":")
    return int(h) * 60 + int(m)


def is_between_hhmm(now_hhmm: str, start_hhmm: str, end_hhmm: str) -> bool:
    now = hhmm_to_minutes(now_hhmm)
    start = hhmm_to_minutes(start_hhmm)
    end = hhmm_to_minutes(end_hhmm)
    if start <= end:
        return start <= now <= end
    return now >= start or now <= end


def classify_symbol(symbol: str) -> str:
    if symbol.endswith("-USD"):
        return "crypto"
    if "=X" in symbol:
        return "forex"
    return "us"


class SignalEngine:
    def __init__(self, min_confidence: int, rr_min: float, intraday_filter: Optional[Dict] = None):
        self.min_confidence = min_confidence
        self.rr_min = rr_min
        self.intraday_filter = intraday_filter or {}

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
        out["atr_pct"] = (out["atr14"] / out["Close"]) * 100
        out["dist_ema20_pct"] = ((out["Close"] - out["ema20"]) / out["ema20"]) * 100
        out["dist_vwap_pct"] = ((out["Close"] - out["vwap"]) / out["vwap"]) * 100
        out["candle_body_pct"] = ((out["Close"] - out["Open"]).abs() / out["Close"]) * 100
        return out.dropna()

    def _intraday_session_ok(self, symbol: str) -> Tuple[bool, str]:
        cfg = self.intraday_filter
        if not cfg.get("enabled", True):
            return True, "filtro horario desactivado"
        now = utc_hhmm_now()
        cls = classify_symbol(symbol)
        if cls == "crypto":
            sess = cfg.get("crypto_session", {"start_utc": "00:00", "end_utc": "23:59"})
        elif cls == "forex":
            sess = cfg.get("forex_session", {"start_utc": "06:00", "end_utc": "20:00"})
        else:
            sess = cfg.get("us_session", {"start_utc": "13:35", "end_utc": "20:00"})
        ok = is_between_hhmm(now, sess["start_utc"], sess["end_utc"])
        return ok, f"horario UTC {now} fuera de ventana {sess['start_utc']}-{sess['end_utc']}"

    def evaluate_intraday(self, symbol: str, df: pd.DataFrame) -> Tuple[List[Signal], List[Signal], Optional[str]]:
        session_ok, session_reason = self._intraday_session_ok(symbol)
        if not session_ok:
            return [], [], session_reason

        df = self.enrich(df, intraday=True)
        if len(df) < 220:
            return [], [], "datos insuficientes"
        row = df.iloc[-1]
        prev = df.iloc[-2]
        candidates: List[Signal] = []
        confirmed: List[Signal] = []

        min_atr_pct = float(self.intraday_filter.get("min_atr_pct", 0.35))
        max_dist_ema = float(self.intraday_filter.get("max_distance_from_ema20_pct", 1.2))
        max_dist_vwap = float(self.intraday_filter.get("max_distance_from_vwap_pct", 1.0))

        # LONG continuation filtered
        reasons = []
        confidence = 0
        if row["atr_pct"] >= min_atr_pct:
            confidence += 10; reasons.append(f"volatilidad válida ATR {row['atr_pct']:.2f}%")
        if row["ema20"] > row["ema50"] > row["ema200"]:
            confidence += 25; reasons.append("tendencia alcista alineada (EMA20>EMA50>EMA200)")
        if row["Close"] > row["vwap"] and abs(row["dist_vwap_pct"]) <= max_dist_vwap:
            confidence += 15; reasons.append("precio sobre VWAP sin extensión excesiva")
        if 50 <= row["rsi14"] <= 66 and row["rsi14"] > prev["rsi14"]:
            confidence += 15; reasons.append("RSI acompaña sin sobrecompra extrema")
        if row["macd_hist"] > prev["macd_hist"] and row["macd"] > row["macd_signal"]:
            confidence += 10; reasons.append("MACD confirma aceleración alcista")
        if row["rel_vol"] >= 1.2:
            confidence += 10; reasons.append(f"volumen relativo {row['rel_vol']:.2f}x")
        if row["Close"] >= row["ema20"] and abs(row["dist_ema20_pct"]) <= max_dist_ema:
            confidence += 10; reasons.append("precio recuperó EMA20 sin quedar extendido")
        if row["candle_body_pct"] <= 0.9:
            confidence += 5; reasons.append("vela no está demasiado extendida")
        entry = float(row["Close"])
        stop = float(entry - 1.15 * row["atr14"])
        target = float(entry + 2.30 * row["atr14"])
        rr = (target - entry) / max(entry - stop, 1e-9)
        sig_long = Signal(symbol, "continuación intradía filtrada", "LONG", "15m", entry, entry, stop, target, rr, int(confidence), reasons, datetime.now(timezone.utc).isoformat())
        candidates.append(sig_long)
        if confidence >= self.min_confidence and rr >= self.rr_min:
            confirmed.append(sig_long)

        # SHORT continuation filtered
        reasons = []
        confidence = 0
        if row["atr_pct"] >= min_atr_pct:
            confidence += 10; reasons.append(f"volatilidad válida ATR {row['atr_pct']:.2f}%")
        if row["ema20"] < row["ema50"] < row["ema200"]:
            confidence += 25; reasons.append("tendencia bajista alineada (EMA20<EMA50<EMA200)")
        if row["Close"] < row["vwap"] and abs(row["dist_vwap_pct"]) <= max_dist_vwap:
            confidence += 15; reasons.append("precio bajo VWAP sin extensión excesiva")
        if 34 <= row["rsi14"] <= 50 and row["rsi14"] < prev["rsi14"]:
            confidence += 15; reasons.append("RSI confirma debilitamiento")
        if row["macd_hist"] < prev["macd_hist"] and row["macd"] < row["macd_signal"]:
            confidence += 10; reasons.append("MACD confirma aceleración bajista")
        if row["rel_vol"] >= 1.2:
            confidence += 10; reasons.append(f"volumen relativo {row['rel_vol']:.2f}x")
        if row["Close"] <= row["ema20"] and abs(row["dist_ema20_pct"]) <= max_dist_ema:
            confidence += 10; reasons.append("precio perdió EMA20 sin quedar extendido")
        if row["candle_body_pct"] <= 0.9:
            confidence += 5; reasons.append("vela no está demasiado extendida")
        entry = float(row["Close"])
        stop = float(entry + 1.15 * row["atr14"])
        target = float(entry - 2.30 * row["atr14"])
        rr = (entry - target) / max(stop - entry, 1e-9)
        sig_short = Signal(symbol, "continuación intradía filtrada", "SHORT", "15m", entry, entry, stop, target, rr, int(confidence), reasons, datetime.now(timezone.utc).isoformat())
        candidates.append(sig_short)
        if confidence >= self.min_confidence and rr >= self.rr_min:
            confirmed.append(sig_short)

        spike_candidates, spike_confirmed = self._evaluate_spike(symbol, df, timeframe="15m")
        candidates.extend(spike_candidates)
        confirmed.extend(spike_confirmed)
        return confirmed, candidates, None

    def evaluate_swing(self, symbol: str, df: pd.DataFrame) -> Tuple[List[Signal], List[Signal]]:
        df = self.enrich(df, intraday=False)
        if len(df) < 220:
            return [], []
        row = df.iloc[-1]
        prev = df.iloc[-2]
        highs20 = df["High"].rolling(20).max()
        lows20 = df["Low"].rolling(20).min()
        candidates = []
        confirmed = []

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
        sig = Signal(symbol, "breakout intrasemana", "LONG", "1h", entry, entry, stop, target, rr, int(confidence), reasons, datetime.now(timezone.utc).isoformat())
        candidates.append(sig)
        if confidence >= self.min_confidence and rr >= self.rr_min:
            confirmed.append(sig)

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
        sig = Signal(symbol, "breakdown intrasemana", "SHORT", "1h", entry, entry, stop, target, rr, int(confidence), reasons, datetime.now(timezone.utc).isoformat())
        candidates.append(sig)
        if confidence >= self.min_confidence and rr >= self.rr_min:
            confirmed.append(sig)

        spike_candidates, spike_confirmed = self._evaluate_spike(symbol, df, timeframe="1h")
        candidates.extend(spike_candidates)
        confirmed.extend(spike_confirmed)
        return confirmed, candidates

    def evaluate_position(self, symbol: str, df: pd.DataFrame) -> Tuple[List[Signal], List[Signal]]:
        df = self.enrich(df, intraday=False)
        if len(df) < 220:
            return [], []
        row = df.iloc[-1]
        prev = df.iloc[-2]
        highs120 = df["High"].rolling(120).max()
        lows120 = df["Low"].rolling(120).min()
        candidates = []
        confirmed = []

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
        sig = Signal(symbol, "tendencia posicional", "LONG", "1d", entry, entry, stop, target, rr, int(confidence), reasons, datetime.now(timezone.utc).isoformat())
        candidates.append(sig)
        if confidence >= self.min_confidence and rr >= self.rr_min:
            confirmed.append(sig)

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
        sig = Signal(symbol, "tendencia posicional", "SHORT", "1d", entry, entry, stop, target, rr, int(confidence), reasons, datetime.now(timezone.utc).isoformat())
        candidates.append(sig)
        if confidence >= self.min_confidence and rr >= self.rr_min:
            confirmed.append(sig)

        spike_candidates, spike_confirmed = self._evaluate_spike(symbol, df, timeframe="1d")
        candidates.extend(spike_candidates)
        confirmed.extend(spike_confirmed)
        return confirmed, candidates

    def _evaluate_spike(self, symbol: str, df: pd.DataFrame, timeframe: str) -> Tuple[List[Signal], List[Signal]]:
        row = df.iloc[-1]
        prev = df.iloc[-2]
        candidates = []
        confirmed = []
        move_pct = ((row["Close"] / prev["Close"]) - 1) * 100
        atr_pct = (row["atr14"] / row["Close"]) * 100 if row["Close"] else 0

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
        entry = float(row["Close"])
        stop = float(entry - 1.0 * row["atr14"])
        target = float(entry + 2.0 * row["atr14"])
        rr = (target - entry) / max(entry - stop, 1e-9)
        sig = Signal(symbol, "caída brusca: posible rebote", "LONG", timeframe, entry, entry, stop, target, rr, int(confidence), reasons, datetime.now(timezone.utc).isoformat())
        if confidence > 0:
            candidates.append(sig)
        if confidence >= self.min_confidence and rr >= self.rr_min:
            confirmed.append(sig)

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
        entry = float(row["Close"])
        stop = float(entry - 1.1 * row["atr14"])
        target = float(entry + 2.2 * row["atr14"])
        rr = (target - entry) / max(entry - stop, 1e-9)
        sig = Signal(symbol, "subida brusca: continuación", "LONG", timeframe, entry, entry, stop, target, rr, int(confidence), reasons, datetime.now(timezone.utc).isoformat())
        if confidence > 0:
            candidates.append(sig)
        if confidence >= self.min_confidence and rr >= self.rr_min:
            confirmed.append(sig)

        return candidates, confirmed


class AlertManager:
    def __init__(self, cooldown_minutes: int, telegram_cfg: Dict, heartbeat_cfg: Dict):
        self.cooldown_seconds = cooldown_minutes * 60
        self.telegram_cfg = telegram_cfg
        self.heartbeat_cfg = heartbeat_cfg
        self.sent_cache: Dict[str, float] = {}
        self.last_heartbeat = 0.0
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

    def maybe_send_heartbeat(self, summary: Dict):
        if not self.heartbeat_cfg.get("enabled", False):
            return
        now = time.time()
        interval = int(self.heartbeat_cfg.get("interval_minutes", 60)) * 60
        if now - self.last_heartbeat < interval:
            return

        message_lines = [
            "🟢 Heartbeat inteligente",
            f"UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
            f"Escaneo OK | Señales enviadas: {summary.get('sent_count', 0)}",
            f"Activos analizados: {summary.get('symbols_scanned', 0)} | omitidos: {summary.get('symbols_skipped', 0)}",
        ]

        if self.heartbeat_cfg.get("include_market_summary", True):
            market = summary.get("market_summary", {})
            if market:
                message_lines.append("")
                message_lines.append("Resumen del mercado:")
                if market.get("risk") is not None:
                    message_lines.append(f"- Sesgo riesgo: {market['risk']}")
                if market.get("us_trend") is not None:
                    message_lines.append(f"- Tendencia USA: {market['us_trend']}")
                if market.get("btc_trend") is not None:
                    message_lines.append(f"- Tendencia BTC: {market['btc_trend']}")
                if market.get("avg_intraday_atr_pct") is not None:
                    message_lines.append(f"- Volatilidad intradía media ATR: {market['avg_intraday_atr_pct']:.2f}%")
                if market.get("hot_symbols"):
                    message_lines.append(f"- Activos más activos: {', '.join(market['hot_symbols'])}")

        if self.heartbeat_cfg.get("include_top_candidates", True):
            top_n = int(self.heartbeat_cfg.get("top_n", 5))
            top_candidates = summary.get("top_candidates", [])[:top_n]
            if top_candidates:
                message_lines.append("")
                message_lines.append("Top oportunidades (sin pasar filtro final):")
                for i, sig in enumerate(top_candidates, 1):
                    message_lines.append(
                        f"{i}. {sig.symbol} | {sig.side} | {sig.strategy} | TF {sig.timeframe} | Conf {sig.confidence} | R/R {sig.risk_reward:.2f}"
                    )

        msg = "\n".join(message_lines)
        print(msg)
        print("-" * 100)
        if self.telegram_cfg.get("enabled"):
            self._send_telegram(msg)
        self.last_heartbeat = now

    def _send_telegram(self, text: str):
        token = self.telegram_cfg.get("bot_token", "")
        chat_id = self.telegram_cfg.get("chat_id", "")
        if not token or not chat_id:
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=15)
        except Exception:
            pass

    @staticmethod
    def format_signal(sig: Signal, include_warning: bool = True) -> str:
        msg = (
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
            msg += "\nAviso: esto es una alerta cuantitativa, no una orden automática. Confirma liquidez, spread, noticias y horario en eToro antes de operar."
        return msg


class Scanner:
    def __init__(self, config: Dict):
        self.config = config
        self.engine = SignalEngine(
            config["min_confidence"],
            config["risk_reward_min"],
            config.get("intraday_filter", {})
        )
        self.alerts = AlertManager(
            config["cooldown_minutes"],
            config["telegram"],
            config.get("heartbeat", {})
        )

    def _market_summary(self, intraday_stats: List[Dict]) -> Dict:
        summary = {
            "risk": None,
            "us_trend": None,
            "btc_trend": None,
            "avg_intraday_atr_pct": None,
            "hot_symbols": []
        }
        if intraday_stats:
            avg_atr_pct = sum(x["atr_pct"] for x in intraday_stats) / len(intraday_stats)
            summary["avg_intraday_atr_pct"] = avg_atr_pct
            hot = sorted(intraday_stats, key=lambda x: x["rel_vol"], reverse=True)[:3]
            summary["hot_symbols"] = [x["symbol"] for x in hot]

            spy = next((x for x in intraday_stats if x["symbol"] == "SPY"), None)
            qqq = next((x for x in intraday_stats if x["symbol"] == "QQQ"), None)
            btc = next((x for x in intraday_stats if x["symbol"] == "BTC-USD"), None)

            if spy and qqq:
                score = int(spy["trend_score"] + qqq["trend_score"])
                if score >= 2:
                    summary["us_trend"] = "alcista"
                elif score <= -2:
                    summary["us_trend"] = "bajista"
                else:
                    summary["us_trend"] = "mixta"
                summary["risk"] = "risk-on" if score >= 1 else ("risk-off" if score <= -1 else "neutral")
            if btc:
                if btc["trend_score"] >= 1:
                    summary["btc_trend"] = "alcista"
                elif btc["trend_score"] <= -1:
                    summary["btc_trend"] = "bajista"
                else:
                    summary["btc_trend"] = "lateral"
        return summary

    def scan_once(self):
        confirmed_signals: List[Signal] = []
        candidate_signals: List[Signal] = []
        symbols_scanned = 0
        symbols_skipped = 0
        intraday_stats: List[Dict] = []

        for symbol in self.config["watchlists"].get("intraday", []):
            df = download_ohlcv(symbol, interval="15m", period="30d")
            if df is None:
                symbols_skipped += 1
                continue
            conf, cand, skip_reason = self.engine.evaluate_intraday(symbol, df)
            if skip_reason:
                print(f"{datetime.now().isoformat()} | {symbol} omitido intradía: {skip_reason}")
                symbols_skipped += 1
                continue
            symbols_scanned += 1
            confirmed_signals.extend(conf)
            candidate_signals.extend(cand)

            enriched = self.engine.enrich(df, intraday=True)
            if not enriched.empty:
                row = enriched.iloc[-1]
                trend_score = 0
                if row["ema20"] > row["ema50"] > row["ema200"]:
                    trend_score = 1
                elif row["ema20"] < row["ema50"] < row["ema200"]:
                    trend_score = -1
                intraday_stats.append({
                    "symbol": symbol,
                    "atr_pct": float(row["atr_pct"]),
                    "rel_vol": float(row["rel_vol"]),
                    "trend_score": trend_score,
                })

        for symbol in self.config["watchlists"].get("swing", []):
            df = download_ohlcv(symbol, interval="60m", period="180d")
            if df is not None:
                symbols_scanned += 1
                conf, cand = self.engine.evaluate_swing(symbol, df)
                confirmed_signals.extend(conf)
                candidate_signals.extend(cand)
            else:
                symbols_skipped += 1

        for symbol in self.config["watchlists"].get("position", []):
            df = download_ohlcv(symbol, interval="1d", period="2y")
            if df is not None:
                symbols_scanned += 1
                conf, cand = self.engine.evaluate_position(symbol, df)
                confirmed_signals.extend(conf)
                candidate_signals.extend(cand)
            else:
                symbols_skipped += 1

        confirmed_signals.sort(key=lambda s: (s.confidence, s.risk_reward), reverse=True)
        candidate_signals.sort(key=lambda s: (s.confidence, s.risk_reward), reverse=True)

        sent = 0
        sent_keys = set()
        for sig in confirmed_signals:
            if self.alerts.should_send(sig):
                self.alerts.send(sig)
                sent += 1
                sent_keys.add(signal_key(sig))

        filtered_candidates = [s for s in candidate_signals if signal_key(s) not in sent_keys and s.confidence >= max(35, self.config["min_confidence"] - 15)]

        if sent == 0:
            print(f"{datetime.now().isoformat()} | Escaneo completado sin alertas nuevas.")
        else:
            print(f"{datetime.now().isoformat()} | Escaneo completado. Alertas enviadas: {sent}")

        self.alerts.maybe_send_heartbeat({
            "sent_count": sent,
            "symbols_scanned": symbols_scanned,
            "symbols_skipped": symbols_skipped,
            "market_summary": self._market_summary(intraday_stats),
            "top_candidates": filtered_candidates
        })

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


def running_in_github_actions() -> bool:
    return os.getenv("GITHUB_ACTIONS", "").lower() == "true"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Ejecuta un solo escaneo y termina.")
    args = parser.parse_args()

    config = load_config()
    scanner = Scanner(config)

    if args.once or running_in_github_actions():
        print("Iniciando scanner en modo una sola ejecución.")
        scanner.scan_once()
    else:
        scanner.run_forever()


if __name__ == "__main__":
    main()
