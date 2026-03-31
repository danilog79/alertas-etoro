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
    def __init__(self, min_confidence: int, rr_min: float):
        self.min_confidence = min_confidence
        self.rr_min = rr_min

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
        return out.dropna()

    def evaluate_intraday(self, symbol: str, df: pd.DataFrame) -> List[Signal]:
        df = self.enrich(df, intraday=True)
        if len(df) < 220:
            return []
        row = df.iloc[-1]
        prev = df.iloc[-2]
        signals = []

        # Long pullback continuation
        reasons = []
        confidence = 0
        if row["ema20"] > row["ema50"] > row["ema200"]:
            confidence += 25; reasons.append("tendencia alcista alineada (EMA20>EMA50>EMA200)")
        if row["Close"] > row["vwap"]:
            confidence += 15; reasons.append("precio sobre VWAP")
        if 48 <= row["rsi14"] <= 68 and row["rsi14"] > prev["rsi14"]:
            confidence += 15; reasons.append("RSI recuperando sin sobrecompra extrema")
        if row["macd_hist"] > prev["macd_hist"]:
            confidence += 10; reasons.append("momentum MACD mejorando")
        if row["rel_vol"] >= 1.2:
            confidence += 10; reasons.append(f"volumen relativo {row['rel_vol']:.2f}x")
        if row["Close"] >= row["ema20"]:
            confidence += 10; reasons.append("precio recuperó EMA20")
        entry = float(row["Close"])
        stop = float(entry - 1.2 * row["atr14"])
        target = float(entry + 2.2 * row["atr14"])
        rr = (target - entry) / max(entry - stop, 1e-9)
        if confidence >= self.min_confidence and rr >= self.rr_min:
            signals.append(Signal(symbol, "continuación intradía", "LONG", "15m", entry, entry, stop, target, rr, confidence, reasons, datetime.now(timezone.utc).isoformat()))

        # Short breakdown continuation
        reasons = []
        confidence = 0
        if row["ema20"] < row["ema50"] < row["ema200"]:
            confidence += 25; reasons.append("tendencia bajista alineada (EMA20<EMA50<EMA200)")
        if row["Close"] < row["vwap"]:
            confidence += 15; reasons.append("precio bajo VWAP")
        if 32 <= row["rsi14"] <= 52 and row["rsi14"] < prev["rsi14"]:
            confidence += 15; reasons.append("RSI debilitándose")
        if row["macd_hist"] < prev["macd_hist"]:
            confidence += 10; reasons.append("momentum MACD empeorando")
        if row["rel_vol"] >= 1.2:
            confidence += 10; reasons.append(f"volumen relativo {row['rel_vol']:.2f}x")
        if row["Close"] <= row["ema20"]:
            confidence += 10; reasons.append("precio perdió EMA20")
        entry = float(row["Close"])
        stop = float(entry + 1.2 * row["atr14"])
        target = float(entry - 2.2 * row["atr14"])
        rr = (entry - target) / max(stop - entry, 1e-9)
        if confidence >= self.min_confidence and rr >= self.rr_min:
            signals.append(Signal(symbol, "continuación intradía", "SHORT", "15m", entry, entry, stop, target, rr, confidence, reasons, datetime.now(timezone.utc).isoformat()))

        # Spike / dip detector for opportunistic alerts
        signals.extend(self._evaluate_spike(symbol, df, timeframe="15m"))
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

        signals.extend(self._evaluate_spike(symbol, df, timeframe="1h"))
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

        signals.extend(self._evaluate_spike(symbol, df, timeframe="1d"))
        return signals

    def _evaluate_spike(self, symbol: str, df: pd.DataFrame, timeframe: str) -> List[Signal]:
        row = df.iloc[-1]
        prev = df.iloc[-2]
        signals = []
        move_pct = ((row["Close"] / prev["Close"]) - 1) * 100
        atr_pct = (row["atr14"] / row["Close"]) * 100 if row["Close"] else 0

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
        message = self.format_signal(sig)
        print(message)
        print("-" * 100)
        self.persist(sig)
        if self.telegram_cfg.get("enabled"):
            self._send_telegram(message)

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
    def format_signal(sig: Signal) -> str:
        return (
            f"ALERTA {sig.side} | {sig.symbol} | {sig.strategy} | TF {sig.timeframe}\n"
            f"Precio: {sig.price:.4f}\n"
            f"Entrada: {sig.entry:.4f}\n"
            f"Stop: {sig.stop:.4f}\n"
            f"Objetivo: {sig.target:.4f}\n"
            f"R/R: {sig.risk_reward:.2f}\n"
            f"Confianza: {sig.confidence}/100\n"
            f"Motivos: {'; '.join(sig.reasons)}\n"
            f"UTC: {sig.timestamp_utc}\n"
            f"Aviso: esto es una alerta cuantitativa, no una orden automática. Confirma liquidez, spread, noticias y horario en eToro antes de operar."
        )


# -----------------------------
# Scanner orchestration
# -----------------------------
class Scanner:
    def __init__(self, config: Dict):
        self.config = config
        self.engine = SignalEngine(config["min_confidence"], config["risk_reward_min"])
        self.alerts = AlertManager(config["cooldown_minutes"], config["telegram"])

    def scan_once(self):
        all_signals: List[Signal] = []

        for symbol in self.config["watchlists"].get("intraday", []):
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

        # sort by confidence then RR
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
            except Exception as e:
                print("Error en el escaneo:")
                traceback.print_exc()
            time.sleep(interval)


if __name__ == "__main__":
    config = load_config()
    scanner = Scanner(config)
    scanner.run_forever()
