# ==============================
# ALERT SYSTEM V4 FINAL
# ==============================

# IMPORTS (mantén los mismos que ya tienes arriba)
# NO cambies tus imports actuales

# ==============================
# AJUSTES CLAVE (NUEVOS)
# ==============================

MIN_CONFIDENCE = 58
MIN_RR = 1.3

TOP_N_INTRADAY = 3
TOP_N_SWING = 3

# ==============================
# MODIFICAR scan_once()
# ==============================

def scan_once(self):
    actionable = []
    candidates = []
    all_signals = []
    scanned = 0
    intraday_atr_pcts = []

    universe_intraday = list(dict.fromkeys(
        self.config["watchlists"].get("intraday", []) +
        self.config["watchlists"].get("etoro_featured_winners", []) +
        self.config["watchlists"].get("etoro_featured_losers", []) +
        self.regime.get("etoro_recommendations", [])
    ))

    universe_swing = list(dict.fromkeys(
        self.config["watchlists"].get("swing", []) +
        self.config["watchlists"].get("etoro_featured_winners", []) +
        self.config["watchlists"].get("etoro_featured_losers", [])
    ))

    seen = set()

    # ==========================
    # INTRADAY
    # ==========================
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

            # ATR protegido
            df15 = download_ohlcv(symbol, "15m", "5d")
            if df15 is not None and len(df15) > 30:
                edf = self.engine.enrich(df15, intraday=True)
                if edf is not None and not edf.empty and "atr_pct" in edf.columns:
                    val = edf.iloc[-1].get("atr_pct")
                    if val is not None and pd.notna(val):
                        intraday_atr_pcts.append(float(val))

        except Exception:
            traceback.print_exc()

    # ==========================
    # SWING
    # ==========================
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

    # ==========================
    # 🔥 RANKING GLOBAL (CLAVE)
    # ==========================
    all_signals.sort(key=lambda s: (s.score, s.risk_reward), reverse=True)

    actionable.sort(key=lambda s: (s.score, s.risk_reward), reverse=True)
    candidates.sort(key=lambda s: (s.score, s.risk_reward), reverse=True)

    # ==========================
    # ENVÍO NORMAL
    # ==========================
    sent = 0

    for sig in actionable[:10]:
        if sig.confidence >= MIN_CONFIDENCE and sig.risk_reward >= MIN_RR:
            if self.alerts.should_send(sig):
                self.alerts.send(sig)
                sent += 1

    # ==========================
    # 🔥 TOP OPORTUNIDADES (NUEVO)
    # ==========================
    top_intraday = [s for s in all_signals if "intradía" in s.strategy][:TOP_N_INTRADAY]
    top_swing = [s for s in all_signals if "swing" in s.strategy][:TOP_N_SWING]

    if top_intraday or top_swing:
        msg = "🔥 TOP OPORTUNIDADES ACTUALES\n\n"

        if top_intraday:
            msg += "📊 INTRADÍA:\n"
            for i, s in enumerate(top_intraday, 1):
                msg += f"{i}. {s.symbol} {s.side} | score {int(s.score)} | RR {s.risk_reward:.2f}\n"

        if top_swing:
            msg += "\n📈 SWING:\n"
            for i, s in enumerate(top_swing, 1):
                msg += f"{i}. {s.symbol} {s.side} | score {int(s.score)} | RR {s.risk_reward:.2f}\n"

        print(msg)
        self.alerts._send_telegram(msg)

    # ==========================
    # SUMMARY
    # ==========================
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "regime_tag": self.regime.get("tag"),
        "market_bias": self.regime.get("market_bias"),
        "avg_intraday_atr_pct": round(sum(intraday_atr_pcts) / max(len(intraday_atr_pcts), 1), 3),
        "scanned_symbols": scanned,
        "signals_sent": sent,
        "candidate_count": len(candidates),
        "top_candidates": [
            {
                "symbol": s.symbol,
                "side": s.side,
                "score": s.score,
                "risk_reward": s.risk_reward,
            } for s in all_signals[:5]
        ],
    }

    self.alerts.save_summary(summary)
    self.alerts.maybe_send_heartbeat(summary)

    print(f"{datetime.now().isoformat()} | Señales: {sent} | Total oportunidades: {len(all_signals)}")
