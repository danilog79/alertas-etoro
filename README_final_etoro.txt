Archivos a subir al repositorio:

1) Subir alert_system_final_etoro.py a la raiz del repo.
2) Reemplazar .github/workflows/alertas.yml por alertas_final_etoro.yml.
3) Mantener requirements_alert_system.txt sin cambios.
4) Confirmar que siguen creados los secretos de GitHub:
   - TELEGRAM_BOT_TOKEN
   - TELEGRAM_CHAT_ID
5) Ejecutar Run workflow una vez.

Qué agrega esta version:
- Integra los activos que mostraste en destacados de eToro.
- Agrega mas de 20 instrumentos extra al analisis.
- Da peso adicional a activos destacados de eToro.
- Usa confirmacion 15m + 1h para intradia.
- Aplica filtro de regimen de mercado por benchmarks.
- Separa señales operables de candidatas watchlist.
- Heartbeat inteligente con resumen de mercado, volatilidad media y top candidatas.
- Quita el texto de Aviso del mensaje de Telegram.

Notas:
- GitHub Actions no garantiza exactitud total de horario.
- Esta version busca mejorar la calidad de señal; no garantiza aumento de P/L.
- Si recibes muy pocas señales, prueba bajar min_confidence de 70 a 67.
