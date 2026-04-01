"""
Telegram bot for the options screener API.

Commands:
  /market    — current market context
  /top       — top 5 picks from liquid_large_cap watchlist
  /scan      — /scan AAPL — scan a single ticker
  /watchlist — /watchlist high_iv — scan a named watchlist

Environment variables required (in .env):
  TELEGRAM_BOT_TOKEN  — token from @BotFather
  API_BASE_URL        — defaults to http://127.0.0.1:8000
"""

import os
import sys
import logging
import httpx
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [telegram] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
API_BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

if not TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN not set. Set it in .env and restart.")
    sys.exit(1)

# Telegram has a 4096-char limit per message; cap picks to avoid truncation
MAX_PICKS = 5


# ── Formatting helpers ─────────────────────────────────────────────────────

def _regime_emoji(regime: str) -> str:
    mapping = {"Low": "🟢", "Normal": "🟡", "High": "🔴", "Unknown": "⚪"}
    return mapping.get(regime, "⚪")


def _trend_emoji(trend: str) -> str:
    return "📈" if trend == "Bull" else "📉" if trend == "Bear" else "➡️"


def _format_market(data: dict) -> str:
    trend = data.get("market_trend", "Unknown")
    regime = data.get("volatility_regime", "Unknown")
    vix = data.get("vix_level")
    vix_regime = data.get("vix_regime", "Unknown")
    macro = data.get("macro_risk_active", False)
    tnx = data.get("tnx_change_pct") or 0.0

    lines = [
        "*Market Context*",
        f"{_trend_emoji(trend)} Trend: *{trend}*",
        f"{_regime_emoji(vix_regime)} VIX Regime: *{vix_regime}*" + (f" ({vix:.1f})" if vix else ""),
        f"Vol Regime: *{regime}*",
        f"Macro Risk: *{'⚠️ Active' if macro else '✅ Clear'}*",
    ]
    if tnx:
        lines.append(f"10Y Yield Δ: *{tnx:+.2f}%*")
    return "\n".join(lines)


def _format_picks(picks: list, title: str) -> str:
    if not picks:
        return f"*{title}*\n\nNo picks found."

    lines = [f"*{title}*\n"]
    for i, p in enumerate(picks[:MAX_PICKS], start=1):
        symbol = p.get("symbol", "?")
        opt_type = (p.get("type") or "?").upper()
        strike = p.get("strike")
        dte = p.get("dte")
        premium = p.get("premium")
        pop = p.get("prob_profit")
        qs = p.get("quality_score")
        ev = p.get("ev_per_contract")

        header = f"#{i} `{symbol} {opt_type}"
        if strike is not None:
            header += f" ${strike}"
        if dte is not None:
            header += f" {dte}d`"
        else:
            header += "`"

        details = []
        if premium is not None:
            details.append(f"Prem ${premium:.2f}")
        if pop is not None:
            details.append(f"PoP {pop:.0%}")
        if qs is not None:
            details.append(f"QS {qs:.2f}")
        if ev is not None:
            details.append(f"EV ${ev:.0f}")

        lines.append(header)
        if details:
            lines.append("  " + " | ".join(details))
        lines.append("")

    return "\n".join(lines).strip()


# ── Command handlers ───────────────────────────────────────────────────────

async def cmd_market(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("Fetching market context...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.get(f"{API_BASE}/market")
            resp.raise_for_status()
            data = resp.json()
        text = _format_market(data)
    except Exception as exc:
        text = f"Error: {exc}"
    await msg.edit_text(text, parse_mode="Markdown")


async def cmd_top(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("Scanning liquid_large_cap watchlist... (this may take ~30s)")
    try:
        async with httpx.AsyncClient(timeout=300.0) as http:
            resp = await http.get(f"{API_BASE}/top", params={"n": MAX_PICKS})
            resp.raise_for_status()
            data = resp.json()
        picks = data.get("picks", [])
        text = _format_picks(picks, title=f"Top Picks — liquid_large_cap")
    except Exception as exc:
        text = f"Error: {exc}"
    await msg.edit_text(text, parse_mode="Markdown")


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /scan SYMBOL\nExample: /scan AAPL")
        return

    symbol = args[0].upper().strip()
    msg = await update.message.reply_text(f"Scanning {symbol}...")
    try:
        async with httpx.AsyncClient(timeout=120.0) as http:
            resp = await http.get(f"{API_BASE}/scan/{symbol}")
            if resp.status_code == 404:
                text = f"Symbol not found: {symbol}"
            elif not resp.is_success:
                detail = resp.json().get("detail", resp.text)
                text = f"Error scanning {symbol}: {detail}"
            else:
                data = resp.json()
                picks = data.get("picks", [])
                text = _format_picks(picks, title=f"Scan: {symbol}")
    except Exception as exc:
        text = f"Error: {exc}"
    await msg.edit_text(text, parse_mode="Markdown")


async def cmd_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text(
            "Usage: /watchlist NAME\n"
            "Available: liquid_large_cap, sector_etfs, high_iv, income"
        )
        return

    name = args[0].lower().strip()
    msg = await update.message.reply_text(f"Scanning watchlist '{name}'... (this may take a while)")
    try:
        async with httpx.AsyncClient(timeout=300.0) as http:
            resp = await http.get(f"{API_BASE}/watchlist/{name}", params={"n": MAX_PICKS})
            if resp.status_code == 404:
                detail = resp.json().get("detail", "Watchlist not found")
                text = f"Error: {detail}"
            elif not resp.is_success:
                detail = resp.json().get("detail", resp.text)
                text = f"Error: {detail}"
            else:
                data = resp.json()
                picks = data.get("picks", [])
                scanned = data.get("tickers_scanned", "?")
                text = _format_picks(picks, title=f"Watchlist: {name} ({scanned} tickers)")
    except Exception as exc:
        text = f"Error: {exc}"
    await msg.edit_text(text, parse_mode="Markdown")


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("market", cmd_market))
    app.add_handler(CommandHandler("top", cmd_top))
    app.add_handler(CommandHandler("scan", cmd_scan))
    app.add_handler(CommandHandler("watchlist", cmd_watchlist))
    logger.info("Telegram bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
