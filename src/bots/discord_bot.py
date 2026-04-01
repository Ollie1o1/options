"""
Discord slash-command bot for the options screener API.

Commands:
  /market   — current market context
  /top      — top 10 picks from liquid_large_cap watchlist
  /scan     — scan a single ticker
  /watchlist — scan a named watchlist from config.json

Environment variables required (in .env):
  DISCORD_BOT_TOKEN   — bot token from Discord Developer Portal
  DISCORD_GUILD_ID    — server ID for instant slash-command sync (optional)
  API_BASE_URL        — defaults to http://127.0.0.1:8000
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

import discord
from discord import app_commands
import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [discord] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
GUILD_ID = os.environ.get("DISCORD_GUILD_ID", "")
API_BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

if not TOKEN:
    logger.error("DISCORD_BOT_TOKEN not set. Set it in .env and restart.")
    sys.exit(1)


# ── Helpers ────────────────────────────────────────────────────────────────

def _regime_emoji(regime: str) -> str:
    mapping = {"Low": "🟢", "Normal": "🟡", "High": "🔴", "Unknown": "⚪"}
    return mapping.get(regime, "⚪")


def _trend_emoji(trend: str) -> str:
    return "📈" if trend == "Bull" else "📉" if trend == "Bear" else "➡️"


def _format_pick(pick: dict, idx: int) -> str:
    symbol = pick.get("symbol", "?")
    opt_type = (pick.get("type") or "?").upper()
    strike = pick.get("strike")
    expiry = pick.get("expiration") or pick.get("dte")
    dte = pick.get("dte")
    premium = pick.get("premium")
    pop = pick.get("prob_profit")
    qs = pick.get("quality_score")
    ev = pick.get("ev_per_contract")

    parts = [f"**#{idx} {symbol} {opt_type}"]
    if strike is not None:
        parts[0] += f" ${strike}"
    if expiry:
        parts[0] += f" exp {expiry}"
    parts[0] += "**"

    details = []
    if dte is not None:
        details.append(f"DTE: {dte}")
    if premium is not None:
        details.append(f"Prem: ${premium:.2f}")
    if pop is not None:
        details.append(f"PoP: {pop:.0%}")
    if qs is not None:
        details.append(f"QS: {qs:.2f}")
    if ev is not None:
        details.append(f"EV: ${ev:.0f}")

    return "\n".join(parts) + ("\n" + " | ".join(details) if details else "")


def _picks_embed(title: str, picks: list, footer: str = "") -> discord.Embed:
    embed = discord.Embed(title=title, color=0x00B4D8)
    if not picks:
        embed.description = "No picks found."
        return embed

    for i, pick in enumerate(picks[:10], start=1):
        symbol = pick.get("symbol", "?")
        opt_type = (pick.get("type") or "?").upper()
        strike = pick.get("strike")
        dte = pick.get("dte")
        premium = pick.get("premium")
        pop = pick.get("prob_profit")
        qs = pick.get("quality_score")

        name_parts = [f"{symbol} {opt_type}"]
        if strike is not None:
            name_parts.append(f"${strike}")
        if dte is not None:
            name_parts.append(f"{dte}d")

        value_parts = []
        if premium is not None:
            value_parts.append(f"Prem: **${premium:.2f}**")
        if pop is not None:
            value_parts.append(f"PoP: **{pop:.0%}**")
        if qs is not None:
            value_parts.append(f"QS: **{qs:.2f}**")

        embed.add_field(
            name=f"#{i} " + " ".join(name_parts),
            value=" | ".join(value_parts) if value_parts else "—",
            inline=False,
        )

    if footer:
        embed.set_footer(text=footer)
    return embed


# ── Bot setup ──────────────────────────────────────────────────────────────

intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)
guild_obj = discord.Object(id=int(GUILD_ID)) if GUILD_ID else None


@client.event
async def on_ready():
    if guild_obj:
        tree.copy_global_to(guild=guild_obj)
        await tree.sync(guild=guild_obj)
        logger.info("Slash commands synced to guild %s", GUILD_ID)
    else:
        await tree.sync()
        logger.info("Slash commands synced globally (may take up to 1 hour)")
    logger.info("Logged in as %s (id=%s)", client.user, client.user.id)


# ── Slash commands ─────────────────────────────────────────────────────────

@tree.command(name="market", description="Show current market context (VIX, trend, macro risk)")
async def cmd_market(interaction: discord.Interaction):
    await interaction.response.defer()
    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.get(f"{API_BASE}/market")
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        await interaction.followup.send(f"API error: {exc}")
        return

    trend = data.get("market_trend", "Unknown")
    regime = data.get("volatility_regime", "Unknown")
    vix = data.get("vix_level")
    vix_regime = data.get("vix_regime", "Unknown")
    macro = data.get("macro_risk_active", False)
    tnx = data.get("tnx_change_pct", 0.0) or 0.0

    embed = discord.Embed(title="Market Context", color=0x00B4D8)
    embed.add_field(
        name=f"{_trend_emoji(trend)} Trend",
        value=f"**{trend}**",
        inline=True,
    )
    embed.add_field(
        name=f"{_regime_emoji(vix_regime)} VIX Regime",
        value=f"**{vix_regime}**" + (f" ({vix:.1f})" if vix else ""),
        inline=True,
    )
    embed.add_field(
        name="⚠️ Macro Risk" if macro else "✅ Macro Risk",
        value="**Active**" if macro else "**Clear**",
        inline=True,
    )
    embed.add_field(
        name="📊 Vol Regime",
        value=f"**{regime}**",
        inline=True,
    )
    if tnx:
        embed.add_field(
            name="📉 10Y Yield Δ",
            value=f"**{tnx:+.2f}%**",
            inline=True,
        )
    await interaction.followup.send(embed=embed)


@tree.command(name="top", description="Top picks from the liquid_large_cap watchlist")
@app_commands.describe(n="Number of picks to show (default 10, max 25)")
async def cmd_top(interaction: discord.Interaction, n: int = 10):
    await interaction.response.defer()
    n = max(1, min(n, 25))
    try:
        async with httpx.AsyncClient(timeout=300.0) as http:
            resp = await http.get(f"{API_BASE}/top", params={"n": n})
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        await interaction.followup.send(f"API error: {exc}")
        return

    picks = data.get("picks", [])
    embed = _picks_embed(
        title=f"Top {len(picks)} Picks — liquid_large_cap",
        picks=picks,
        footer=f"Scanned {data.get('count', len(picks))} results",
    )
    await interaction.followup.send(embed=embed)


@tree.command(name="scan", description="Scan a single ticker for options setups")
@app_commands.describe(symbol="Ticker symbol, e.g. AAPL")
async def cmd_scan(interaction: discord.Interaction, symbol: str):
    await interaction.response.defer()
    symbol = symbol.upper().strip()
    try:
        async with httpx.AsyncClient(timeout=120.0) as http:
            resp = await http.get(f"{API_BASE}/scan/{symbol}")
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("detail", str(exc)) if exc.response else str(exc)
        await interaction.followup.send(f"Error scanning {symbol}: {detail}")
        return
    except Exception as exc:
        await interaction.followup.send(f"API error: {exc}")
        return

    picks = data.get("picks", [])
    embed = _picks_embed(
        title=f"Scan: {symbol}",
        picks=picks,
        footer=f"{len(picks)} pick(s) found",
    )
    # Attach market context as a footer field if available
    mc = data.get("market_context", {})
    if mc:
        trend = mc.get("market_trend", "?")
        vix = mc.get("vix_level")
        footer_text = f"Trend: {trend}"
        if vix:
            footer_text += f" | VIX: {vix:.1f}"
        embed.set_footer(text=footer_text)
    await interaction.followup.send(embed=embed)


@tree.command(name="watchlist", description="Scan a named watchlist from config.json")
@app_commands.describe(
    name="Watchlist name (liquid_large_cap, sector_etfs, high_iv, income)",
    n="Number of picks to show (default 10)",
)
async def cmd_watchlist(interaction: discord.Interaction, name: str, n: int = 10):
    await interaction.response.defer()
    n = max(1, min(n, 25))
    try:
        async with httpx.AsyncClient(timeout=300.0) as http:
            resp = await http.get(f"{API_BASE}/watchlist/{name}", params={"n": n})
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("detail", str(exc)) if exc.response else str(exc)
        await interaction.followup.send(f"Error: {detail}")
        return
    except Exception as exc:
        await interaction.followup.send(f"API error: {exc}")
        return

    picks = data.get("picks", [])
    scanned = data.get("tickers_scanned", "?")
    embed = _picks_embed(
        title=f"Watchlist: {name}",
        picks=picks,
        footer=f"Scanned {scanned} tickers, {len(picks)} picks",
    )
    await interaction.followup.send(embed=embed)


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    client.run(TOKEN, log_handler=None)


if __name__ == "__main__":
    main()
