"""`opt notify` — Discord Rich Embeds + Zero-Clutter Single-Slot macOS Alerts."""
from __future__ import annotations

import datetime as dt
import json
import os
from typing import Any, Dict

import click

from src.notifications import (
    DEFAULT_STATE_PATH,
    build_events_embed,
    build_morning_embed,
    detect_action_events,
    filter_unsent_events,
    format_status_pill,
    load_notify_state,
    save_notify_state,
    send_discord_embed,
    send_macos_notification,
    summarize_for_macos,
)
from src.paths import load_project_env
from src.tui.state import DeskDataProvider


@click.command("notify")
@click.option("--morning", is_flag=True, help="Send the daily Morning Desk Dispatch to Discord (+ macOS only if exits/issues).")
@click.option("--check", is_flag=True, help="Check for new action-only events (open pick exits, clean TFSA setups, stale pipes).")
@click.option("--pill", is_flag=True, help="Print compact 1-line status pill for macOS menu bar / prompt.")
@click.option("--test-macos", is_flag=True, help="Send one clean test macOS notification.")
@click.option("--test-discord", is_flag=True, help="Send one test Bloomberg embed to DISCORD_WEBHOOK_URL.")
@click.option("--dry-run", is_flag=True, help="Preview payloads without sending or advancing dedupe state.")
@click.option("--refresh", is_flag=True, help="Refresh local radar before checking.")
@click.pass_context
def notify_cmd(
    ctx: click.Context,
    morning: bool,
    check: bool,
    pill: bool,
    test_macos: bool,
    test_discord: bool,
    dry_run: bool,
    refresh: bool,
) -> None:
    """Push Discord rich embeds, quiet single-slot macOS notifications, or a 1-line status pill."""
    load_project_env()
    use_json = bool((ctx.obj or {}).get("json"))

    if test_macos:
        ok, backend = send_macos_notification(
            "Options Desk",
            "Zero-clutter alerts active — banners only fire on open-pick exits or broken pipes.",
            subtitle="Test Notification",
            sound=False,
        )
        out = {"ok": ok, "backend": backend}
        click.echo(json.dumps(out, indent=2) if use_json else f"{'✓' if ok else '✗'} macOS notification ({backend})")
        return

    if test_discord:
        embed = {
            "title": "Options Desk · Discord Channel Connected",
            "color": 0x5EC98D,
            "description": "Morning Dispatches & action-only TFSA exit/setup alerts are armed. Live execution remains OFF.",
            "footer": {"text": "Run `opt notify --morning` anytime to send a full brief."},
        }
        if dry_run:
            click.echo(json.dumps(embed, indent=2))
            return
        if not os.environ.get("DISCORD_WEBHOOK_URL", "").strip():
            click.echo("✗ DISCORD_WEBHOOK_URL is not set in .env — add it and re-run.", err=True)
            raise SystemExit(1)
        ok = send_discord_embed(embed)
        click.echo("✓ Test embed sent to Discord." if ok else "✗ Discord webhook post failed.")
        if not ok:
            raise SystemExit(1)
        return

    provider = DeskDataProvider(
        ohlcv_db=(ctx.obj or {}).get("ohlcv_db", "data/equity_ohlcv.db"),
        cache_path=(ctx.obj or {}).get("cache_path", "data/radar_cache.json"),
        own_picks_db=(ctx.obj or {}).get("own_picks_db", "data/own_picks.db"),
    )
    snap = provider.get_radar(refresh=refresh)
    book = provider.get_book()
    health = provider.get_health()

    if pill or not (morning or check):
        line = format_status_pill(snap, book, health)
        if use_json:
            click.echo(json.dumps({"pill": line, "regime": snap.regime_posture, "exit_alerts": book.get("exit_alerts") or []}))
        else:
            click.echo(line)
        return

    today = dt.date.today().isoformat()
    state = load_notify_state(DEFAULT_STATE_PATH)
    result: Dict[str, Any] = {"mode": "morning" if morning else "check", "dry_run": dry_run}

    if morning:
        embed = build_morning_embed(snap, book, health)
        events = detect_action_events(snap, book, health)
        mac = summarize_for_macos(events)
        result["discord_embed"] = embed
        result["macos_banner"] = mac
        if not dry_run:
            result["discord_sent"] = send_discord_embed(embed)
            if mac is not None:
                sub, body, snd = mac
                ok_m, bk = send_macos_notification("Options Desk", body, subtitle=sub, sound=snd)
                result["macos_sent"] = ok_m
                result["macos_backend"] = bk
            else:
                result["macos_sent"] = False
                result["macos_note"] = "Suppressed (zero exits or pipe failures — keeping macOS quiet)"
            state["last_morning"] = today
            save_notify_state(state, DEFAULT_STATE_PATH)
    else:
        all_events = detect_action_events(snap, book, health)
        unsent = filter_unsent_events(all_events, state, today=today)
        result["events_total"] = len(all_events)
        result["events_new"] = [e.title for e in unsent]
        if unsent:
            embed = build_events_embed(unsent)
            mac = summarize_for_macos(unsent)
            result["discord_embed"] = embed
            result["macos_banner"] = mac
            if not dry_run:
                result["discord_sent"] = send_discord_embed(embed)
                if mac is not None:
                    sub, body, snd = mac
                    ok_m, bk = send_macos_notification("Options Desk", body, subtitle=sub, sound=snd)
                    result["macos_sent"] = ok_m
                    result["macos_backend"] = bk
                for e in unsent:
                    state["sent"][e.key] = today
                save_notify_state(state, DEFAULT_STATE_PATH)

    if use_json or dry_run:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        d_ok = result.get("discord_sent")
        m_ok = result.get("macos_sent")
        click.echo(f"Notify ({result['mode']}): discord={'sent' if d_ok else 'skipped'} · macos={'sent' if m_ok else 'quiet'}")
