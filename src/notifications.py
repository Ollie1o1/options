"""Discord Rich Embeds + Zero-Clutter Single-Slot macOS Notifications (`src/notifications.py`).

Design invariants:
  1. Discord gets rich, structured Bloomberg-style embeds (1 compact Morning Dispatch +
     action-only trigger embeds).
  2. macOS native notifications NEVER spam:
     - Only fire on genuine action-required events (open pick exit trigger, monthly TFSA
       rebalance due, broken data pipe) — never on "do nothing today" days.
     - Coalesced into ONE single banner via `terminal-notifier -group options-desk` (replaces
       prior banner instead of stacking; click opens Terminal into `opt`) with automatic
       zero-dependency `osascript` fallback.
     - Deduplicated per `(event_key, date)` in `data/notify_state.json`.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.paths import load_project_env, repo_path
from src.radar.schema import RadarSnapshot

DEFAULT_STATE_PATH = "data/notify_state.json"

# Semantic Discord embed colors (matching desk_kit / Quant Desk tokens)
COLOR_GOOD = 0x5EC98D
COLOR_WARN = 0xD6A452
COLOR_BAD = 0xE06C75
COLOR_INFO = 0x82AAD2


@dataclass(frozen=True)
class AlertEvent:
    key: str
    severity: str       # "CRITICAL" | "WARN" | "INFO"
    category: str       # "EXIT_ALERT" | "ALPHA_SETUP" | "PIPE_RED" | "REBALANCE"
    title: str
    body: str
    cli_cmd: str = "opt today"


def load_notify_state(path: str = DEFAULT_STATE_PATH) -> Dict[str, Any]:
    full = Path(repo_path(path))
    try:
        raw = json.loads(full.read_text(encoding="utf-8"))
        return {"sent": dict(raw.get("sent") or {}), "last_morning": raw.get("last_morning")}
    except (OSError, ValueError):
        return {"sent": {}, "last_morning": None}


def save_notify_state(state: Dict[str, Any], path: str = DEFAULT_STATE_PATH) -> None:
    full = Path(repo_path(path))
    full.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(full) + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, full)


# ── Status Pill (Zero-Popup macOS Menu Bar / Prompt Summary) ────────────────


def format_status_pill(
    snap: RadarSnapshot,
    book: Dict[str, Any],
    health: Dict[str, Any],
) -> str:
    """Compact 1-line status pill suitable for SwiftBar/xbar or tmux."""
    exits = book.get("exit_alerts") or []
    pipes = health.get("pipes") or {}
    stale_pipes = [k for k, v in pipes.items() if v.get("status") == "STALE"]
    core_syms = "/".join(
        str(c["ticker"]) for c in (snap.core_allocation or []) if str(c.get("ticker", "")) != "CASH"
    )
    if exits:
        return f"▲ {len(exits)} EXIT · {snap.regime_posture}"
    if stale_pipes:
        return f"⚠ {len(stale_pipes)} PIPE · {core_syms or snap.regime_posture}"
    icon = "●" if snap.regime_posture == "RISK_ON" else ("◆" if snap.regime_posture == "RISK_OFF" else "◐")
    vix_s = f" VIX {snap.vix:.0f}" if snap.vix is not None else ""
    return f"{icon} {core_syms or snap.regime_posture}{vix_s}"


# ── Event Detection ─────────────────────────────────────────────────────────


def detect_action_events(
    snap: RadarSnapshot,
    book: Dict[str, Any],
    health: Dict[str, Any],
) -> List[AlertEvent]:
    """Identify genuinely actionable events worthy of pushing to Discord / macOS."""
    events: List[AlertEvent] = []

    # 1. Open Pick Exit Signals (HIGHEST PRIORITY — real capital on the line)
    for al in book.get("exit_alerts") or []:
        sig_key = f"exit:{al}"
        events.append(
            AlertEvent(
                key=sig_key,
                severity="CRITICAL",
                category="EXIT_ALERT",
                title="Open Pick Exit Triggered",
                body=str(al),
                cli_cmd="opt desk --tab book",
            )
        )

    # 2. Clean TFSA-Eligible Swing / Buy-Zone Setups (no risk flags)
    for o in snap.opportunities:
        if o.playbook_id in ("equity_pullback", "holdings_buy_zone") and not o.risk_flags:
            events.append(
                AlertEvent(
                    key=f"alpha:{o.playbook_id}:{o.ticker}",
                    severity="INFO",
                    category="ALPHA_SETUP",
                    title=f"Clean TFSA Setup: {o.ticker} ({o.playbook_label})",
                    body=f"{o.setup_summary} · cost {o.friction_label}",
                    cli_cmd=f"opt card {o.ticker}",
                )
            )

    # 3. Monthly TFSA Core Rebalance Due (when user has saved core holdings)
    try:
        from src.equity_playbooks.rebalance import compute_tfsa_rebalance
        from src.profile import load_profile, rebalance_due_status

        prof = load_profile()
        if prof.core_holdings and snap.core_allocation:
            plan = compute_tfsa_rebalance(
                snap.core_allocation,
                account_usd=prof.tfsa_usd,
                current_shares=prof.core_holdings,
                fractional=prof.fractional_shares,
                drift_threshold_pct=prof.drift_threshold_pct,
            )
            due, msg = rebalance_due_status(
                prof,
                max_drift_pct=plan.max_drift_pct,
                rebalance_recommended=plan.rebalance_recommended,
            )
            if due:
                events.append(
                    AlertEvent(
                        key=f"rebalance:{dt.date.today().strftime('%Y-%m')}",
                        severity="WARN",
                        category="REBALANCE",
                        title="TFSA Core Rebalance Due (VM_OUTLOOK_TOP3)",
                        body=msg,
                        cli_cmd="opt rebalance",
                    )
                )
    except Exception:
        pass

    # 4. Critical Data Pipe Failures (STALE only — avoids noisy WARN flapping)
    pipes = health.get("pipes") or {}
    stale_names = [k for k, v in pipes.items() if k in ("ohlcv", "chains", "earnings", "disk") and v.get("status") == "STALE"]
    if stale_names:
        details = " · ".join(f"{k.upper()}: {pipes[k].get('detail', 'stale')}" for k in stale_names)
        events.append(
            AlertEvent(
                key="pipe:" + ",".join(sorted(stale_names)),
                severity="WARN",
                category="PIPE_RED",
                title="Data Pipeline Needs Attention",
                body=details,
                cli_cmd="opt desk --tab lab",
            )
        )

    return events


def filter_unsent_events(
    events: Sequence[AlertEvent],
    state: Dict[str, Any],
    *,
    today: Optional[str] = None,
) -> List[AlertEvent]:
    day = today or dt.date.today().isoformat()
    sent = state.get("sent") or {}
    return [e for e in events if sent.get(e.key) != day]


# ── Discord Rich Embed Builders & Sender ────────────────────────────────────


def _pipe_led_str(health: Dict[str, Any]) -> str:
    pipes = health.get("pipes") or {}
    if not pipes:
        return f"Disk {health.get('disk_free_gb', '?')} GB [{health.get('worst', 'OK')}]"
    parts = []
    for name, info in pipes.items():
        st = str(info.get("status") or "OK")
        dot = "🟢" if st == "OK" else ("🟡" if st == "WARN" else "🔴")
        parts.append(f"{dot} `{name.upper()}`")
    return " ".join(parts)


def build_morning_embed(
    snap: RadarSnapshot,
    book: Dict[str, Any],
    health: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a compact, Bloomberg-style Discord Rich Embed for the daily Morning Dispatch."""
    posture = snap.regime_posture
    color = COLOR_GOOD if posture == "RISK_ON" else (COLOR_BAD if posture == "RISK_OFF" else COLOR_WARN)
    exits = book.get("exit_alerts") or []
    if exits:
        color = COLOR_BAD

    spy_s = f"${snap.spy_last:.2f} ({snap.spy_chg_1d_pct:+.1f}%)" if snap.spy_last is not None else "—"
    qqq_s = f"{snap.qqq_chg_1d_pct:+.1f}%" if snap.qqq_chg_1d_pct is not None else "—"
    vix_s = f"{snap.vix:.1f} ({snap.vix_term or '—'})" if snap.vix is not None else "—"
    rv_s = f"{snap.spy_rv20:.1f}%" if getattr(snap, "spy_rv20", None) is not None else "—"
    kalshi_s = (
        f" · Kalshi: {snap.kalshi_macro[0]['label']} ({snap.kalshi_macro[0]['prob_pct']:.0f}%)"
        if snap.kalshi_macro
        else ""
    )

    core_lines = []
    for c in snap.core_allocation or []:
        tk = "SGOV/CASH" if str(c["ticker"]) == "CASH" else str(c["ticker"])
        core_lines.append(f"**{tk}** `{float(c['target_weight']) * 100:.0f}%`")
    core_txt = " · ".join(core_lines) if core_lines else "Run `opt refresh`"

    tfsa_opps = [
        o for o in snap.opportunities
        if "TAXABLE_ONLY" not in o.risk_flags
        and o.playbook_id not in ("vol_managed_outlook", "etf_wide_spread", "etf_bear_call")
    ]
    if tfsa_opps:
        setup_txt = "\n".join(
            f"• **{o.ticker}** ({o.playbook_label}) — {o.setup_summary[:70]}"
            + (f" ⚠ `{','.join(o.risk_flags[:2])}`" if o.risk_flags else "")
            for o in tfsa_opps[:4]
        )
    else:
        setup_txt = "No satellite TFSA setups firing today — stay in Core ETF sleeve."

    open_eq = [p for p in (book.get("equity_picks") or []) if str(p.get("status") or "").upper() == "OPEN"]
    if exits:
        book_txt = "**EXIT ALERTS:**\n" + "\n".join(f"• {a}" for a in exits[:4])
    else:
        book_txt = f"{len(open_eq)} open stock/ETF pick(s) · 0 exit alerts"

    today_str = dt.date.today().isoformat()
    return {
        "title": f"Options Desk Dispatch · {today_str} · [{posture}]",
        "color": color,
        "fields": [
            {
                "name": "Market Telemetry",
                "value": f"SPY `{spy_s}` · QQQ `{qqq_s}` · VIX `{vix_s}` · RV20 `{rv_s}`\nNext: `{snap.next_macro_event}`{kalshi_s}",
                "inline": False,
            },
            {
                "name": "Core TFSA Sleeve (PROVEN · Wealthsimple USD)",
                "value": f"{core_txt}\n_Monthly rebalance; park defensive cash in SGOV/BIL._",
                "inline": False,
            },
            {
                "name": "TFSA Satellite Watch",
                "value": setup_txt[:1000],
                "inline": False,
            },
            {
                "name": "Open Book",
                "value": book_txt[:1000],
                "inline": False,
            },
            {
                "name": "Data Pipeline LEDs",
                "value": _pipe_led_str(health),
                "inline": False,
            },
        ],
        "footer": {"text": "Options Desk · Run `opt` for full cockpit · Live execution OFF"},
    }


def build_events_embed(events: Sequence[AlertEvent]) -> Dict[str, Any]:
    """Bundle 1..N action events into a single clean Discord embed (never floods channel)."""
    has_crit = any(e.severity == "CRITICAL" for e in events)
    has_warn = any(e.severity == "WARN" for e in events)
    color = COLOR_BAD if has_crit else (COLOR_WARN if has_warn else COLOR_INFO)
    fields = [
        {
            "name": f"[{e.category}] {e.title}",
            "value": f"{e.body}\n`{e.cli_cmd}`",
            "inline": False,
        }
        for e in events[:8]
    ]
    return {
        "title": f"Options Desk Action Alert ({len(events)})",
        "color": color,
        "fields": fields,
        "footer": {"text": f"Options Desk · {dt.datetime.now().strftime('%H:%M local')}"},
    }


def send_discord_embed(
    embed: Dict[str, Any],
    *,
    webhook_url: Optional[str] = None,
    timeout: float = 15.0,
) -> bool:
    """Post a single Discord Rich Embed payload. Returns True on success, False if unconfigured/failed."""
    load_project_env()
    url = (webhook_url or os.environ.get("DISCORD_WEBHOOK_URL") or "").strip()
    if not url:
        return False
    payload = json.dumps({"username": "Options Desk", "embeds": [embed]}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "options-screener-desk/2.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status) < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


# ── Zero-Clutter Single-Slot macOS Notification ─────────────────────────────


def _escape_applescript(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def send_macos_notification(
    title: str,
    message: str,
    *,
    subtitle: str = "",
    group: str = "options-desk",
    sound: bool = False,
    open_cmd: Optional[str] = None,
    runner: Optional[Callable[[List[str]], Any]] = None,
) -> Tuple[bool, str]:
    """Deliver a single non-stacking macOS notification (no-op on non-macOS).

    Uses `terminal-notifier -group <group>` when installed so each notification
    replaces the previous one (zero Notification Center clutter) and supports
    click-to-open; otherwise falls back cleanly to built-in `/usr/bin/osascript`.
    Returns `(ok, backend_used)`.
    """
    if sys.platform != "darwin" and runner is None:
        return False, "non-macos"

    exec_fn = runner or (lambda cmd: subprocess.run(cmd, check=False, capture_output=True, timeout=5))
    tn = shutil.which("terminal-notifier")
    if tn:
        cmd = [tn, "-title", title, "-message", message, "-group", group]
        if subtitle:
            cmd += ["-subtitle", subtitle]
        if sound:
            cmd += ["-sound", "Glass"]
        if open_cmd:
            cmd += ["-execute", open_cmd]
        try:
            res = exec_fn(cmd)
            rc = getattr(res, "returncode", 0)
            return (rc == 0), "terminal-notifier"
        except Exception:
            pass

    osa = shutil.which("osascript") or "/usr/bin/osascript"
    if not os.path.exists(osa) and runner is None:
        return False, "unavailable"

    script = f'display notification "{_escape_applescript(message)}" with title "{_escape_applescript(title)}"'
    if subtitle:
        script += f' subtitle "{_escape_applescript(subtitle)}"'
    if sound:
        script += ' sound name "Glass"'
    try:
        res = exec_fn([osa, "-e", script])
        rc = getattr(res, "returncode", 0)
        return (rc == 0), "osascript"
    except Exception:
        return False, "error"


def summarize_for_macos(events: Sequence[AlertEvent]) -> Optional[Tuple[str, str, bool]]:
    """Coalesce action-required events into ONE crisp macOS banner `(subtitle, body, sound)`.

    Returns None when there are only low-priority INFO events (keeping macOS desktop silent
    unless an exit alert or broken data pipe genuinely needs attention).
    """
    actionable = [e for e in events if e.severity in ("CRITICAL", "WARN")]
    if not actionable:
        return None
    crits = [e for e in actionable if e.severity == "CRITICAL"]
    warns = [e for e in actionable if e.severity == "WARN"]
    if crits:
        sub = f"{len(crits)} Exit Alert" + ("s" if len(crits) > 1 else "")
        body = crits[0].body + (f" (+{len(actionable) - 1} more)" if len(actionable) > 1 else "")
        return sub, body[:160], True
    sub = "Data Pipeline Check"
    body = warns[0].body[:160]
    return sub, body, False
