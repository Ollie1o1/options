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


BLOCKED_ALERT_FLAGS = ("HIST_WEAK_EDGE", "WIDE_SPREAD", "IV_RICH_FOR_LONG")


def filter_good_option_alerts(
    rows: Sequence[Any],
    *,
    max_spread_friction: float = 0.20,
    max_tfsa_friction: float = 0.12,
    max_tfsa_iv_vs_rv: float = 0.95,
    min_oi: int = 50,
) -> List[Any]:
    """Return only high-conviction options setups worthy of an automated Discord ping.

    Rules (no spam on dead/choppy sessions):
      - Rejects any candidate carrying earnings or execution/edge red flags
        (`EARNINGS*`, `HIST_WEAK_EDGE`, `WIDE_SPREAD`, `IV_RICH_FOR_LONG`).
      - Rejects thin contracts (`min_oi < min_oi`).
      - Keeps `PROMISING` defined-risk spreads with `friction_pct <= max_spread_friction`.
      - Keeps TFSA-legal single-leg options (`tfsa_legal=True`) with tight spreads
        (`friction_pct <= max_tfsa_friction`) AND cheap/fair implied vol
        (`iv_vs_rv` present and `<= max_tfsa_iv_vs_rv`).
    """
    good: List[Any] = []
    for r in rows:
        flags = tuple(str(f).upper() for f in (getattr(r, "risk_flags", None) or ()))
        if any(f.startswith("EARNINGS") or f in BLOCKED_ALERT_FLAGS for f in flags):
            continue
        oi = int(getattr(r, "min_oi", 0) or 0)
        if oi < min_oi:
            continue
        fric = float(getattr(r, "friction_pct", 1.0) or 0.0)
        badge = str(getattr(r, "evidence_badge", "") or "").upper()
        tfsa_ok = bool(getattr(r, "tfsa_legal", False))
        iv_rv = getattr(r, "iv_vs_rv", None)

        if badge == "PROMISING" and fric <= max_spread_friction:
            good.append(r)
        elif tfsa_ok and fric <= max_tfsa_friction and iv_rv is not None and float(iv_rv) <= max_tfsa_iv_vs_rv:
            good.append(r)
    return good


def build_options_embed(
    rows: Sequence[Any],
    *,
    note: str = "",
    strategy: str = "all",
    title_prefix: str = "Options Desk · Top Setups",
) -> Dict[str, Any]:
    """Build a crisp Discord Rich Embed from `[7] OPTIONS` Scanner rows (`OptionScanRow`)."""
    today_str = dt.date.today().isoformat()
    promising = [r for r in rows if getattr(r, "evidence_badge", "") == "PROMISING"]
    tfsa_rows = [r for r in rows if getattr(r, "tfsa_legal", False)]
    other_rows = [r for r in rows if r not in promising and r not in tfsa_rows]

    def _fmt_row(r: Any) -> str:
        side_s = f"+${r.mid_price:.2f} cr" if r.is_credit else f"-${r.mid_price:.2f} deb"
        ivrv = f" · IV/RV `{r.iv_vs_rv:.2f}x`" if getattr(r, "iv_vs_rv", None) is not None else ""
        acct = "TFSA OK" if r.tfsa_legal else "TAXABLE"
        return (
            f"• **{r.ticker}** {r.strategy} `{r.strikes_label}` ({r.dte}d) — "
            f"`{side_s}` · Limit `${r.walk_away_limit:.2f}` · Fric `{r.friction_pct*100:.1f}%` · "
            f"Risk `${r.max_risk_usd:,.0f}`{ivrv} · `{acct}`"
        )

    fields: List[Dict[str, Any]] = []
    if promising:
        fields.append(
            {
                "name": f"Promising Defined-Risk Spreads ({len(promising)})",
                "value": "\n".join(_fmt_row(r) for r in promising[:5])[:1000],
                "inline": False,
            }
        )
    if tfsa_rows:
        fields.append(
            {
                "name": f"TFSA-Legal Single-Leg Options ({len(tfsa_rows)})",
                "value": "\n".join(_fmt_row(r) for r in tfsa_rows[:5])[:1000],
                "inline": False,
            }
        )
    if not fields and other_rows:
        fields.append(
            {
                "name": f"Watchlist / Caution Candidates ({len(other_rows)})",
                "value": "\n".join(_fmt_row(r) for r in other_rows[:6])[:1000],
                "inline": False,
            }
        )
    if not fields:
        fields.append(
            {
                "name": "No Setups Cleared Filters",
                "value": "0 candidates passed friction/liquidity rules right now (markets may be closed or spreads too wide).",
                "inline": False,
            }
        )

    return {
        "title": f"{title_prefix} · {today_str} ({strategy.upper()})",
        "color": COLOR_GOOD if promising else (COLOR_INFO if tfsa_rows else COLOR_WARN),
        "description": note or f"{len(rows)} candidate(s) scanned · limit orders only · live execution OFF",
        "fields": fields,
        "footer": {"text": "Options Desk · Run `opt desk --tab options` for Chain Matrix & 2D Heatmap"},
    }


def build_option_ticket_embed(
    ticker: str,
    structure_name: str,
    *,
    spot: float,
    legs_summary: str,
    net_label: str,
    friction_pct: float,
    max_risk_usd: float,
    tfsa_legal: bool,
    greeks_line: str = "",
    ladder_lines: Sequence[str] = (),
    flags: Sequence[str] = (),
) -> Dict[str, Any]:
    """Build a mobile-ready Discord trade ticket embed for one selected option or staged multi-leg spread."""
    acct = "TFSA OK" if tfsa_legal else "TAXABLE / IBKR ONLY (CRA TFSA ban)"
    fields: List[Dict[str, Any]] = [
        {
            "name": "Structure & Legs",
            "value": f"Spot `${spot:.2f}` · `{acct}`\n{legs_summary}",
            "inline": False,
        },
        {
            "name": "Pricing & Risk",
            "value": f"{net_label} · Friction `{friction_pct*100:.1f}%` · Max Risk `${max_risk_usd:,.0f}`"
            + (f"\n{greeks_line}" if greeks_line else ""),
            "inline": False,
        },
    ]
    if ladder_lines:
        fields.append(
            {
                "name": "Walk-Away Limit Ladder (Never Chase)",
                "value": "\n".join(f"• {ln}" for ln in ladder_lines)[:1000],
                "inline": False,
            }
        )
    if flags:
        fields.append({"name": "Flags", "value": ", ".join(f"`{f}`" for f in flags), "inline": False})
    return {
        "title": f"Trade Ticket · {ticker.upper()} {structure_name}",
        "color": COLOR_GOOD if friction_pct <= 0.15 else (COLOR_WARN if friction_pct <= 0.25 else COLOR_BAD),
        "fields": fields,
        "footer": {"text": f"Options Desk · {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} · Paper/Limit discipline"},
    }


def _safe_float(val: Any) -> Optional[float]:
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return None


def _as_dict(val: Any) -> Dict[str, Any]:
    return val if isinstance(val, dict) else {}


def _as_list(val: Any) -> List[Any]:
    return val if isinstance(val, list) else []


def build_options_scan_api_embed(
    payload: Dict[str, Any],
    *,
    strategy: str = "all",
    query: str = "",
) -> Dict[str, Any]:
    """Build a Discord embed dict from `GET /desk/api/options/scan` JSON (used by `/options`)."""
    raw_rows = payload.get("rows")
    rows: List[Dict[str, Any]] = [r for r in raw_rows if isinstance(r, dict)] if isinstance(raw_rows, list) else []
    note = str(payload.get("note") or "")
    today_str = dt.date.today().isoformat()

    promising = [r for r in rows if str(r.get("evidence_badge") or "").upper() == "PROMISING"]
    tfsa_rows = [r for r in rows if bool(r.get("tfsa_legal"))]
    other_rows = [r for r in rows if r not in promising and r not in tfsa_rows]

    def _fmt_dict_row(r: Dict[str, Any]) -> str:
        tk = str(r.get("ticker") or "?")
        strat = str(r.get("strategy") or "Option")
        strikes = str(r.get("strikes_label") or "")
        dte = int(_safe_float(r.get("dte")) or 0)
        mid = _safe_float(r.get("mid_price")) or 0.0
        is_cr = bool(r.get("is_credit"))
        walk = _safe_float(r.get("walk_away_limit")) or mid
        fric = (_safe_float(r.get("friction_pct")) or 0.0) * 100.0
        risk = _safe_float(r.get("max_risk_usd")) or 0.0
        iv_rv = _safe_float(r.get("iv_vs_rv"))
        iv_s = f" · IV/RV `{iv_rv:.2f}x`" if iv_rv is not None else ""
        acct = "TFSA OK" if bool(r.get("tfsa_legal")) else "TAXABLE"
        side_s = f"+${mid:.2f} cr" if is_cr else f"-${mid:.2f} deb"
        return (
            f"• **{tk}** {strat} `{strikes}` ({dte}d) — "
            f"`{side_s}` · Limit `${walk:.2f}` · Fric `{fric:.1f}%` · Risk `${risk:,.0f}`{iv_s} · `{acct}`"
        )

    fields: List[Dict[str, Any]] = []
    if promising:
        fields.append(
            {
                "name": f"Promising Defined-Risk Spreads ({len(promising)})",
                "value": "\n".join(_fmt_dict_row(r) for r in promising[:5])[:1000],
                "inline": False,
            }
        )
    if tfsa_rows:
        fields.append(
            {
                "name": f"TFSA-Legal Single-Leg Options ({len(tfsa_rows)})",
                "value": "\n".join(_fmt_dict_row(r) for r in tfsa_rows[:5])[:1000],
                "inline": False,
            }
        )
    if not fields and other_rows:
        fields.append(
            {
                "name": f"Watchlist / Caution Candidates ({len(other_rows)})",
                "value": "\n".join(_fmt_dict_row(r) for r in other_rows[:6])[:1000],
                "inline": False,
            }
        )
    if not fields:
        fields.append(
            {
                "name": "No Setups Found",
                "value": "0 candidates matched your filter right now.",
                "inline": False,
            }
        )

    q_tag = f" · `{query}`" if query.strip() else ""
    return {
        "title": f"Options Hub Scan · {today_str} ({strategy.upper()}){q_tag}",
        "color": COLOR_GOOD if promising else (COLOR_INFO if tfsa_rows else COLOR_WARN),
        "description": note or f"{len(rows)} setup(s) returned · limit orders only",
        "fields": fields,
        "footer": {"text": "Options Desk · Use /chain <ticker> to inspect strikes & GEX"},
    }


def build_chain_api_embed(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build a Discord embed dict from `GET /desk/api/options/chain` JSON (used by `/chain`)."""
    if not payload.get("found") or not isinstance(payload.get("chain"), dict):
        tk = str(payload.get("ticker") or "?").upper()
        err = str(payload.get("error") or f"No chain available for {tk}.")
        return {
            "title": f"Chain Matrix · {tk}",
            "color": COLOR_WARN,
            "description": err,
            "footer": {"text": "Options Desk · Try live=True during market hours"},
        }

    ch: Dict[str, Any] = payload["chain"]
    tk = str(ch.get("ticker") or "?").upper()
    spot = _safe_float(ch.get("spot")) or 0.0
    exp = str(ch.get("expiration") or "—")
    dte = int(_safe_float(ch.get("dte")) or 0)
    source = str(ch.get("source") or "chain_archive")
    atm_iv = _safe_float(ch.get("atm_iv_pct"))
    rv20 = _safe_float(ch.get("rv20_pct"))
    iv_rv = _safe_float(ch.get("iv_vs_rv"))
    em_usd = _safe_float(ch.get("expected_move_usd"))
    em_pct = _safe_float(ch.get("expected_move_pct"))
    net_gex = _safe_float(ch.get("net_gex_m"))
    gex_reg = str(ch.get("gex_regime") or "UNKNOWN")
    gamma_flip = _safe_float(ch.get("gamma_flip"))
    call_wall = _safe_float(ch.get("call_wall"))
    put_wall = _safe_float(ch.get("put_wall"))

    vol_parts: List[str] = [f"Spot `${spot:.2f}`"]
    if atm_iv is not None:
        vol_parts.append(f"ATM IV `{atm_iv:.1f}%`")
    if rv20 is not None:
        vol_parts.append(f"RV20 `{rv20:.1f}%`")
    if iv_rv is not None:
        vol_parts.append(f"IV/RV `{iv_rv:.2f}x`")
    if em_usd is not None and em_pct is not None:
        vol_parts.append(f"±1σ `${em_usd:.2f}` (`±{em_pct:.1f}%`)")

    gex_parts: List[str] = []
    if net_gex is not None:
        gex_parts.append(f"Net GEX `{net_gex:+.1f}M` ({gex_reg})")
    if gamma_flip is not None:
        gex_parts.append(f"Flip `${gamma_flip:.0f}`")
    if call_wall is not None:
        gex_parts.append(f"Call Wall `${call_wall:.0f}`")
    if put_wall is not None:
        gex_parts.append(f"Put Wall `${put_wall:.0f}`")

    raw_rows = ch.get("rows")
    pair_rows: List[Dict[str, Any]] = [r for r in raw_rows if isinstance(r, dict)] if isinstance(raw_rows, list) else []
    sorted_near = sorted(pair_rows, key=lambda r: abs((_safe_float(r.get("strike")) or 0.0) - spot))[:7]
    sorted_near.sort(key=lambda r: _safe_float(r.get("strike")) or 0.0)

    ladder_lines: List[str] = []
    for pr in sorted_near:
        k = _safe_float(pr.get("strike")) or 0.0
        c_obj = _as_dict(pr.get("call"))
        p_obj = _as_dict(pr.get("put"))
        c_mid = _safe_float(c_obj.get("mid")) if c_obj else None
        p_mid = _safe_float(p_obj.get("mid")) if p_obj else None
        c_del = _safe_float(c_obj.get("delta")) if c_obj else None
        p_del = _safe_float(p_obj.get("delta")) if p_obj else None
        c_s = f"${c_mid:5.2f} Δ{c_del:+.2f}" if (c_mid is not None and c_del is not None) else (f"${c_mid:5.2f}" if c_mid is not None else "  —  ")
        p_s = f"${p_mid:5.2f} Δ{p_del:+.2f}" if (p_mid is not None and p_del is not None) else (f"${p_mid:5.2f}" if p_mid is not None else "  —  ")
        tags = _as_list(pr.get("tags"))
        mark = "*" if "SPOT" in tags else " "
        ladder_lines.append(f"{mark}C {c_s:<13} | {k:6.1f} | P {p_s}")

    fields: List[Dict[str, Any]] = [
        {"name": "Vol & Expected Move", "value": " · ".join(vol_parts), "inline": False},
    ]
    if gex_parts:
        fields.append({"name": "Dealer GEX & OI Walls", "value": " · ".join(gex_parts), "inline": False})
    if ladder_lines:
        fields.append(
            {
                "name": "Near-ATM Call | Strike | Put",
                "value": "```\n" + "\n".join(ladder_lines) + "\n```",
                "inline": False,
            }
        )
    return {
        "title": f"Chain Matrix · {tk} · {exp} ({dte}d)",
        "color": COLOR_GOOD if gex_reg == "POSITIVE_GEX" else (COLOR_WARN if gex_reg == "NEGATIVE_GEX" else COLOR_INFO),
        "description": f"Source: `{source}` · Snapshot `{ch.get('snap_date') or '—'}`",
        "fields": fields,
        "footer": {"text": "Options Desk · Run `opt desk --tab options` for full GEX bars & 2D heatmap"},
    }


def build_morning_api_embed(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build a Discord embed dict from `GET /desk/api/overview` JSON (used by `/morning`)."""
    snap = _as_dict(payload.get("snapshot"))
    book = _as_dict(payload.get("book"))
    posture = str(snap.get("regime_posture") or "NEUTRAL")
    exits = _as_list(book.get("exit_alerts"))

    spy_last = _safe_float(snap.get("spy_last"))
    spy_chg = _safe_float(snap.get("spy_chg_1d_pct"))
    qqq_chg = _safe_float(snap.get("qqq_chg_1d_pct"))
    vix = _safe_float(snap.get("vix"))
    rv20 = _safe_float(snap.get("spy_rv20"))
    spy_s = f"${spy_last:.2f} ({spy_chg:+.1f}%)" if (spy_last is not None and spy_chg is not None) else "—"
    qqq_s = f"{qqq_chg:+.1f}%" if qqq_chg is not None else "—"
    vix_s = f"{vix:.1f} ({snap.get('vix_term') or '—'})" if vix is not None else "—"
    rv_s = f"{rv20:.1f}%" if rv20 is not None else "—"

    core_raw = _as_list(snap.get("core_allocation"))
    core_lines: List[str] = []
    for c in core_raw:
        if not isinstance(c, dict):
            continue
        tk = "SGOV/CASH" if str(c.get("ticker") or "") == "CASH" else str(c.get("ticker") or "?")
        wt = (_safe_float(c.get("target_weight")) or 0.0) * 100.0
        core_lines.append(f"**{tk}** `{wt:.0f}%`")
    core_txt = " · ".join(core_lines) if core_lines else "Run `opt refresh`"

    opps_raw = _as_list(snap.get("opportunities"))
    tfsa_lines: List[str] = []
    for o in opps_raw:
        if not isinstance(o, dict):
            continue
        rflags = _as_list(o.get("risk_flags"))
        pb_id = str(o.get("playbook_id") or "")
        if "TAXABLE_ONLY" in rflags or pb_id in ("vol_managed_outlook", "etf_wide_spread", "etf_bear_call"):
            continue
        tfsa_lines.append(f"• **{o.get('ticker', '?')}** ({o.get('playbook_label', '')}) — {str(o.get('setup_summary') or '')[:70]}")
    setup_txt = "\n".join(tfsa_lines[:4]) if tfsa_lines else "No satellite TFSA setups firing — stay in Core ETF sleeve."

    eq_picks = _as_list(book.get("equity_picks"))
    open_eq = [p for p in eq_picks if isinstance(p, dict) and str(p.get("status") or "").upper() == "OPEN"]
    book_txt = "**EXIT ALERTS:**\n" + "\n".join(f"• {a}" for a in exits[:4]) if exits else f"{len(open_eq)} open stock/ETF pick(s) · 0 exit alerts"

    color = COLOR_BAD if exits else (COLOR_GOOD if posture == "RISK_ON" else (COLOR_BAD if posture == "RISK_OFF" else COLOR_WARN))
    return {
        "title": f"Options Desk Brief · {dt.date.today().isoformat()} · [{posture}]",
        "color": color,
        "fields": [
            {
                "name": "Market Telemetry",
                "value": f"SPY `{spy_s}` · QQQ `{qqq_s}` · VIX `{vix_s}` · RV20 `{rv_s}`\nNext: `{snap.get('next_macro_event') or '—'}`",
                "inline": False,
            },
            {"name": "Core TFSA Sleeve", "value": core_txt, "inline": False},
            {"name": "TFSA Satellite Watch", "value": setup_txt[:1000], "inline": False},
            {"name": "Open Book", "value": book_txt[:1000], "inline": False},
        ],
        "footer": {"text": "Options Desk · Use /options or /chain <ticker> for live options scans"},
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
