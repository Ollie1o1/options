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
    *,
    cloud_mode: Optional[bool] = None,
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

    # 4. Critical Data Pipe Failures (STALE only — avoids noisy WARN flapping).
    # On ephemeral cloud runners (GitHub Actions), local-only archives (chains/earnings)
    # are not persisted, so only monitor OHLCV and disk there to prevent daily false alarms.
    is_cloud = (os.environ.get("GITHUB_ACTIONS", "").lower() == "true") if cloud_mode is None else bool(cloud_mode)
    monitored_pipes = ("ohlcv", "disk") if is_cloud else ("ohlcv", "chains", "earnings", "disk")
    pipes = health.get("pipes") or {}
    stale_names = [k for k, v in pipes.items() if k in monitored_pipes and v.get("status") == "STALE"]
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


def _load_fred_telemetry_snippet() -> str:
    """Return compact cached FRED credit & yield-curve snippet (zero network I/O)."""
    try:
        from src.topdown.macro import load_fred_cache

        cache = load_fred_cache()
        parts: List[str] = []
        hy_obs = _as_list(_as_dict(cache.get("BAMLH0A0HYM2")).get("obs"))
        if hy_obs and isinstance(hy_obs[-1], (list, tuple)) and len(hy_obs[-1]) >= 2:
            parts.append(f"HY OAS `{float(hy_obs[-1][1]):.2f}%`")
        yc_obs = _as_list(_as_dict(cache.get("T10Y2Y")).get("obs"))
        if yc_obs and isinstance(yc_obs[-1], (list, tuple)) and len(yc_obs[-1]) >= 2:
            parts.append(f"10Y-2Y `{float(yc_obs[-1][1]):+.2f}%`")
        return " · ".join(parts)
    except Exception:
        return ""


def _regime_directive(posture: str, rv20: Optional[float]) -> str:
    rv_tag = f"SPY RV20 {rv20:.1f}%" if rv20 is not None else "elevated vol"
    if posture == "RISK_ON":
        return "Desk Stance: Full Core TFSA allocation active · selective pullback longs & rich-vol credit spreads permitted."
    if posture == "RISK_OFF":
        return f"Desk Stance: Defensive ({rv_tag}) · no unhedged longs; stay in Core TFSA defensive sleeve & cheap-vol hedges."
    return "Desk Stance: Selective · enforce walk-away limits and avoid chasing extended breakouts."


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
    rv20_val = getattr(snap, "spy_rv20", None)
    rv_s = f"{rv20_val:.1f}%" if rv20_val is not None else "—"
    kalshi_s = (
        f" · Kalshi: {snap.kalshi_macro[0]['label']} ({snap.kalshi_macro[0]['prob_pct']:.0f}%)"
        if snap.kalshi_macro
        else ""
    )
    fred_s = _load_fred_telemetry_snippet()
    as_of_s = getattr(snap, "ohlcv_as_of", None)
    macro_line = f"Next: `{snap.next_macro_event}`{kalshi_s}"
    if fred_s:
        macro_line = f"{fred_s} · {macro_line}"
    if as_of_s:
        macro_line += f"\n_Data as of `{as_of_s}` close_"

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
    clean_opps = [o for o in tfsa_opps if "BEAR_REGIME" not in o.risk_flags]
    blocked_opps = [o for o in tfsa_opps if "BEAR_REGIME" in o.risk_flags]
    if clean_opps:
        setup_txt = "\n".join(
            f"• **{o.ticker}** ({o.playbook_label}) — {o.setup_summary[:70]}"
            + (f" ⚠ `{','.join(o.risk_flags[:2])}`" if o.risk_flags else "")
            for o in clean_opps[:4]
        )
    elif blocked_opps:
        blocked_lines = [
            f"• ~~**{o.ticker}**~~ — {o.setup_summary[:76]}"
            for o in blocked_opps[:3]
        ]
        setup_txt = "_Blocked by `BEAR_REGIME` (watchlist only — do not force longs):_\n" + "\n".join(blocked_lines)
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
        "description": _regime_directive(posture, rv20_val),
        "fields": [
            {
                "name": "Market Telemetry",
                "value": f"SPY `{spy_s}` · QQQ `{qqq_s}` · VIX `{vix_s}` · RV20 `{rv_s}`\n{macro_line}",
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
    max_tfsa_friction: float = 0.08,
    max_tfsa_iv_vs_rv: float = 0.95,
    min_oi: int = 100,
) -> List[Any]:
    """Return only high-conviction options setups worthy of an automated Discord ping.

    Rules (quant execution discipline — zero spam on dead/wide sessions):
      - Rejects any candidate carrying earnings or execution/edge red flags
        (`EARNINGS*`, `HIST_WEAK_EDGE`, `WIDE_SPREAD`, `IV_RICH_FOR_LONG`).
      - Rejects thin contracts (`min_oi < min_oi`).
      - For defined-risk spreads: rejects `VOL_CHEAP_SHORT` (never sell cheap vol),
        keeps `PROMISING` spreads with `friction_pct <= max_spread_friction`, plus
        tight rich-vol Bull Puts (`friction_pct <= 0.15`, `iv_vs_rv >= 1.00`, `ror_pct >= 12%`).
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
        is_cr = bool(getattr(r, "is_credit", False))
        strat_key = str(getattr(r, "strategy_key", "") or "").lower()
        ror = _safe_float(getattr(r, "ror_pct", None))

        if is_cr and "VOL_CHEAP_SHORT" in flags:
            continue

        if badge == "PROMISING" and fric <= max_spread_friction:
            good.append(r)
        elif (
            strat_key == "bull_put"
            and is_cr
            and fric <= 0.15
            and iv_rv is not None
            and float(iv_rv) >= 1.00
            and ror is not None
            and ror >= 12.0
        ):
            good.append(r)
        elif tfsa_ok and fric <= max_tfsa_friction and iv_rv is not None and float(iv_rv) <= max_tfsa_iv_vs_rv:
            good.append(r)
    return good


SESSION_TITLES: Dict[str, Tuple[str, str]] = {
    "morning": ("Morning Desk Sweep", "Pre-market & open radar across liquid ETFs & leaders"),
    "midday": ("Mid-Morning Market Sweep", "11:00 AM ET · Opening dust settled, tightest intraday spreads"),
    "afternoon": ("Afternoon Pre-Close Sweep", "2:30 PM ET · Late-session leadership & overnight swing setups"),
    "manual": ("On-Demand Market Sweep", "Live broad-market scan across ETFs & large-cap leaders"),
}


def resolve_market_session(session: str = "auto", now_utc: Optional[dt.datetime] = None) -> str:
    """Resolve `auto` into `morning` / `midday` / `afternoon` / `manual` using US Eastern hours."""
    s = (session or "auto").strip().lower()
    if s in SESSION_TITLES:
        return s
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    # Approximate ET (UTC-5 EST / UTC-4 EDT — close enough for session bucketing)
    et_hour = (now.hour - 5) % 24
    if 6 <= et_hour < 10:
        return "morning"
    if 10 <= et_hour < 14:
        return "midday"
    if 14 <= et_hour < 17:
        return "afternoon"
    return "manual"


def _lookup_sector(ticker: str) -> Optional[str]:
    try:
        from src.radar.scanners import get_sector_etf

        return get_sector_etf(ticker)
    except Exception:
        return None


def _dedupe_by_ticker(rows: Sequence[Any], limit: int = 5, *, max_per_sector: int = 2) -> List[Any]:
    """Keep at most 1 setup per ticker and cap correlated sector concentration."""
    out: List[Any] = []
    seen: set[str] = set()
    sector_counts: Dict[str, int] = {}
    overflow: List[Any] = []
    for r in rows:
        tk = str(getattr(r, "ticker", "") or "").upper()
        if not tk or tk in seen:
            continue
        seen.add(tk)
        sec = _lookup_sector(tk)
        if sec and max_per_sector > 0 and sector_counts.get(sec, 0) >= max_per_sector:
            overflow.append(r)
            continue
        if sec:
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
        out.append(r)
        if len(out) >= limit:
            return out
    for r in overflow:
        if len(out) >= limit:
            break
        out.append(r)
    return out


def _fmt_short_exp(exp_str: str) -> str:
    try:
        d = dt.date.fromisoformat(str(exp_str)[:10])
        return d.strftime("%b%d")
    except ValueError:
        return ""


def _fmt_oi_compact(oi: int) -> str:
    if oi >= 1000:
        return f"{oi / 1000.0:.1f}k"
    return str(oi)


def _single_leg_breakeven(r: Any) -> Optional[Tuple[float, float]]:
    """Compute (breakeven_price, pct_move_from_spot) for single-leg long calls/puts."""
    spot = _safe_float(getattr(r, "spot", None))
    if spot is None or spot <= 0:
        return None
    strat = str(getattr(r, "strategy_key", "") or "").lower()
    if strat not in ("long_call", "long_put"):
        return None
    strike: Optional[float] = None
    legs = getattr(r, "legs", None) or ()
    if legs:
        strike = _safe_float(getattr(legs[0], "strike", None))
    if strike is None:
        raw_lbl = str(getattr(r, "strikes_label", "") or "").strip().lstrip("$")
        num = "".join(ch for ch in raw_lbl if ch.isdigit() or ch == ".")
        try:
            strike = float(num) if num else None
        except ValueError:
            strike = None
    if strike is None or strike <= 0:
        return None
    cost = _safe_float(getattr(r, "walk_away_limit", None)) or _safe_float(getattr(r, "mid_price", None)) or 0.0
    be = (strike + cost) if strat == "long_call" else (strike - cost)
    pct = (be / spot - 1.0) * 100.0
    return be, pct


def build_options_embed(
    rows: Sequence[Any],
    *,
    note: str = "",
    strategy: str = "all",
    title_prefix: str = "Options Desk · Top Setups",
    session: str = "",
    market_pulse: Optional[Dict[str, Any]] = None,
    all_rows: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Build a broad-market, session-aware Discord Rich Embed from `[7] OPTIONS` Scanner rows."""
    today_str = dt.date.today().isoformat()
    promising_all = [
        r for r in rows
        if getattr(r, "evidence_badge", "") == "PROMISING"
        or (bool(getattr(r, "is_credit", False)) and not bool(getattr(r, "tfsa_legal", False)))
    ]
    tfsa_all = [r for r in rows if getattr(r, "tfsa_legal", False)]
    other_all = [r for r in rows if r not in promising_all and r not in tfsa_all]

    # Rank spreads by tightest friction + highest RoR; TFSA single-legs by cheap IV/RV + tight friction
    promising_ranked = sorted(
        promising_all,
        key=lambda r: (
            getattr(r, "evidence_badge", "") != "PROMISING",
            float(getattr(r, "friction_pct", 1.0) or 1.0),
            -(_safe_float(getattr(r, "ror_pct", None)) or 0.0),
        ),
    )
    tfsa_ranked = sorted(
        tfsa_all,
        key=lambda r: (
            getattr(r, "iv_vs_rv", None) is None,
            float(getattr(r, "iv_vs_rv", 9.9) or 9.9) + 2.5 * float(getattr(r, "friction_pct", 1.0) or 1.0),
            -int(getattr(r, "min_oi", 0) or 0),
        ),
    )
    promising_top = _dedupe_by_ticker(promising_ranked, limit=5, max_per_sector=2)
    tfsa_top = _dedupe_by_ticker(tfsa_ranked, limit=5, max_per_sector=2)
    other_top = _dedupe_by_ticker(other_all, limit=5, max_per_sector=2)

    distinct_promising = len({str(getattr(r, "ticker", "")).upper() for r in promising_all})
    distinct_tfsa = len({str(getattr(r, "ticker", "")).upper() for r in tfsa_all})

    def _fmt_row(r: Any, *, show_acct: bool = False) -> str:
        exp_short = _fmt_short_exp(str(getattr(r, "expiration", "") or ""))
        exp_lbl = f"{exp_short} ({r.dte}d)" if exp_short else f"({r.dte}d)"
        spot = _safe_float(getattr(r, "spot", None))
        spot_s = f" `${spot:.2f}`" if spot is not None and spot > 0 else ""
        iv_pct = _safe_float(getattr(r, "iv_pct", None))
        iv_rv = _safe_float(getattr(r, "iv_vs_rv", None))
        if iv_pct is not None and iv_rv is not None:
            vol_s = f" · IV `{iv_pct:.0f}%` (`{iv_rv:.2f}x` RV)"
        elif iv_rv is not None:
            vol_s = f" · IV/RV `{iv_rv:.2f}x`"
        else:
            vol_s = ""
        oi_i = int(getattr(r, "min_oi", 0) or 0)
        oi_s = f" · OI `{_fmt_oi_compact(oi_i)}`" if oi_i > 0 else ""
        acct_s = (" · `TFSA OK`" if r.tfsa_legal else " · `TAXABLE`") if show_acct else ""

        if r.is_credit:
            sd = _safe_float(getattr(r, "short_delta", None))
            delta_s = f" · Δ`{sd:.2f}` (≈`{(1.0 - abs(sd)) * 100:.0f}%` PoP)" if sd is not None else ""
            ror = _safe_float(getattr(r, "ror_pct", None))
            ror_s = f" · RoR `{ror:.0f}%`" if ror is not None else ""
            return (
                f"• **{r.ticker}**{spot_s} · {r.strategy} `{r.strikes_label}` {exp_lbl}{delta_s}\n"
                f"  Cr `+${r.mid_price:.2f}` (Min `${r.walk_away_limit:.2f}`){ror_s} · "
                f"Risk `${r.max_risk_usd:,.0f}` · Fric `{r.friction_pct*100:.1f}%`{vol_s}{oi_s}{acct_s}"
            )

        sd = _safe_float(getattr(r, "short_delta", None))
        strat_k = str(getattr(r, "strategy_key", "") or "").lower()
        if sd is not None:
            signed_d = -abs(sd) if "put" in strat_k else abs(sd)
            delta_s = f" · Δ`{signed_d:+.2f}`"
        else:
            delta_s = ""
        be_info = _single_leg_breakeven(r)
        be_s = f" · BE `${be_info[0]:.2f}` (`{be_info[1]:+.1f}%`)" if be_info is not None else ""
        return (
            f"• **{r.ticker}**{spot_s} · {r.strategy} `{r.strikes_label}` {exp_lbl}{delta_s}{be_s}\n"
            f"  Limit `${r.walk_away_limit:.2f}` · Fric `{r.friction_pct*100:.1f}%`{vol_s}{oi_s}{acct_s}"
        )

    fields: List[Dict[str, Any]] = []

    if isinstance(market_pulse, dict) and int(market_pulse.get("breadth_total") or 0) > 0:
        pulse_parts: List[str] = []
        for idx_lbl, key in (("SPY", "spy_chg"), ("QQQ", "qqq_chg"), ("IWM", "iwm_chg")):
            val = _safe_float(market_pulse.get(key))
            if val is not None:
                pulse_parts.append(f"{idx_lbl} `{val:+.1f}%`")
        adv = int(market_pulse.get("advancers") or 0)
        dec = int(market_pulse.get("decliners") or 0)
        pulse_parts.append(f"Breadth `{adv}↑ / {dec}↓`")
        lines: List[str] = [" · ".join(pulse_parts)]
        leaders = _as_list(market_pulse.get("leaders"))
        laggards = _as_list(market_pulse.get("laggards"))
        if leaders:
            ld_txt = ", ".join(f"**{tk}** `{float(ch):+.1f}%`" for tk, ch in leaders[:3])
            lg_txt = ", ".join(f"**{tk}** `{float(ch):+.1f}%`" for tk, ch in laggards[:3])
            lines.append(f"Leaders: {ld_txt}" + (f" · Laggards: {lg_txt}" if lg_txt else ""))
        fields.append({"name": "Market Pulse & Breadth", "value": "\n".join(lines)[:1000], "inline": False})

    if promising_top:
        tk_word = "ticker" if distinct_promising == 1 else "tickers"
        fields.append(
            {
                "name": f"Best Defined-Risk Spreads · TAXABLE ({distinct_promising} {tk_word} · Top {len(promising_top)})",
                "value": "\n".join(_fmt_row(r) for r in promising_top)[:1000],
                "inline": False,
            }
        )
    if tfsa_top:
        tk_word = "ticker" if distinct_tfsa == 1 else "tickers"
        fields.append(
            {
                "name": f"Best TFSA-Legal Options ({distinct_tfsa} {tk_word} · Top {len(tfsa_top)})",
                "value": "\n".join(_fmt_row(r) for r in tfsa_top)[:1000],
                "inline": False,
            }
        )
    if not promising_top and not tfsa_top and other_top:
        fields.append(
            {
                "name": f"Watchlist / Caution Candidates ({len(other_all)})",
                "value": "\n".join(_fmt_row(r, show_acct=True) for r in other_top)[:1000],
                "inline": False,
            }
        )

    # Cross-market vol extremes computed across the FULL scanned universe (never post-filtered subset!)
    vol_universe = all_rows if all_rows is not None else rows
    iv_by_tk: Dict[str, float] = {}
    for r in vol_universe:
        iv_v = _safe_float(getattr(r, "iv_vs_rv", None))
        tk = str(getattr(r, "ticker", "") or "").upper()
        if tk and iv_v is not None and tk not in iv_by_tk:
            iv_by_tk[tk] = iv_v
    if len(iv_by_tk) >= 4:
        sorted_iv = sorted(iv_by_tk.items(), key=lambda x: x[1])
        cheap = [(tk, v) for tk, v in sorted_iv if v <= 0.95][:3]
        rich = [(tk, v) for tk, v in reversed(sorted_iv) if v >= 1.05][:3]
        cheap_s = (
            ", ".join(f"**{tk}** `{v:.2f}x`" for tk, v in cheap)
            if cheap
            else "None ≤ 0.95x (vol priced rich across board)"
        )
        rich_s = (
            ", ".join(f"**{tk}** `{v:.2f}x`" for tk, v in rich)
            if rich
            else "None ≥ 1.05x (vol priced below realized)"
        )
        fields.append(
            {
                "name": f"Cross-Market Vol Extremes · IV/RV20 ({len(iv_by_tk)} tickers)",
                "value": f"Cheapest (buy vol ≤0.95x): {cheap_s}\nRichest (sell vol ≥1.05x): {rich_s}",
                "inline": False,
            }
        )

    if not promising_top and not tfsa_top and not other_top:
        fields.append(
            {
                "name": "No Setups Cleared Filters",
                "value": "0 candidates passed friction/liquidity rules right now (markets may be closed or spreads too wide).",
                "inline": False,
            }
        )

    sess_key = resolve_market_session(session) if session else ""
    sess_title, sess_desc = SESSION_TITLES.get(sess_key, ("", ""))
    if sess_title:
        tag = "High-Conviction · " if "High-Conviction" in title_prefix else ""
        final_title = f"Options Desk · {tag}{sess_title} · {today_str} ({strategy.upper()})"
    else:
        final_title = f"{title_prefix} · {today_str} ({strategy.upper()})"
    desc_parts = [p for p in (sess_desc, note) if p]
    return {
        "title": final_title,
        "color": COLOR_GOOD if promising_top else (COLOR_INFO if tfsa_top else COLOR_WARN),
        "description": " · ".join(desc_parts) or f"{len(rows)} candidate(s) scanned · limit orders only · live execution OFF",
        "fields": fields,
        "footer": {"text": "Options Desk · Sector-diversified (max 2/sector) · Limit orders only · Live execution OFF"},
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
    except urllib.error.HTTPError as exc:
        print(f"[discord] Webhook HTTP error: {exc.code} {exc.reason}", file=sys.stderr)
        return False
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"[discord] Webhook delivery failed: {type(exc).__name__}", file=sys.stderr)
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
