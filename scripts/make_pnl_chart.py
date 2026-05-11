#!/usr/bin/env python3
"""
Build a stand-alone SVG equity curve from paper_trades.db.

Marks the 2026-05-11 11:17 calibration apply with a vertical line so
pre/post-calibration performance is visually separable.

Output:
  reports/equity_curve_<YYYY-MM-DD>.svg

Pure stdlib — no matplotlib / pandas required.
"""
from __future__ import annotations

import datetime as _dt
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "reports"

# Charts to render: (DB filename, slug used in output file, title suffix)
TARGETS = [
    ("paper_trades.db",        "equity",  "Equity Options"),
    ("paper_trades_crypto.db", "crypto",  "Crypto Options"),
]

# Calibration cutoffs. Trades with entry_id < first_post_entry_id are PRE;
# >= are POST. The iso_dt is used solely for the vertical marker on charts.
# Format: (first_post_entry_id, label, iso_dt_for_marker)
CALIBRATION_CUTOFFS: List[Tuple[int, str, str]] = [
    (220, "Calibration #1 applied", "2026-05-11 11:17:22"),
]
# Back-compat alias for the chart marker section below.
CALIBRATION_EVENTS: List[Tuple[str, str]] = [(c[2], c[1]) for c in CALIBRATION_CUTOFFS]


def _parse_dt(s: str) -> _dt.datetime:
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return _dt.datetime.strptime(s[:19] if " " in s else s[:10], fmt)
        except ValueError:
            continue
    raise ValueError(f"unparseable date: {s!r}")


def _load_closed_trades(db_path: Path, cohort: str = "all") -> List[dict]:
    """Load closed trades. Cohort filter uses entry_id vs the calibration cutoff
    in CALIBRATION_CUTOFFS: 'pre' = entry_id < first_post, 'post' = >=, 'all' = no
    filter. The crypto DB has no cohort split — caller passes cohort='all'."""
    first_post = CALIBRATION_CUTOFFS[-1][0] if CALIBRATION_CUTOFFS else None
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT entry_id, date, exit_date, ticker, strategy_name, pnl_usd "
            "FROM trades WHERE status='CLOSED' AND exit_date IS NOT NULL "
            "ORDER BY exit_date ASC"
        ).fetchall()
    out: List[dict] = []
    for r in rows:
        try:
            ed = _parse_dt(str(r["exit_date"]))
        except ValueError:
            continue
        if cohort in ("pre", "post") and first_post is not None:
            eid = int(r["entry_id"] or 0)
            is_post = eid >= first_post
            if cohort == "pre" and is_post:
                continue
            if cohort == "post" and not is_post:
                continue
        out.append({
            "exit_dt": ed,
            "ticker": r["ticker"] or "",
            "strategy": r["strategy_name"] or "",
            "pnl_usd": float(r["pnl_usd"] or 0.0),
        })
    return out


def _xfrac(t: _dt.datetime, t0: _dt.datetime, t1: _dt.datetime) -> float:
    if t1 == t0:
        return 0.0
    return (t - t0).total_seconds() / (t1 - t0).total_seconds()


def _yfrac(v: float, vmin: float, vmax: float) -> float:
    if vmax == vmin:
        return 0.5
    return (v - vmin) / (vmax - vmin)


def _render(db_path: Path, slug: str, title_suffix: str, show_calibration: bool,
            cohort: str = "all") -> int:
    trades = _load_closed_trades(db_path, cohort=cohort)
    if not trades:
        print(f"[{slug}] no closed trades for cohort={cohort} — skipping")
        return 0

    cum: List[Tuple[_dt.datetime, float, dict]] = []
    running = 0.0
    for t in trades:
        running += t["pnl_usd"]
        cum.append((t["exit_dt"], running, t))

    t0 = cum[0][0]
    t1 = cum[-1][0]
    vmin = min(0.0, min(v for _, v, _ in cum))
    vmax = max(0.0, max(v for _, v, _ in cum))
    pad = (vmax - vmin) * 0.08 or 100.0
    vmin -= pad
    vmax += pad

    W, H = 1100, 540
    LM, RM, TM, BM = 80, 30, 60, 70  # margins
    iw = W - LM - RM
    ih = H - TM - BM

    def px(t: _dt.datetime) -> float:
        return LM + _xfrac(t, t0, t1) * iw

    def py(v: float) -> float:
        return TM + (1.0 - _yfrac(v, vmin, vmax)) * ih

    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="ui-monospace,Menlo,Consolas,monospace" font-size="12">'
    )
    parts.append(f'<rect width="{W}" height="{H}" fill="#0f1115"/>')

    # title
    last_pnl = cum[-1][1]
    n_wins = sum(1 for t in trades if t["pnl_usd"] > 0)
    wr = n_wins / len(trades) * 100
    cohort_tag = "" if cohort == "all" else f"  ({cohort.upper()}-CALIBRATION)"
    parts.append(
        f'<text x="{LM}" y="28" fill="#e6e6e6" font-size="16" font-weight="600">'
        f'Equity Curve — {title_suffix}{cohort_tag}</text>'
    )
    parts.append(
        f'<text x="{LM}" y="46" fill="#a0a0a0" font-size="11">'
        f'{len(trades)} closed trades  |  realized P&amp;L ${last_pnl:+,.0f}  |  '
        f'win rate {wr:.1f}%  |  '
        f'{t0.strftime("%Y-%m-%d")} → {t1.strftime("%Y-%m-%d")}</text>'
    )

    # grid lines (5 horizontal)
    for i in range(6):
        y = TM + (ih * i / 5)
        v = vmax - (vmax - vmin) * i / 5
        parts.append(
            f'<line x1="{LM}" y1="{y:.1f}" x2="{LM+iw}" y2="{y:.1f}" '
            f'stroke="#222" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{LM-8}" y="{y+4:.1f}" fill="#888" text-anchor="end">'
            f'${v:,.0f}</text>'
        )

    # zero line emphasized
    yzero = py(0.0)
    parts.append(
        f'<line x1="{LM}" y1="{yzero:.1f}" x2="{LM+iw}" y2="{yzero:.1f}" '
        f'stroke="#555" stroke-width="1" stroke-dasharray="3,4"/>'
    )

    # x-axis date ticks: ~7 evenly spaced
    span = t1 - t0
    for i in range(7):
        ti = t0 + span * (i / 6)
        x = px(ti)
        parts.append(
            f'<line x1="{x:.1f}" y1="{TM+ih}" x2="{x:.1f}" y2="{TM+ih+5}" '
            f'stroke="#666" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{TM+ih+20}" fill="#888" text-anchor="middle">'
            f'{ti.strftime("%m-%d")}</text>'
        )

    # equity curve polyline
    pts = " ".join(f"{px(t):.1f},{py(v):.1f}" for t, v, _ in cum)
    parts.append(
        f'<polyline points="{pts}" fill="none" stroke="#4cc9f0" stroke-width="2"/>'
    )

    # win/loss dots
    for t, v, info in cum:
        color = "#21c55d" if info["pnl_usd"] > 0 else ("#ef4444" if info["pnl_usd"] < 0 else "#888")
        parts.append(
            f'<circle cx="{px(t):.1f}" cy="{py(v):.1f}" r="3" '
            f'fill="{color}" opacity="0.85"/>'
        )

    # calibration event markers (equity book only; suppressed for cohort views
    # since a pre/post chart's data is already on one side of the line).
    cal_events = CALIBRATION_EVENTS if (show_calibration and cohort == "all") else []
    for ev_dt_s, ev_label in cal_events:
        ev_dt = _parse_dt(ev_dt_s)
        if not (t0 <= ev_dt <= t1):
            continue
        x = px(ev_dt)
        parts.append(
            f'<line x1="{x:.1f}" y1="{TM}" x2="{x:.1f}" y2="{TM+ih}" '
            f'stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="5,3"/>'
        )
        parts.append(
            f'<text x="{x+6:.1f}" y="{TM+14:.1f}" fill="#f59e0b" font-size="11" '
            f'font-weight="600">▼ {ev_label}</text>'
        )
        parts.append(
            f'<text x="{x+6:.1f}" y="{TM+28:.1f}" fill="#f59e0b" font-size="10">'
            f'{ev_dt.strftime("%Y-%m-%d %H:%M")}</text>'
        )

    # legend
    lx = LM
    ly = H - 18
    parts.append(
        f'<circle cx="{lx+6}" cy="{ly-3}" r="3" fill="#21c55d"/>'
        f'<text x="{lx+14}" y="{ly}" fill="#bbb">winners</text>'
        f'<circle cx="{lx+82}" cy="{ly-3}" r="3" fill="#ef4444"/>'
        f'<text x="{lx+90}" y="{ly}" fill="#bbb">losers</text>'
        f'<line x1="{lx+150}" y1="{ly-3}" x2="{lx+170}" y2="{ly-3}" '
        f'stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="5,3"/>'
        f'<text x="{lx+176}" y="{ly}" fill="#bbb">calibration apply</text>'
    )

    parts.append("</svg>")

    today = _dt.date.today().isoformat()
    suffix = "" if cohort == "all" else f"_{cohort}"
    out_path = OUT_DIR / f"{slug}_curve{suffix}_{today}.svg"
    out_path.write_text("\n".join(parts))
    print(f"[{slug}/{cohort}] wrote {out_path}")
    print(f"        trades={len(trades)}  P&L=${last_pnl:+,.2f}  win_rate={wr:.1f}%")
    print(f"        span={t0.strftime('%Y-%m-%d')} → {t1.strftime('%Y-%m-%d')}")
    print(f"        open: file://{out_path}")
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=("all", "pre", "post"), default="all",
                    help="Filter equity trades by calibration cohort.")
    ap.add_argument("--equity-only", action="store_true",
                    help="Skip the crypto chart (useful from menu).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for db_name, slug, title in TARGETS:
        if args.equity_only and slug != "equity":
            continue
        db_path = PROJECT_ROOT / db_name
        if not db_path.exists():
            print(f"[{slug}] {db_path} not found — skipping")
            continue
        # Cohort filter only applies to the equity book — the crypto ledger
        # pre-dates the calibration and has no cohort split.
        effective_cohort = args.cohort if slug == "equity" else "all"
        _render(db_path, slug, title,
                show_calibration=(slug == "equity"),
                cohort=effective_cohort)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
