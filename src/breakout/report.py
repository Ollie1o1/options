# src/breakout/report.py
"""Rendering for the breakout engine: backtest metric tables and live
breakout/breakdown leaderboards. Uses src/ui.py table/rule helpers; no raw ANSI
(color discipline is enforced by tests/test_color_discipline.py)."""
from __future__ import annotations
from typing import List
from src import ui


# Column specs — ui.table requires list[{'h': str, 'w': int, 'align': str}]
_BACKTEST_COLS = [
    {"h": "Horizon",       "w": 8,  "align": "left"},
    {"h": "n",             "w": 5,  "align": "right"},
    {"h": "Brier",         "w": 6,  "align": "right"},
    {"h": "ECE",           "w": 6,  "align": "right"},
    {"h": "AUC",           "w": 6,  "align": "right"},
    {"h": "Cover",         "w": 6,  "align": "right"},
    {"h": "Skill vs base", "w": 13, "align": "right"},
]

_FORECAST_COLS = [
    {"h": "Ticker",   "w": 7,  "align": "left"},
    {"h": "Hzn",      "w": 5,  "align": "left"},
    {"h": "Median",   "w": 8,  "align": "right"},
    {"h": "80% band", "w": 12, "align": "right"},
    {"h": "P(+10%)",  "w": 8,  "align": "right"},
    {"h": "P(-10%)",  "w": 8,  "align": "right"},
]


def render_backtest(result: dict) -> str:
    lines = [ui.rule(80, "BREAKOUT BACKTEST — calibration of P(+10%)")]
    rows = []
    for label, m in result.items():
        if not m or m.get("n", 0) == 0:
            rows.append([label, "0", "-", "-", "-", "-", "-"])
            continue
        auc = f"{m['auc']:.3f}" if m.get("auc") is not None else "n/a"
        rows.append([label, str(m["n"]), f"{m['brier']:.3f}", f"{m['ece']:.3f}",
                     auc, f"{m['coverage']:.0%}", f"{m['skill_vs_baseline']:+.3f}"])
    lines.append(ui.table(_BACKTEST_COLS, rows))
    lines.append("")
    lines.append("  Skill>0 means it beats the unconditional vol baseline. Caveat:")
    lines.append("  yfinance has survivorship bias — downside-tail accuracy is optimistic.")
    return "\n".join(lines)


def render_forecasts(rows: List[dict], top: int = 10) -> str:
    if not rows:
        return "  No forecast data — run `python -m src.breakout --update-data` first."

    def _fmt(r):
        lo, hi = r["band"]
        return [r["ticker"], r["horizon"], f"{r['point']:+.1%}",
                f"[{lo:+.0%},{hi:+.0%}]", f"{r['up_prob']:.0%}", f"{r['down_prob']:.0%}"]
    ups = sorted(rows, key=lambda r: r["up_prob"], reverse=True)[:top]
    downs = sorted(rows, key=lambda r: r["down_prob"], reverse=True)[:top]
    out = [ui.rule(80, "BREAKOUT CANDIDATES (highest upper-tail probability)"),
           ui.table(_FORECAST_COLS, [_fmt(r) for r in ups]), "",
           ui.rule(80, "BREAKDOWN CANDIDATES (highest lower-tail probability)"),
           ui.table(_FORECAST_COLS, [_fmt(r) for r in downs]),
           "  P(-10%) is model-raw; calibration is scored for P(+10%) only."]
    return "\n".join(out)
