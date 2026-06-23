"""Rendering for the vol-intelligence report: IV movers + implied-vs-realized
(VRP) blocks via src/ui.py. No raw ANSI (color discipline is enforced by a
source scan in the tests)."""
from __future__ import annotations
from typing import List, Optional
from src import ui

_MOVER_COLS = [{"h": "Symbol", "w": 8, "align": "left"},
               {"h": "IV", "w": 7, "align": "right"},
               {"h": "dIV(vp)", "w": 9, "align": "right"},
               {"h": "RV%ile", "w": 8, "align": "right"}]
_VRP_COLS = [{"h": "Symbol", "w": 8, "align": "left"},
             {"h": "Impl", "w": 7, "align": "right"},
             {"h": "Rlzd", "w": 7, "align": "right"},
             {"h": "VRP(vp)", "w": 9, "align": "right"},
             {"h": "Label", "w": 7, "align": "left"}]
W = 74


def _pct(x: Optional[float]) -> str:
    return f"{x * 100:.1f}%" if x is not None else "-"


def _vp(x: Optional[float]) -> str:
    return f"{x * 100:+.1f}" if x is not None else "-"


def _pctile(x: Optional[float]) -> str:
    return f"{x * 100:.0f}%" if x is not None else "-"


def render_iv_movers(rows: List[dict], top: int = 10) -> str:
    have = [r for r in rows if r.get("d_iv") is not None]
    have.sort(key=lambda r: abs(r["d_iv"]), reverse=True)
    body = [[r["symbol"], _pct(r.get("iv")), _vp(r.get("d_iv")), _pctile(r.get("rv_pctile"))]
            for r in have[:top]]
    return "\n".join([ui.rule(W, "IV MOVERS - dATM-IV vs prior snapshot"),
                      ui.table(_MOVER_COLS, body)])


def render_vrp(rows: List[dict], top: int = 10) -> str:
    have = [r for r in rows if r.get("vrp") is not None]
    rich = sorted(have, key=lambda r: r["vrp"], reverse=True)[:top]
    cheap = sorted(have, key=lambda r: r["vrp"])[:top]

    def _fmt(r):
        return [r["symbol"], _pct(r.get("iv")), _pct(r.get("rv")),
                _vp(r.get("vrp")), r.get("label", "-")]

    return "\n".join([ui.rule(W, "IMPLIED vs REALIZED - RICH = sell-vol candidate"),
                      ui.table(_VRP_COLS, [_fmt(r) for r in rich]), "",
                      ui.rule(W, "CHEAP = buy-vol candidate"),
                      ui.table(_VRP_COLS, [_fmt(r) for r in cheap])])


def render_report(movers: List[dict], vrp_rows: List[dict], n_cov: int) -> str:
    return "\n".join([
        render_iv_movers(movers), "",
        render_vrp(vrp_rows), "",
        f"  coverage: {n_cov} symbols (chain_archive intersect equity_ohlcv).",
        "  VRP uses TRAILING realized vol (a live proxy); forward VRP is the",
        "  Track-4 backtest. Sparse archive - treat as a watchlist, not a signal.",
    ])
