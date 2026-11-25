"""Zero-reload Bloomberg Terminal × Amber CRT local Web Cockpit (`src/web_terminal.py`).

Served via `opt web` (automatically uses FastAPI/uvicorn if installed, or built-in stdlib
`ThreadingHTTPServer` with ZERO extra dependencies) and also mounted on `src/api.py` at `/desk`.
100% self-contained HTML + CSS + Vanilla JS + crisp inline SVG charts (zero external CDN
dependencies, works fully offline, <50ms tab switches, keyboard shortcuts `1-6`, `:`, `t`).
"""
from __future__ import annotations

import json
import math
import urllib.parse
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional


def _clean_json(v: Any) -> Any:
    """Recursively convert numpy scalars, NaN/inf, and Timestamps to JSON-safe primitives."""
    try:
        import numpy as np

        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            fv = float(v)
            return None if (math.isnan(fv) or math.isinf(fv)) else fv
        if isinstance(v, np.ndarray):
            return [_clean_json(x) for x in v.tolist()]
    except ImportError:
        pass
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, dict):
        return {str(k): _clean_json(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean_json(x) for x in v]
    if hasattr(v, "isoformat") and callable(v.isoformat):
        return v.isoformat()
    return v


def build_desk_overview(
    *,
    refresh: bool = False,
    provider: Optional[Any] = None,
) -> Dict[str, Any]:
    """Assemble full local cockpit JSON payload for `/desk` (100% offline-safe)."""
    from src.alpha.edgar_events import load_cached_edgar_events
    from src.alpha.formatters import format_macro_text
    from src.alpha.macro_plumbing import load_macro_plumbing
    from src.alpha.options_for_shares import scan_option_signals_for_shares
    from src.cli.guide import plain_english_action_plan
    from src.equity_playbooks.rebalance import compute_tfsa_rebalance
    from src.profile import estimate_cash_yield_annual, load_profile, rebalance_due_status
    from src.tui.state import DeskDataProvider

    prov = provider or DeskDataProvider()
    snap = prov.get_radar(refresh=refresh)
    prof = load_profile()
    _y_ann, y_lbl = estimate_cash_yield_annual()

    sector_board: List[Dict[str, Any]] = []
    spot_lookup: Dict[str, float] = {}
    try:
        hub = prov.get_hub(refresh=refresh)
        sector_board = hub.board.to_dict().get("rows", [])
        spot_lookup = {
            sym: float(cs[-1])
            for sym, (_ds, cs, _vs) in (hub.ohlcv or {}).items()
            if len(cs) > 0
        }
    except Exception:
        pass

    skew_signals: List[Dict[str, Any]] = []
    try:
        skew_signals = [
            s.to_dict()
            for s in scan_option_signals_for_shares(chain_db=prov.chain_db)[:20]
        ]
    except Exception:
        pass

    macro_text = ""
    try:
        macro_text = format_macro_text(load_macro_plumbing(), rich=False)
    except Exception:
        pass

    edgar_events: List[Dict[str, Any]] = []
    try:
        edgar_events = [e.to_dict() for e in load_cached_edgar_events(limit=25)]
    except Exception:
        pass

    ghost_race: Dict[str, Any] = {}
    try:
        race = prov.get_ghost_race()
        if race is not None:
            ghost_race = race.to_dict()
    except Exception:
        pass

    book: Dict[str, Any] = {}
    try:
        book = prov.get_book()
    except Exception:
        pass

    rebalance: Dict[str, Any] = {}
    try:
        plan = compute_tfsa_rebalance(
            snap.core_allocation,
            account_usd=prof.tfsa_usd,
            current_shares=prof.core_holdings,
            spot_lookup=spot_lookup,
            fractional=prof.fractional_shares,
            drift_threshold_pct=prof.drift_threshold_pct,
        )
        due, due_msg = rebalance_due_status(
            prof,
            max_drift_pct=plan.max_drift_pct,
            rebalance_recommended=plan.rebalance_recommended,
        )
        rebalance = {
            **plan.to_dict(),
            "due": due,
            "due_message": due_msg,
            "cash_etf": prof.cash_etf,
        }
    except Exception:
        pass

    health_info: Dict[str, Any] = {}
    try:
        health_info = prov.get_health()
    except Exception:
        pass

    playbooks: List[Dict[str, Any]] = []
    try:
        playbooks = [asdict(p) for p in prov.get_playbooks()]
    except Exception:
        pass

    return _clean_json(
        {
            "snapshot": snap.to_dict(),
            "action_plan": plain_english_action_plan(snap),
            "profile": prof.to_dict(),
            "cash_yield_label": y_lbl,
            "sector_board": sector_board,
            "alpha": {
                "skew_signals": skew_signals,
                "macro_text": macro_text,
                "edgar_events": edgar_events,
            },
            "ghost_race": ghost_race,
            "book": book,
            "rebalance": rebalance,
            "health": health_info,
            "playbooks": playbooks,
        }
    )


def build_desk_card(
    ticker: str,
    *,
    provider: Optional[Any] = None,
) -> Dict[str, Any]:
    """Assemble local Evidence Card + deterministic Pre-Mortem Devil's Advocate for one ticker."""
    from src.alpha.macro_plumbing import load_macro_plumbing
    from src.alpha.options_for_shares import scan_option_signals_for_shares
    from src.alpha.premortem import build_premortem
    from src.profile import load_profile
    from src.tui.state import DeskDataProvider

    tk = ticker.upper().strip()
    prov = provider or DeskDataProvider()
    card = prov.get_card(tk)
    sk_rows = scan_option_signals_for_shares(chain_db=prov.chain_db, symbols=(tk,))
    mac = load_macro_plumbing()
    prof = load_profile()
    pm = build_premortem(
        tk,
        card.dossier,
        skew_signal=sk_rows[0] if sk_rows else None,
        macro=mac,
        account_type=prof.account_type,
    )
    return _clean_json(
        {
            "ticker": tk,
            "verdict": card.verdict,
            "verdict_style": card.verdict_style,
            "reasons": list(card.reasons),
            "warnings": list(card.warnings),
            "dossier": card.dossier.to_dict(),
            "premortem": pm.to_dict(),
        }
    )


def serve_web_terminal(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Zero-dependency stdlib ThreadingHTTPServer serving `/desk` and its JSON endpoints."""

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            qs = urllib.parse.parse_qs(parsed.query)
            if path in ("/", "/desk"):
                self._send(200, TERMINAL_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/health":
                self._send(200, b'{"status":"ok"}', "application/json")
                return
            if path == "/desk/api/overview":
                ref = qs.get("refresh", ["0"])[0] in ("1", "true", "yes")
                try:
                    data = build_desk_overview(refresh=ref)
                    self._send(200, json.dumps(data).encode("utf-8"), "application/json")
                except Exception as exc:
                    self._send(500, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")
                return
            if path.startswith("/desk/api/card/"):
                tk = urllib.parse.unquote(path.split("/desk/api/card/", 1)[1]).strip()
                try:
                    data = build_desk_card(tk)
                    self._send(200, json.dumps(data).encode("utf-8"), "application/json")
                except Exception as exc:
                    self._send(500, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")
                return
            self._send(404, b'{"error":"not found"}', "application/json")

    srv = ThreadingHTTPServer((host, int(port)), _Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


TERMINAL_HTML = """<!doctype html>
<html lang="en" data-theme="amber_crt">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>OPTIONS DESK // BLOOMBERG CRT TERMINAL</title>
<style>
:root, [data-theme="amber_crt"] {
  --bg: #110d07; --surface: #1b150c; --panel: #241b0f; --border: #4a371b;
  --ink: #f2d7a7; --ink-strong: #ffeed2; --muted: #a88d65;
  --primary: #ffb03b; --accent: #e69628;
  --good: #5ec98d; --warn: #d6a452; --bad: #e06c75; --info: #82aad2;
}
[data-theme="quant_desk"] {
  --bg: #0c1016; --surface: #131922; --panel: #1a222e; --border: #2a3545;
  --ink: #cbd5e1; --ink-strong: #f8fafc; --muted: #7c8b9e;
  --primary: #82aad2; --accent: #58a6ff;
  --good: #5ec98d; --warn: #d6a452; --bad: #e06c75; --info: #82aad2;
}
[data-theme="paper"] {
  --bg: #faf9f6; --surface: #f3f1ec; --panel: #ebe7de; --border: #d5cfc3;
  --ink: #23282e; --ink-strong: #111518; --muted: #787266;
  --primary: #9a6a15; --accent: #8a5a0e;
  --good: #1f6f43; --warn: #8a6516; --bad: #9d2c33; --info: #2d5f8b;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  font-size: 13px; line-height: 1.45;
}
.topbar {
  position: sticky; top: 0; z-index: 20;
  display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
  background: var(--panel); border-bottom: 1px solid var(--border);
  padding: 8px 16px;
}
.brand { color: var(--primary); font-weight: 800; letter-spacing: 0.08em; }
.pill {
  padding: 2px 8px; border-radius: 3px; font-weight: 700; font-size: 11px;
  border: 1px solid var(--border); background: var(--surface);
}
.pill.good { color: var(--good); border-color: var(--good); }
.pill.bad  { color: var(--bad);  border-color: var(--bad); }
.pill.warn { color: var(--warn); border-color: var(--warn); }
.pill.info { color: var(--info); border-color: var(--info); }
.cmdbar {
  margin-left: auto; display: flex; gap: 6px; align-items: center;
}
.cmdbar input {
  background: var(--bg); color: var(--ink-strong); border: 1px solid var(--primary);
  padding: 4px 9px; font-family: inherit; font-size: 12px; width: 220px; border-radius: 3px;
}
.btn {
  background: var(--surface); color: var(--primary); border: 1px solid var(--border);
  padding: 4px 10px; font-family: inherit; font-size: 11px; cursor: pointer; border-radius: 3px;
}
.btn:hover, .btn.active { background: var(--primary); color: var(--bg); font-weight: 700; }
.nav {
  display: flex; gap: 6px; padding: 8px 16px; background: var(--surface);
  border-bottom: 1px solid var(--border); flex-wrap: wrap;
}
.wrap { padding: 14px 16px; max-width: 1480px; margin: 0 auto; }
.grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 12px; }
.c12 { grid-column: span 12; } .c8 { grid-column: span 8; } .c7 { grid-column: span 7; }
.c6  { grid-column: span 6; }  .c5 { grid-column: span 5; } .c4 { grid-column: span 4; }
@media (max-width: 960px) { .c8,.c7,.c6,.c5,.c4 { grid-column: span 12; } }
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 4px; padding: 12px 14px;
}
.card h3 {
  margin: 0 0 8px 0; font-size: 12px; letter-spacing: 0.08em;
  color: var(--primary); text-transform: uppercase;
  border-bottom: 1px solid var(--border); padding-bottom: 5px;
  display: flex; justify-content: space-between; align-items: center;
}
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { padding: 5px 7px; border-bottom: 1px solid var(--border); text-align: left; font-variant-numeric: tabular-nums; }
th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; }
tr.clickable { cursor: pointer; }
tr.clickable:hover { background: var(--panel); }
.num { text-align: right; }
.good { color: var(--good); } .bad { color: var(--bad); } .warn { color: var(--warn); } .muted { color: var(--muted); }
.led-row { display: flex; flex-wrap: wrap; gap: 8px; }
.led { padding: 3px 8px; border-radius: 3px; border: 1px solid var(--border); background: var(--panel); font-size: 11px; }
.led .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 5px; }
.dot.OK { background: var(--good); } .dot.WARN { background: var(--warn); } .dot.STALE { background: var(--bad); }
.tab { display: none; } .tab.active { display: block; }
svg { width: 100%; height: auto; display: block; }
.bar-wrap { height: 7px; background: var(--panel); border-radius: 3px; overflow: hidden; margin-top: 3px; }
.bar-fill { height: 100%; background: var(--primary); }
.mono-pre { white-space: pre-wrap; font-size: 12px; color: var(--ink); margin: 0; }
</style>
</head>
<body>
<div class="topbar">
  <span class="brand">◆ OPTIONS DESK</span>
  <span id="tb-spy" class="pill">SPY —</span>
  <span id="tb-vix" class="pill">VIX —</span>
  <span id="tb-regime" class="pill">—</span>
  <span id="tb-macro" class="muted">Next: —</span>
  <div class="cmdbar">
    <input id="cmd-in" placeholder="[:] NVDA · SKEW · REB · MACRO"/>
    <button class="btn" onclick="runCmd()">GO</button>
    <button class="btn" onclick="cycleTheme()" title="Shortcut: t">THEME</button>
    <button class="btn" onclick="loadAll(true)">REFRESH</button>
  </div>
</div>

<div class="nav">
  <button class="btn active" data-tab="today" onclick="showTab('today')">[1] TODAY &amp; TFSA</button>
  <button class="btn" data-tab="sectors" onclick="showTab('sectors')">[2] SECTORS &amp; RRG</button>
  <button class="btn" data-tab="alpha" onclick="showTab('alpha')">[3] ALPHA LABS</button>
  <button class="btn" data-tab="book" onclick="showTab('book')">[4] GHOST RACE &amp; BOOK</button>
  <button class="btn" data-tab="lab" onclick="showTab('lab')">[5] MISSION LAB</button>
  <button class="btn" data-tab="card" onclick="showTab('card')">[6] EVIDENCE CARD</button>
</div>

<div class="wrap">
  <!-- [1] TODAY -->
  <div id="tab-today" class="tab active">
    <div class="grid">
      <div class="card c7">
        <h3>Morning Desk Dispatch <span id="t-asof" class="muted"></span></h3>
        <div id="t-plan">Loading…</div>
      </div>
      <div class="card c5">
        <h3>Core TFSA Sleeve (PROVEN · Wealthsimple USD) <span id="t-sgov" class="good"></span></h3>
        <div id="t-core"></div>
      </div>
      <div class="card c12">
        <h3>Unified Opportunity Radar <span class="muted">Click any ticker for Evidence Card</span></h3>
        <table id="t-radar"><thead><tr><th>Tier</th><th>Ticker</th><th>Strategy</th><th>Dir</th><th class="num">Match</th><th class="num">Cost</th><th>Flags</th><th>Setup</th></tr></thead><tbody></tbody></table>
      </div>
    </div>
  </div>

  <!-- [2] SECTORS RRG -->
  <div id="tab-sectors" class="tab">
    <div class="grid">
      <div class="card c6">
        <h3>4-Quadrant Sector Rotation Graph (RRG) <span class="muted">RS 3m vs Acceleration</span></h3>
        <div id="s-rrg"></div>
      </div>
      <div class="card c6">
        <h3>16-ETF Outlook Ranking (PROVEN Order)</h3>
        <table id="s-board"><thead><tr><th>#</th><th>ETF</th><th>Sector</th><th>Call</th><th class="num">Core%</th><th class="num">RS 1m</th><th class="num">RS 3m</th><th class="num">&gt;200d</th></tr></thead><tbody></tbody></table>
      </div>
    </div>
  </div>

  <!-- [3] ALPHA LABS (PHASE 3) -->
  <div id="tab-alpha" class="tab">
    <div class="grid">
      <div class="card c6">
        <h3>Option-Implied Signals → Trade SHARES <span class="pill warn">WATCHLIST</span></h3>
        <table id="a-skew"><thead><tr><th>Ticker</th><th class="num">Spot</th><th class="num">CW IV Spread</th><th class="num">Put Smirk</th><th class="num">Net GEX ($M)</th><th>Read</th></tr></thead><tbody></tbody></table>
      </div>
      <div class="card c6">
        <h3>Macro Plumbing &amp; Alt-Data <span class="pill info">CONTEXT</span></h3>
        <div id="a-macro">Loading…</div>
      </div>
      <div class="card c12">
        <h3>SEC EDGAR Forced-Flow Events (Form 10 Spinoffs · Schedule 13D · Net Buybacks) <span class="pill warn">WATCHLIST</span></h3>
        <table id="a-edgar"><thead><tr><th>Type</th><th>Ticker</th><th>Filed</th><th>Headline / Metric</th><th>Source</th></tr></thead><tbody></tbody></table>
      </div>
    </div>
  </div>

  <!-- [4] GHOST RACE & BOOK -->
  <div id="tab-book" class="tab">
    <div class="grid">
      <div class="card c12">
        <h3>Ghost Benchmark Race (Normalized Base = 100) <span id="b-race-meta" class="muted"></span></h3>
        <div id="b-race-svg"></div>
      </div>
      <div class="card c7">
        <h3>Open &amp; Closed Stock/ETF Picks (data/own_picks.db)</h3>
        <table id="b-picks"><thead><tr><th>ID</th><th>Date</th><th>Ticker</th><th>Playbook</th><th class="num">Entry</th><th class="num">Now</th><th class="num">P&amp;L%</th><th>Status / Exit</th></tr></thead><tbody></tbody></table>
      </div>
      <div class="card c5">
        <h3>Saved TFSA Profile &amp; Rebalance Orders</h3>
        <div id="b-rebal"></div>
      </div>
    </div>
  </div>

  <!-- [5] MISSION LAB -->
  <div id="tab-lab" class="tab">
    <div class="grid">
      <div class="card c12">
        <h3>Data Pipeline Heartbeat LEDs</h3>
        <div id="l-leds" class="led-row"></div>
      </div>
      <div class="card c6">
        <h3>Preregistered Experiment Clocks</h3>
        <div id="l-clocks"></div>
      </div>
      <div class="card c6">
        <h3>Playbook Evidence Registry (configs/playbooks.json)</h3>
        <table id="l-playbooks"><thead><tr><th>Tier</th><th>Playbook</th><th>Accounts</th><th>Horizon</th></tr></thead><tbody></tbody></table>
      </div>
    </div>
  </div>

  <!-- [6] EVIDENCE CARD -->
  <div id="tab-card" class="tab">
    <div class="card">
      <h3 id="c-title">Evidence Card <span class="muted">Type any ticker in the command bar above</span></h3>
      <div id="c-body" class="muted">Select a ticker from Radar, Sectors, or type `NVDA` in the top command bar.</div>
    </div>
  </div>
</div>

<script>
const THEMES = ['amber_crt', 'quant_desk', 'paper'];
let STATE = null;

function cycleTheme() {
  const cur = document.documentElement.getAttribute('data-theme') || 'amber_crt';
  const nxt = THEMES[(THEMES.indexOf(cur) + 1) % THEMES.length];
  document.documentElement.setAttribute('data-theme', nxt);
  try { localStorage.setItem('desk-bloomberg-theme', nxt); } catch {}
}
(function initTheme() {
  try {
    const saved = localStorage.getItem('desk-bloomberg-theme');
    if (saved && THEMES.includes(saved)) document.documentElement.setAttribute('data-theme', saved);
  } catch {}
})();

function showTab(id) {
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav .btn').forEach(el => el.classList.toggle('active', el.dataset.tab === id));
  const pane = document.getElementById('tab-' + id);
  if (pane) pane.classList.add('active');
}

document.addEventListener('keydown', e => {
  if (document.activeElement && document.activeElement.tagName === 'INPUT') {
    if (e.key === 'Enter') runCmd();
    if (e.key === 'Escape') document.activeElement.blur();
    return;
  }
  const map = {'1':'today','2':'sectors','3':'alpha','4':'book','5':'lab','6':'card'};
  if (map[e.key]) showTab(map[e.key]);
  else if (e.key === ':' || e.key === '/') { e.preventDefault(); document.getElementById('cmd-in').focus(); }
  else if (e.key === 't' || e.key === 'T') cycleTheme();
  else if (e.key === 'r') loadAll(true);
});

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}

async function runCmd() {
  const raw = document.getElementById('cmd-in').value.trim();
  if (!raw) return;
  const parts = raw.split(/\\s+/);
  const cmd = parts[0].toUpperCase();
  if (['TODAY','SECTORS','ALPHA','BOOK','LAB'].includes(cmd)) { showTab(cmd.toLowerCase()); return; }
  if (cmd === 'SKEW' || cmd === 'GEX' || cmd === 'EDGAR' || cmd === 'MACRO') { showTab('alpha'); return; }
  if (cmd === 'REB' || cmd === 'REBAL') { showTab('book'); return; }
  await openCard(cmd);
}

async function openCard(ticker) {
  showTab('card');
  document.getElementById('c-title').innerHTML = `EVIDENCE CARD // ${esc(ticker)}`;
  document.getElementById('c-body').innerHTML = 'Loading local evidence card…';
  try {
    const resp = await fetch('/desk/api/card/' + encodeURIComponent(ticker));
    const d = await resp.json();
    const dos = d.dossier || {};
    const prem = d.premortem || {};
    const risks = (prem.kill_switches || []).map(x => `<li class="warn">${esc(x)}</li>`).join('');
    const flags = (dos.risk_flags || []).map(f => `<span class="pill bad">${esc(f)}</span>`).join(' ');
    document.getElementById('c-body').innerHTML = `
      <div class="grid">
        <div class="card c6">
          <h3>Verdict &amp; Structure</h3>
          <p><span class="pill ${d.verdict_style === 'green' ? 'good' : d.verdict_style === 'red' ? 'bad' : 'warn'}">${esc(d.verdict)}</span>
          <b>${esc(ticker)}</b> $${Number(dos.spot || 0).toFixed(2)} ${flags}</p>
          <ul>${(d.reasons || []).map(r => `<li>${esc(r)}</li>`).join('')}${(d.warnings || []).map(w => `<li class="warn">⚠ ${esc(w)}</li>`).join('')}</ul>
          <p class="muted">Trend vs 200d: ${dos.dist_sma200_pct ?? '—'}% · 12-1m: ${dos.mom_12_1_pct ?? '—'}% · RSI(2): ${dos.rsi_2 ?? '—'} · ATM IV: ${dos.atm_iv ?? '—'}%</p>
        </div>
        <div class="card c6">
          <h3>Pre-Mortem Devil's Advocate</h3>
          <p class="muted">${esc(prem.summary || 'Deterministic pre-trade risk check.')}</p>
          <ul>${risks || '<li class="good">No immediate deterministic red flags.</li>'}</ul>
        </div>
      </div>`;
  } catch (err) {
    document.getElementById('c-body').innerHTML = `<span class="bad">Failed to load card: ${esc(err)}</span>`;
  }
}

function renderRRG(rows) {
  const pts = (rows || []).filter(r => r.rs_63d_pct != null && r.rs_21d_pct != null).map(r => ({
    etf: r.etf, x: Number(r.rs_63d_pct), y: Number(r.rs_21d_pct) - Number(r.rs_63d_pct) / 3.0,
    core: Number(r.weight || 0) > 0, avoid: r.posture === 'AVOID'
  }));
  if (!pts.length) return '<div class="muted">No sector RS data.</div>';
  const W = 540, H = 310, pad = 34;
  const maxX = Math.max(5, ...pts.map(p => Math.abs(p.x))) * 1.15;
  const maxY = Math.max(3, ...pts.map(p => Math.abs(p.y))) * 1.15;
  const sx = x => pad + ((x + maxX) / (2 * maxX)) * (W - 2 * pad);
  const sy = y => pad + ((maxY - y) / (2 * maxY)) * (H - 2 * pad);
  const mx = sx(0), my = sy(0);
  const dots = pts.map(p => {
    const cx = sx(p.x).toFixed(1), cy = sy(p.y).toFixed(1);
    const col = p.core ? 'var(--good)' : (p.avoid ? 'var(--bad)' : 'var(--primary)');
    const r = p.core ? 6 : 4;
    return `<g style="cursor:pointer" onclick="openCard('${esc(p.etf)}')">
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="${col}" stroke="var(--bg)" stroke-width="1.5"/>
      <text x="${Number(cx)+8}" y="${Number(cy)+4}" fill="${col}" font-size="10" font-weight="${p.core?800:500}">${p.core?'★':''}${esc(p.etf)}</text>
    </g>`;
  }).join('');
  return `<svg viewBox="0 0 ${W} ${H}">
    <rect x="${mx}" y="${pad}" width="${W-pad-mx}" height="${my-pad}" fill="var(--good)" opacity="0.06"/>
    <rect x="${pad}" y="${my}" width="${mx-pad}" height="${H-pad-my}" fill="var(--bad)" opacity="0.06"/>
    <line x1="${pad}" y1="${my}" x2="${W-pad}" y2="${my}" stroke="var(--border)" stroke-dasharray="4,3"/>
    <line x1="${mx}" y1="${pad}" x2="${mx}" y2="${H-pad}" stroke="var(--border)" stroke-dasharray="4,3"/>
    <text x="${W-pad-4}" y="${pad+14}" text-anchor="end" fill="var(--good)" font-size="10">LEADING ↗</text>
    <text x="${pad+4}" y="${pad+14}" fill="var(--warn)" font-size="10">↖ IMPROVING</text>
    <text x="${pad+4}" y="${H-pad-6}" fill="var(--bad)" font-size="10">↙ LAGGING</text>
    <text x="${W-pad-4}" y="${H-pad-6}" text-anchor="end" fill="var(--muted)" font-size="10">WEAKENING ↘</text>
    ${dots}
  </svg>`;
}

function renderGhostSVG(race) {
  const spy = race?.spy || [], vm = race?.vm_outlook_sgov || [], real = race?.real_book || [];
  if (spy.length < 2) return '<div class="muted">No OHLCV history available for Ghost Race.</div>';
  const W = 920, H = 210, pad = 34;
  const all = [...spy, ...vm, ...real];
  const lo = Math.min(...all) - 1, hi = Math.max(...all) + 1;
  const sx = i => pad + (i / Math.max(spy.length - 1, 1)) * (W - 2 * pad);
  const sy = v => pad + ((hi - v) / Math.max(hi - lo, 1e-6)) * (H - 2 * pad);
  const poly = arr => arr.map((v, i) => `${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(' ');
  const y100 = sy(100).toFixed(1);
  return `<svg viewBox="0 0 ${W} ${H}">
    <line x1="${pad}" y1="${y100}" x2="${W-pad}" y2="${y100}" stroke="var(--border)" stroke-dasharray="4,4"/>
    <polyline points="${poly(spy)}" fill="none" stroke="var(--muted)" stroke-width="1.6" stroke-dasharray="4,3"/>
    <polyline points="${poly(vm)}" fill="none" stroke="var(--primary)" stroke-width="2.4"/>
    ${real.length >= 2 ? `<polyline points="${poly(real)}" fill="none" stroke="var(--good)" stroke-width="2.2"/>` : ''}
    <text x="${pad}" y="18" fill="var(--primary)" font-size="11" font-weight="700">━ VM_OUTLOOK+SGOV ${race.vm_ret_pct >= 0 ? '+' : ''}${race.vm_ret_pct}% (+${race.vm_cash_only_boost_pct}% T-bill)</text>
    <text x="${pad+310}" y="18" fill="var(--muted)" font-size="11">┄ SPY ${race.spy_ret_pct >= 0 ? '+' : ''}${race.spy_ret_pct}%</text>
    <text x="${pad+460}" y="18" fill="var(--good)" font-size="11">━ Real Book ${race.real_ret_pct == null ? '— (log picks to race)' : (race.real_ret_pct >= 0 ? '+' : '') + race.real_ret_pct + '%'}</text>
  </svg>`;
}

async function loadAll(refresh = false) {
  const url = '/desk/api/overview' + (refresh ? '?refresh=1' : '');
  const resp = await fetch(url);
  STATE = await resp.json();
  const s = STATE.snapshot || {};
  document.getElementById('tb-spy').textContent = `SPY $${Number(s.spy_last||0).toFixed(2)} (${s.spy_chg_1d_pct>=0?'+':''}${Number(s.spy_chg_1d_pct||0).toFixed(1)}%)`;
  document.getElementById('tb-vix').textContent = `VIX ${s.vix!=null?Number(s.vix).toFixed(1):'—'} ${s.vix_term||''}`;
  const reg = document.getElementById('tb-regime');
  reg.textContent = s.regime_posture || '—';
  reg.className = 'pill ' + (s.regime_posture === 'RISK_ON' ? 'good' : s.regime_posture === 'RISK_OFF' ? 'bad' : 'warn');
  document.getElementById('tb-macro').textContent = 'Next: ' + (s.next_macro_event || '—');
  document.getElementById('t-asof').textContent = (s.generated_at || '').slice(0, 16);

  // Today plan & Core
  document.getElementById('t-plan').innerHTML = (STATE.action_plan || []).map(x => `<p style="margin:6px 0">${esc(x)}</p>`).join('');
  document.getElementById('t-sgov').textContent = `${STATE.profile?.cash_etf || 'SGOV'} @ ${STATE.cash_yield_label || ''}`;
  document.getElementById('t-core').innerHTML = (s.core_allocation || []).map(c => {
    const pct = (Number(c.target_weight||0)*100).toFixed(1);
    const tk = c.ticker === 'CASH' ? (STATE.profile?.cash_etf || 'SGOV') + ' (Cash)' : c.ticker;
    return `<div style="margin-bottom:8px"><b>${esc(tk)}</b> <span style="float:right">${pct}%</span><div class="bar-wrap"><div class="bar-fill" style="width:${pct}%"></div></div></div>`;
  }).join('');

  // Radar table
  const rb = document.querySelector('#t-radar tbody');
  rb.innerHTML = (s.opportunities || []).map(o => `<tr class="clickable" onclick="openCard('${esc(o.ticker)}')">
    <td><span class="pill ${o.evidence_tier==='VALIDATED'?'good':o.evidence_tier==='LEAD'?'info':'warn'}">${esc(o.evidence_tier)}</span></td>
    <td><b>${esc(o.ticker)}</b></td><td>${esc(o.playbook_label)}</td>
    <td class="${o.direction==='LONG'?'good':o.direction==='SHORT'?'bad':'muted'}">${esc(o.direction)}</td>
    <td class="num">${o.confluence_count}</td><td class="num">${esc(o.friction_label)}</td>
    <td class="warn">${esc((o.risk_flags||[]).join(', ')||'—')}</td><td>${esc(o.setup_summary)}</td></tr>`).join('');

  // Sectors + RRG
  const rows = STATE.sector_board || [];
  document.getElementById('s-rrg').innerHTML = renderRRG(rows);
  document.querySelector('#s-board tbody').innerHTML = rows.map(r => `<tr class="clickable" onclick="openCard('${esc(r.etf)}')">
    <td>${r.rank}</td><td><b>${esc(r.etf)}</b></td><td>${esc(r.name)}</td>
    <td class="${r.posture==='FAVOR'?'good':r.posture==='AVOID'?'bad':'muted'}">${esc(r.posture)}</td>
    <td class="num">${(Number(r.weight||0)*100).toFixed(0)}%</td>
    <td class="num ${r.rs_21d_pct>=0?'good':'bad'}">${r.rs_21d_pct!=null?Number(r.rs_21d_pct).toFixed(1)+'%':'—'}</td>
    <td class="num ${r.rs_63d_pct>=0?'good':'bad'}">${r.rs_63d_pct!=null?Number(r.rs_63d_pct).toFixed(1)+'%':'—'}</td>
    <td class="num">${r.pct_above_200d!=null?Number(r.pct_above_200d).toFixed(0)+'%':'—'}</td></tr>`).join('');

  // Alpha
  const alpha = STATE.alpha || {};
  document.querySelector('#a-skew tbody').innerHTML = (alpha.skew_signals || []).slice(0, 12).map(x => `<tr class="clickable" onclick="openCard('${esc(x.ticker)}')">
    <td><b>${esc(x.ticker)}</b></td><td class="num">$${Number(x.spot||0).toFixed(2)}</td>
    <td class="num ${x.cw_iv_spread_vp>=0?'good':'bad'}">${x.cw_iv_spread_vp!=null?x.cw_iv_spread_vp.toFixed(2)+'vp':'—'}</td>
    <td class="num">${x.put_smirk_vp!=null?x.put_smirk_vp.toFixed(2)+'vp':'—'}</td>
    <td class="num ${x.net_gex_m>=0?'good':'bad'}">${x.net_gex_m!=null?x.net_gex_m.toFixed(1):'—'}</td>
    <td>${esc(x.signal_label)}</td></tr>`).join('') || '<tr><td colspan="6" class="muted">No archived chains with skew signals yet.</td></tr>';
  document.getElementById('a-macro').innerHTML = `<pre class="mono-pre">${esc(alpha.macro_text || 'Run `opt alpha macro --refresh` to populate macro plumbing.')}</pre>`;
  document.querySelector('#a-edgar tbody').innerHTML = (alpha.edgar_events || []).slice(0, 15).map(e => `<tr class="clickable" onclick="openCard('${esc(e.ticker)}')">
    <td><span class="pill warn">${esc(e.event_type)}</span></td><td><b>${esc(e.ticker)}</b></td>
    <td>${esc(e.filed_date)}</td><td>${esc(e.summary)}</td><td class="muted">${esc(e.source)}</td></tr>`).join('') || '<tr><td colspan="5" class="muted">Run `opt alpha edgar --refresh` to fetch SEC Form 10 / 13D / XBRL buyback events.</td></tr>';

  // Book & Ghost Race
  const race = STATE.ghost_race || {};
  document.getElementById('b-race-meta').textContent = `${(race.dates||[])[0]||''} .. ${(race.dates||[]).slice(-1)[0]||''}`;
  document.getElementById('b-race-svg').innerHTML = renderGhostSVG(race);
  document.querySelector('#b-picks tbody').innerHTML = (STATE.book?.equity_picks || []).map(p => `<tr>
    <td>#${p.pick_id}</td><td>${esc((p.date||'').slice(0,10))}</td><td><b>${esc(p.ticker)}</b></td>
    <td>${esc(p.playbook_id)}</td><td class="num">$${Number(p.entry_price||0).toFixed(2)}</td>
    <td class="num">${p.current_price!=null?'$'+Number(p.current_price).toFixed(2):'—'}</td>
    <td class="num ${(p.unrealized_pct||0)>=0?'good':'bad'}">${p.unrealized_pct!=null?Number(p.unrealized_pct).toFixed(1)+'%':'—'}</td>
    <td class="${p.exit_signal?'warn':'muted'}">${esc(p.exit_signal||p.status)}</td></tr>`).join('') || '<tr><td colspan="8" class="muted">No stock/ETF picks logged yet (`opt log` or TUI `[l]`).</td></tr>';
  const reb = STATE.rebalance || {};
  document.getElementById('b-rebal').innerHTML = `<p><b>${esc(reb.due_message||'')}</b><br><span class="muted">Account $${Number(reb.account_usd||0).toLocaleString()} USD · ${esc(STATE.profile?.cash_etf||'SGOV')} reserve $${Number(reb.estimated_cash_after_usd||0).toLocaleString()}</span></p>` +
    `<table><thead><tr><th>Action</th><th>ETF</th><th class="num">Order Sh</th><th class="num">Delta $</th></tr></thead><tbody>` +
    (reb.orders||[]).map(o => `<tr><td class="${o.action==='BUY'?'good':o.action==='SELL'?'bad':'muted'}">${esc(o.action)}</td><td><b>${esc(o.ticker)}</b></td><td class="num">${o.delta_shares||0}</td><td class="num">$${Number(o.delta_usd||0).toFixed(0)}</td></tr>`).join('') + '</tbody></table>';

  // Lab LEDs & Playbooks
  const pipes = STATE.health?.pipes || {};
  document.getElementById('l-leds').innerHTML = Object.entries(pipes).map(([k, v]) =>
    `<div class="led"><span class="dot ${esc(v.status)}"></span><b>${esc(k.toUpperCase())}</b> <span class="muted">${esc(v.detail)}</span></div>`).join('');
  document.getElementById('l-clocks').innerHTML = (STATE.health?.readouts || []).map(r => `<p style="margin:6px 0">⏱ ${esc(r)}</p>`).join('');
  document.querySelector('#l-playbooks tbody').innerHTML = (STATE.playbooks || []).map(p => `<tr>
    <td><span class="pill ${p.evidence_tier==='VALIDATED'?'good':p.evidence_tier==='LEAD'?'info':'warn'}">${esc(p.evidence_tier)}</span></td>
    <td><b>${esc(p.id)}</b></td><td>${esc((p.accounts||[]).join(', '))}</td><td>${esc(p.horizon_label)}</td></tr>`).join('');
}
loadAll(false);
</script>
</body>
</html>
"""
