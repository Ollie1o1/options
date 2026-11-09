"""Push trade alerts to Discord and/or Telegram for new candidates that match
the evidence-backed alert rule.

Reads `data/candidates.db` (every scan records its board there), picks new
gate-passed rows since the last run, filters them by `config.json` →
`trade_alerts`, de-duplicates, and sends a short message per pick.

Credentials go in `.env` (git-ignored), never in config.json:
    DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
    TELEGRAM_BOT_TOKEN=123456:ABC...
    TELEGRAM_CHAT_ID=123456789

    python -m src.trade_alerts                  # send new alerts (scheduled use)
    python -m src.trade_alerts --dry-run        # print what would be sent
    python -m src.trade_alerts --test           # send a test message
    python -m src.trade_alerts --telegram-chat-id   # find your chat id
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "candidates.db"
DEFAULT_STATE = ROOT / "data" / "trade_alerts_state.json"

DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    # Where the evidence points (docs/RESEARCH_FINDINGS.md, 2026-11-09):
    # gate-passed Bear Calls on index/sector ETFs. Bull Put is off: the live
    # rule measured as costly beta (2026-10-15).
    "strategies": ["Bear Call"],
    "symbols": ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLU", "XLV",
                "XLI", "XLP", "XLY", "XLB", "XLRE", "XLC", "EEM", "EFA", "GLD",
                "SLV", "TLT", "HYG", "SMH", "XBI", "KRE", "GDX", "USO", "ARKK"],
    "min_dte": 7,
    "max_dte": 45,
    "max_per_run": 3,
    "dedupe_days": 5,
    "max_age_minutes": 120,
    "evidence_note": ("Paper evidence only: gate-passed Bear Calls on ETFs, "
                      "Aug-Oct 2026, PF 2.2, but not yet shown to beat their "
                      "own market direction. Forward verdict due 2027-01-20. "
                      "Size small."),
}


# ── config / env ────────────────────────────────────────────────────────────

def load_settings(config_path: Path = ROOT / "config.json") -> Dict[str, Any]:
    try:
        cfg = json.loads(Path(config_path).read_text())
    except (OSError, ValueError):
        cfg = {}
    return {**DEFAULTS, **(cfg.get("trade_alerts") or {})}


def load_env(env_path: Path = ROOT / ".env") -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        pass


def channels() -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    hook = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if hook:
        out["discord"] = {"url": hook}
    tok = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if tok and chat:
        out["telegram"] = {"token": tok, "chat_id": chat}
    return out


# ── senders ─────────────────────────────────────────────────────────────────

def _post(url: str, data: bytes, ctype: str, timeout: float = 15.0) -> None:
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": ctype, "User-Agent": "options-screener-alerts/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if r.status >= 300:
            raise RuntimeError(f"HTTP {r.status}")


def send_discord(cfg: Dict[str, str], text: str) -> None:
    _post(cfg["url"], json.dumps({"content": text[:1990]}).encode(), "application/json")


def send_telegram(cfg: Dict[str, str], text: str) -> None:
    url = f"https://api.telegram.org/bot{cfg['token']}/sendMessage"
    body = urllib.parse.urlencode({"chat_id": cfg["chat_id"], "text": text[:4000],
                                   "disable_web_page_preview": "true"}).encode()
    _post(url, body, "application/x-www-form-urlencoded")


SENDERS = {"discord": send_discord, "telegram": send_telegram}


def broadcast(chans: Dict[str, Dict[str, str]], text: str) -> List[str]:
    """Send to every channel; return the names that succeeded."""
    ok = []
    for name, cfg in chans.items():
        try:
            SENDERS[name](cfg, text)
            ok.append(name)
        except (urllib.error.URLError, RuntimeError, OSError, ValueError) as e:
            print(f"[trade-alerts] {name} send failed: {type(e).__name__}: {e}")
    return ok


# ── state ───────────────────────────────────────────────────────────────────

def load_state(path: Path) -> Dict[str, Any]:
    try:
        s = json.loads(Path(path).read_text())
        return {"cursor": s.get("cursor"), "sent": dict(s.get("sent") or {})}
    except (OSError, ValueError):
        return {"cursor": None, "sent": {}}


def save_state(path: Path, state: Dict[str, Any]) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
    os.replace(tmp, path)


# ── selection ───────────────────────────────────────────────────────────────

def _utc(ts: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def fetch_new(db: Path, since: str, s: Dict[str, Any]) -> List[Dict[str, Any]]:
    strategies, symbols = list(s["strategies"]), list(s["symbols"])
    if not strategies or not symbols:
        return []
    q = f"""SELECT ts, board, contract_key, symbol, strategy_name, expiration,
                   premium, ev_net, round_trip_pct, features_json
            FROM candidates
            WHERE ts > ? AND gate_passed = 1
              AND strategy_name IN ({','.join('?' * len(strategies))})
              AND symbol IN ({','.join('?' * len(symbols))})
            ORDER BY ts"""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in con.execute(q, [since, *strategies, *symbols])]
    finally:
        con.close()
    for r in rows:
        try:
            r["f"] = json.loads(r.pop("features_json") or "{}")
        except ValueError:
            r["f"] = {}
    return rows


def select(rows: List[Dict[str, Any]], state: Dict[str, Any], s: Dict[str, Any],
           now: dt.datetime) -> List[Dict[str, Any]]:
    """Latest row per contract; drop stale, out-of-window, recently-sent; one
    per (symbol, strategy); best EV after costs first; cap per run."""
    latest: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        latest[r["contract_key"]] = r          # rows are ts-ordered
    cutoff = now - dt.timedelta(days=int(s["dedupe_days"]))
    picks = []
    for r in latest.values():
        if now - _utc(r["ts"]) > dt.timedelta(minutes=int(s["max_age_minutes"])):
            continue
        dte = (dt.date.fromisoformat(r["expiration"][:10]) - now.date()).days
        if not (int(s["min_dte"]) <= dte <= int(s["max_dte"])):
            continue
        sent = state["sent"].get(r["contract_key"])
        if sent and _utc(sent) > cutoff:
            continue
        r["dte"] = dte
        picks.append(r)
    picks.sort(key=lambda r: -(r["ev_net"] if r["ev_net"] is not None else -1e9))
    seen, out = set(), []
    for r in picks:
        k = (r["symbol"], r["strategy_name"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
        if len(out) >= int(s["max_per_run"]):
            break
    return out


# ── formatting ──────────────────────────────────────────────────────────────

def _m(x: Optional[float]) -> str:
    return "?" if x is None else f"{x:.2f}"


def format_alert(r: Dict[str, Any], s: Dict[str, Any]) -> str:
    f = r["f"]
    strat = r["strategy_name"]
    ks, kl = f.get("short_strike"), f.get("long_strike")
    sb, sa, lb, la = (f.get(k) for k in ("short_bid", "short_ask", "long_bid", "long_ask"))
    width = f.get("spread_width")
    right = "C" if strat == "Bear Call" else "P"
    mid = None
    if None not in (sb, sa, lb, la):
        mid = (sb + sa) / 2 - (lb + la) / 2
    nat = (sb - la) if None not in (sb, la) else None
    lines = [f"Trade lead: {r['symbol']} {strat} {ks}/{kl}{right}  exp {r['expiration'][:10]} "
             f"({r['dte']} DTE)",
             f"  SELL {ks}{right} (bid {_m(sb)} / ask {_m(sa)})",
             f"  BUY  {kl}{right} (bid {_m(lb)} / ask {_m(la)})"]
    if mid is not None and width:
        max_loss = (width - mid) * 100
        be = (ks + mid) if strat == "Bear Call" else (ks - mid)
        lines.append(f"  Credit: mid {mid:.2f} (natural {_m(nat)}). Limit order at mid; "
                     f"don't take less than ~{(mid + (nat or mid)) / 2:.2f}.")
        lines.append(f"  Per contract: max profit ${mid * 100:.0f}, max loss ${max_loss:.0f}, "
                     f"breakeven {be:.2f}")
    if r.get("ev_net") is not None:
        lines.append(f"  Model EV after costs: ${r['ev_net']:.2f}/contract")
    lines.append("  Plan (as paper-tested): take profit at 50% of credit; stop at a loss "
                 "equal to the credit; otherwise close after 3 days once <= 21 DTE.")
    lines.append(f"  {s['evidence_note']}")
    lines.append(f"  scan {r['ts'][:16]}Z")
    return "\n".join(lines)


# ── main ────────────────────────────────────────────────────────────────────

def run(db: Path = DEFAULT_DB, state_path: Path = DEFAULT_STATE, *, dry_run: bool = False,
        now: Optional[dt.datetime] = None, settings: Optional[Dict[str, Any]] = None,
        chans: Optional[Dict[str, Dict[str, str]]] = None) -> int:
    s = settings or load_settings()
    if not s.get("enabled", True):
        print("[trade-alerts] disabled in config (trade_alerts.enabled=false)")
        return 0
    now = now or dt.datetime.now(dt.timezone.utc)
    chans = channels() if chans is None else chans
    if not chans and not dry_run:
        print("[trade-alerts] no channel configured (set DISCORD_WEBHOOK_URL or "
              "TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env); nothing sent")
        return 0
    state = load_state(state_path)
    since = state["cursor"] or (now - dt.timedelta(minutes=int(s["max_age_minutes"]))).isoformat()
    rows = fetch_new(db, since, s)
    picks = select(rows, state, s, now)
    print(f"[trade-alerts] {len(rows)} matching rows since {since[:19]}, {len(picks)} to send")
    sent_any = 0
    for r in picks:
        text = format_alert(r, s)
        if dry_run:
            print(text + "\n")
            continue
        if broadcast(chans, text):
            state["sent"][r["contract_key"]] = now.isoformat()
            sent_any += 1
    if not dry_run:
        if rows:
            state["cursor"] = max(r["ts"] for r in rows)
        keep = now - dt.timedelta(days=30)
        state["sent"] = {k: v for k, v in state["sent"].items() if _utc(v) > keep}
        save_state(state_path, state)
    return sent_any


def telegram_chat_ids() -> None:
    tok = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not tok:
        print("Set TELEGRAM_BOT_TOKEN in .env first.")
        return
    with urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/getUpdates", timeout=15) as r:
        data = json.load(r)
    ids = {(u.get("message") or {}).get("chat", {}).get("id") for u in data.get("result", [])}
    ids.discard(None)
    if not ids:
        print("No messages yet: send your bot any message in Telegram, then re-run.")
    for i in ids:
        print(f"TELEGRAM_CHAT_ID={i}")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--telegram-chat-id", action="store_true")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--state", default=str(DEFAULT_STATE))
    a = ap.parse_args()
    load_env()
    if a.telegram_chat_id:
        telegram_chat_ids()
        return
    if a.test:
        chans = channels()
        if not chans:
            print("No channel configured in .env.")
            return
        ok = broadcast(chans, "Options screener: test alert. Alerts are working.")
        print(f"[trade-alerts] test sent to: {', '.join(ok) or 'none'}")
        return
    run(Path(a.db), Path(a.state), dry_run=a.dry_run)


if __name__ == "__main__":
    main()
