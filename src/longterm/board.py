"""Terminal surfaces for the holdings desk: startup banner + board.

House style: everything through src/ui.py + fmt.style semantic names —
never raw Colors. Plain mode must stay readable."""
from typing import Dict, List, Optional, Tuple

import src.formatting as fmt
from src import ui
from .discover import CandidateRead, DeepRead, insight_line
from .fills import DEFAULT_DB
from .plan import Plan, PlanName, Tranche, tranche_size_usd
from .wizard import (build_add_command, build_edit_command, build_fill_command,
                     open_tranche_levels, parse_levels)
from .zones import FILLED, IN_ZONE, NEAR, WATCHING, ZoneRead

_STATE_STYLE = {IN_ZONE: "good", NEAR: "warn", WATCHING: "muted", FILLED: "label"}
_LEVEL_TOL = 1e-6


def suggested_size(plan: Plan, name: PlanName, tranche: Tranche,
                   remaining_usd: float) -> float:
    return min(tranche_size_usd(plan, name, tranche), max(0.0, remaining_usd))


def _name_for(plan: Plan, ticker: str) -> Optional[PlanName]:
    for n in plan.names:
        if n.ticker == ticker:
            return n
    return None


def _tranche_at(name: PlanName, level: float) -> Optional[Tranche]:
    for t in name.tranches:
        if abs(t.level - level) < _LEVEL_TOL:
            return t
    return None


def banner(reads: List[ZoneRead], plan: Plan, remaining_usd: float,
           earnings: Optional[Dict[str, str]] = None, width: int = 100) -> str:
    triggered = [r for r in reads if r.state in (IN_ZONE, NEAR)]
    if not triggered:
        return ""
    earnings = earnings or {}
    lines = [ui.rule(width, "LONG-TERM BUY ZONES")]
    for r in triggered:
        name = _name_for(plan, r.ticker)
        tranche = _tranche_at(name, r.next_level) if name and r.next_level is not None else None
        size = suggested_size(plan, name, tranche, remaining_usd) if tranche else 0.0
        segs = [
            fmt.style(r.ticker, "emph"),
            fmt.style(f"{r.spot:,.2f}", "value"),
            fmt.style(r.state, _STATE_STYLE[r.state]),
            fmt.style(f"tranche @ {r.next_level:g} ({r.distance_pct:+.1f}%)", "value"),
            fmt.style(f"drawdown {r.drawdown_pct:+.1f}% ATH", "label"),
            fmt.style(f"tranche ${size:,.0f}", "accent"),
        ]
        if r.ticker in earnings:
            segs.append(fmt.style(f"{fmt.GLYPHS['warn']} earnings {earnings[r.ticker]}", "warn"))
        lines.append("  " + f"  {fmt.style(fmt.GLYPHS['dot'], 'muted')}  ".join(segs))
    lines.append("  " + fmt.style(
        f"cash pool: ${remaining_usd:,.0f} remaining of ${plan.cash_pool_usd:,.0f}", "muted"))
    return "\n".join(lines)


def render_board(plan: Plan, reads: List[ZoneRead], book: Dict[str, dict],
                 remaining_usd: float, earnings: Optional[Dict[str, str]] = None,
                 width: int = 100) -> str:
    earnings = earnings or {}
    lines = [ui.rule(width, "HOLDINGS — LONG-TERM ACCUMULATION")]
    if not plan.names:
        lines.append("  " + fmt.style(
            "plan is empty — ADD MU 750/650/550 to start a ladder", "label"))
        return "\n".join(lines)
    by_ticker = {r.ticker: r for r in reads}
    for name in plan.names:
        r = by_ticker.get(name.ticker)
        head = [fmt.style(name.ticker, "heading")]
        if r:
            head.append(fmt.style(f"{r.spot:,.2f}", "emph"))
            head.append(fmt.style(r.state, _STATE_STYLE[r.state]))
            head.append(fmt.style(f"{r.drawdown_pct:+.1f}% ATH", "label"))
            if r.sigma_dist is not None:
                head.append(fmt.style(f"{r.sigma_dist:+.1f}σ to zone", "label"))
            if r.above_ma200 is not None:
                head.append(fmt.style(
                    "above 200dma" if r.above_ma200 else "below 200dma", "muted"))
        else:
            head.append(fmt.style("no data", "bad"))
        if name.ticker in earnings:
            head.append(fmt.style(
                f"{fmt.GLYPHS['warn']} earnings {earnings[name.ticker]}", "warn"))
        lines.append("")
        lines.append("  " + f"  {fmt.style(fmt.GLYPHS['dot'], 'muted')}  ".join(head))
        if name.thesis:
            lines.append("    " + fmt.style(name.thesis, "muted"))
        open_levels = set(open_tranche_levels(name, r))
        filled_here = {t.level for t in name.tranches} - open_levels
        for i, t in enumerate(name.tranches):
            joint = "└─" if i == len(name.tranches) - 1 else "├─"
            size = suggested_size(plan, name, t, remaining_usd)
            if t.level in filled_here:
                mark = fmt.style(f"{fmt.GLYPHS['check']} filled", "good")
            elif r and r.next_level is not None and abs(t.level - r.next_level) < _LEVEL_TOL:
                mark = fmt.style(f"{r.state}  ({r.distance_pct:+.1f}%)",
                                 _STATE_STYLE[r.state])
            else:
                mark = fmt.style("open", "muted")
            lines.append(f"    {joint} " + fmt.style(f"{t.level:g}", "value")
                         + "  " + fmt.style(f"${size:,.0f}", "label") + "  " + mark)
        held = book.get(name.ticker)
        if held and held["shares"]:
            pnl = (r.spot - held["avg_price"]) * held["shares"] if r else None
            seg = (f"held {held['shares']:,.2f} sh @ avg {held['avg_price']:,.2f}"
                   + (f"  P&L {pnl:+,.0f}" if pnl is not None else ""))
            lines.append("    " + fmt.style(seg, "good" if (pnl or 0) >= 0 else "bad"))
    lines.append("")
    lines.append("  " + fmt.style(
        f"cash pool ${plan.cash_pool_usd:,.0f}  {fmt.GLYPHS['dot']}  "
        f"remaining ${remaining_usd:,.0f}", "label"))
    return "\n".join(lines)


def render_discover_board(results: List[Tuple[CandidateRead, Optional[DeepRead]]],
                          sector_keyword: str, width: int = 100) -> str:
    """Discovery scan results: a numbered table for every candidate, plus
    the narrative insight line for whichever entries carry a DeepRead
    (the top-ranked handful — see discover.scan's deep_limit).

    `results` is discover.scan()'s return shape, ranked most-beaten-down
    first (most negative drawdown_pct sorts first) — a ranking by historical
    drawdown, never a prediction or a "top picks" claim.

    Args:
      results: (CandidateRead, Optional[DeepRead]) pairs in scan() order.
      sector_keyword: the sector this scan covered, shown in the header.
      width: rule width, matching the other board renderers in this file.

    Returns:
      The full board as one string (rule header, one row per candidate,
      then a "deeper read" section for entries with a DeepRead). An empty
      `results` renders just the header plus a "no candidates" message.
    """
    lines = [ui.rule(width, f"DISCOVER — {sector_keyword.upper()}")]
    if not results:
        lines.append("  " + fmt.style(
            "no candidates found — check the sector keyword or try again", "label"))
        return "\n".join(lines)

    for i, (candidate, _deep) in enumerate(results, start=1):
        support_label = candidate.supports[0]["label"] if candidate.supports else "—"
        momentum_txt = (f"{candidate.momentum_12_1 * 100:+.0f}%"
                        if candidate.momentum_12_1 is not None else "n/a")
        ma_txt = (f"{candidate.ma200_distance_pct:+.0f}%"
                  if candidate.ma200_distance_pct is not None else "n/a")
        segs = [
            fmt.style(f"{i:>2}", "muted"),
            fmt.style(candidate.ticker, "emph"),
            fmt.style(f"{candidate.spot:,.2f}", "value"),
            fmt.style(f"{candidate.drawdown_pct:+.0f}% ATH", "bad"),
            fmt.style(f"{ma_txt} vs 200dma", "label"),
            fmt.style(f"mom {momentum_txt}", "muted"),
            fmt.style(f"near {support_label}", "label"),
        ]
        lines.append("  " + f"  {fmt.style(fmt.GLYPHS['dot'], 'muted')}  ".join(segs))

    deep_entries = [(i, c, d) for i, (c, d) in enumerate(results, start=1) if d]
    if deep_entries:
        lines.append("")
        lines.append("  " + fmt.style("deeper read:", "heading"))
        for i, candidate, deep in deep_entries:
            lines.append(f"    {i}. " + fmt.style(insight_line(candidate, deep), "value"))
    return "\n".join(lines)


def resolve_add_target(arg_line: str,
                       last_results: Optional[List[Tuple[CandidateRead, Optional[DeepRead]]]]
                       ) -> str:
    """Translate `ADD <n>` (a 1-based index into the last discovery scan)
    into the canonical `ADD <TICKER> <level>/<level>/...` command, built
    from that candidate's suggested_ladder. Pure function: no I/O, never
    mutates `last_results` or anything else.

    Anything else — a ticker-first ADD, an out-of-range index, a
    non-numeric second token, no prior scan, or a non-ADD command — passes
    through unchanged, so today's `ADD MU 750/650/550` grammar keeps
    working exactly as it does without this function in the loop.

    Args:
      arg_line: the raw command line as typed (case-insensitive verb).
      last_results: the most recent discover.scan() output, or None if no
        discovery scan has run yet this session.

    Returns:
      The canonical ADD command string if `arg_line` was a valid `ADD <n>`
      referring to an in-range candidate; otherwise `arg_line` unchanged.
    """
    parts = arg_line.split()
    if len(parts) != 2 or parts[0].upper() != "ADD" or not last_results:
        return arg_line
    try:
        index = int(parts[1])
    except ValueError:
        return arg_line
    if not (1 <= index <= len(last_results)):
        return arg_line
    candidate, _deep = last_results[index - 1]
    ladder = "/".join(f"{t.level:g}" for t in candidate.suggested_ladder)
    return f"ADD {candidate.ticker} {ladder}"


# ── Commands + interactive menu ──────────────────────────────────────────────
import datetime as _dt
import sys as _sys

from .fills import book as _book
from .fills import deployed_usd as _deployed
from .fills import filled_levels as _filled_levels
from .fills import record_fill as _record_fill
from .plan import DEFAULT_PATH as _PLAN_PATH
from .plan import save_plan as _save_plan

_GRAMMAR = ("commands: ADD MU 750/650/550  |  FILL MU 750 2.5 748.20  |  "
            "EDIT MU 800/700/600  |  REMOVE MU  |  CASH 6000  |  "
            "R report  |  B back")


def _parse_ladder(spec: str):
    levels = [float(x) for x in spec.split("/") if x.strip()]
    if not levels:
        raise ValueError("empty ladder")
    w = 1.0 / len(levels)
    return [Tranche(lvl, w) for lvl in levels]


def handle_command(line: str, plan: Plan, plan_path: str = _PLAN_PATH,
                   db_path: str = DEFAULT_DB):
    parts = line.split()
    verb = parts[0].upper() if parts else ""
    try:
        if verb == "ADD" and len(parts) == 3:
            ticker = parts[1].upper()
            if _name_for(plan, ticker):
                return plan, fmt.style(f"{ticker} already on the plan — use EDIT", "warn")
            plan.names.append(PlanName(ticker, _parse_ladder(parts[2])))
            _save_plan(plan, plan_path)
            return plan, fmt.style(f"{ticker} added with {len(plan.names[-1].tranches)}"
                                   f"-tranche ladder", "good")
        if verb == "FILL" and len(parts) == 5:
            ticker, level, shares, price = (parts[1].upper(), float(parts[2]),
                                            float(parts[3]), float(parts[4]))
            name = _name_for(plan, ticker)
            if not name or not _tranche_at(name, level):
                return plan, fmt.style(f"no tranche @ {level:g} on {ticker}", "bad")
            if any(abs(level - f) < _LEVEL_TOL
                   for f in _filled_levels(ticker, db_path=db_path)):
                return plan, fmt.style(f"{ticker} @ {level:g} is already filled", "warn")
            _record_fill(ticker, level, shares, price, db_path=db_path)
            return plan, fmt.style(
                f"filled {ticker} tranche @ {level:g}: {shares:g} sh @ {price:,.2f}", "good")
        if verb == "EDIT" and len(parts) == 3:
            name = _name_for(plan, parts[1].upper())
            if not name:
                return plan, fmt.style(f"{parts[1].upper()} not on the plan", "bad")
            name.tranches = _parse_ladder(parts[2])
            _save_plan(plan, plan_path)
            return plan, fmt.style(f"{name.ticker} ladder replaced", "good")
        if verb == "REMOVE" and len(parts) == 2:
            ticker = parts[1].upper()
            held = _book(db_path=db_path).get(ticker)
            if held and held["shares"] > 0:
                return plan, fmt.style(
                    f"{ticker} has {held['shares']:g} shares held — can't remove from "
                    f"the plan while holding a position; the ladder just won't alert "
                    f"further once fully filled", "bad")
            plan.names = [n for n in plan.names if n.ticker != ticker]
            _save_plan(plan, plan_path)
            return plan, fmt.style(f"{ticker} removed (fill history kept)", "good")
        if verb == "CASH" and len(parts) == 2:
            plan.cash_pool_usd = float(parts[1])
            _save_plan(plan, plan_path)
            return plan, fmt.style(f"cash pool set to ${plan.cash_pool_usd:,.0f}", "good")
    except ValueError as exc:
        return plan, fmt.style(f"bad input ({exc}) — {_GRAMMAR}", "bad")
    return plan, fmt.style(_GRAMMAR, "label")


def _earnings_flags(tickers):
    out = {}
    for t in tickers:
        try:
            import json as _json

            from src.earnings_provider import next_earnings_date, resolve_api_key
            try:
                with open("config.json") as f:
                    cfg = _json.load(f)
            except Exception:
                cfg = None
            when = next_earnings_date(t, api_key=resolve_api_key(cfg))
            if when and 0 <= (when.date() - _dt.date.today()).days <= 7:
                out[t] = when.strftime("%m-%d")
        except Exception:
            pass
    return out


def _gather(plan: Plan, db_path: str = DEFAULT_DB):
    """Snapshots → reads/book/remaining. One batched fetch."""
    from .data import fetch_snapshots
    from .zones import assess
    snaps = fetch_snapshots([n.ticker for n in plan.names])
    reads = [assess(n, snaps[n.ticker], _filled_levels(n.ticker, db_path=db_path))
             for n in plan.names if n.ticker in snaps]
    remaining = max(0.0, plan.cash_pool_usd - _deployed(db_path=db_path))
    return snaps, reads, _book(db_path=db_path), remaining


def _gather_cached(plan: Plan, snaps, db_path: str = DEFAULT_DB):
    """Re-derive reads/book/remaining from already-fetched snapshots
    (commands change fills/plan, not prices)."""
    from .zones import assess
    reads = [assess(n, snaps[n.ticker], _filled_levels(n.ticker, db_path=db_path))
             for n in plan.names if n.ticker in snaps]
    remaining = max(0.0, plan.cash_pool_usd - _deployed(db_path=db_path))
    return reads, _book(db_path=db_path), remaining


_ACTIONS = [
    ("1", "Add a stock", "track a new ticker with a buy ladder"),
    ("2", "Record a buy", "log a fill against an open tranche"),
    ("3", "Edit buy levels", "replace a ticker's ladder"),
    ("4", "Remove a stock", "drop a ticker from the plan"),
    ("5", "Set cash budget", "change the pool tranche sizes are drawn from"),
    ("6", "Find candidates", "scan a sector for new buy-the-dip ideas"),
    ("7", "Write & open report", "render the HTML holdings report"),
]


def render_actions_menu(width: int = 100) -> str:
    lines = [ui.rule(width, "ACTIONS")]
    for num, name, desc in _ACTIONS:
        n = fmt.style(f"[{num}]", "accent")
        t = fmt.style(ui.pad(name, 20), "emph", bold=True)
        d = fmt.style(f"— {desc}", "muted")
        lines.append(f"  {n} {t} {d}")
    lines.append(f"  {fmt.style('[B]', 'muted')} {fmt.style('Back', 'muted')}")
    lines.append(ui.rule(width))
    return "\n".join(lines)


def _ask(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"  {prompt}{suffix}: ").strip()
    return default if (not val and default is not None) else val


def _ask_float(prompt: str, default: Optional[str] = None) -> float:
    while True:
        raw = _ask(prompt, default)
        try:
            return float(raw)
        except ValueError:
            print(ui.error_line(f"'{raw}' isn't a number — try again"))


def _ask_levels(prompt: str) -> List[float]:
    while True:
        raw = _ask(prompt)
        try:
            return parse_levels(raw)
        except ValueError as exc:
            print(ui.error_line(str(exc)))


def _choose(items: List[str], prompt: str) -> int:
    for i, item in enumerate(items, start=1):
        print(f"    {fmt.style(f'[{i}]', 'accent')} {item}")
    while True:
        raw = _ask(prompt)
        try:
            idx = int(raw)
        except ValueError:
            print(ui.error_line(f"'{raw}' isn't a number — try again"))
            continue
        if 1 <= idx <= len(items):
            return idx - 1
        print(ui.error_line(f"pick a number between 1 and {len(items)}"))


def _guided_add(plan: Plan, last_discovery,
                plan_path: str = _PLAN_PATH, db_path: str = DEFAULT_DB):
    if last_discovery:
        target = _ask("ticker to add, or a number from your last scan")
    else:
        target = _ask("ticker to add")
    resolved = resolve_add_target(f"ADD {target.upper()}", last_discovery)
    if resolved == f"ADD {target.upper()}":
        levels = _ask_levels("buy levels, e.g. 750, 650, 550")
        resolved = build_add_command(target, levels)
    plan, msg = handle_command(resolved, plan, plan_path=plan_path, db_path=db_path)
    print("  " + msg)
    return plan, last_discovery


def _guided_fill(plan: Plan, reads: List[ZoneRead],
                 plan_path: str = _PLAN_PATH, db_path: str = DEFAULT_DB) -> Plan:
    if not plan.names:
        print(ui.error_line("nothing on your plan yet — add a stock first"))
        return plan
    by_ticker = {r.ticker: r for r in reads}
    tickers_with_opens = []
    for name in plan.names:
        r = by_ticker.get(name.ticker)
        levels = open_tranche_levels(name, r)
        if levels:
            tickers_with_opens.append(name.ticker)
    if not tickers_with_opens:
        print(ui.error_line("nothing to fill — no open tranches on any ticker"))
        return plan
    idx = _choose(tickers_with_opens, "which ticker did you buy?")
    ticker = tickers_with_opens[idx]
    name = _name_for(plan, ticker)
    r = by_ticker.get(ticker)
    levels = open_tranche_levels(name, r)
    lidx = _choose([f"{lvl:g}" for lvl in levels],
                   f"which tranche did you fill on {ticker}?")
    level = levels[lidx]
    shares = _ask_float("how many shares")
    default_price = f"{r.spot:g}" if r else None
    price = _ask_float("price paid", default=default_price)
    plan, msg = handle_command(build_fill_command(ticker, level, shares, price),
                               plan, plan_path=plan_path, db_path=db_path)
    print("  " + msg)
    return plan


def _guided_edit(plan: Plan, plan_path: str = _PLAN_PATH, db_path: str = DEFAULT_DB) -> Plan:
    if not plan.names:
        print(ui.error_line("nothing on your plan yet — add a stock first"))
        return plan
    tickers = [n.ticker for n in plan.names]
    idx = _choose(tickers, "which ticker?")
    name = plan.names[idx]
    current = "/".join(f"{t.level:g}" for t in name.tranches)
    print(f"  {name.ticker}'s current ladder: {current}")
    levels = _ask_levels("new buy levels, e.g. 800, 700, 600")
    plan, msg = handle_command(build_edit_command(name.ticker, levels),
                               plan, plan_path=plan_path, db_path=db_path)
    print("  " + msg)
    return plan


def menu(width: int = 100) -> None:
    from .plan import load_plan
    plan = load_plan()
    interactive = _sys.stdin.isatty()
    with ui.spinner("pricing holdings…"):
        snaps, reads, held, remaining = _gather(plan)
        flags = _earnings_flags([r.ticker for r in reads
                                 if r.state in (IN_ZONE, NEAR)])
    print(render_board(plan, reads, held, remaining, earnings=flags, width=width))
    if not interactive:
        return
    last_discovery: Optional[List[Tuple[CandidateRead, Optional[DeepRead]]]] = None
    while True:
        print(render_actions_menu(width))
        try:
            raw = input("\n  holdings> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        up = raw.upper()
        if up in ("B", "BACK", "Q", "QUIT", "X", ""):
            return
        if up == "1":
            try:
                plan, last_discovery = _guided_add(plan, last_discovery)
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            reads, held, remaining = _gather_cached(plan, snaps)
            print(render_board(plan, reads, held, remaining, earnings=flags, width=width))
            continue
        if up == "2":
            try:
                plan = _guided_fill(plan, reads)
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            reads, held, remaining = _gather_cached(plan, snaps)
            print(render_board(plan, reads, held, remaining, earnings=flags, width=width))
            continue
        if up == "3":
            try:
                plan = _guided_edit(plan)
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            reads, held, remaining = _gather_cached(plan, snaps)
            print(render_board(plan, reads, held, remaining, earnings=flags, width=width))
            continue
        if up in ("R", "REPORT"):
            from .report import write_report
            with ui.spinner("rendering report…"):
                html_path, _ = write_report()
            print("  " + fmt.style(f"report written: {html_path}", "good"))
            continue
        # Token-based verb match (mirrors handle_command's own parts[0].upper()
        # idiom) rather than raw.upper().startswith("DISCOVER") — a prefix
        # check would also fire on a mistyped/concatenated word like
        # "DISCOVERSEMI" (verb "DISCOVERSEMI" != "DISCOVER", so it correctly
        # falls through to the grammar-help message below instead).
        parts = raw.split(None, 1)
        verb = parts[0].upper() if parts else ""
        if verb in ("D", "DISCOVER"):
            arg = parts[1] if len(parts) > 1 else ""
            if not arg:
                print("  " + fmt.style(
                    "usage: D <SECTOR>  e.g. D SEMICONDUCTORS", "label"))
                continue
            from .discover import scan
            try:
                with ui.spinner(f"scanning {arg.upper()}…"):
                    last_discovery = scan(arg)
                print(render_discover_board(last_discovery, arg, width=width))
            except ValueError as exc:
                print("  " + fmt.style(str(exc), "bad"))
            continue
        resolved = resolve_add_target(raw, last_discovery)
        plan, msg = handle_command(resolved, plan)
        print("  " + msg)
        reads, held, remaining = _gather_cached(plan, snaps)
        print(render_board(plan, reads, held, remaining, earnings=flags, width=width))
