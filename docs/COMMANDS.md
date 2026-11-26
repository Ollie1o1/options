# Master Command Directory (`opt` CLI, Bloomberg TUI, Web Cockpit & Legacy Tools)

Every command in this repository lives on this single page. You can also print a compact terminal cheatsheet anytime with:

```bash
opt commands        # Complete categorized CLI + TUI + `:` mnemonic cheatsheet
opt --help          # Top-level Click command list
opt alpha --help    # Phase 3 Alpha & Alt-Data subcommands
opt lab --help      # Ops health, real-fill harness & disk cleanup
```

> **Tip:** If you haven't installed the `opt` entrypoint into your venv yet, run `pip install -e . --no-deps` once (or invoke `python -m src.cli.main ...`).

---

## 1. Daily Routine (2 Minutes)

```bash
opt refresh         # Update local daily prices (Yahoo) + FRED credit/rates + Radar cache
opt                 # Launch 6-tab Bloomberg × Amber CRT Textual TUI
opt web             # Launch zero-reload Bloomberg Web Cockpit at http://127.0.0.1:8000/desk
opt today           # 10-second Morning Action Brief in plain terminal output
opt guide           # Zero-memorization step-by-step interactive coach
```

---

## 2. Wealthsimple USD TFSA Core Loop (`VM_OUTLOOK_TOP3` + `SGOV`)

CRA prohibits margin and multi-leg credit/debit spreads inside a TFSA. The desk enforces this automatically whenever your account type is `tfsa` (default).

| Command | What it does |
|---|---|
| `opt profile` | Show your saved TFSA balance, cash ETF (`SGOV`), risk budget, and Core ETF shares (`data/tfsa_profile.json`) |
| `opt profile --tfsa 10000 --cash-etf SGOV` | Set your Wealthsimple USD TFSA balance and defensive T-bill buffer ETF |
| `opt profile --risk 500 --broker wealthsimple_usd` | Set default per-trade risk budget and broker |
| `opt profile --holdings XLE:10,XLV:4,XLP:5` | Manually set current Core ETF share counts |
| `opt profile --import-ws ~/Downloads/activity.csv` | Idempotently import a Wealthsimple Activity/Statement CSV (updates Core ETFs + logs stock picks, never double-counts) |
| `opt rebalance` | Compute exact Wealthsimple fractional BUY/SELL orders for `VM_OUTLOOK_TOP3` + `SGOV` reserve & estimated annual T-bill yield |
| `opt rebalance --apply` | Persist resulting target shares & today's rebalance date into `data/tfsa_profile.json` |
| `opt rebalance --account 15000 --whole` | Preview whole-share orders for a custom dollar amount |

---

## 3. Top-Down Discovery, Ticker Checks & Trade Planning

| Command | What it does |
|---|---|
| `opt discover` | Top-down research hub: ① Market → ② 16 Sectors (PROVEN Outlook rank) → ③ Labelled Idea Lenses |
| `opt discover --sector XLE` | Drill into a sector ETF's constituents sorted by 12-1m trend |
| `opt discover --lens opt_skew` | Run a specific lens (`core_rotation`, `pullbacks`, `etf_spreads`, `leaders`, `revisions`, `insiders`, `opt_skew`, `edgar_events`, `holdings`, `landmines`) |
| `opt discover --ledger` | Print the `NULL_LEDGER` of ideas tested here that **failed** (so you never re-trade dead edges) |
| `opt discover --forward` | Inspect preregistered forward-tracking record of discovery lenses |
| `opt card NVDA` | ④ Evidence Card — verdict, structure, peers, options odds, portfolio fit & source provenance |
| `opt radar` | Unified Stock + Option Opportunity Radar sorted by Tier → Confluence → Friction |
| `opt radar --tier lead --clean` | Filter to `VALIDATED` + `LEAD` setups with zero risk flags |
| `opt radar --live-options` | Fetch live 60–120d 5%-wide ETF Bull Put spreads (Test O1 limit-only; `[TAXABLE_ONLY]`) |
| `opt check AAPL` | 1-screen Ticker Dossier with Braille 60d price tape, S/R levels, landmines & Vol/RND |
| `opt express AAPL --dir bull --target +8% --days 30` | Compare Shares, LEAPS, Covered Call, and (taxable-only) Spreads with Walk-Away Limit Ladders |
| `opt express AAPL --dir bull --target +8% --owns-shares` | Include Covered Call comparison when you already hold 100+ shares |

---

## 4. Trade Book & Position Lifecycle

| Command | What it does |
|---|---|
| `opt book` | View open/closed stock & ETF picks (`own_picks.db`), live P&L, exit alerts (`EXIT >SMA5`, `STOP HIT`, `TARGET HIT`), paper options & long-term plan |
| `opt log AAPL --price 210 --shares 5 --playbook equity_pullback` | Fast 1-line logger for a manual stock/ETF buy |
| `opt log AAPL --price 210 --shares 5 --stop 198 --target 228` | Log a buy with explicit stop and target levels |
| `opt close 1 --price 224.50 --note "Hit >SMA5"` | Close an open stock/ETF pick by Pick ID (`1`) or Ticker (`AAPL`) and record realized P&L |

---

## 5. Phase 3 Alpha & Alt-Data Labs (`opt alpha`)

All Phase 3 lenses are free public feeds labelled `WATCHLIST` or `CONTEXT` (never `PROVEN` until they pass preregistered OOS validation) and built to harvest edge in **$0-commission TFSA shares** rather than wide-spread options.

| Command | What it does |
|---|---|
| `opt alpha skew` | Cremers-Weinbaum ATM Call-Put IV spread (`BULL_FLOW_SHARES`) & Xing-Zhang-Zhao Put Smirk (`CRASH_SMIRK_AVOID`) from `data/chain_archive.db` |
| `opt alpha gex SPY` | Dealer Net Gamma Exposure ($M per 1% move), Gamma Flip strike, and Call/Put OI Walls |
| `opt alpha edgar` | View cached SEC EDGAR forced-flow events (Form 10 spinoffs, 5-day 13D activists, XBRL Net Buyback Yield + GP/A) |
| `opt alpha edgar --refresh` | Fetch fresh filings from free SEC EDGAR API into `data/edgar_events.db` |
| `opt alpha edgar --type SPINOFF_FORM10` | Filter by `SPINOFF_FORM10`, `ACTIVIST_13D`, or `NET_BUYBACK_QUALITY` |
| `opt alpha macro` | View cached 4-point CBOE VIX curve, FRED Fed Net Liquidity, CFTC COT positioning & Polymarket macro odds |
| `opt alpha macro --refresh` | Refresh all 4 free macro plumbing feeds into `data/macro_plumbing.json` |
| `opt alpha premortem NVDA` | Deterministic Pre-Mortem Devil's Advocate: bear case, local landmines, objective kill-switches & TFSA rules |
| `opt alpha premortem NVDA --ai` | Also append an opt-in OpenRouter LLM Devil's Advocate risk audit |
| `opt alpha archive-chains --limit 60` | Snapshot the expanded 60-ticker liquid ETF + large-cap universe from free CBOE chains into `data/chain_archive.db` |

---

## 6. Discord Rich Embeds & Zero-Clutter macOS Notifications (`opt notify`)

| Command | What it does |
|---|---|
| `opt notify --morning` | Push one daily Bloomberg-style Morning Dispatch embed to Discord (with TFSA Core weights, TFSA-legal satellites & Data Pipeline LEDs) |
| `opt notify --morning --dry-run` | Preview the Discord embed JSON locally without sending |
| `opt notify --check` | Signal-only alert check: pushes ONLY on Open Pick Exits, Clean TFSA Setups, Monthly Rebalance Due, or `STALE` Data Pipes (deduped daily; coalesced into one single-slot macOS banner) |
| `opt notify --pill` | Print ultra-compact status pill (`◆ XLE/XLV/XLP VIX 27`) for menu bar / shell prompts |
| `opt notify --test-macos` | Send a single test macOS notification banner |
| `opt notify --test-discord` | Send a test embed to your `DISCORD_WEBHOOK_URL` |
| `scripts/menubar_pill.py` | Optional zero-clutter macOS menu-bar widget (symlink into SwiftBar as `options_desk.15m.py`) |

> **Zero-clutter macOS tip:** `brew install terminal-notifier` is auto-detected and forces all alerts into a single replaceable notification slot (`-group options-desk`) so banners never pile up. Without it, `opt notify` falls back cleanly to built-in `osascript`.

---

## 7. Mission Lab, Execution Harness & Disk Cleanup (`opt lab`)

| Command | What it does |
|---|---|
| `opt lab health` | Check disk space, per-pipe freshness LEDs (`ohlcv`, `chains`, `earnings`, `estimates`, `fred`, `kalshi`), and upcoming preregistered readout dates |
| `opt lab playbooks` | List every playbook in `configs/playbooks.json` (including `CLOSED` ones) with evidence tier & allowed accounts |
| `opt lab fills` | Summarize empirical broker limit-order fill rates & mean concession `k` vs the `0.35` Test O1 ceiling |
| `opt lab log-fill --broker ibkr --ticker SPY --structure "Bull Put" --side SELL --bid 3.90 --ask 4.10 --limit 4.00 --filled` | Log one real broker limit-order attempt into `data/fills.db` |
| `opt lab cleanup` | Dry-run report of reclaimable disk space from `CLOSED` research studies (deletes nothing) |
| `opt lab cleanup --confirm` | Actually reclaim disk space from `CLOSED` research studies |

*Global flags on any `opt` command:* `opt --json <subcommand>` (machine-readable JSON), `opt --no-color <subcommand>` (plain text for scripts/pipes).

---

## 8. Interactive TUI (`opt`) Keyboard Shortcuts & Bloomberg `:` Mnemonics

### Navigation & Global Keys

| Key | Action |
|---|---|
| `1` – `7` | Switch tabs: `[1] TODAY`, `[2] RADAR`, `[3] WORKBENCH`, `[4] BOOK`, `[5] LAB`, `[6] DISCOVER`, `[7] OPTIONS` |
| `:` | Open **Bloomberg Command Bar** (see mnemonics below) |
| `Ctrl+P` | Open Textual Command Palette (fuzzy search actions, tickers, chain, scanner & themes) |
| `c` | Jump straight to the **Interactive T-Shaped Options Chain Matrix** for the focused ticker |
| `p` | Paper-log an option or spread directly into `paper_trades.db` (`PaperOptionModal`) |
| `h` | Toggle **2D Price × IV Shock Risk Surface Heatmap** inside Workbench |
| `,` / `.` | Scrub scenario time `-5d` / `+5d` (`T+Nd` theta decay preview in Options & Leg Composer) |
| `{` / `}` | Shock scenario IV `-5%` / `+5%` live across Workbench, Scanner & Leg Composer |
| `b` | Open **TFSA Core Rebalance Modal** (`VM_OUTLOOK_TOP3` + `SGOV` reserve + one-click Apply) |
| `t` | Cycle Desk Theme (`amber_crt` → `quant_desk` → `matrix_terminal` → `cyberpunk_neon`) — persisted to `settings.json` |
| `r` | Refresh Radar scanners & local caches |
| `u` | Run audited Yahoo daily price refresh into `data/equity_ohlcv.db` |
| `o` | Scan live 60–120d Wide ETF Option Spreads (market hours, `[TAXABLE_ONLY]`) |
| `l` | Quick-log a stock/ETF pick (`QuickLogModal` with stop & target fields) |
| `x` | Close an open stock/ETF pick (`ClosePickModal` — or press `Enter` on a row in `[4] BOOK`) |
| `f` | Log a real broker limit-order fill attempt (`LogFillModal`) |
| `?` | Open Plain-English Guide & Keyboard Cheatsheet (`HelpModal`) |
| `q` | Quit |

### Tab-Specific Power Keys

| Tab | Key | Action |
|---|---|---|
| `[2] RADAR` | `/` / `s` / `j` `k` / `Enter` | Focus micro-query filter (`stock`, `option`, `clean`, `actionable`, `conf>=2`, `fric<10%`) · Cycle sort · Move · Open Workbench |
| `[3] WORKBENCH` | `[` / `]` | Nudge target move `-2%` / `+2%` live (updates Braille RND curve & vehicle EVs) |
| `[3] WORKBENCH` | `-` / `+` (or `=`) | Nudge horizon `-10d` / `+10d` live |
| `[3] WORKBENCH` | `a` | Toggle Account Type (`TFSA` ↔ `Taxable`) |
| `[6] DISCOVER` | `g` | Toggle 4-Quadrant **Sector Relative Rotation Graph (RRG)** |
| `[6] DISCOVER` | `Enter` | On Sector Board → drill into constituents; on Idea/Constituent row → full-screen ④ Evidence Card (`w` jumps to Workbench) |

### Bloomberg `:` Command Bar Mnemonics

Press `:` anywhere in `opt` (or `opt web`) and enter:

| Mnemonic | Action |
|---|---|
| `NVDA` | Open ④ Evidence Card for `NVDA` |
| `CHAIN NVDA` (or `NVDA CHAIN`) | Open interactive **T-Shaped Options Chain Matrix + IV Smile + Dealer GEX** for `NVDA` |
| `SCAN` / `SCAN bull_put` | Open **`[7] OPTIONS` Multi-Strategy Scanner** (optionally filtered to a strategy) |
| `FLOW` / `VOL` | Open **Options Flow (UOA/OCC)** or **Vol-Intel (VRP)** Hub inside `[7] OPTIONS` |
| `SURF NVDA` | Open `[3] WORKBENCH` on `NVDA` with **2D Risk Surface Heatmap** enabled |
| `NVDA WB` | Open `[3] WORKBENCH` directly on `NVDA` |
| `REB` | Open TFSA Core Rebalance Modal |
| `SKEW` | Open Option-Implied Signals → Shares table (`CW IV Spread` & `Put Smirk`) |
| `GEX SPY` | Open Dealer Gamma Exposure (Net GEX, Gamma Flip, Call/Put OI Walls) for `SPY` |
| `EDGAR` / `13D` / `SPIN` / `BUYBACK` | Open SEC EDGAR Forced-Flow Events viewer |
| `MACRO` | Open Macro Plumbing viewer (4-pt VIX Curve, Fed Net Liquidity, COT, Polymarket) |
| `PM NVDA` | Open Pre-Mortem Devil's Advocate for `NVDA` |
| `CLOSE` | Open Close Pick modal |
| `THEME amber_crt` | Switch theme (`amber_crt`, `quant_desk`, `matrix_terminal`, `cyberpunk_neon`) |

---

## 9. Bloomberg Web Cockpit (`opt web`)

Run `opt web` (or `opt web --port 8080 --no-open`) to open `http://127.0.0.1:8000/desk`. Works out of the box with zero extra packages (uses `uvicorn`/`fastapi` when installed, or built-in stdlib `ThreadingHTTPServer`).

- **`1` – `6`**: Switch tabs (`TODAY & TFSA`, `SECTORS & RRG`, `ALPHA LABS`, `GHOST RACE & BOOK`, `MISSION LAB`, `EVIDENCE CARD`)
- **`:` or `/`**: Focus top command bar (`NVDA`, `SKEW`, `MACRO`, `REB`, ...)
- **`t`**: Cycle `Amber CRT` → `Quant Desk` → `Paper`
- **`r`**: Refresh local data

---

## 10. Legacy Screener (`run.py`) & Standalone Utilities

All legacy commands remain 100% compatible:

| Command | What it does |
|---|---|
| `python run.py` | Interactive Quant Desk Launcher (`[G] GUIDE`, `[D] DESK`, `[W] WEB`, `[1-8]` legacy tools) |
| `python run.py -ds` / `-sps` / `-ics` / `-ss` | Unattended auto-log scans (discovery / spreads / iron condors / short premium) |
| `python run.py --enforce-exits` | Close paper options trades past TP / stop / time-exit |
| `python -m src.doctor` | Read-only environment, dependency & DB health check |
| `python -m src.check_pnl` | Paper options portfolio viewer with Greeks |
| `python -m src.backtester --calibrate` | Preview paper-trade IC calibration (read-only) |
| `streamlit run src/dashboard.py` | Legacy multi-tab Streamlit cockpit |
| `bash scripts/test.sh [pattern]` | Run local stdlib `unittest` suite (e.g. `bash scripts/test.sh phases_1_to_3`) |

---

## 11. `.env` Environment Variables (All Optional)

`opt`, `run.py`, `dashboard.py`, `api.py`, and `notifications.py` auto-load `.env` from the repo root:

```bash
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...   # `opt notify` rich embeds
SEC_EDGAR_CONTACT=you@example.com                          # Polite SEC User-Agent for EDGAR/insiders
POLYGON_API_KEY=...                                        # Optional news/VWAP enrichment
ALPHAVANTAGE_API_KEY=...                                   # Optional historical option backfill
OPENROUTER_API_KEY=...                                     # Optional AI ranking & `opt alpha premortem --ai`
```
