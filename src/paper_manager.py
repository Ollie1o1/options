#!/usr/bin/env python3
"""
Paper Trading Manager for Options Screener.
Handles logging forward tests and updating open positions using SQLite.
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from .data_fetching import get_risk_free_rate as _get_rfr
    _HAS_RFR = True
except ImportError:
    _HAS_RFR = False

from .utils import is_short_position as _is_short_position

# Realistic execution cost constants (deprecated fallbacks — use config.json paper_trading section)
COMMISSION_PER_CONTRACT = 0.65   # $ per contract per leg (retail ~$0.65, e.g. Tastytrade/TDA)
SLIPPAGE_PER_SHARE = 0.05        # $ per share (1 typical options tick, ~half spread)
# Round-trip friction per share = entry slippage + exit slippage + 2 commissions
_FRICTION_PER_SHARE = (2 * SLIPPAGE_PER_SHARE) + (2 * COMMISSION_PER_CONTRACT / 100.0)

_SCHEMA_VERSION = 6
_MIGRATIONS = {
    1: [],
    2: ["ALTER TABLE trades ADD COLUMN pnl_usd REAL"],
    3: [
        "ALTER TABLE trades ADD COLUMN pop_score REAL",
        "ALTER TABLE trades ADD COLUMN ev_score REAL",
        "ALTER TABLE trades ADD COLUMN rr_score REAL",
        "ALTER TABLE trades ADD COLUMN liquidity_score REAL",
        "ALTER TABLE trades ADD COLUMN momentum_score REAL",
        "ALTER TABLE trades ADD COLUMN iv_rank_score REAL",
        "ALTER TABLE trades ADD COLUMN theta_score REAL",
    ],
    4: [
        # Track AI score at entry to measure AI IC vs technical IC separately
        "ALTER TABLE trades ADD COLUMN ai_score REAL",
        "ALTER TABLE trades ADD COLUMN ai_confidence REAL",
    ],
    5: [
        # Store entry Greeks and IV for accurate stress testing and P&L attribution
        "ALTER TABLE trades ADD COLUMN entry_iv REAL",
        "ALTER TABLE trades ADD COLUMN entry_delta REAL",
        "ALTER TABLE trades ADD COLUMN entry_gamma REAL",
        "ALTER TABLE trades ADD COLUMN entry_vega REAL",
        "ALTER TABLE trades ADD COLUMN entry_theta REAL",
        "ALTER TABLE trades ADD COLUMN dividend_yield REAL",
    ],
    6: [
        # Expanded component scores for per-component IC validation
        "ALTER TABLE trades ADD COLUMN iv_edge_score REAL",
        "ALTER TABLE trades ADD COLUMN vrp_score REAL",
        "ALTER TABLE trades ADD COLUMN iv_mispricing_score REAL",
        "ALTER TABLE trades ADD COLUMN skew_align_score REAL",
        "ALTER TABLE trades ADD COLUMN vega_risk_score REAL",
        "ALTER TABLE trades ADD COLUMN term_structure_score REAL",
    ],
}


class PaperManager:
    """Manages paper trades stored in a SQLite database."""
    
    def __init__(self, db_path: str = "paper_trades.db", config_path: str = "config.json"):
        self.db_path = db_path
        self.config_path = config_path
        # Load friction costs from config (fall back to module-level constants)
        try:
            with open(config_path, 'r') as f:
                _cfg = json.load(f)
            _pt = _cfg.get("paper_trading", {})
            self._commission_per_contract = float(_pt.get("commission_per_contract", COMMISSION_PER_CONTRACT))
            self._slippage_per_share = float(_pt.get("slippage_per_share", SLIPPAGE_PER_SHARE))
        except Exception:
            self._commission_per_contract = COMMISSION_PER_CONTRACT
            self._slippage_per_share = SLIPPAGE_PER_SHARE
        self._friction_per_share = (2 * self._slippage_per_share) + (2 * self._commission_per_contract / 100.0)
        self._init_db()

    def _get_connection(self):
        """Returns a new sqlite3 connection with WAL mode for concurrent access."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """Creates the trades table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS trades (
            entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            ticker TEXT,
            expiration TEXT,
            strike REAL,
            type TEXT,
            entry_price REAL,
            quality_score REAL,
            strategy_name TEXT,
            status TEXT,
            exit_price REAL,
            exit_date TEXT,
            pnl_pct REAL,
            pnl_usd REAL
        )
        """
        with self._get_connection() as conn:
            conn.execute(query)
        self._migrate_db()

    def _migrate_db(self):
        """Apply incremental schema migrations up to _SCHEMA_VERSION."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA user_version")
            current_version = cur.fetchone()[0]
            for ver in range(current_version + 1, _SCHEMA_VERSION + 1):
                for sql in _MIGRATIONS.get(ver, []):
                    try:
                        cur.execute(sql)
                    except sqlite3.OperationalError:
                        pass  # column may already exist
                cur.execute(f"PRAGMA user_version = {int(ver)}")
            conn.commit()

    def _load_config(self) -> Dict[str, Any]:
        """Loads configuration for exit rules."""
        _default = {
            "exit_rules": {
                "take_profit": 0.50,
                "stop_loss": -0.25
            }
        }
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.debug("Config file not found at %s — using defaults", self.config_path)
            return _default
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON in config %s: %s — using defaults", self.config_path, exc)
            return _default

    def log_trade(self, trade_dict: Dict[str, Any]):
        """
        Logs a new paper trade to the SQLite database.
        Required keys: ticker, expiration, strike, type, entry_price, quality_score, strategy_name
        Optional keys: ai_score, ai_confidence, entry_iv, entry_delta, entry_gamma, entry_vega, entry_theta, dividend_yield,
                       pop_score, ev_score, rr_score, liquidity_score, momentum_score, iv_rank_score, theta_score,
                       iv_edge_score, vrp_score, iv_mispricing_score, skew_align_score, vega_risk_score, term_structure_score
        """
        if not trade_dict.get("strategy_name"):
            raise ValueError("strategy_name is required; must include 'short'/'long' to set P&L direction")
        if float(trade_dict.get("entry_price", 0)) <= 0:
            raise ValueError(f"Cannot log trade: entry_price must be > 0, got {trade_dict.get('entry_price')}")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        query = """
        INSERT INTO trades (
            date, ticker, expiration, strike, type,
            entry_price, quality_score, strategy_name,
            status, exit_price, exit_date, pnl_pct,
            ai_score, ai_confidence,
            entry_iv, entry_delta, entry_gamma, entry_vega, entry_theta, dividend_yield,
            pop_score, ev_score, rr_score, liquidity_score, momentum_score, iv_rank_score, theta_score,
            iv_edge_score, vrp_score, iv_mispricing_score, skew_align_score, vega_risk_score, term_structure_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        def _float_or_none(key):
            v = trade_dict.get(key)
            if v is None:
                return None
            try:
                fv = float(v)
                return fv if np.isfinite(fv) else None
            except (ValueError, TypeError):
                return None

        params = (
            trade_dict.get("date", now),
            trade_dict["ticker"].upper(),
            trade_dict["expiration"],
            float(trade_dict["strike"]),
            trade_dict["type"].lower(),
            float(trade_dict["entry_price"]),
            float(trade_dict["quality_score"]),
            trade_dict["strategy_name"],
            "OPEN",
            None,   # exit_price
            "",     # exit_date
            None,   # pnl_pct
            trade_dict.get("ai_score"),        # None if not ranked yet
            trade_dict.get("ai_confidence"),   # None if not ranked yet
            _float_or_none("entry_iv"),
            _float_or_none("entry_delta"),
            _float_or_none("entry_gamma"),
            _float_or_none("entry_vega"),
            _float_or_none("entry_theta"),
            _float_or_none("dividend_yield"),
            _float_or_none("pop_score"),
            _float_or_none("ev_score"),
            _float_or_none("rr_score"),
            _float_or_none("liquidity_score"),
            _float_or_none("momentum_score"),
            _float_or_none("iv_rank_score"),
            _float_or_none("theta_score"),
            _float_or_none("iv_edge_score"),
            _float_or_none("vrp_score"),
            _float_or_none("iv_mispricing_score"),
            _float_or_none("skew_align_score"),
            _float_or_none("vega_risk_score"),
            _float_or_none("term_structure_score"),
        )

        with self._get_connection() as conn:
            conn.execute(query, params)

        print(f"Logged {trade_dict['type'].upper()} on {trade_dict['ticker']} at ${float(trade_dict['entry_price']):.2f}")

    def log_spread(self, spread_dict: dict) -> None:
        """
        Log a multi-leg spread (credit spread, calendar, etc.) as a single paper trade.
        spread_dict keys: date, ticker, expiration, short_strike, long_strike, type,
                          net_credit, max_profit, max_loss, quality_score
        """
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO trades (date, ticker, expiration, strike, type, entry_price, quality_score, strategy_name, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
            """, (
                spread_dict.get("date", datetime.now().strftime("%Y-%m-%d")),
                spread_dict.get("ticker", ""),
                spread_dict.get("expiration", ""),
                spread_dict.get("short_strike", 0),
                spread_dict.get("type", "Spread"),
                spread_dict.get("net_credit", 0),
                spread_dict.get("quality_score", 0.5),
                f"SPREAD:{spread_dict.get('long_strike', 0)}:{spread_dict.get('max_profit', 0):.2f}:{spread_dict.get('max_loss', 0):.2f}"
            ))
            conn.commit()

    def _get_option_symbol(self, ticker: str, expiration: str, strike: float, option_type: str) -> str:
        """Generates a yfinance-compatible option symbol."""
        try:
            exp_date = pd.to_datetime(expiration).strftime('%y%m%d')
            otype = 'C' if option_type.lower() == 'call' else 'P'
            strike_price = f"{int(strike * 1000):08d}"
            return f"{ticker}{exp_date}{otype}{strike_price}"
        except Exception:
            return ""

    def _get_spread_slippage(self, ticker: str, expiration: str, strike: float, option_type: str, entry_price: float) -> float:
        """Return per-share slippage as 30% of the bid-ask spread width, capped at $0.50 and floored at self._slippage_per_share."""
        try:
            symbol = self._get_option_symbol(ticker, expiration, strike, option_type)
            if not symbol:
                return self._slippage_per_share
            tkr = yf.Ticker(symbol)
            bid = getattr(tkr.fast_info, "bid", None)
            ask = getattr(tkr.fast_info, "ask", None)
            if bid is None or ask is None:
                # fallback: use 10% of entry price as spread estimate
                spread = entry_price * 0.10
            else:
                spread = max(0.0, float(ask) - float(bid))
            slippage = spread * 0.30
            return max(self._slippage_per_share, min(slippage, 0.50))
        except Exception:
            return self._slippage_per_share

    def update_positions(self):
        """Updates all OPEN positions using SQLite and checks exit rules."""
        config = self._load_config()
        exit_rules = config.get("exit_rules", {"take_profit": 0.50, "stop_loss": -0.25})
        tp = exit_rules.get("take_profit", 0.50)
        sl = exit_rules.get("stop_loss", -0.25)
        time_exit_dte = exit_rules.get("time_exit_dte", 21)  # close at ≤21 DTE regardless

        # Fetch open trades
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM trades WHERE status='OPEN'")
            open_trades = cursor.fetchall()

        if not open_trades:
            return

        today = date.today()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        closed_this_run = []

        import warnings

        # Batch-fetch spot prices for all unique underlying tickers
        unique_tickers = list({row["ticker"] for row in open_trades})
        spot_cache: Dict[str, float] = {}

        def _fetch_spot(t: str) -> Tuple[str, Optional[float]]:
            try:
                tkr = yf.Ticker(t)
                s = getattr(tkr.fast_info, "last_price", None)
                if s and float(s) > 0:
                    return t, float(s)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hist = tkr.history(period="5d")
                if not hist.empty:
                    val = float(hist["Close"].iloc[-1])
                    if val > 0:
                        return t, val
            except Exception as exc:
                logger.debug("Spot fetch failed for %s: %s", t, exc)
            return t, None

        max_workers = min(len(unique_tickers), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            try:
                for ticker, spot in ex.map(_fetch_spot, unique_tickers, timeout=30):
                    if spot is not None:
                        spot_cache[ticker] = spot
            except TimeoutError:
                logger.warning("Spot price fetch timed out after 30s — proceeding with partial data")

        # Pre-build symbol list and parallel-fetch option prices for non-spread positions
        _option_fetch_tasks: List[Tuple[int, str, str, str, float, str]] = []
        for row in open_trades:
            strategy_name = row["strategy_name"] or ""
            if strategy_name.startswith("SPREAD:"):
                continue
            if row["ticker"] not in spot_cache:
                continue
            symbol = self._get_option_symbol(row["ticker"], row["expiration"], row["strike"], row["type"])
            if symbol:
                _option_fetch_tasks.append((row["entry_id"], symbol, row["ticker"], row["expiration"], row["strike"], row["type"]))

        option_price_cache: Dict[int, float] = {}

        def _fetch_option_price(task_tuple):
            entry_id, symbol, ticker, expiration, strike, option_type = task_tuple
            try:
                tkr = yf.Ticker(symbol)
                price = None
                try:
                    price = getattr(tkr.fast_info, "last_price", None)
                except Exception:
                    pass
                if price is None or np.isnan(price) or price <= 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        hist = tkr.history(period="1d")
                    if not hist.empty:
                        price = float(hist["Close"].iloc[-1])
                if price is None or np.isnan(price) or price <= 0:
                    try:
                        from .utils import american_price
                        S = spot_cache.get(ticker)
                        if S:
                            T = max((datetime.strptime(expiration[:10], "%Y-%m-%d") - datetime.now()).days / 365, 1/365)
                            _rfr = _get_rfr() if _HAS_RFR else 0.045
                            price = american_price(option_type, float(S), float(strike), T, _rfr, 0.30)
                    except Exception:
                        pass
                if price is not None and not np.isnan(price) and price > 0:
                    return entry_id, float(price)
            except Exception as exc:
                logger.debug("Option price fetch failed for %s: %s", symbol, exc)
            return entry_id, None

        if _option_fetch_tasks:
            _opt_workers = min(len(_option_fetch_tasks), 8)
            with ThreadPoolExecutor(max_workers=_opt_workers) as ex:
                try:
                    for eid, price in ex.map(_fetch_option_price, _option_fetch_tasks, timeout=30):
                        if price is not None:
                            option_price_cache[eid] = price
                except TimeoutError:
                    logger.warning("Option price fetch timed out — proceeding with partial data")

        for row in open_trades:
            entry_id    = row["entry_id"]
            ticker      = row["ticker"]
            expiration  = row["expiration"]
            strike      = row["strike"]
            option_type = row["type"]
            entry_price = row["entry_price"]

            # Time-based exit: close if DTE ≤ time_exit_dte (avoids gamma risk near expiry)
            try:
                exp_date = datetime.strptime(expiration[:10], "%Y-%m-%d").date()
                dte = (exp_date - today).days
            except Exception:
                dte = 999

            # Days held — don't time-exit a trade logged today or yesterday
            try:
                trade_date = datetime.strptime(str(row["date"])[:10], "%Y-%m-%d").date()
                days_held = (today - trade_date).days
            except Exception:
                days_held = 999

            if ticker not in spot_cache:
                continue

            # Spread P&L: detect and handle spread rows before single-leg symbol fetch
            strategy_name = row["strategy_name"] or ""
            if strategy_name.startswith("SPREAD:"):
                try:
                    parts = strategy_name.split(":")
                    long_strike = float(parts[1])
                    max_profit = float(parts[2])
                    max_loss = float(parts[3])
                    spot = spot_cache.get(ticker)
                    if spot is None:
                        continue
                    short_k = float(strike)  # always the short leg (see log_spread)
                    spread_width = abs(long_strike - short_k)
                    if spread_width > 0:
                        if short_k > long_strike:
                            # Bull put spread: loses when spot falls below short put strike
                            intrinsic_frac = max(0.0, min(1.0, (short_k - spot) / spread_width))
                        else:
                            # Bear call spread: loses when spot rises above short call strike
                            intrinsic_frac = max(0.0, min(1.0, (spot - short_k) / spread_width))
                    else:
                        intrinsic_frac = 0.5
                    # For a short credit spread: loses when intrinsic_frac → 1
                    current_value = float(max_loss) * intrinsic_frac   # cost to close (positive = loss)
                    net_credit = float(max_profit)
                    if net_credit > 0:
                        pnl_raw = (net_credit - current_value) / net_credit
                    else:
                        continue
                    hit_tp = pnl_raw >= tp
                    hit_sl = pnl_raw <= sl
                    hit_time = (0 < dte <= time_exit_dte) and days_held >= 3
                    if hit_tp or hit_sl or hit_time:
                        if hit_tp:
                            reason = "Take Profit"
                        elif hit_sl:
                            reason = "Stop Loss"
                        else:
                            reason = f"Time Exit ({dte}d to expiry)"
                        friction_fraction = self._friction_per_share / entry_price if entry_price > 0 else 0.0
                        pnl_realistic = max(pnl_raw - friction_fraction, -1.0)
                        closed_this_run.append(
                            f"{ticker} SPREAD ${strike:.0f}/{long_strike:.0f} → {reason} "
                            f"(mkt: {pnl_raw:+.1%}, after costs: {pnl_realistic:+.1%})"
                        )
                        with self._get_connection() as conn:
                            conn.execute(
                                "UPDATE trades SET status='CLOSED', exit_price=?, exit_date=?, pnl_pct=? WHERE entry_id=?",
                                (current_value, now, pnl_realistic, entry_id),
                            )
                except Exception as exc:
                    logger.debug("Spread P&L calc failed for %s: %s", ticker, exc)
                continue

            # Use pre-fetched option price from parallel batch
            current_price = option_price_cache.get(entry_id)

            if current_price is not None:
                # Raw market P&L — flip sign for short/credit positions
                # (seller profits when option loses value)
                is_short = _is_short_position(row["strategy_name"] or "")
                if is_short:
                    pnl_raw = (entry_price - current_price) / entry_price
                else:
                    pnl_raw = (current_price - entry_price) / entry_price
                hit_tp = pnl_raw >= tp
                hit_sl = pnl_raw <= sl
                # Time exit needs at least 3 days held to avoid closing freshly-logged trades
                hit_time = (0 < dte <= time_exit_dte) and days_held >= 3

                if hit_tp or hit_sl or hit_time:
                    if hit_tp:
                        reason = "Take Profit"
                    elif hit_sl:
                        reason = "Stop Loss"
                    else:
                        reason = f"Time Exit ({dte}d to expiry)"

                    # Realistic P&L: proportional slippage (30% of bid-ask) + commissions
                    _slip = self._get_spread_slippage(ticker, expiration, strike, option_type, entry_price)
                    _friction = (2 * _slip) + (2 * self._commission_per_contract / 100.0)
                    friction_fraction = _friction / entry_price if entry_price > 0 else 0.0
                    pnl_realistic = max(pnl_raw - friction_fraction, -1.0)

                    closed_this_run.append(
                        f"{ticker} {option_type.upper()} ${strike:.0f} → {reason} "
                        f"(mkt: {pnl_raw:+.1%}, after costs: {pnl_realistic:+.1%})"
                    )
                    update_query = """
                    UPDATE trades
                    SET status='CLOSED', exit_price=?, exit_date=?, pnl_pct=?
                    WHERE entry_id=?
                    """
                    with self._get_connection() as conn:
                        conn.execute(update_query, (current_price, now, pnl_realistic, entry_id))

        if closed_this_run:
            print(f"  Auto-closed {len(closed_this_run)} position(s):")
            for msg in closed_this_run:
                print(f"    \u2713 {msg}")
            print(f"    [costs: ${self._slippage_per_share:.2f}/share slippage ×2 + ${self._commission_per_contract:.2f}/contract commissions ×2]")

    def get_correlated_open_positions(
        self,
        ticker: str,
        lookback_days: int = 60,
        correlation_threshold: float = 0.80,
    ) -> List[Dict]:
        """Return open positions whose underlying is highly correlated with `ticker`.

        Fetches `lookback_days` of daily closes via yfinance for `ticker` and each
        distinct ticker in OPEN trades, then computes Pearson correlation of daily
        returns.  Returns a list of dicts with keys "ticker" and "correlation" for
        any pair where abs(correlation) > correlation_threshold.
        """
        try:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT ticker FROM trades WHERE status='OPEN'"
                ).fetchall()
        except Exception:
            return []

        open_tickers = [r[0] for r in rows if r[0] and r[0].upper() != ticker.upper()]
        if not open_tickers:
            return []

        period = f"{lookback_days}d"

        all_tickers = [ticker] + open_tickers

        def _fetch_returns(sym: str):
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                hist = yf.download(sym, period=period, interval="1d", progress=False, auto_adjust=True)
            if hist.empty:
                return sym, None
            close_col = hist["Close"]
            if isinstance(close_col, pd.DataFrame):
                # MultiIndex result — take first column
                close_col = close_col.iloc[:, 0]
            close_col = close_col.dropna()
            if not isinstance(close_col, pd.Series) or len(close_col) < 5:
                return sym, None
            return sym, close_col.pct_change().dropna()

        # Parallel download — capped at 8 workers to avoid rate-limiting
        hist_map: dict = {}
        try:
            with ThreadPoolExecutor(max_workers=min(len(all_tickers), 8)) as exe:
                futures = {exe.submit(_fetch_returns, sym): sym for sym in all_tickers}
                for fut in as_completed(futures):
                    sym, returns = fut.result()
                    if returns is not None:
                        hist_map[sym] = returns
        except Exception:
            return []

        ref_returns = hist_map.get(ticker)
        if ref_returns is None:
            return []

        correlated = []
        for ot in open_tickers:
            other_returns = hist_map.get(ot)
            if other_returns is None:
                continue
            combined = pd.concat([ref_returns, other_returns], axis=1, join="inner").dropna()
            if len(combined) < 10:
                continue
            corr = float(combined.iloc[:, 0].corr(combined.iloc[:, 1]))
            if abs(corr) > correlation_threshold:
                correlated.append({"ticker": ot, "correlation": corr})

        return correlated

    def get_position_size_with_correlation(
        self,
        ticker: str,
        base_blended_fraction: float,
    ) -> Tuple[float, str]:
        """Return an (adjusted_fraction, reason_string) pair.

        Loads correlation_threshold from config (default 0.80).  If any open
        position is highly correlated with `ticker`, halves the fraction
        (reduction factor from config, default 0.50).
        """
        try:
            config = self._load_config()
            threshold = config.get("correlation_threshold", 0.80)
            reduction = config.get("correlation_size_reduction", 0.50)
            if not config.get("correlation_aware_sizing", True):
                return base_blended_fraction, ""
        except Exception:
            threshold, reduction = 0.80, 0.50

        correlated = self.get_correlated_open_positions(ticker, correlation_threshold=threshold)
        if not correlated:
            return base_blended_fraction, ""

        top = max(correlated, key=lambda x: abs(x["correlation"]))
        reason = (
            f"Correlation-adjusted: {top['ticker']} r={top['correlation']:.2f} "
            f"→ size reduced by {(1-reduction)*100:.0f}%"
        )
        return base_blended_fraction * reduction, reason

    def get_strategy_breakdown(self) -> List[Dict]:
        """Return win/loss/avg P&L grouped by strategy_name."""
        query = """
            SELECT strategy_name,
                   COUNT(*) as total,
                   SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN pnl_pct <= 0 THEN 1 ELSE 0 END) as losses,
                   AVG(pnl_pct) as avg_pnl,
                   SUM(pnl_pct) as total_pnl
            FROM trades
            WHERE status = 'CLOSED' AND pnl_pct IS NOT NULL
            GROUP BY strategy_name
            ORDER BY total DESC
        """
        with self._get_connection() as conn:
            rows = conn.execute(query).fetchall()
        return [
            {"strategy": r[0] or "Unknown", "total": r[1], "wins": r[2],
             "losses": r[3], "win_rate": r[2] / r[1] if r[1] else 0,
             "avg_pnl": r[4], "total_pnl": r[5]}
            for r in rows
        ]

    def get_all_trades(self) -> pd.DataFrame:
        """Returns all trades as a pandas DataFrame."""
        with self._get_connection() as conn:
            return pd.read_sql_query("SELECT * FROM trades", conn)

    def get_performance_summary(self) -> pd.DataFrame:
        """Returns a summary of trading performance with Sharpe, Sortino, and win rate."""
        with self._get_connection() as conn:
            total_count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] or 0
            closed_count = conn.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED'").fetchone()[0] or 0
            win_count = conn.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND pnl_pct > 0").fetchone()[0] or 0
            avg_pnl = conn.execute("SELECT AVG(pnl_pct) FROM trades WHERE status='CLOSED'").fetchone()[0] or 0.0
            sum_pnl = conn.execute("SELECT SUM(pnl_pct) FROM trades WHERE status='CLOSED'").fetchone()[0] or 0.0

        win_rate = win_count / closed_count if closed_count > 0 else 0.0

        # Sharpe and Sortino from closed-trade returns (per-trade, not annualized)
        sharpe_str = "n/a"
        sortino_str = "n/a"
        if closed_count >= 5:
            try:
                with self._get_connection() as conn:
                    rows = conn.execute(
                        "SELECT pnl_pct FROM trades WHERE status='CLOSED' AND pnl_pct IS NOT NULL"
                    ).fetchall()
                returns = np.array([r[0] for r in rows], dtype=float)
                mean_r = np.mean(returns)
                std_r = np.std(returns, ddof=1)
                if std_r > 0:
                    sharpe_str = f"{mean_r / std_r:.3f}"
                downside = returns[returns < 0]
                if len(downside) > 1:
                    sortino_std = np.std(downside, ddof=1)
                    if sortino_std > 0:
                        sortino_str = f"{mean_r / sortino_std:.3f}"
            except Exception as exc:
                logger.debug("Sharpe/Sortino calculation failed: %s", exc)

        summary = {
            "Total Trades": [total_count],
            "Closed Trades": [closed_count],
            "Win Rate": [f"{win_rate:.1%}"],
            "Total PnL %": [f"{sum_pnl:.1%}"],
            "Avg Return": [f"{avg_pnl:.1%}"],
            "Per-Trade Sharpe": [sharpe_str],
            "Per-Trade Sortino": [sortino_str],
        }
        return pd.DataFrame(summary)

    def compute_ic(self) -> dict:
        """Compute Information Coefficient between quality_score and realized pnl_pct.

        IC (Pearson correlation between predicted score and actual P&L) is the key
        metric for validating whether the model has real edge.

        Interpretation:
          IC > 0.10, p < 0.05  →  solid edge, model is predictive
          IC > 0.05, p < 0.20  →  some edge, keep trading to confirm
          IC > 0, not sig      →  weak positive, need more trades
          IC ≤ 0               →  no edge detected

        Requires at least 10 closed trades for a meaningful result.
        """
        df = self.get_all_trades()
        closed = df[
            (df["status"] == "CLOSED")
            & df["pnl_pct"].notna()
            & df["quality_score"].notna()
        ].copy()

        result: dict = {"n": len(closed)}

        if len(closed) < 10:
            result["message"] = (
                f"Need at least 10 closed trades for IC (have {len(closed)}). "
                "Keep paper trading and check back."
            )
            return result

        try:
            from scipy.stats import pearsonr, spearmanr
        except ImportError:
            result["message"] = "scipy not installed — pip install scipy"
            return result

        q_scores = closed["quality_score"].values.astype(float)
        pnl = closed["pnl_pct"].values.astype(float)

        ic_p, pval_p = pearsonr(q_scores, pnl)
        ic_s, pval_s = spearmanr(q_scores, pnl)

        if ic_p > 0.10 and pval_p < 0.05:
            interp = "SOLID EDGE — model is predictive of returns"
        elif ic_p > 0.05 and pval_p < 0.20:
            interp = "SOME EDGE — statistically weak, keep trading to confirm"
        elif ic_p > 0:
            interp = "WEAK POSITIVE — not yet statistically significant"
        else:
            interp = "NO EDGE DETECTED — model is not predictive of returns"

        result.update({
            "ic_technical_pearson": round(float(ic_p), 4),
            "p_technical": round(float(pval_p), 4),
            "ic_technical_spearman": round(float(ic_s), 4),
            "interpretation": interp,
        })

        # AI IC if ai_score was recorded at entry
        if "ai_score" in closed.columns:
            ai_valid = closed[closed["ai_score"].notna()].copy()
            if len(ai_valid) >= 10:
                ai_q = ai_valid["ai_score"].values.astype(float) / 100.0
                ai_pnl = ai_valid["pnl_pct"].values.astype(float)
                ai_ic, ai_pval = pearsonr(ai_q, ai_pnl)
                result["ic_ai_pearson"] = round(float(ai_ic), 4)
                result["p_ai"] = round(float(ai_pval), 4)
                result["ai_adds_value"] = bool(ai_ic > ic_p)
                result["ai_ic_note"] = (
                    "AI score outperforms technical" if ai_ic > ic_p
                    else "Technical score outperforms AI"
                )

        return result

if __name__ == "__main__":
    # Test script with temporary database
    test_db = "test_paper_trades.db"
    manager = PaperManager(db_path=test_db)
    
    test_trade = {
        "ticker": "AAPL",
        "expiration": "2026-06-19",
        "strike": 150.0,
        "type": "call",
        "entry_price": 50.0,
        "quality_score": 0.85,
        "strategy_name": "Test Strategy"
    }
    
    manager.log_trade(test_trade)
    manager.update_positions()
    print("\nPerformance Summary:")
    print(manager.get_performance_summary())
    
    # Cleanup test database
    if os.path.exists(test_db):
        os.remove(test_db)
