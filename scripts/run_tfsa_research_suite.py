"""CLI runner for the TFSA & Options Research Backtest Suite (`scripts/run_tfsa_research_suite.py`).

Usage:
    python -m scripts.run_tfsa_research_suite
    python -m scripts.run_tfsa_research_suite --out reports/tfsa_research_suite.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.equity_playbooks.tfsa_studies import run_all_tfsa_studies
from src.paths import repo_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all TFSA & Options research backtest studies.")
    ap.add_argument("--ohlcv-db", default="data/equity_ohlcv.db")
    ap.add_argument("--out", default="reports/tfsa_research_suite.json")
    args = ap.parse_args()

    res = run_all_tfsa_studies(ohlcv_db=args.ohlcv_db)
    out_p = Path(repo_path(args.out))
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("=== TFSA & OPTIONS RESEARCH BACKTEST SUITE ===")
    cc = res["vrp_covered_calls"]
    if "error" not in cc:
        print(
            f"1. VRP Covered Calls ({cc['window']}):\n"
            f"   Bare VM_OUTLOOK : CAGR {cc['bare_vm_outlook']['cagr_pct']}% · Sharpe {cc['bare_vm_outlook']['sharpe']} · MaxDD {cc['bare_vm_outlook']['max_dd_pct']}%\n"
            f"   Uncond 0.25d CC : CAGR {cc['unconditional_cc_25d']['cagr_pct']}% · Sharpe {cc['unconditional_cc_25d']['sharpe']} · MaxDD {cc['unconditional_cc_25d']['max_dd_pct']}%\n"
            f"   VRP-Gated CC    : CAGR {cc['vrp_gated_cc_25d']['cagr_pct']}% · Sharpe {cc['vrp_gated_cc_25d']['sharpe']} · MaxDD {cc['vrp_gated_cc_25d']['max_dd_pct']}% (overwrote {cc['vrp_gated_cc_25d']['overwrite_pct_of_legs']}% of legs)"
        )
    lp = res["leaps_plus_sgov"]
    if "error" not in lp:
        print(
            f"2. Deep-ITM LEAPS + SGOV ({lp['window']}):\n"
            f"   Bare VM_OUTLOOK : CAGR {lp['bare_vm_outlook']['cagr_pct']}% · Sharpe {lp['bare_vm_outlook']['sharpe']} · MaxDD {lp['bare_vm_outlook']['max_dd_pct']}%\n"
            f"   0.80d LEAPS+SGOV: CAGR {lp['leaps_80d_plus_sgov']['cagr_pct']}% · Sharpe {lp['leaps_80d_plus_sgov']['sharpe']} · MaxDD {lp['leaps_80d_plus_sgov']['max_dd_pct']}%"
        )
    wo = res["wider_outlook_universe"]
    if "error" not in wo:
        print(
            f"3. Wider Outlook Universe ({wo['window_aligned']}):\n"
            f"   16 ETFs         : CAGR {wo['baseline_16']['cagr_pct']}% · Sharpe {wo['baseline_16']['sharpe']} · MaxDD {wo['baseline_16']['max_dd_pct']}% · 2022 {wo['bear_2022_ret_pct']['baseline_16']}%\n"
            f"   {wo['expanded_count']} ETFs         : CAGR {wo['expanded_universe']['cagr_pct']}% · Sharpe {wo['expanded_universe']['sharpe']} · MaxDD {wo['expanded_universe']['max_dd_pct']}% · 2022 {wo['bear_2022_ret_pct']['expanded_universe']}%"
        )
    rg = res["regime_gated_pullback"]
    if "error" not in rg:
        u, g = rg["unfiltered"], rg["regime_gated"]
        print(
            f"4. Regime-Gated Uptrend Pullback ({rg['gate']}):\n"
            f"   Unfiltered      : n={u['n']} · Win {u['win_rate_pct']}% · Mean {u['mean_ret_pct']}% · PF {u['pf']} · 2018/22 {u['bear_2018_2022_mean_pct']}%\n"
            f"   Regime-Gated    : n={g['n']} · Win {g['win_rate_pct']}% · Mean {g['mean_ret_pct']}% · PF {g['pf']} · 2018/22 {g['bear_2018_2022_mean_pct']}%"
        )
    p3 = res["phase3_alpha_readiness"]
    print(
        f"5. Phase 3 Alpha Archive Readiness: {p3['readiness_verdict']} "
        f"(chains={p3['chain_archive']['symbols']} syms/{p3['chain_archive']['days']}d · "
        f"EDGAR={p3['edgar_events_count']} events · OCC={p3['occ_symbols_latest']} syms)"
    )
    print(f"Saved full JSON -> {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
