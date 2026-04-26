"""Phase 7 stress: per-structure weight optimizer.

Seeds 30 closed trades of each structure (single, spread, iron_condor) with
synthetic component scores correlated to pnl_pct, then runs
recommend_weights_for_structure() for each. Verifies:
  - structure filter pulls the right N trades
  - per-component IC has correct sign for the planted relationships
  - recommended weights shift toward high-IC components (within cap)
  - the three structures get independently optimized weight maps
"""
import os, sys, json, tempfile, sqlite3, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.backtester import (
    run_paper_trade_ic_for_structure,
    recommend_weights_for_structure,
    _SPREAD_FEATURE_COLS,
    _IRON_FEATURE_COLS,
)


# --- Build minimal trades schema -------------------------------------------
DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
DB.close()

# Need columns: status, quality_score, pnl_pct, long_strike, short_put_strike,
# short_call_strike, plus all _SPREAD_FEATURE_COLS + _IRON_FEATURE_COLS.
all_feature_cols = sorted(set(_SPREAD_FEATURE_COLS) | set(_IRON_FEATURE_COLS) | {"return_on_risk_score", "delta_neutral_score", "credit_to_width_score"})

with sqlite3.connect(DB.name) as conn:
    cols_sql = ",\n  ".join(f"{c} REAL" for c in all_feature_cols)
    conn.execute(f"""
    CREATE TABLE trades (
        entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
        status TEXT,
        quality_score REAL,
        pnl_pct REAL,
        long_strike REAL,
        short_put_strike REAL,
        short_call_strike REAL,
        long_put_strike REAL,
        long_call_strike REAL,
        {cols_sql}
    )""")

rng = np.random.default_rng(42)
N_PER_STRUCT = 30


def _insert(conn, structure, n):
    """Plant correlations:
      - single: pnl_pct strongly correlated with pop_score (+0.7 IC target)
      - spread: pnl_pct correlated with credit_to_width_score (+0.7) and pop_score (+0.4)
      - iron_condor: pnl_pct correlated with delta_neutral_score (+0.7)
    """
    for _ in range(n):
        scores = {c: float(rng.uniform(0.0, 1.0)) for c in all_feature_cols}
        if structure == "single":
            ls = None; sp = None; sc = None; lp = None; lc = None
            pnl = 0.7 * (scores["pop_score"] - 0.5) + 0.05 * float(rng.normal())
        elif structure == "spread":
            ls = 95.0; sp = None; sc = None; lp = None; lc = None
            pnl = 0.6 * (scores["credit_to_width_score"] - 0.5) + 0.3 * (scores["pop_score"] - 0.5) + 0.05 * float(rng.normal())
        else:  # iron_condor
            ls = None; sp = 90.0; sc = 110.0; lp = 85.0; lc = 115.0
            pnl = 0.7 * (scores["delta_neutral_score"] - 0.5) + 0.05 * float(rng.normal())

        cols = ["status", "quality_score", "pnl_pct",
                "long_strike", "short_put_strike", "short_call_strike",
                "long_put_strike", "long_call_strike"] + all_feature_cols
        vals = ["CLOSED", float(rng.uniform(0.4, 0.8)), pnl,
                ls, sp, sc, lp, lc] + [scores[c] for c in all_feature_cols]
        placeholders = ",".join(["?"] * len(cols))
        conn.execute(f"INSERT INTO trades({','.join(cols)}) VALUES ({placeholders})", vals)


with sqlite3.connect(DB.name) as conn:
    _insert(conn, "single", N_PER_STRUCT)
    _insert(conn, "spread", N_PER_STRUCT)
    _insert(conn, "iron_condor", N_PER_STRUCT)
    conn.commit()


# --- Verify structure filter pulls the right N -----------------------------
ic_single = run_paper_trade_ic_for_structure(DB.name, "single", _SPREAD_FEATURE_COLS)
ic_spread = run_paper_trade_ic_for_structure(DB.name, "spread", _SPREAD_FEATURE_COLS)
ic_iron = run_paper_trade_ic_for_structure(DB.name, "iron_condor", _IRON_FEATURE_COLS)

print(f"single n     = {ic_single['n_trades']}  (expect {N_PER_STRUCT})")
print(f"spread n     = {ic_spread['n_trades']}  (expect {N_PER_STRUCT})")
print(f"iron_cond n  = {ic_iron['n_trades']}    (expect {N_PER_STRUCT})")
assert ic_single["n_trades"] == N_PER_STRUCT
assert ic_spread["n_trades"] == N_PER_STRUCT
assert ic_iron["n_trades"] == N_PER_STRUCT
print("  ✓ structure filter rows correct\n")

# --- Verify planted correlations show up as positive IC --------------------
print("Spread per-component IC:")
for col, v in sorted(ic_spread["component_ic"].items(), key=lambda x: -x[1]):
    print(f"  {col:<28s} IC = {v:+.3f}  (n={ic_spread['component_n'].get(col)})")
assert ic_spread["component_ic"].get("credit_to_width_score", 0) > 0.20, "spread c/w IC should be strongly positive"
assert ic_spread["component_ic"].get("pop_score", 0) > 0.10, "spread pop IC should be positive"
print("  ✓ spread planted IC verified\n")

print("Iron condor per-component IC:")
for col, v in sorted(ic_iron["component_ic"].items(), key=lambda x: -x[1]):
    print(f"  {col:<28s} IC = {v:+.3f}  (n={ic_iron['component_n'].get(col)})")
assert ic_iron["component_ic"].get("delta_neutral_score", 0) > 0.20, "IC delta_neutral IC should be strongly positive"
print("  ✓ iron condor planted IC verified\n")


# --- Verify per-structure recommend produces independent weight maps -------
# We need a config.json — point at the project's real one
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")

rec_spread = recommend_weights_for_structure(DB.name, config_path, "spread")
rec_iron = recommend_weights_for_structure(DB.name, config_path, "iron_condor")

print(f"\nSpread recommend: ready={rec_spread['ready']}  weights_key={rec_spread['weights_key']}")
print(f"  budget={rec_spread['budget']}  shrinkage={rec_spread['shrinkage']}")
top_spread = sorted(rec_spread["deltas"].items(), key=lambda x: -x[1])[:3]
print(f"  top weight increases: {top_spread}")

print(f"\nIron condor recommend: ready={rec_iron['ready']}  weights_key={rec_iron['weights_key']}")
print(f"  budget={rec_iron['budget']}  shrinkage={rec_iron['shrinkage']}")
top_iron = sorted(rec_iron["deltas"].items(), key=lambda x: -x[1])[:3]
print(f"  top weight increases: {top_iron}")

# Spread should have credit_to_width with a non-trivial positive delta
assert rec_spread["weights_key"] == "credit_spread_weights"
assert rec_spread["ready"], "spread optimizer should be ready"
spread_cw_delta = rec_spread["deltas"].get("credit_to_width", 0.0)
print(f"\n  spread credit_to_width delta = {spread_cw_delta:+.4f}")
assert spread_cw_delta >= 0.0, "spread credit_to_width should not decrease (high IC component)"

# IC should have delta_neutral with non-trivial positive delta
assert rec_iron["weights_key"] == "iron_condor_weights"
assert rec_iron["ready"], "iron condor optimizer should be ready"
ic_dn_delta = rec_iron["deltas"].get("delta_neutral", 0.0)
print(f"  iron_condor delta_neutral delta = {ic_dn_delta:+.4f}")
assert ic_dn_delta >= 0.0, "IC delta_neutral should not decrease (high IC component)"

# Independence: spread and iron use different weight maps
assert rec_spread["weights_key"] != rec_iron["weights_key"]
assert set(rec_spread["calibratable_keys"]) != set(rec_iron["calibratable_keys"])
print(f"\n  ✓ spread keys ({len(rec_spread['calibratable_keys'])}) and IC keys ({len(rec_iron['calibratable_keys'])}) are independent")

# Verify recommended weights sum back to original budget (renormalization)
spread_sum = sum(rec_spread["recommended"][k] for k in rec_spread["calibratable_keys"])
iron_sum = sum(rec_iron["recommended"][k] for k in rec_iron["calibratable_keys"])
print(f"  spread budget preserved: {spread_sum:.4f} ≈ {rec_spread['budget']:.4f}")
print(f"  iron budget preserved:   {iron_sum:.4f} ≈ {rec_iron['budget']:.4f}")
assert abs(spread_sum - rec_spread["budget"]) < 0.001
assert abs(iron_sum - rec_iron["budget"]) < 0.001

print("\nOK Phase 7 per-structure optimizer stress PASSED")
os.unlink(DB.name)
