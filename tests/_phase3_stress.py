"""Phase 3 end-to-end stress: scored df → find → enrich → log → verify DB."""
import os, sys, tempfile

import pandas as pd
import numpy as np
from src.options_screener import find_credit_spreads, find_iron_condors
from src.spread_scoring import enrich_credit_spreads, enrich_iron_condors
from src.paper_manager import PaperManager
import json, sqlite3

with open("config.json") as f:
    config = json.load(f)

exp = "2026-05-20"
sym = "TEST"

chain_spec = [
    (80,  -0.08, 0.30, 0.92, 20.00),
    (85,  -0.16, 0.70, 0.85, 16.00),
    (90,  -0.26, 1.60, 0.75, 12.00),
    (95,  -0.38, 2.80, 0.62, 8.50),
    (100, -0.50, 4.50, 0.50, 5.50),
    (105, -0.62, 7.00, 0.38, 2.80),
    (110, -0.74, 9.50, 0.26, 1.60),
    (115, -0.84, 12.50, 0.16, 0.70),
    (120, -0.92, 16.00, 0.08, 0.30),
]

rows = []
for strike, pd_, pp, cd, cp in chain_spec:
    base = dict(symbol=sym, expiration=exp, strike=strike, volume=1500, openInterest=2500,
                impliedVolatility=0.30, gamma=0.02, vega=0.15, theta=-0.04,
                premium=0.0, delta=0.0, type="")
    base_p = dict(base); base_p.update(type="put", delta=pd_, premium=pp)
    base_c = dict(base); base_c.update(type="call", delta=cd, premium=cp)
    rows.append(base_p)
    rows.append(base_c)

df = pd.DataFrame(rows)

SCORE_COLS = [
    "pop_score","ev_score","rr_score","liquidity_score","momentum_score",
    "iv_rank_score","theta_score","iv_advantage_score","vrp_score","iv_mispricing_score",
    "skew_align_score","vega_risk_score","term_structure_score","catalyst_score",
    "em_realism_score","gamma_theta_score","gex_score","gamma_magnitude_score",
    "gamma_pin_score","iv_velocity_score","max_pain_score","oi_change_score",
    "option_rvol_score","pcr_score","sentiment_score_norm","spread_score","trader_pref_score",
]
rng = np.random.default_rng(42)
for c in SCORE_COLS:
    df[c] = rng.uniform(0.4, 0.8, len(df))
df["quality_score"] = rng.uniform(0.5, 0.7, len(df))

print(f"chain: {len(df)} rows")
spreads = find_credit_spreads(df)
print(f"find_credit_spreads -> {len(spreads)} spreads")
if not spreads.empty:
    print(spreads[["type","short_strike","long_strike","net_credit"]].to_string())

condors = find_iron_condors(df)
print(f"find_iron_condors -> {len(condors)} condors")
if not condors.empty:
    print(condors[["short_put_strike","long_put_strike","short_call_strike","long_call_strike","total_credit","net_delta"]].to_string())

assert not spreads.empty, "stress fail: find_credit_spreads empty"
assert not condors.empty, "stress fail: find_iron_condors empty"

sp_e = enrich_credit_spreads(spreads, df, config)
ic_e = enrich_iron_condors(condors, df, config)
print(f"\nspread quality_score head={sp_e['quality_score'].head(3).round(3).tolist()}")
print(f"spread credit_to_width_score: {sp_e['credit_to_width_score'].head(3).round(3).tolist()}")
print(f"ic qs: {ic_e['quality_score'].head(3).round(3).tolist()}")
print(f"ic delta_neutral_score: {ic_e['delta_neutral_score'].head(3).round(3).tolist()}")

tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
tmp.close()
pm = PaperManager(db_path=tmp.name)

sp_row = sp_e.iloc[0]
sp_payload = {
    "date": "2026-04-24", "ticker": sym, "expiration": exp,
    "short_strike": sp_row["short_strike"], "long_strike": sp_row["long_strike"],
    "type": sp_row["type"], "net_credit": sp_row["net_credit"],
    "max_profit": sp_row["max_profit"], "max_loss": sp_row["max_loss"],
    "quality_score": sp_row["quality_score"],
    "weight_profile": "stress_test_v1",
}
for c in SCORE_COLS:
    if c in sp_row.index:
        sp_payload[c] = sp_row[c]
sp_payload["iv_edge_score"] = sp_row.get("iv_advantage_score")
for g in ("entry_iv","entry_delta","entry_gamma","entry_vega","entry_theta"):
    if g in sp_row.index:
        sp_payload[g] = sp_row[g]
ok = pm.log_spread_if_new(sp_payload)
print(f"\nlog_spread_if_new: inserted={ok}")

ic_row = ic_e.iloc[0]
ic_payload = {
    "date": "2026-04-24", "ticker": sym, "expiration": exp,
    "short_put_strike": ic_row["short_put_strike"], "long_put_strike": ic_row["long_put_strike"],
    "short_call_strike": ic_row["short_call_strike"], "long_call_strike": ic_row["long_call_strike"],
    "total_credit": ic_row["total_credit"],
    "max_profit": ic_row.get("total_credit", 0)*100,
    "max_risk": ic_row.get("max_risk", 0),
    "net_delta": ic_row.get("net_delta"),
    "quality_score": ic_row["quality_score"],
    "weight_profile": "stress_test_v1",
}
for c in SCORE_COLS:
    if c in ic_row.index:
        ic_payload[c] = ic_row[c]
ic_payload["iv_edge_score"] = ic_row.get("iv_advantage_score")
for g in ("entry_iv","entry_delta","entry_gamma","entry_vega","entry_theta"):
    if g in ic_row.index:
        ic_payload[g] = ic_row[g]
ok2 = pm.log_iron_condor_if_new(ic_payload)
print(f"log_iron_condor_if_new: inserted={ok2}")

con = sqlite3.connect(tmp.name)
cur = con.cursor()
cur.execute("SELECT entry_id, ticker, strategy_name, type, strike, long_strike, spread_width, "
            "net_credit, max_profit_usd, max_loss_usd, "
            "short_put_strike, long_put_strike, short_call_strike, long_call_strike, "
            "net_delta, quality_score, pop_score, ev_score, iv_rank_score, weight_profile "
            "FROM trades")
for row in cur.fetchall():
    print(f"\nDB row id={row[0]} {row[1]} {row[2]}")
    print(f"  short={row[4]} long={row[5]} width={row[6]} credit={row[7]} maxP={row[8]} maxL={row[9]}")
    print(f"  IC: sp={row[10]} lp={row[11]} sc={row[12]} lc={row[13]} netD={row[14]}")
    print(f"  scores: q={row[15]} pop={row[16]} ev={row[17]} ivrank={row[18]} profile={row[19]}")

ok3 = pm.log_spread_if_new(sp_payload)
ok4 = pm.log_iron_condor_if_new(ic_payload)
print(f"\nRe-log dedup: spread={ok3} condor={ok4} (both should be False)")
assert not ok3 and not ok4, "dedup broken"
print("\nOK Phase 3 end-to-end stress PASSED")
os.unlink(tmp.name)
