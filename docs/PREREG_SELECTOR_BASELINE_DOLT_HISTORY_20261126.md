# Preregistration — selector vs liquidity-first baseline, historical hold-to-expiry companion

**Registered:** 2026-11-26. The download was running and the code was tested on a partial cache
(6 SPY entries, one of them settled). No full-sample outcome had been computed.
**Companion to:** [PREREG_SELECTOR_LIQUIDITY_BASELINE_20261126.md](PREREG_SELECTOR_LIQUIDITY_BASELINE_20261126.md)
(forward, daily marks, book exits). Everything not listed below is identical to that study:
the generic selector call, the baseline rule and tie-breaks, the 14–58 DTE window, the friction cap,
fees, the pair definition and the ISO-week clustering.
**Code:** `src/selector_study/dolt_history.py`. **Command:**
`opt lab study history --fetch` (download, resumable) then `opt lab study history`.

## Data

- **Source:** DoltHub `post-no-preference/options`, end-of-day bid/ask/IV/delta, free and keyless.
  Coverage 2019-02 → 2026-06-12.
- **Symbols (17):** SPY, DIA, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, AAPL, AMZN, GOOGL, JPM, META,
  MSFT, NVDA. These are the study's agreed symbols that Dolt carries. QQQ, IWM, GLD, TLT, SMH, EEM
  and HYG are absent.
- **Entry days:** one per ISO week: the Wednesday. If Dolt has no chain that day, Thursday, then
  Tuesday. A week with no chain on any of the three is a missing week for that symbol.
- **Spot:** Dolt has no underlying price. Spot is the forward implied by regressing call mid − put
  mid on strike for the nearest expiration, within 8% of the money (slope must be −0.8 to −1.2).
- **Open interest:** not in Dolt, so the baseline's OI ≥ 1 condition is dropped (set to 0).

## Differences from the forward study (frozen)

| | Forward study | This companion |
|---|---|---|
| Exit | book spread rules on daily marks | **held to expiration**, settled at intrinsic value |
| Crossings | entry and exit | **entry only** (crossed = primary, mid = sensitivity) |
| Capture time | 14:00–20:00 ET check | end-of-day source; no time check |
| Bootstrap block | 3 weeks | **8 weeks** (holds last 2–8 weeks) |
| Read | ≥ 2027-03-01 | at registration (historical data) |

**Why hold to expiry:** Dolt keeps ~27 strikes per expiration and re-chooses them daily. On SPY
2023-05-19 puts, 2–12 of 27 strikes survived from 03-15 to later dates. Daily marks for a held
contract are mostly absent, and substituting nearby strikes would add an error that differs
between arms. Holding to expiration needs only the settlement price.

**Consequence, stated before data:** one crossing instead of two **understates** whatever a
tighter-spread baseline gains. A baseline advantage found here is a lower bound. A null here does
not rule out an advantage under the book's exits.

## Settlement

Underlying close at expiration = local adjusted close × (parity spot / adjusted close) on the
fetched day nearest the expiration (within 7 days). If that ratio differs by more than 5% from the
entry day's ratio (a split during the hold), the entry day's ratio is used, which keeps
pre-split strikes on their own basis. No close → the outcome is `missing`, excluded and counted.

## Read rule (frozen)

The same as the forward study: ≥ 12 weeks and ≥ 40 resolved pairs in the crossed case, otherwise
*Insufficient data*. Verdict from the 95% CI of the mean weekly paired difference. The readout
also reports the share of identical picks. On Dolt's coarse strike grid that share is expected
to be high, which lowers power. It is reported rather than worked around.

## Known limitations

- Dolt's strike grid is coarser than the live chains. This measures selection *on that grid*.
- 2022–24 Dolt cadence is roughly every other day, so some weeks fall back to Thursday or Tuesday.
- End-of-day quotes. No fill is measured.
