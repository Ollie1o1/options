# Trust audit — 2026-08-03

Every number this tool prints that feeds a decision, walked back to the code
that produces it. The question asked of each was not "does it look right" but
"can it be wrong without anyone noticing".

Three defects were found. All three are the same shape: **a number that could
not change when the thing it described changed.**

---

## 1. Exit status is not proof a scheduled job ran — FIXED (`0e8db47`)

`maintenance_health` decided the scheduler was alive from `launchctl list`'s
last exit status. On 2026-08-03 all three agents reported `0` while
`logs/launchagent.log` — which only the agents themselves write — had not been
touched since **2026-06-15**. Status `0` means the last run succeeded, whenever
that was. It does not mean a run happened.

So the guard built specifically to catch "three agents sat dead for six weeks"
was blind to exactly that, and `next_launchd_dead_state` was *clearing* the
marker that arms the acknowledgement block.

**Fix:** a second symptom, `launchd_silence_days()` — business days since the
agents last wrote their own log. A loaded job silent past
`LAUNCHD_SILENCE_DEAD_DAYS` is dead whatever launchctl claims. Both symptoms
route through one predicate, `_scheduler_is_dead()`, so the banner, the duration
clock and the state marker cannot disagree. Wired into the screener's startup,
`check_pnl`, `doctor`, and `python -m src.maintenance --health`. Fails open: no
jobs loaded, or no log to read, stays quiet.

**Verified:** the banner now fires on the real machine, reporting 35 business
days of silence.

## 2. One ledger, two answers about what counts as evidence — FIXED (`a4fb3e9`)

The 2026-08-01 ruling marked `entry_id 91` a double-log. Both gate cohorts
honoured it. `backtester.run_paper_trade_ic`, `backtester.get_calibration_status`
and `walk_forward.load_trades` did not — they filter on status, `quality_score`
and `pnl_pct` and nothing else, so the same row was excluded from the gate and
counted in the pooled IC. **821 rows against the ledger's 820.**

The filter lived in `phase1_checkpoint`, which is why only its own callers had
it. `walk_forward` was not leaking *today*, but only because the one ruled row
is a Short Put and that path filters Long Call — an accident of the row, not a
rule.

**Fix:** moved to `src/ledger_filters.py`, deliberately free of pandas/numpy
because `backtester` treats those as optional and the shared filter must not
turn an optional dependency into a required one. Pooled IC cohort now reads 820.

## 3. A caveat that cannot stop being true — FIXED (`3d6ad08`)

The short-premium gate's exit-fidelity caveat stated two facts as string
literals: that the scheduler died on 2026-06-15, and that 94% of stopped trades
ran past their stop. Both are currently correct and neither was derived — so the
number could not move when the ledger moved, and the sentence would have gone on
reporting a dead scheduler after the Login Items toggle fixed it.

**Fix:** the measurement moved from `scripts/overshoot_report.py` into
`src/overshoot.py` so code that has to qualify a verdict can reach it, not only
a human running a report. `cohort_caveats(rows, db_path)` now measures it.
Degrades honestly: an unreadable ledger still prints the warning without a
number; a ledger with no stop exits says nothing rather than reciting a
remembered figure. On the real ledger the text is unchanged — 94% is what it
measures.

---

## Verified sound, no change needed

**The Long Call gate reproduces exactly.** Recomputed from raw SQL and scipy
with no project reporting code in the path, restricted to trades closed on or
before the report date:

| | reproduced | reported 2026-08-01 |
|---|---|---|
| nominal cohort | n=92, Pearson −0.0646 (p=0.541), Spearman −0.1321 (p=0.209) | n=92, −0.065 (p=0.541), −0.132 (p=0.209) |
| affordable subset | n=90, Pearson −0.0920 (p=0.388), Spearman −0.1611 (p=0.129) | n=90, −0.092 (p=0.388), −0.161 (p=0.129) |

Matches to four decimals on both cohorts. The STOP verdict is sound. On today's
larger cohort (n=95) it has gone *further* negative (−0.118 / −0.175), which
reinforces it.

**The cost model threads config correctly.** `$0` commission and `$0` FX reach
the EV calculation from `config.json`; the screener does not fall back to the
constant. `FALLBACK_COMMISSION_PER_CONTRACT = 0.65` fires only for helpers
constructed without a commission threaded through, and is deliberately not `0.0`
— overstating cost can make a strategy look worse than it is, never better. The
index-option exception (SPX/VIX $1, RUT $0.50, XSP $0.25) is documented at the
constant and in `config.json`, and a single scalar still cannot express it.

**The $4,000 affordability cap is read from config everywhere** — `auto_log.max_capital_at_risk`
in both the screener and the checkpoint. No hardcoded copy exists.

**The evidence banner reads artifacts, not literals.** Confirmed loading from
`reports/` after the repo move and rendering figures that match the checkpoint.

**`reports/TRACK_RECORD.md` already excluded ruled duplicates** before this pass.

---

## CI — red for weeks, now green and blocking (2026-08-04)

Two independent causes, neither visible from a local test run.

**The test failure.** `tests/test_cache_release.py` installs a fake
`yfinance.cache`. Its helper calls `setdefault("yfinance", ...)`, which returns
the *real* module whenever yfinance is already imported, then assigns `.cache`
on it. `tearDown` restored both `sys.modules` entries — but they point at the
object that was mutated, so restoring them undid nothing, and every later test
in the session saw `_FakeManager`.

It passed locally because the local runner imports yfinance *after* the fake is
installed; under pytest the order flips. Which is the lesson worth keeping:
**`scripts/test.sh` passing has never proved CI passes.** There is no pytest in
the venv, and 6 test modules use pytest-only constructs and are skipped locally
altogether. They are genuinely different test runs.

**The mypy job.** It was `continue-on-error: true` *and* red — the worst
combination, because it cost time on every push while proving nothing. It
reported **349 errors in 76 of 289 modules**; the other **213 were already
clean**.

Rather than make 349 type edits across the pricing and ledger code, the job now
blocks on the 213 clean modules, with the 76 exempted one section each in
`mypy.ini`. A module leaves the list by deleting its own two lines.

**The exemption list is debt, not a clean bill of health.** Among the 349:

| code | n | meaning |
|---|---|---|
| `union-attr` | 54 | attribute access on a possibly-`None` value |
| `operator` | 11 | arithmetic on a possibly-`None` value |
| `index` | 11 | indexing a possibly-`None` value |

Those 76 are latent crashes, not style — e.g. `src/structure/candidates.py:197`,
`Unsupported operand types for * ("None" and "float")`. Shrinking the list is
real work and the possible-`None` codes are where it should start.

CI findings are readable without admin rights: job logs return 403 anonymously,
but check-run annotations are public. `scripts/mypy_annotate.py` publishes each
diagnostic plus a histogram by error code and by file.

## Not fixed here

**Exit fidelity itself.** The measurement is now honest, but the underlying
problem — exits checked only when someone opens the screener — needs the
operator-only Login Items toggle. Until then the exit rules in this repo
describe what was intended, not what was enforced. The bias runs *down* on
recorded P&L, so it does not threaten a positive verdict.

**The unobserved tail.** Arm A's READY rests on under two months containing no
volatility shock, and the affordability cap excludes every large loss the wider
ledger contains — so the measured cohort has never actually taken a big loss.
This is a limit of the evidence, not a defect in the code, and the gate already
prints it on every run.
