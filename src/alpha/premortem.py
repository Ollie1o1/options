"""Pre-Mortem Devil's Advocate & Structured Risk Audit (`src/alpha/premortem.py`).

Before entering ANY trade, automatically answers:
  "Assume this trade loses 15% over the next 30 days — what are the most likely reasons,
   and what exact kill-switches invalidate the thesis?"

Layer 1 (Always-On, Deterministic, 0ms, 100% Offline):
  Reads Dossier structure, Earnings/POST_EARN_CRASH flags, Sector posture, Option Skew/GEX,
  VIX term structure, and Canadian TFSA legality to generate concrete, factual kill-switches.

Layer 2 (Opt-In `--ai` Only, OpenRouter):
  Asks a structured LLM prompt strictly for the bear case & filing risks (never directional guesses).
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.alpha.macro_plumbing import MacroPlumbingSnapshot
from src.alpha.options_for_shares import OptionShareSignal
from src.paths import load_project_env
from src.radar.schema import TickerDossier


@dataclass
class PreMortemReport:
    ticker: str
    spot: Optional[float]
    risk_grade: str               # "CLEAN" | "CAUTION" | "HIGH_RISK"
    summary: str
    kill_switches: List[str] = field(default_factory=list)
    bear_thesis_points: List[str] = field(default_factory=list)
    tfsa_notes: List[str] = field(default_factory=list)
    ai_audit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_premortem(
    ticker: str,
    dossier: Optional[TickerDossier],
    *,
    skew_signal: Optional[OptionShareSignal] = None,
    macro: Optional[MacroPlumbingSnapshot] = None,
    account_type: str = "tfsa",
) -> PreMortemReport:
    """Deterministic offline Pre-Mortem Devil's Advocate for one ticker."""
    sym = ticker.upper().strip()
    kill_switches: List[str] = []
    bear_points: List[str] = []
    tfsa_notes: List[str] = []
    red_count = 0
    warn_count = 0

    spot = dossier.spot if dossier is not None else None

    if dossier is not None:
        # 1. Trend / Regime failure modes
        if dossier.dist_sma200_pct is not None and dossier.dist_sma200_pct < 0:
            red_count += 1
            bear_points.append(
                f"Trading {dossier.dist_sma200_pct:.1f}% BELOW its 200-day moving average (structural downtrend)."
            )
            kill_switches.append("Do not size as a momentum/pullback long until price reclaims 200d MA.")
        elif dossier.dist_sma200_pct is not None and dossier.dist_sma200_pct > 25.0:
            warn_count += 1
            bear_points.append(
                f"Extended +{dossier.dist_sma200_pct:.1f}% above 200d MA — mean-reversion pullback risk is elevated."
            )

        # 2. Earnings & Post-Earnings Breakdown landmines
        if dossier.days_to_earnings is not None and dossier.days_to_earnings <= 10:
            red_count += 1
            bear_points.append(
                f"Binary earnings gap in {dossier.days_to_earnings}d ({dossier.earnings_date}) — overnight gap bypasses stops."
            )
            kill_switches.append(f"Close or reduce position BEFORE {dossier.earnings_date} report.")
        for fl in dossier.risk_flags:
            if "POST_EARN_CRASH" in fl:
                red_count += 1
                bear_points.append(
                    f"Active {fl}: measured in this repo to lag SPY by -1.16% [-2.19%, -0.19%] over the next 20-40d."
                )
                kill_switches.append("Hard avoid for 20+ trading sessions after a high-volume earnings breakdown.")
            elif "BEAR_REGIME" in fl:
                warn_count += 1
                bear_points.append("Broad market in BEAR_REGIME — equity pullbacks lost money in 2018 & 2022 bears.")

        # 3. Sector headwind
        if dossier.sector_posture == "AVOID":
            warn_count += 1
            bear_points.append(
                f"Parent sector {dossier.sector_etf} is regime-gated AVOID in Outlook — swimming upstream."
            )

        # 4. Support stop kill-switch
        if dossier.supports:
            sup0 = dossier.supports[0]
            kill_switches.append(
                f"Price invalidation: close below {sup0['label']} (${float(sup0['level']):.2f}, {float(sup0['pct'])*100:+.1f}%)."
            )

        # 5. Volatility cost check
        if dossier.iv_vs_rv_ratio is not None and dossier.iv_vs_rv_ratio >= 1.30:
            warn_count += 1
            bear_points.append(
                f"Options IV is rich vs realized vol ({dossier.iv_vs_rv_ratio:.2f}x) — buying options faces severe vol-crush drag."
            )
            tfsa_notes.append("Prefer $0-commission Wealthsimple SHARES (or Covered Call if holding 100 sh) over buying calls.")

        if dossier.cdr_ticker:
            tfsa_notes.append(f"Canadian CDR available: {dossier.cdr_ticker} (or buy USD shares directly in Wealthsimple USD TFSA).")

    # 6. Option-implied flow & Dealer GEX warnings
    if skew_signal is not None:
        if skew_signal.signal_label == "CRASH_SMIRK_AVOID":
            red_count += 1
            bear_points.append(
                f"Options market prices crash fear (Put Smirk {skew_signal.put_smirk_vp}vp · CW IV spread {skew_signal.cw_iv_spread_vp}vp)."
            )
            kill_switches.append("Wait for put smirk to normalize before adding long equity exposure.")
        if skew_signal.gex_regime == "NEGATIVE_GEX":
            warn_count += 1
            bear_points.append(
                f"Negative Dealer GEX ({skew_signal.net_gex_m:+.1f}$M/1%) — dealer hedging amplifies swings; widen stops or halve size."
            )

    # 7. Macro volatility curve warning
    if macro is not None:
        if macro.vix_term_label == "BACKWARDATION":
            warn_count += 1
            bear_points.append(
                f"VIX curve inverted (VIX {macro.vix_curve.get('vix')} > VIX3M {macro.vix_curve.get('vix3m')}) — elevated correlation crash regime."
            )
        if macro.fed_liq_chg_4w_b is not None and macro.fed_liq_chg_4w_b < -100.0:
            warn_count += 1
            bear_points.append(f"Fed Net Liquidity contracting ({macro.fed_liq_chg_4w_b:+.0f}$B over 4w) — macro headwind.")

    if account_type.lower() == "tfsa":
        tfsa_notes.append(
            "Wealthsimple TFSA rule: shares, ETFs, covered calls & single-leg long options ONLY — no spreads; avoid rapid day-trading."
        )

    if red_count >= 1:
        grade = "HIGH_RISK"
        summary = f"{sym} carries {red_count} hard landmine(s) and {warn_count} caution(s) — size down or pass."
    elif warn_count >= 2:
        grade = "CAUTION"
        summary = f"{sym} has {warn_count} regime/structure cautions — require strict stop discipline."
    else:
        grade = "CLEAN"
        summary = f"{sym} cleared deterministic pre-mortem checks ({warn_count} minor caution)."

    return PreMortemReport(
        ticker=sym,
        spot=spot,
        risk_grade=grade,
        summary=summary,
        kill_switches=kill_switches or ["Define exact hard stop & max dollar loss BEFORE clicking buy."],
        bear_thesis_points=bear_points or ["No immediate technical, earnings, or skew red flags detected locally."],
        tfsa_notes=tfsa_notes,
    )


def run_llm_premortem_audit(report: PreMortemReport, *, timeout: float = 20.0) -> Optional[str]:
    """Opt-in OpenRouter Devil's Advocate (strictly bear-case / risk audit, never price prediction)."""
    load_project_env()
    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not key:
        return None
    prompt = (
        f"You are a ruthless institutional risk manager running a Pre-Mortem on {report.ticker} "
        f"(spot ${report.spot or 0:.2f}). Current deterministic flags: {report.bear_thesis_points}. "
        "Do NOT predict direction. In 3 concise bullets (<=120 words total), state: "
        "(1) the sharpest fundamental/structural bear case, (2) hidden correlation or macro tail risk, "
        "and (3) one objective falsification trigger that proves a long thesis wrong."
    )
    body = json.dumps(
        {
            "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 220,
            "temperature": 0.2,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Ollie1o1/options",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        return str(raw["choices"][0]["message"]["content"]).strip()
    except Exception:
        return None
