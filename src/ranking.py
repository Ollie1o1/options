"""Combine screener quality scores with AI scores to produce a final ranked list.

Usage:
    from src.ranking import combine_scores, print_ranked_table, to_csv, to_json

    ranked = combine_scores(picks_df, ai_df, vix_regime="high")
    print_ranked_table(ranked, verbose_reasoning=True)
    to_csv(ranked, "output/ranked.csv")
    to_json(ranked, "output/ranked.json")
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config_ai import AI_CONFIG

# ── Score combination ──────────────────────────────────────────────────────────


def combine_scores(
    picks_df: pd.DataFrame,
    ai_df: pd.DataFrame,
    ai_weight: Optional[float] = None,
    technical_weight: Optional[float] = None,
    vix_regime: str = "normal",
    thematic_context=None,
) -> pd.DataFrame:
    """Merge AI results into *picks_df* and compute a dynamic weighted *final_score*.

    final_score = dynamic_tw * quality_score + dynamic_aw * (ai_score / 100)

    The ai_weight is dynamically adjusted per row based on:
      - VIX regime multiplier (high VIX -> more AI weight)
      - AI confidence (higher confidence -> more AI weight)
      - Liquidity quality (illiquid contracts -> slightly more AI weight)
    """
    cfg = AI_CONFIG
    base_aw = ai_weight if ai_weight is not None else cfg["ai_weight"]
    base_tw = technical_weight if technical_weight is not None else cfg["technical_weight"]

    # Regime multiplier
    regime_mults = cfg.get("regime_weight_multipliers", {"low": 0.80, "normal": 1.00, "high": 1.30})
    regime_key = vix_regime.lower() if vix_regime.lower() in regime_mults else "normal"
    regime_mult = regime_mults[regime_key]

    df = picks_df.copy()
    ai_cols = ["ai_score", "ai_reasoning", "ai_flags", "catalyst_risk", "iv_justified", "ai_confidence"]
    present = [c for c in ai_cols if c in ai_df.columns]
    df = df.reset_index(drop=True)
    ai_df = ai_df.reset_index(drop=True)
    df = df.join(ai_df[present])

    # Fill missing AI columns
    for col, default in [("ai_score", 50.0), ("ai_reasoning", ""), ("ai_flags", ""),
                          ("catalyst_risk", "medium"), ("iv_justified", True), ("ai_confidence", 5.0)]:
        if col not in df.columns:
            df[col] = default

    tech = df["quality_score"].fillna(0).clip(0, 1)
    ai_norm = df["ai_score"].fillna(50).clip(0, 100) / 100.0
    confidence = df["ai_confidence"].fillna(5.0).clip(0, 10) / 10.0

    # Per-row dynamic ai_weight
    # confidence_mult: ranges 0.5 (conf=0) to 1.0 (conf=10)
    confidence_mult = 0.5 + 0.5 * confidence

    # Liquidity adjustment: boost AI weight for illiquid/wide-spread contracts
    liquidity_adj = pd.Series(1.0, index=df.index)
    if "spread_pct" in df.columns:
        wide_spread = df["spread_pct"].fillna(0) > 0.20
        liquidity_adj = np.where(wide_spread, 1.10, 1.0)

    aw_dynamic = (base_aw * regime_mult * confidence_mult * liquidity_adj).clip(0, 0.55)
    tw_dynamic = 1.0 - aw_dynamic

    df["final_score"] = (tw_dynamic * tech + aw_dynamic * ai_norm).round(4)
    df["ai_weight_used"] = aw_dynamic.round(3)

    # Divergence detection
    thr = cfg.get("narrative_thresholds", {})
    div_threshold = thr.get("divergence_flag_threshold", 0.20)
    df["score_divergence"] = (ai_norm - tech).abs().round(3)
    df["divergence_flag"] = df["score_divergence"] > div_threshold
    df["divergence_direction"] = "---"
    ai_higher = (ai_norm - tech) > div_threshold
    tech_higher = (tech - ai_norm) > div_threshold
    df.loc[ai_higher,  "divergence_direction"] = "AI>TECH"
    df.loc[tech_higher, "divergence_direction"] = "TECH>AI"

    # Divergence adjustments
    df["divergence_adjusted"] = False
    penalty_factor = cfg.get("divergence_penalty_factor", 0.15)
    boost_factor   = cfg.get("divergence_boost_factor",   0.10)

    # TECH>AI: quant bullish, AI bearish -> penalise final_score
    tech_higher_mask = df["divergence_direction"] == "TECH>AI"
    if tech_higher_mask.any():
        penalty = df.loc[tech_higher_mask, "score_divergence"] * penalty_factor
        df.loc[tech_higher_mask, "final_score"] -= penalty
        df.loc[tech_higher_mask, "divergence_adjusted"] = True

    # AI>TECH: AI bullish, quant bearish -> boost final_score symmetrically
    # Only boost when AI is sufficiently confident (>= 7.0 / 10)
    high_conf_mask = df["ai_confidence"].fillna(5.0) >= 7.0
    ai_higher_mask = df["divergence_direction"] == "AI>TECH"
    if ai_higher_mask.any():
        boost_mask = ai_higher_mask & high_conf_mask
        if boost_mask.any():
            boost = df.loc[boost_mask, "score_divergence"] * boost_factor
            df.loc[boost_mask, "final_score"] += boost
            df.loc[boost_mask, "ai_weight_used"] = (
                df.loc[boost_mask, "ai_weight_used"] + boost
            ).clip(0, 0.55)
        df.loc[ai_higher_mask, "divergence_adjusted"] = True

    df["final_score"] = df["final_score"].clip(0, 1).round(4)

    # Override for AI-skipped rows
    if "ai_skipped" in df.columns:
        mask = df["ai_skipped"].fillna(False).astype(bool)
        df.loc[mask, "divergence_flag"] = False
        df.loc[mask, "divergence_direction"] = "---"
        df.loc[mask, "divergence_adjusted"] = False
        df.loc[mask, "ai_weight_used"] = 0.0
        df.loc[mask, "final_score"] = tech[mask].values

    # Thematic multiplier — applied post-scoring so it doesn't distort divergence logic
    df["thematic_multiplier"] = 1.0
    df["thematic_edge"] = ""

    if thematic_context is not None:
        from src.data_fetching import SECTOR_MAP as _SM
        top_set = set(getattr(thematic_context, "top_sectors", []) or [])
        bot_set = set(getattr(thematic_context, "bottom_sectors", []) or [])
        mr_set  = set(getattr(thematic_context, "mean_reversion_setups", []) or [])

        etf_s  = df["symbol"].map(lambda s: _SM.get(str(s).upper()))
        mult_s = pd.Series(1.0, index=df.index)
        edge_s = pd.Series("", index=df.index)

        if top_set:
            m = etf_s.isin(top_set)
            mult_s[m] = 1.10
            edge_s[m] = "TAILWIND"
        if bot_set:
            m = etf_s.isin(bot_set)
            mult_s[m] = 0.95
            edge_s[m] = "HEADWIND"
        if mr_set:
            m = etf_s.isin(mr_set)
            mult_s[m] = (mult_s[m] + 0.03)
            edge_s[m] = (edge_s[m] + "+MEAN-REV").str.lstrip("+")

        df["thematic_multiplier"] = mult_s.round(3)
        df["thematic_edge"] = edge_s
        df["final_score"] = (df["final_score"] * mult_s).clip(0, 1).round(4)

    df["rank"] = df["final_score"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("rank")


# ── Rich table output ──────────────────────────────────────────────────────────

_RISK_STYLE    = {"low": "green", "medium": "yellow", "high": "red"}
_DIV_STYLE     = {"AI>TECH": "cyan", "TECH>AI": "magenta", "---": "dim"}

_SCORE_BANDS: list[tuple[float, float, str]] = [
    (80, 101, "bold green"),
    (60, 80,  "green"),
    (40, 60,  "yellow"),
    (20, 40,  "red"),
    (0,  20,  "bold red"),
]


def _score_style(score: float) -> str:
    for lo, hi, style in _SCORE_BANDS:
        if lo <= score < hi:
            return style
    return "white"


def print_ranked_table(
    df: pd.DataFrame,
    top_n: int = 20,
    verbose_reasoning: bool = False,
) -> None:
    """Print a ranked options table with divergence highlighting."""
    subset = df.head(top_n)
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        from rich.text import Text
        from rich.panel import Panel

        console = Console(width=180, highlight=False)
        _rich_table(subset, console, verbose_reasoning)
    except ImportError:
        _plain_table(subset)


def _rich_table(df: pd.DataFrame, console, verbose_reasoning: bool) -> None:
    from rich.table import Table
    from rich import box
    from rich.text import Text

    # Print divergence legend
    divergent = df[df["divergence_flag"]] if "divergence_flag" in df.columns else pd.DataFrame()

    table = Table(
        title="[bold cyan]AI-Enhanced Options Ranking[/bold cyan]",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold magenta",
        expand=False,
    )

    cols = [
        ("Rank",     "right",  4,  True),
        ("Symbol",   "left",   6,  True),
        ("Type",     "center", 4,  True),
        ("Strike",   "right",  7,  True),
        ("Expiry",   "center", 10, True),
        ("DTE",      "right",  4,  True),
        ("Premium",  "right",  8,  True),
        ("IV%",      "right",  6,  True),
        ("PoP%",     "right",  5,  True),
        ("Tech",     "right",  5,  True),
        ("AI",       "right",  5,  True),
        ("Conf",     "right",  4,  True),
        ("Final",    "right",  5,  True),
        ("Cat",      "center", 5,  True),
        ("Div",      "center", 8,  True),
        ("Flags / Reasoning", "left", 35, False),
    ]
    for col, justify, min_w, no_wrap in cols:
        table.add_column(col, justify=justify, min_width=min_w, no_wrap=no_wrap)

    now = datetime.now(timezone.utc)

    for _, row in df.iterrows():
        final_pct = float(row.get("final_score", 0)) * 100
        ai_s      = float(row.get("ai_score", 50))
        tech_pct  = float(row.get("quality_score", 0)) * 100
        conf      = float(row.get("ai_confidence", 5))
        cat_risk  = str(row.get("catalyst_risk", "medium"))
        div_dir   = str(row.get("divergence_direction", "---"))
        is_div    = bool(row.get("divergence_flag", False))

        flags     = str(row.get("ai_flags", "") or "")
        reasoning = str(row.get("ai_reasoning", "") or "")

        if verbose_reasoning:
            detail = reasoning if reasoning else flags
        else:
            detail = flags or reasoning[:80]

        # Divergence note appended
        if is_div:
            detail = f"[DIVERGE:{div_dir}] " + detail

        dte = ""
        exp_dt = row.get("exp_dt")
        if exp_dt is not None and not (isinstance(exp_dt, float) and pd.isna(exp_dt)):
            try:
                dte = str(max(0, (exp_dt - now).days))
            except Exception:
                pass

        row_style = "bold" if is_div else ""
        div_text = Text(div_dir, style=_DIV_STYLE.get(div_dir, "white"))

        is_adjusted = bool(row.get("divergence_adjusted", False))
        rank_str = str(int(row.get("rank", 0))) + ("*" if is_adjusted else "")
        type_raw = str(row.get("type", "")).upper()
        if bool(row.get("_is_spread", False)):
            type_display = f"[SPREAD] {str(row.get('_spread_type', '')).upper()}"
        else:
            type_display = type_raw

        table.add_row(
            rank_str,
            str(row.get("symbol", "")),
            type_display,
            f"${float(row.get('strike', 0)):.1f}",
            str(row.get("expiration", ""))[:10],
            dte,
            f"${float(row.get('premium', 0)):.2f}",
            f"{float(row.get('impliedVolatility', 0)) * 100:.1f}%",
            f"{float(row.get('pop_sim', row.get('prob_profit', 0))) * 100:.0f}%",
            Text(f"{tech_pct:.1f}", style=_score_style(tech_pct)),
            Text(f"{ai_s:.1f}",    style=_score_style(ai_s)),
            f"{conf:.1f}",
            Text(f"{final_pct:.1f}", style=_score_style(final_pct)),
            Text(cat_risk[:3].upper(), style=_RISK_STYLE.get(cat_risk, "white")),
            div_text,
            detail,
            style=row_style,
        )

    console.print(table)

    # Show divergence summary
    if not divergent.empty:
        console.print()
        console.print("[bold yellow]Score Divergence Detected[/bold yellow] -- these picks scored very differently between AI and technical models:")
        for _, r in divergent.iterrows():
            ai_s   = float(r.get("ai_score", 50))
            tech_s = float(r.get("quality_score", 0)) * 100
            sym    = r.get("symbol", "")
            strike = r.get("strike", "")
            opt_t  = r.get("type", "")
            direction = r.get("divergence_direction", "")
            reasoning = str(r.get("ai_reasoning", ""))[:120]
            console.print(f"  [{direction}] {sym} {opt_t} ${strike} -- Tech: {tech_s:.1f} | AI: {ai_s:.1f} -- {reasoning}")
        console.print()


def _plain_table(df: pd.DataFrame) -> None:
    cols = ["rank", "symbol", "type", "strike", "expiration", "premium",
            "quality_score", "ai_score", "ai_confidence", "final_score",
            "catalyst_risk", "divergence_flag", "ai_reasoning"]
    available = [c for c in cols if c in df.columns]
    print("\n=== AI-Enhanced Options Ranking ===")
    print(df[available].to_string(index=False))
    print()


# ── Export helpers ─────────────────────────────────────────────────────────────

def to_csv(df: pd.DataFrame, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved ranked results -> {out}")


def to_json(df: pd.DataFrame, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    export = df.copy()
    for col in export.columns:
        if pd.api.types.is_datetime64_any_dtype(export[col]):
            export[col] = export[col].astype(str)
    export.to_json(out, orient="records", indent=2)
    print(f"Saved ranked results -> {out}")
