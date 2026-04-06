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
    technical_weight if technical_weight is not None else cfg["technical_weight"]

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

_RISK_STYLE    = {"low": "bold green", "medium": "bold yellow", "high": "bold red"}
_DIV_STYLE     = {"AI>TECH": "bold cyan", "TECH>AI": "bold magenta", "---": "dim white"}

_SCORE_BANDS: list[tuple[float, float, str]] = [
    (80, 101, "bold bright_green"),
    (60, 80,  "green"),
    (40, 60,  "yellow"),
    (20, 40,  "red"),
    (0,  20,  "bold red"),
]

_TYPE_STYLE = {"CALL": "bold bright_green", "PUT": "bold bright_red"}
_CONF_BANDS: list[tuple[float, float, str]] = [
    (8, 11,  "bold bright_green"),
    (6, 8,   "green"),
    (4, 6,   "yellow"),
    (0, 4,   "dim red"),
]


def _score_style(score: float) -> str:
    for lo, hi, style in _SCORE_BANDS:
        if lo <= score < hi:
            return style
    return "white"


def _conf_style(conf: float) -> str:
    for lo, hi, style in _CONF_BANDS:
        if lo <= conf < hi:
            return style
    return "white"


def _dte_style(dte_val: int) -> str:
    if dte_val <= 7:
        return "bold red"
    elif dte_val <= 14:
        return "yellow"
    elif dte_val <= 30:
        return "green"
    return "dim white"


def _pop_style(pop_pct: float) -> str:
    if pop_pct >= 65:
        return "bold bright_green"
    elif pop_pct >= 50:
        return "green"
    elif pop_pct >= 40:
        return "yellow"
    return "red"


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
    from rich.panel import Panel

    if "divergence_flag" in df.columns:
        try:
            divergent = df[df["divergence_flag"].fillna(False).astype(bool)]
        except Exception:
            divergent = pd.DataFrame()
    else:
        divergent = pd.DataFrame()

    # ── Summary header panel ──────────────────────────────────────────────
    n_picks = len(df)
    avg_final = df["final_score"].mean() * 100 if "final_score" in df.columns else 0
    avg_ai = df["ai_score"].mean() if "ai_score" in df.columns else 0
    avg_tech = df["quality_score"].mean() * 100 if "quality_score" in df.columns else 0
    n_diverge = len(divergent)
    top_sym = str(df.iloc[0]["symbol"]) if (not df.empty and "symbol" in df.columns) else "?"
    top_final = float(df.iloc[0]["final_score"]) * 100 if (not df.empty and "final_score" in df.columns) else 0

    summary_parts = [
        f"[bold bright_cyan]{n_picks}[/bold bright_cyan] picks ranked",
        f"Avg Final: [bold]{avg_final:.1f}[/bold]",
        f"Avg Tech: [green]{avg_tech:.1f}[/green]",
        f"Avg AI: [cyan]{avg_ai:.1f}[/cyan]",
    ]
    if n_diverge > 0:
        summary_parts.append(f"[bold yellow]{n_diverge} divergence(s)[/bold yellow]")
    summary_parts.append(f"Top: [bold bright_green]{top_sym} ({top_final:.1f}%)[/bold bright_green]")

    console.print()
    console.print(Panel(
        "  ".join(summary_parts),
        title="[bold bright_cyan]AI-Enhanced Options Ranking[/bold bright_cyan]",
        subtitle="[dim]Tech + AI composite scoring with divergence detection[/dim]",
        border_style="bright_cyan",
        padding=(0, 2),
    ))

    # ── Main table ────────────────────────────────────────────────────────
    table = Table(
        box=box.HEAVY_HEAD,
        show_lines=False,
        row_styles=["", "on grey7"],
        header_style="bold bright_magenta on grey11",
        border_style="bright_blue",
        expand=False,
        padding=(0, 1),
    )

    cols = [
        ("#",        "right",  3,  True),
        ("Symbol",   "left",   6,  True),
        ("Type",     "center", 4,  True),
        ("Strike",   "right",  7,  True),
        ("Expiry",   "center", 10, True),
        ("DTE",      "right",  3,  True),
        ("Prem",     "right",  7,  True),
        ("IV%",      "right",  5,  True),
        ("PoP",      "right",  4,  True),
        ("Tech",     "right",  4,  True),
        ("AI",       "right",  4,  True),
        ("Conf",     "right",  3,  True),
        ("Final",    "right",  5,  True),
        ("Cat",      "center", 4,  True),
        ("Div",      "center", 7,  True),
        ("Reasoning", "left",  40, False),
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

        # DTE calculation
        dte_val = 0
        dte_str = ""
        exp_dt = row.get("exp_dt")
        if exp_dt is not None and not (isinstance(exp_dt, float) and pd.isna(exp_dt)):
            try:
                dte_val = max(0, (exp_dt - now).days)
                dte_str = str(dte_val)
            except Exception:
                pass
        if not dte_str and "T_years" in row.index:
            try:
                dte_val = max(0, int(float(row["T_years"]) * 365))
                dte_str = str(dte_val)
            except Exception:
                pass

        is_adjusted = bool(row.get("divergence_adjusted", False))
        rank_num = int(row.get("rank", 0))
        rank_style = "bold bright_yellow" if rank_num <= 3 else ("white" if rank_num <= 7 else "dim")
        rank_str = Text(f"{rank_num}" + ("*" if is_adjusted else ""), style=rank_style)

        type_raw = str(row.get("type", "")).upper()
        if bool(row.get("_is_spread", False)):
            type_display = Text(str(row.get("_spread_type", "SPR")).upper(), style="bold bright_blue")
        else:
            type_display = Text(type_raw, style=_TYPE_STYLE.get(type_raw, "white"))

        sym_text = Text(str(row.get("symbol", "")), style="bold bright_white")

        # Premium coloring
        prem_val = float(row.get("premium", 0))
        prem_text = Text(f"${prem_val:.2f}", style="bright_white")

        # IV coloring
        iv_val = float(row.get("impliedVolatility", 0)) * 100
        iv_style = "bright_red" if iv_val > 60 else ("yellow" if iv_val > 35 else "green")
        iv_text = Text(f"{iv_val:.0f}%", style=iv_style)

        # PoP coloring
        pop_val = float(row.get("pop_sim", row.get("prob_profit", 0))) * 100
        pop_text = Text(f"{pop_val:.0f}%", style=_pop_style(pop_val))

        # Divergence styling
        if is_div:
            div_text = Text(div_dir, style=_DIV_STYLE.get(div_dir, "white"))
            detail_text = Text(detail, style="bold")
        else:
            div_text = Text("---", style="dim")
            detail_text = Text(detail, style="dim white" if not detail else "white")

        # Catalyst risk with icon
        cat_icons = {"low": "LOW", "medium": "MED", "high": "HI!"}
        cat_display = cat_icons.get(cat_risk, cat_risk[:3].upper())

        table.add_row(
            rank_str,
            sym_text,
            type_display,
            f"${float(row.get('strike', 0)):.1f}",
            str(row.get("expiration", ""))[:10],
            Text(dte_str, style=_dte_style(dte_val)),
            prem_text,
            iv_text,
            pop_text,
            Text(f"{tech_pct:.0f}", style=_score_style(tech_pct)),
            Text(f"{ai_s:.0f}",    style=_score_style(ai_s)),
            Text(f"{conf:.0f}", style=_conf_style(conf)),
            Text(f"{final_pct:.0f}%", style=_score_style(final_pct)),
            Text(cat_display, style=_RISK_STYLE.get(cat_risk, "white")),
            div_text,
            detail_text,
        )

    console.print(table)

    # ── Divergence detail panel ───────────────────────────────────────────
    if not divergent.empty:
        console.print()
        div_lines = []
        for _, r in divergent.iterrows():
            ai_s   = float(r.get("ai_score", 50))
            tech_s = float(r.get("quality_score", 0)) * 100
            sym    = r.get("symbol", "")
            strike = r.get("strike", "")
            opt_t  = str(r.get("type", "")).upper()
            direction = r.get("divergence_direction", "")
            reason = str(r.get("ai_reasoning", ""))[:100]
            dir_color = "cyan" if direction == "AI>TECH" else "magenta"
            div_lines.append(
                f"  [{dir_color}]{direction}[/{dir_color}] "
                f"[bold]{sym}[/bold] {opt_t} ${strike} "
                f"-- Tech: [green]{tech_s:.0f}[/green] | AI: [cyan]{ai_s:.0f}[/cyan] "
                f"-- [dim]{reason}[/dim]"
            )
        console.print(Panel(
            "\n".join(div_lines),
            title="[bold yellow]Score Divergence Detected[/bold yellow]",
            border_style="yellow",
            padding=(0, 1),
        ))


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
