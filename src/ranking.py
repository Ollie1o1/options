"""Combine screener quality scores with AI scores to produce a final ranked list.

Usage:
    from src.ranking import combine_scores, print_ranked_table, to_csv, to_json

    ranked = combine_scores(picks_df, ai_df)
    print_ranked_table(ranked)
    to_csv(ranked, "output/ranked.csv")
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config_ai import AI_CONFIG

# ── Score combination ──────────────────────────────────────────────────────────


def combine_scores(
    picks_df: pd.DataFrame,
    ai_df: pd.DataFrame,
    ai_weight: Optional[float] = None,
    technical_weight: Optional[float] = None,
) -> pd.DataFrame:
    """Merge AI results into *picks_df* and compute a weighted *final_score*.

    final_score = technical_weight * quality_score
                + ai_weight        * (ai_score / 100)

    Parameters
    ----------
    picks_df:
        Screener output DataFrame (must contain ``quality_score``).
    ai_df:
        AI scorer output returned by ``AIScorer.score_candidates()``.
    ai_weight:
        Override ``AI_CONFIG["ai_weight"]``.  Pass together with
        *technical_weight* to keep the two weights summing to 1.
    technical_weight:
        Override ``AI_CONFIG["technical_weight"]``.
    """
    aw = ai_weight if ai_weight is not None else AI_CONFIG["ai_weight"]
    tw = technical_weight if technical_weight is not None else AI_CONFIG["technical_weight"]

    df = picks_df.copy()
    ai_cols = ["ai_score", "ai_reasoning", "ai_flags", "catalyst_risk", "iv_justified"]
    present = [c for c in ai_cols if c in ai_df.columns]
    df = df.join(ai_df[present])

    # Ensure columns exist even if ai_df was partial
    if "ai_score" not in df.columns:
        df["ai_score"] = 50.0
    if "ai_reasoning" not in df.columns:
        df["ai_reasoning"] = ""
    if "ai_flags" not in df.columns:
        df["ai_flags"] = ""
    if "catalyst_risk" not in df.columns:
        df["catalyst_risk"] = "medium"
    if "iv_justified" not in df.columns:
        df["iv_justified"] = True

    tech = df["quality_score"].fillna(0).clip(0, 1)
    ai_norm = df["ai_score"].fillna(50).clip(0, 100) / 100.0

    df["final_score"] = (tw * tech + aw * ai_norm).round(4)
    df["rank"] = (
        df["final_score"].rank(ascending=False, method="min").astype(int)
    )

    return df.sort_values("rank")


# ── Rich table output ──────────────────────────────────────────────────────────

_RISK_STYLE = {"low": "green", "medium": "yellow", "high": "red"}

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


def print_ranked_table(df: pd.DataFrame, top_n: int = 20) -> None:
    """Print a ranked options table.

    Uses *rich* for a coloured table if available; falls back to plain text.
    """
    subset = df.head(top_n)
    try:
        from rich.console import Console  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415
        from rich import box  # noqa: PLC0415
        from rich.text import Text  # noqa: PLC0415

        console = Console()
        table = Table(
            title="[bold cyan]AI-Enhanced Options Ranking[/bold cyan]",
            box=box.ROUNDED,
            show_lines=True,
            header_style="bold magenta",
        )

        for col, justify in [
            ("Rank",    "right"),
            ("Symbol",  "left"),
            ("Type",    "center"),
            ("Strike",  "right"),
            ("Expiry",  "center"),
            ("DTE",     "right"),
            ("Premium", "right"),
            ("IV%",     "right"),
            ("PoP%",    "right"),
            ("Tech",    "right"),
            ("AI",      "right"),
            ("Final",   "right"),
            ("Catalyst","center"),
            ("Flags / Reasoning", "left"),
        ]:
            table.add_column(col, justify=justify)

        now = datetime.now(timezone.utc)

        for _, row in subset.iterrows():
            final_pct  = float(row.get("final_score", 0)) * 100
            ai_s       = float(row.get("ai_score", 50))
            tech_pct   = float(row.get("quality_score", 0)) * 100
            cat_risk   = str(row.get("catalyst_risk", "medium"))

            flags     = str(row.get("ai_flags", "") or "")
            reasoning = str(row.get("ai_reasoning", "") or "")
            detail    = flags or reasoning
            if flags and reasoning:
                detail = f"{flags} | {reasoning}"

            dte = ""
            exp_dt = row.get("exp_dt")
            if exp_dt is not None and not (
                isinstance(exp_dt, float) and pd.isna(exp_dt)
            ):
                try:
                    dte = str(max(0, (exp_dt - now).days))
                except Exception:
                    pass

            table.add_row(
                str(int(row.get("rank", 0))),
                str(row.get("symbol", "")),
                str(row.get("type", "")).upper(),
                f"${float(row.get('strike', 0)):.1f}",
                str(row.get("expiration", ""))[:10],
                dte,
                f"${float(row.get('premium', 0)):.2f}",
                f"{float(row.get('impliedVolatility', 0)) * 100:.1f}%",
                f"{float(row.get('pop_sim', row.get('prob_profit', 0))) * 100:.0f}%",
                Text(f"{tech_pct:.1f}", style=_score_style(tech_pct)),
                Text(f"{ai_s:.1f}",    style=_score_style(ai_s)),
                Text(f"{final_pct:.1f}", style=_score_style(final_pct)),
                Text(cat_risk.upper(), style=_RISK_STYLE.get(cat_risk, "white")),
                detail[:90],
            )

        console.print(table)

    except ImportError:
        _plain_table(subset)


def _plain_table(df: pd.DataFrame) -> None:
    """Fallback plain-text rendering when *rich* is not installed."""
    display_cols = [
        "rank", "symbol", "type", "strike", "expiration",
        "premium", "quality_score", "ai_score", "final_score",
        "catalyst_risk", "ai_reasoning",
    ]
    cols = [c for c in display_cols if c in df.columns]
    print("\n=== AI-Enhanced Options Ranking ===")
    print(df[cols].to_string(index=False))
    print()


# ── Export helpers ─────────────────────────────────────────────────────────────


def to_csv(df: pd.DataFrame, path: str) -> None:
    """Write *df* to a CSV file, creating parent directories as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved ranked results → {out}")


def to_json(df: pd.DataFrame, path: str) -> None:
    """Write *df* to a JSON file (records orientation)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    export = df.copy()
    # Stringify datetime columns so they serialise cleanly
    for col in export.columns:
        if pd.api.types.is_datetime64_any_dtype(export[col]):
            export[col] = export[col].astype(str)
    export.to_json(out, orient="records", indent=2)
    print(f"Saved ranked results → {out}")
