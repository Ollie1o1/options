"""AI-powered options candidate scorer using the Claude API (Anthropic).

Takes a DataFrame of screener candidates and returns a parallel DataFrame
of AI-generated scores, reasoning, and flags.

Usage:
    from src.ai_scorer import AIScorer

    scorer = AIScorer()
    ai_df = scorer.score_candidates(picks_df)
    # ai_df columns: ai_score, ai_reasoning, ai_flags, catalyst_risk, iv_justified
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import pandas as pd

from src.config_ai import AI_CONFIG

logger = logging.getLogger(__name__)

# ── Tool schema for structured JSON output ────────────────────────────────────

_SCORING_TOOL: dict[str, Any] = {
    "name": "score_options_batch",
    "description": (
        "Score a batch of options candidates and return structured analysis "
        "for each one."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Candidate ID echoed from the request.",
                        },
                        "ai_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Overall trade quality score 0–100.",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "1–2 sentence analysis of the setup: "
                                "IV edge, catalyst risk, trend alignment, risk/reward."
                            ),
                        },
                        "flags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Short uppercase tags for key factors, e.g. "
                                "'IV_CHEAP', 'EARNINGS_NEAR', 'TREND_ALIGNED', "
                                "'HIGH_SPREAD', 'DECAY_RISK'."
                            ),
                        },
                        "catalyst_risk": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": (
                                "Likelihood of a gap-inducing event before expiry."
                            ),
                        },
                        "iv_justified": {
                            "type": "boolean",
                            "description": (
                                "True if current IV is appropriate given "
                                "upcoming catalysts."
                            ),
                        },
                    },
                    "required": [
                        "id",
                        "ai_score",
                        "reasoning",
                        "flags",
                        "catalyst_risk",
                        "iv_justified",
                    ],
                },
            }
        },
        "required": ["scores"],
    },
}

_SYSTEM_PROMPT = """\
You are an expert options trader and quantitative analyst. You receive batches of
options candidates produced by a technical screener and must evaluate each trade setup.

Scoring guidelines (0–100):
  80–100  Exceptional — strong edge, favorable IV/HV, clean catalyst-free runway
  60– 79  Good — solid thesis, manageable risk, minor concerns
  40– 59  Average — some merit but notable risks or mixed signals
  20– 39  Weak — unfavorable conditions, elevated risk, poor risk/reward
   0– 19  Avoid — multiple red flags, high probability of loss

For each candidate weigh:
1. IV vs HV: positive iv_vs_hv = IV rich (seller edge); negative = IV cheap (buyer edge).
2. Catalyst/earnings risk: does the expiry straddle a known event that could gap the stock?
3. Trend & momentum: does the option direction agree with RSI, ret_5d, Trend_Aligned?
4. Probability of profit vs premium: is the reward worth the defined risk?
5. Screener warnings: macro_warning, sr_warning, decay_warning signal elevated risk.

Keep reasoning concise (1–2 sentences). Use short uppercase strings for flags."""


# ── Public class ───────────────────────────────────────────────────────────────


class AIScorer:
    """Score options candidates using the Claude API.

    Parameters
    ----------
    config:
        Dict of overrides applied on top of ``AI_CONFIG``.  Useful for
        changing weights, model, or batch size at call time without editing
        the config file.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = {**AI_CONFIG, **(config or {})}
        self._client = None  # lazy-initialised on first use

    # ── Public API ─────────────────────────────────────────────────────────────

    def score_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score all rows in *df* and return a parallel DataFrame of AI results.

        Returned DataFrame (same index as *df*) columns:
            ai_score        float  0–100
            ai_reasoning    str    1–2 sentence analysis
            ai_flags        str    comma-separated flag tags
            catalyst_risk   str    "low" | "medium" | "high"
            iv_justified    bool
        """
        if df.empty:
            return _empty_results(df)

        candidates = self._extract_candidates(df)
        all_results: list[dict] = []

        batch_size = self.config["batch_size"]
        batches = [
            candidates[i : i + batch_size]
            for i in range(0, len(candidates), batch_size)
        ]

        for i, batch in enumerate(batches):
            logger.info(
                "Scoring batch %d/%d (%d candidates)…", i + 1, len(batches), len(batch)
            )
            try:
                results = self._score_batch(batch)
                all_results.extend(results)
            except Exception as exc:
                logger.warning("Batch %d failed: %s — using neutral defaults", i + 1, exc)
                for c in batch:
                    all_results.append(_default_result(c["_id"]))

            # Brief pause between batches to respect rate limits
            if i < len(batches) - 1:
                time.sleep(0.5)

        return self._results_to_df(df, all_results)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_client(self):
        """Lazy-initialise the Anthropic client."""
        if self._client is None:
            try:
                import anthropic  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "The 'anthropic' package is not installed. "
                    "Run:  pip install anthropic"
                ) from exc

            api_key = os.environ.get(self.config["api_key_env"])
            if not api_key:
                raise EnvironmentError(
                    f"API key not found. Set the {self.config['api_key_env']!r} "
                    "environment variable before running the AI scorer.\n"
                    "Example:  export ANTHROPIC_API_KEY=sk-ant-..."
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def _extract_candidates(self, df: pd.DataFrame) -> list[dict]:
        """Return a list of dicts with only the configured fields + an _id."""
        wanted = [f for f in self.config["fields_to_include"] if f in df.columns]
        candidates: list[dict] = []
        for idx, row in df.iterrows():
            c: dict[str, Any] = {"_id": str(idx)}
            for field in wanted:
                val = row[field]
                # Convert pandas NA / NaT to None for clean serialisation
                if pd.isna(val) if not isinstance(val, (list, dict)) else False:
                    c[field] = None
                else:
                    c[field] = val
            candidates.append(c)
        return candidates

    def _score_batch(self, batch: list[dict]) -> list[dict]:
        """Make one API call for *batch* and return the list of scored dicts."""
        client = self._get_client()
        prompt = self._build_prompt(batch)

        response = client.messages.create(
            model=self.config["model"],
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"],
            system=_SYSTEM_PROMPT,
            tools=[_SCORING_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": prompt}],
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "score_options_batch":
                return block.input.get("scores", [])

        raise ValueError("API response did not contain the expected tool_use block.")

    def _build_prompt(self, batch: list[dict]) -> str:
        blocks = [_format_candidate(c) for c in batch]
        body = "\n\n".join(blocks)
        return (
            f"Please score the following {len(batch)} option candidate(s).\n\n"
            f"{body}\n\n"
            "Call the score_options_batch tool with your analysis."
        )

    def _results_to_df(
        self, original_df: pd.DataFrame, results: list[dict]
    ) -> pd.DataFrame:
        """Align scored results back to the original DataFrame index."""
        score_map = {r["id"]: r for r in results if "id" in r}
        rows = []
        for idx in original_df.index:
            r = score_map.get(str(idx), _default_result(str(idx)))
            rows.append(
                {
                    "ai_score": float(r.get("ai_score", 50)),
                    "ai_reasoning": str(r.get("reasoning", "")),
                    "ai_flags": ", ".join(r.get("flags") or []),
                    "catalyst_risk": str(r.get("catalyst_risk", "medium")),
                    "iv_justified": bool(r.get("iv_justified", True)),
                }
            )
        return pd.DataFrame(rows, index=original_df.index)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _format_candidate(c: dict[str, Any]) -> str:
    """Format one candidate dict as a readable text block for the prompt."""
    lines = [f"Candidate ID: {c['_id']}"]
    for k, v in c.items():
        if k == "_id":
            continue
        if v is None:
            lines.append(f"  {k}: N/A")
        elif isinstance(v, float):
            lines.append(f"  {k}: {v:.4f}")
        else:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _default_result(id_: str) -> dict:
    return {
        "id": id_,
        "ai_score": 50.0,
        "reasoning": "Scoring unavailable — using neutral default.",
        "flags": [],
        "catalyst_risk": "medium",
        "iv_justified": True,
    }


def _empty_results(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ai_score": pd.Series(dtype=float),
            "ai_reasoning": pd.Series(dtype=str),
            "ai_flags": pd.Series(dtype=str),
            "catalyst_risk": pd.Series(dtype=str),
            "iv_justified": pd.Series(dtype=bool),
        },
        index=df.index,
    )
