"""AI-powered options candidate scorer.

Supports two providers configured in src/config_ai.py:
  - "openrouter"  — OpenAI-compatible API at openrouter.ai (free models available)
  - "anthropic"   — Anthropic Claude API

API keys are loaded from a .env file in the project root (never hard-coded).

Usage:
    from src.ai_scorer import AIScorer

    scorer = AIScorer()
    ai_df = scorer.score_candidates(picks_df)
    # ai_df columns: ai_score, ai_reasoning, ai_flags, catalyst_risk, iv_justified
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.config_ai import AI_CONFIG

logger = logging.getLogger(__name__)

# ── Load .env automatically ────────────────────────────────────────────────────
# Tries python-dotenv if installed; silently skips if not.
def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # noqa: PLC0415
        env_path = Path(__file__).resolve().parent.parent / ".env"
        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        pass

_load_dotenv()


# ── Prompts ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert options trader and quantitative analyst. You receive batches of
options candidates produced by a technical screener and must evaluate each trade setup.

Scoring guidelines (0-100):
  80-100  Exceptional - strong edge, favorable IV/HV, clean catalyst-free runway
  60-79   Good - solid thesis, manageable risk, minor concerns
  40-59   Average - some merit but notable risks or mixed signals
  20-39   Weak - unfavorable conditions, elevated risk, poor risk/reward
  0-19    Avoid - multiple red flags, high probability of loss

For each candidate weigh:
1. IV vs HV: positive iv_vs_hv = IV rich (seller edge); negative = IV cheap (buyer edge).
2. Catalyst/earnings risk: does the expiry straddle a known event?
3. Trend & momentum: does the option direction agree with RSI, ret_5d, Trend_Aligned?
4. Probability of profit vs premium: is the reward worth the defined risk?
5. Screener warnings: macro_warning, sr_warning, decay_warning signal elevated risk.

Keep reasoning concise (1-2 sentences). Use short uppercase strings for flags
(e.g. IV_CHEAP, EARNINGS_NEAR, TREND_ALIGNED, HIGH_SPREAD, DECAY_RISK)."""

_JSON_SCHEMA_DESC = """\
Respond with a JSON object in exactly this format and nothing else:
{
  "scores": [
    {
      "id": "<candidate id string>",
      "ai_score": <number 0-100>,
      "reasoning": "<1-2 sentence analysis>",
      "flags": ["FLAG1", "FLAG2"],
      "catalyst_risk": "<low|medium|high>",
      "iv_justified": <true|false>
    }
  ]
}"""

# Tool schema used by the Anthropic provider
_SCORING_TOOL: dict[str, Any] = {
    "name": "score_options_batch",
    "description": "Score a batch of options candidates and return structured analysis.",
    "input_schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "ai_score": {"type": "number", "minimum": 0, "maximum": 100},
                        "reasoning": {"type": "string"},
                        "flags": {"type": "array", "items": {"type": "string"}},
                        "catalyst_risk": {"type": "string", "enum": ["low", "medium", "high"]},
                        "iv_justified": {"type": "boolean"},
                    },
                    "required": ["id", "ai_score", "reasoning", "flags", "catalyst_risk", "iv_justified"],
                },
            }
        },
        "required": ["scores"],
    },
}


# ── Public class ───────────────────────────────────────────────────────────────


class AIScorer:
    """Score options candidates using a configurable AI provider.

    Parameters
    ----------
    config:
        Dict of overrides applied on top of ``AI_CONFIG``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = {**AI_CONFIG, **(config or {})}
        self._client = None  # lazy-initialised on first use

    # ── Public API ─────────────────────────────────────────────────────────────

    def score_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score all rows in *df* and return a parallel DataFrame of AI results.

        Returned DataFrame (same index as *df*) columns:
            ai_score        float  0-100
            ai_reasoning    str    1-2 sentence analysis
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
                "Scoring batch %d/%d (%d candidates)...", i + 1, len(batches), len(batch)
            )
            results = self._score_batch_with_retry(batch, batch_num=i + 1)
            all_results.extend(results)

            if i < len(batches) - 1:
                time.sleep(0.5)

        return self._results_to_df(df, all_results)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_client(self):
        """Lazy-initialise the API client based on configured provider."""
        if self._client is not None:
            return self._client

        provider = self.config.get("provider", "openrouter")
        api_key = os.environ.get(self.config["api_key_env"])

        if not api_key:
            raise EnvironmentError(
                f"API key not found. Set {self.config['api_key_env']!r} in your .env file "
                "or as an environment variable.\n"
                "See .env.example for the format."
            )

        if provider == "anthropic":
            try:
                import anthropic  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError("Run:  pip install anthropic") from exc
            self._client = anthropic.Anthropic(api_key=api_key)

        else:
            # openrouter (and any other OpenAI-compatible endpoint)
            try:
                import openai  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError("Run:  pip install openai") from exc
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/Ollie1o1/options",
                    "X-Title": "Options Screener AI Ranking",
                },
            )

        return self._client

    def _score_batch_with_retry(self, batch: list[dict], batch_num: int = 1) -> list[dict]:
        """Call _score_batch with exponential backoff on 429 rate-limit errors."""
        max_retries = 4
        delay = 5  # seconds, doubles each attempt
        for attempt in range(1, max_retries + 1):
            try:
                return self._score_batch(batch)
            except Exception as exc:
                is_rate_limit = "429" in str(exc) or "rate" in str(exc).lower()
                if is_rate_limit and attempt < max_retries:
                    print(f"  [ai_scorer] batch {batch_num} rate-limited, retrying in {delay}s... (attempt {attempt}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.warning("Batch %d failed after %d attempt(s): %s", batch_num, attempt, exc)
                    if not is_rate_limit:
                        print(f"  [ai_scorer] batch {batch_num} error: {exc}")
                    return [_default_result(c["_id"]) for c in batch]
        return [_default_result(c["_id"]) for c in batch]

    def _score_batch(self, batch: list[dict]) -> list[dict]:
        """Make one API call and return list of scored dicts."""
        provider = self.config.get("provider", "openrouter")
        if provider == "anthropic":
            return self._score_batch_anthropic(batch)
        return self._score_batch_openai(batch)

    def _score_batch_openai(self, batch: list[dict]) -> list[dict]:
        """OpenAI-compatible call (used for OpenRouter)."""
        client = self._get_client()
        prompt = self._build_prompt(batch, include_schema=True)

        response = client.chat.completions.create(
            model=self.config["model"],
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"],
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )

        raw = response.choices[0].message.content or ""
        return _parse_json_response(raw)

    def _score_batch_anthropic(self, batch: list[dict]) -> list[dict]:
        """Anthropic tool_use call."""
        client = self._get_client()
        prompt = self._build_prompt(batch, include_schema=False)

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

        raise ValueError("Anthropic response did not contain the expected tool_use block.")

    def _build_prompt(self, batch: list[dict], include_schema: bool = True) -> str:
        blocks = [_format_candidate(c) for c in batch]
        body = "\n\n".join(blocks)
        schema_instruction = f"\n\n{_JSON_SCHEMA_DESC}" if include_schema else \
            "\n\nCall the score_options_batch tool with your analysis."
        return (
            f"Please score the following {len(batch)} option candidate(s).\n\n"
            f"{body}"
            f"{schema_instruction}"
        )

    def _extract_candidates(self, df: pd.DataFrame) -> list[dict]:
        wanted = [f for f in self.config["fields_to_include"] if f in df.columns]
        candidates: list[dict] = []
        for idx, row in df.iterrows():
            c: dict[str, Any] = {"_id": str(idx)}
            for field in wanted:
                val = row[field]
                try:
                    is_na = pd.isna(val)
                except (TypeError, ValueError):
                    is_na = False
                c[field] = None if is_na else val
            candidates.append(c)
        return candidates

    def _results_to_df(
        self, original_df: pd.DataFrame, results: list[dict]
    ) -> pd.DataFrame:
        score_map = {r["id"]: r for r in results if "id" in r}
        rows = []
        for idx in original_df.index:
            r = score_map.get(str(idx), _default_result(str(idx)))
            rows.append(
                {
                    "ai_score":     float(r.get("ai_score", 50)),
                    "ai_reasoning": str(r.get("reasoning", "")),
                    "ai_flags":     ", ".join(r.get("flags") or []),
                    "catalyst_risk": str(r.get("catalyst_risk", "medium")),
                    "iv_justified": bool(r.get("iv_justified", True)),
                }
            )
        return pd.DataFrame(rows, index=original_df.index)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _format_candidate(c: dict[str, Any]) -> str:
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


def _parse_json_response(raw: str) -> list[dict]:
    """Extract the scores list from a raw JSON string response."""
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            l for l in lines if not l.startswith("```")
        ).strip()

    # Find the outermost { } block
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response: {raw[:200]}")

    parsed = json.loads(text[start:end])
    scores = parsed.get("scores", [])
    if not isinstance(scores, list):
        raise ValueError(f"Expected 'scores' list, got: {type(scores)}")
    return scores


def _default_result(id_: str) -> dict:
    return {
        "id": id_,
        "ai_score": 50.0,
        "reasoning": "Scoring unavailable - using neutral default.",
        "flags": [],
        "catalyst_risk": "medium",
        "iv_justified": True,
    }


def _empty_results(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ai_score":     pd.Series(dtype=float),
            "ai_reasoning": pd.Series(dtype=str),
            "ai_flags":     pd.Series(dtype=str),
            "catalyst_risk": pd.Series(dtype=str),
            "iv_justified": pd.Series(dtype=bool),
        },
        index=df.index,
    )
