"""Pipeline: fetch -> aggregate -> enrich -> (cache) -> synth -> render.

Cache hit skips the AI call and the history persist. All external seams
(fetch_fn, scorer, db paths) are injectable for tests.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional

from src.macro_pulse import cache as _cache
from src.macro_pulse import context as _ctx
from src.macro_pulse import panel as _panel
from src.macro_pulse import synth as _synth


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def run(*, fetch_fn: Optional[Callable[[], list]] = None, scorer=None,
        db_path: Optional[str] = None, cache_db: Optional[str] = None,
        persist: bool = True, today: Optional[str] = None) -> str:
    from src.worldnews import scoring as _scoring, sources as _sources

    fetch_fn = fetch_fn or _sources.fetch_all
    db_path = db_path or _ctx.default_db_path()
    cache_db = cache_db or _cache.default_db_path()
    today = today or _today_str()

    try:
        items = fetch_fn() or []
    except Exception:
        items = []
    agg = _scoring.aggregate(items)
    ctx = _ctx.enrich(agg, db_path=db_path)

    titles = [t.get("title", "") for t in agg.get("top", [])]
    key = _cache.bundle_key(titles, today)
    cached = _cache.get(key, cache_db)
    if cached is not None:
        _hydrate(ctx, cached)
        return _panel.render(ctx)

    ctx = _synth.narrate(ctx, scorer=scorer)
    _cache.put(key, _dump(ctx), cache_db)
    if persist:
        theme_scores = {t.theme: t.score for t in ctx.themes}
        _ctx.persist_reading(ctx.pulse, theme_scores, db_path)
    return _panel.render(ctx)


def _dump(ctx) -> dict:
    return {"headline": ctx.headline, "what_would_flip": ctx.what_would_flip,
            "narrative_source": ctx.narrative_source,
            "reads": {t.theme: t.read for t in ctx.themes}}


def _hydrate(ctx, payload: dict) -> None:
    ctx.headline = payload.get("headline", "")
    ctx.what_would_flip = payload.get("what_would_flip", "")
    ctx.narrative_source = payload.get("narrative_source", "deterministic")
    reads = payload.get("reads", {})
    for t in ctx.themes:
        t.read = reads.get(t.theme, t.read)
