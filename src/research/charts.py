"""Compatibility shim. The research chart primitives moved to desk_kit.charts
(the desk-wide SVG library); this module re-exports them so older imports and
tests keep working. New code should import src.desk_kit.charts directly."""
from src.desk_kit.charts import (  # noqa: F401
    area_chart, cone_chart, hbar_diverging, price_chart, rsi_strip,
    term_chart,
)
