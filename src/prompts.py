"""AI prompt strings for the options scorer.

Centralises all prompt text so A/B testing and edits don't require
touching scorer logic.
"""


def scoring_system_prompt() -> str:
    return """\
Score 0-100: 80+=multi-signal+low catalyst risk; 60-79=good+manageable; 40-59=neutral/mixed; <40=avoid.
Weight in order: (1)IV justified vs upcoming events+realized vol (2)Catalyst timing vs expiry window \
(3)Trend/momentum alignment with trade direction (4)Breakeven realism vs 1\u03c3 expected move.
reasoning: ONE concrete sentence \u2014 primary reason only, not a list.
Flag iv_justified:false when iv_rank>0.70 AND no earnings within 21d of expiry.
confidence:0-10. flags:\u22643 SHORT_CAPS."""


def ticker_context_prompt() -> str:
    return """\
Analyze ticker conditions. Flag term_structure:"BACKWARDATION" when front-month IV > back-month IV \
\u2014 this shifts edge: sellers edge in contango, buyers edge in backwardation.
JSON only: {"regime":"SELLER_EDGE|BUYER_EDGE|NEUTRAL","catalyst_risk":"low|medium|high",\
"directional_bias":"bullish|bearish|neutral","term_structure":"CONTANGO|BACKWARDATION|FLAT",\
"key_risks":["r1"],"summary":"\u226420 words","confidence":1-10}"""


def json_schema_instruction() -> str:
    return 'JSON only: {"scores":[{"id":"<id>","ai_score":0-100,"reasoning":"\u226412 words","flags":["F"],"catalyst_risk":"low|medium|high","iv_justified":true,"ai_confidence":0-10}]}'
