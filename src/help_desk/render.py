"""Blocks to styled lines.

Pure: same blocks in, same lines out. No I/O, no terminal probing, no clock —
that is what makes a 400-line manual testable at all. Paging is the menu's job
precisely because paging needs the terminal and this does not.

Every colour goes through ``fmt.style`` with a semantic name, never a raw
``Colors`` constant, so the manual re-themes with the rest of the desk.
"""
from __future__ import annotations

import textwrap
from typing import List, Optional, Sequence

from .. import formatting as fmt
from .. import ui

WIDTH = 74
INDENT = "  "
BODY_W = WIDTH - 4

# Definition-list gutter. Wider than ui.LABEL_W because the glossary defines
# field names, not metrics: `entry_price_fill` is 16 characters.
LABEL_W = 17
LABEL_GUTTER = len(INDENT) + LABEL_W + 1

# Tag -> number of elements in the block tuple. The content module is data, so
# this table is the only schema it has; the tests validate every chapter
# against it.
BLOCK_ARITY = {
    "h": 2, "p": 2, "bullet": 2, "num": 2, "kv": 3,
    "table": 3, "callout": 3, "code": 2, "gap": 1, "rule": 1,
}

# A callout's tone picks its border label. Three tones, three meanings: a thing
# worth knowing, a thing that will cost you money, a thing you must not do.
_CALLOUT_TITLE = {"good": "NOTE", "warn": "CAUTION", "bad": "STOP"}


def _wrap(text, width: int) -> List[str]:
    """Wrap, preserving deliberate line breaks and never dropping blank ones."""
    out: List[str] = []
    for para in str(text).split("\n"):
        out.extend(textwrap.wrap(para, width) or [""])
    return out


def _hanging(marker: str, text, width: int) -> List[str]:
    """A marked line whose continuations align under its first word."""
    pad = " " * len(marker)
    wrapped = _wrap(text, width - len(marker))
    return ([INDENT + marker + wrapped[0]]
            + [INDENT + pad + w for w in wrapped[1:]])


def _kv(label, value) -> List[str]:
    """A definition row: dim label in a fixed gutter, wrapped value beside it.

    Deliberately not ui.kv_line. That gutter is 11, sized for data rows like
    "delta" and "spread"; a glossary also has to hold `ev_per_contract` and
    `entry_price_fill`, and it does not wrap at all. A label wider than the
    gutter takes its own line rather than pushing the column out of true.
    """
    label, value = str(label), str(value)
    body_w = WIDTH - LABEL_GUTTER
    styled_label = fmt.style(label, "label")
    chunks = _wrap(value, body_w)
    pad = " " * LABEL_GUTTER

    if len(label) > LABEL_W:
        return ([INDENT + styled_label]
                + [pad + fmt.style(c, "value") for c in chunks])
    first = (INDENT + fmt.style(ui.pad(label, LABEL_W), "label")
             + " " + fmt.style(chunks[0], "value"))
    return [first] + [pad + fmt.style(c, "value") for c in chunks[1:]]


# Tags that read as a list: consecutive siblings stay tight, and only the
# group gets a blank line before it.
_TIGHT = ("bullet", "kv")


def _needs_gap(prev: Optional[str], tag: str) -> bool:
    """Whether a blank line goes between two blocks.

    Without this, consecutive paragraphs render as one wall of text — visible
    the moment you look at a real chapter, and invisible to a width test.
    """
    if prev is None or tag in ("gap", "rule", "h"):
        return False
    if prev in ("h", "gap", "rule"):
        return False
    if tag == prev and tag in _TIGHT:
        return False
    return True


def render_blocks(blocks: Sequence[tuple]) -> List[str]:
    """Render a chapter body to display lines.

    Raises ValueError on an unknown tag or a wrong-length block. Loudly, because
    the content module is hand-written data and a silently skipped block is a
    paragraph that quietly stops being in the manual.
    """
    lines: List[str] = []
    prev: Optional[str] = None
    for block in blocks:
        if not block or block[0] not in BLOCK_ARITY:
            raise ValueError(f"unknown help block: {block!r}")
        tag = block[0]
        if len(block) != BLOCK_ARITY[tag]:
            raise ValueError(
                f"block {tag!r} takes {BLOCK_ARITY[tag]} elements: {block!r}")

        if _needs_gap(prev, tag):
            lines.append("")
        prev = tag

        if tag == "gap":
            lines.append("")
        elif tag == "rule":
            lines.append(ui.rule(WIDTH))
        elif tag == "h":
            lines.append("")
            lines.append(INDENT + fmt.style(str(block[1]).upper(),
                                            "heading", bold=True))
        elif tag == "p":
            lines.extend(INDENT + fmt.style(w, "muted")
                         for w in _wrap(block[1], BODY_W))
        elif tag == "bullet":
            lines.extend(fmt.style(ln, "muted")
                         for ln in _hanging("· ", block[1], BODY_W))
        elif tag == "num":
            for i, item in enumerate(block[1], start=1):
                lines.extend(fmt.style(ln, "muted")
                             for ln in _hanging(f"{i}. ", item, BODY_W))
        elif tag == "kv":
            lines.extend(_kv(block[1], block[2]))
        elif tag == "table":
            # ui.pad does not clip, so an over-long cell pushes the next column
            # out of true without ever making the LINE too wide — a defect a
            # width assertion cannot see. Clip here; the content tests assert
            # cells fit, so this only ever fires on a future edit.
            cols = block[1]
            rows = [[ui.clip(cell, c["w"]) for cell, c in zip(row, cols)]
                    for row in block[2]]
            lines.extend(ui.table(cols, rows).splitlines())
        elif tag == "callout":
            tone = block[1] if block[1] in _CALLOUT_TITLE else "warn"
            body = [fmt.style(w, tone) for w in _wrap(block[2], WIDTH - 6)]
            lines.extend(
                ui.card(_CALLOUT_TITLE[tone], body, WIDTH, boxed=True).splitlines())
        elif tag == "code":
            lines.append(INDENT + fmt.style(str(block[1]), "accent"))
    return lines
