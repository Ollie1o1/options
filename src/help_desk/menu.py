"""The manual's chapter menu.

`input_fn` and `output_fn` are injected rather than called directly so the loop
can be driven from a test without a TTY. That is also why paging lives here and
not in the renderer: paging is the one part that must ask the terminal how tall
it is, and everything that asks the terminal anything is confined to this file.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional, Sequence

from .. import formatting as fmt
from .. import ui
from .content import CHAPTERS, Chapter
from .render import WIDTH, render_blocks

_PROMPT = "  Chapter [B]: "
_MORE = "  — more — [Enter] to continue, Q to stop "

# Index row: 2 indent + "[n]" + 2 + title + 2 + blurb. TITLE_W holds the
# longest chapter title; BLURB_W is whatever is left, and content is tested
# against it so an edited blurb cannot silently push the row over the rule.
TITLE_W = 26
BLURB_W = WIDTH - (2 + 3 + 2 + TITLE_W + 2)


def format_index() -> str:
    """The chapter list, styled like the launcher's own menu rows."""
    out = [ui.banner("HELP  ·  THE MANUAL", width=WIDTH), ""]
    for i, chapter in enumerate(CHAPTERS, start=1):
        key = fmt.style(f"[{i}]", "accent", bold=True)
        name = fmt.style(ui.pad(chapter.title, TITLE_W), "heading")
        blurb = fmt.style(chapter.blurb, "muted")
        out.append(f"  {key}  {name}  {blurb}")
    out.append("")
    out.append("  " + fmt.style("[A]", "accent", bold=True) + "  "
               + fmt.style(ui.pad("READ ALL", TITLE_W), "heading") + "  "
               + fmt.style("every chapter, in order", "muted"))
    out.append("  " + fmt.style("[B]", "muted") + "  "
               + fmt.style("BACK", "muted"))
    out.append(ui.rule(WIDTH))
    return "\n".join(out)


def page_lines(lines: Sequence[str], height: int) -> List[List[str]]:
    """Split into pages of `height` lines. A height of 0 (or one that already
    holds everything) means one page — which is what non-TTY callers get, so
    piping the manual gives you the whole thing."""
    lines = list(lines)
    if height <= 0 or len(lines) <= height:
        return [lines]
    return [lines[i:i + height] for i in range(0, len(lines), height)]


def _terminal_height() -> int:
    """Usable rows, or 0 when there is no interactive terminal to measure."""
    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return 0
    except Exception:
        return 0
    try:
        return max(0, os.get_terminal_size().lines - 3)
    except OSError:
        return 0


def _resolve(token: str) -> Optional[Chapter]:
    """A row number, a chapter key, or a title prefix. Anything else is not a
    selection — the caller reprints rather than guessing."""
    token = token.strip()
    if not token:
        return None
    if token.isdigit():
        idx = int(token) - 1
        return CHAPTERS[idx] if 0 <= idx < len(CHAPTERS) else None
    lowered = token.lower()
    for chapter in CHAPTERS:
        if chapter.key == lowered:
            return chapter
    for chapter in CHAPTERS:
        if chapter.title.lower().startswith(lowered):
            return chapter
    return None


def _show(chapter: Chapter, output_fn: Callable, input_fn: Callable) -> bool:
    """Print one chapter, paged. Returns False if the reader asked to stop."""
    header = [
        "",
        ui.rule(WIDTH, chapter.title),
    ]
    pages = page_lines(header + render_blocks(chapter.body) + [""],
                       _terminal_height())
    for i, page in enumerate(pages):
        for line in page:
            output_fn(line)
        if i < len(pages) - 1:
            try:
                if input_fn(_MORE).strip().upper() in ("Q", "QUIT"):
                    return False
            except (EOFError, KeyboardInterrupt):
                output_fn("")
                return False
    return True


def run_menu(input_fn: Callable = input, output_fn: Callable = print) -> None:
    """Browse the manual. Returns on B / Q / empty input / EOF."""
    while True:
        output_fn("")
        output_fn(format_index())
        try:
            choice = input_fn(_PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return

        upper = choice.upper()
        if upper in ("B", "BACK", "Q", "QUIT", ""):
            return

        if upper in ("A", "ALL", "READ ALL"):
            for chapter in CHAPTERS:
                if not _show(chapter, output_fn, input_fn):
                    break
            continue

        chapter = _resolve(choice)
        if chapter is None:
            output_fn(fmt.style(
                f"  Unknown choice: {choice!r} — pick 1-{len(CHAPTERS)}, A, or B",
                "warn"))
            continue
        _show(chapter, output_fn, input_fn)
