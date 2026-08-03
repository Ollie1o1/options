#!/usr/bin/env python3
"""Turn mypy's output into GitHub Actions annotations.

The typecheck job is `continue-on-error`, so it can never fail the build and
its log goes unread — which is how it stayed red for weeks without anyone
knowing what it objected to. Annotations put each diagnostic on the commit
itself, where it is visible without opening a log (and, unlike job logs,
readable through the public API).

Reads mypy's plain output on stdin or from a path, writes workflow commands
to stdout. Never fails: a diagnostic reporter that can itself break the step
is worse than no reporter, so anything unparseable is passed through as-is.

    mypy src/ --config-file mypy.ini > out.txt; python scripts/mypy_annotate.py out.txt
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from typing import Iterable, List

# `path:line: error: message  [code]` — the column field is optional, and mypy
# emits `note:` lines that elaborate on the error above them rather than
# standing on their own.
_LINE = re.compile(
    r"^(?P<file>[^:]+):(?P<line>\d+):(?:(?P<col>\d+):)?\s+"
    r"(?P<level>error|warning|note):\s+(?P<msg>.*)$"
)

# GitHub renders at most a handful per step and the API caps what it returns;
# past this the list stops being a summary and starts being a log again.
MAX_ANNOTATIONS = 50

_LEVEL = {"error": "error", "warning": "warning", "note": "notice"}

# mypy appends its error code in brackets: `... [arg-type]`.
_CODE = re.compile(r"\[([a-z][a-z0-9-]+)\]\s*$")


def _escape(text: str) -> str:
    """Escape a workflow-command message property.

    Order matters: `%` first, or the escapes introduced below get re-escaped.
    """
    return (text.replace("%", "%25")
                .replace("\r", "%0D")
                .replace("\n", "%0A"))


def annotations_for(lines: Iterable[str],
                    max_annotations: int = MAX_ANNOTATIONS) -> List[str]:
    """Workflow commands for the diagnostics in `lines`.

    Only `error` becomes an annotation. Notes are mypy's own elaboration of the
    error above them, so promoting each to its own annotation would double-count
    a single problem.
    """
    out: List[str] = []
    errors = 0
    for raw in lines:
        m = _LINE.match(raw.rstrip("\n"))
        if not m or m.group("level") != "error":
            continue
        errors += 1
        if len(out) >= max_annotations:
            continue
        parts = [f"file={m.group('file')}", f"line={m.group('line')}"]
        if m.group("col"):
            parts.append(f"col={m.group('col')}")
        parts.append("title=mypy")
        out.append(f"::{_LEVEL['error']} {','.join(parts)}::"
                   f"{_escape(m.group('msg'))}")
    if errors > len(out):
        out.append(f"::notice title=mypy::{errors} errors total; "
                   f"{len(out)} annotated (cap {max_annotations})")
    return out


def breakdown(lines: Iterable[str]) -> dict:
    """Counts of errors by mypy error code and by file.

    With hundreds of diagnostics the capped annotation list says nothing about
    the shape of the problem — whether it is one bad pattern repeated across a
    package, or that many distinct ones. This is the part worth reading first.
    """
    codes: Counter = Counter()
    files: Counter = Counter()
    total = 0
    for raw in lines:
        m = _LINE.match(raw.rstrip("\n"))
        if not m or m.group("level") != "error":
            continue
        total += 1
        files[m.group("file")] += 1
        code = _CODE.search(m.group("msg"))
        codes[code.group(1) if code else "(none)"] += 1
    return {"total": total, "codes": codes, "files": files}


def render_breakdown(breakdown_: dict = None,
                     lines: Iterable[str] = (),
                     top: int = 12) -> str:
    """The histogram as one annotation-sized block of text."""
    b = breakdown_ if breakdown_ is not None else breakdown(lines)
    if not b["total"]:
        return "mypy: no errors"
    out = [f"{b['total']} errors"]
    out.append("by code: " + ", ".join(
        f"{c}={n}" for c, n in b["codes"].most_common(top)))
    out.append("by file: " + ", ".join(
        f"{f}={n}" for f, n in b["files"].most_common(top)))
    return "\n".join(out)


def main(argv: List[str]) -> int:
    if len(argv) > 1:
        try:
            with open(argv[1], encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError as exc:
            print(f"::notice title=mypy::could not read {argv[1]}: {exc}")
            return 0
    else:
        lines = sys.stdin.readlines()
    # Shape first, then the individual diagnostics: with hundreds of errors the
    # histogram is the only part that fits in a glance.
    summary = render_breakdown(lines=lines)
    print(f"::notice title=mypy summary::{_escape(summary)}")
    for line in annotations_for(lines):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
