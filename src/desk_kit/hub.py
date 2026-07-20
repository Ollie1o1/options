"""The desk hub: reports/index.html — one page that knows where everything is.

`build_index(entries)` is pure; `scan(reports_dir)` and `write_index` do the
I/O. Every report writer calls `refresh(reports_dir)` after writing, so the
hub is always current and every masthead's DESK link lands somewhere real.
`refresh_latest` keeps a stable `latest.html` alias beside dated outputs so
cross-page links never dangle on a filename that changes daily.
"""
import os
import re
import shutil
from datetime import datetime

from src.desk_kit import shell

_SECTIONS = (
    ("briefings", "Morning briefings", "briefing"),
    ("research", "Research desk", "research report"),
    ("tearsheets", "Tearsheets", "tearsheet"),
    ("holdings", "Holdings", "holdings report"),
)
_SKIP = {"latest.html", "index.html"}


def _title(kind, fname):
    stem = fname[:-5]  # drop .html
    if kind == "tearsheets":
        parts = stem.split("_")
        if len(parts) >= 3 and re.fullmatch(r"\d{8}", parts[-1]):
            d = parts[-1]
            return "{} {} · exp {}-{}-{}".format(
                parts[0], " ".join(parts[1:-1]), d[:4], d[4:6], d[6:])
    if kind == "research":
        m = re.fullmatch(r"research_(\d{8})_(\d{4})(?:_(\w+))?", stem)
        if m:
            d, t, sym = m.groups()
            base = "{}-{}-{} {}:{}".format(d[:4], d[4:6], d[6:], t[:2], t[2:])
            return base + (" · {}".format(sym) if sym else " · market")
    return stem


def scan(reports_dir):
    """{section: [{name, href, title, mtime, when}]}, newest first."""
    out = {}
    for sub, _label, _noun in _SECTIONS:
        d = os.path.join(reports_dir, sub)
        items = []
        if os.path.isdir(d):
            for f in os.listdir(d):
                if not f.endswith(".html") or f in _SKIP:
                    continue
                path = os.path.join(d, f)
                try:
                    mt = os.path.getmtime(path)
                except OSError:
                    continue
                items.append({
                    "name": f, "href": "{}/{}".format(sub, f),
                    "title": _title(sub, f), "mtime": mt,
                    "when": datetime.fromtimestamp(mt).strftime(
                        "%Y-%m-%d %H:%M"),
                })
        items.sort(key=lambda x: x["mtime"], reverse=True)
        out[sub] = items
    return out


def build_index(entries, generated_at=None):
    """Pure HTML over a scan() result."""
    cards = []
    for sub, label, noun in _SECTIONS:
        items = entries.get(sub) or []
        if not items:
            body = shell.ph("no {}s generated yet".format(noun))
        else:
            newest, rest = items[0], items[1:9]
            body = ('<div class="evt"><a href="{h}"><strong>{t}</strong></a> '
                    '<span class="mut">· latest · {w}</span></div>').format(
                        h=shell.esc(newest["href"]),
                        t=shell.esc(newest["title"]),
                        w=shell.esc(newest["when"]))
            body += "".join(
                '<div class="evt"><a href="{h}">{t}</a> '
                '<span class="mut">· {w}</span></div>'.format(
                    h=shell.esc(i["href"]), t=shell.esc(i["title"]),
                    w=shell.esc(i["when"])) for i in rest)
            if len(items) > 9:
                body += ('<div class="evt mut">… {} more in reports/{}/'
                         "</div>").format(len(items) - 9, shell.esc(sub))
        cards.append(shell.card(label, body, span=4))
    mast = shell.masthead(
        "HUB", "", meta_html=shell.chipline(
            [("generated", shell.esc(generated_at or ""))]), where="hub")
    body = shell.grid(cards) + (
        '<div class="foot">every report writer refreshes this page and its '
        "section's latest.html alias · regenerate: python -m src.desk_kit.hub"
        "</div>")
    return shell.page("Options Desk — Hub", mast, body)


def write_index(reports_dir="reports"):
    html = build_index(scan(reports_dir),
                       generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"))
    path = os.path.join(reports_dir, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def refresh_latest(out_dir, newest_path, alias="latest.html"):
    """Copy the just-written dated output over the stable alias."""
    target = os.path.join(out_dir, alias)
    try:
        shutil.copyfile(newest_path, target)
    except OSError:
        return None
    return target


def refresh(reports_dir="reports"):
    """Best-effort hub refresh; a hub failure must never fail a report."""
    try:
        return write_index(reports_dir)
    except OSError:
        return None


if __name__ == "__main__":
    print(write_index())
