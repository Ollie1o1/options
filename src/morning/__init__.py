"""Market-wide morning-briefing HTML tearsheet. Display-only; no AI calls."""
import json
import os

from .collect import build  # noqa: F401


def write_briefing(data: dict, out_dir: str = "reports/briefings"):
    """Write `<date>.html` + `<date>.json` sidecar. Returns both paths.
    The sidecar makes the page reproducible: render is pure over it."""
    from .render import render as render_html
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(data["meta"]["sidecar"])[0]
    json_path = os.path.join(out_dir, base + ".json")
    html_path = os.path.join(out_dir, base + ".html")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
    with open(html_path, "w") as f:
        f.write(render_html(data))
    return html_path, json_path


def build_and_write(out_dir: str = "reports/briefings", slow: bool = True):
    return write_briefing(build(slow=slow), out_dir=out_dir)
