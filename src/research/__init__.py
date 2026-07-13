"""Research Desk: unified tabbed HTML research report. Display-only; no AI calls."""
import json
import os

from .collect import build  # noqa: F401


def write_desk(data: dict, out_dir: str = "reports/research"):
    """Write `<base>.html` + `<base>.json` sidecar. Returns both paths.
    The sidecar makes the page reproducible: render is pure over it."""
    from .render import render as render_html
    os.makedirs(out_dir, exist_ok=True)
    base = data["meta"]["base"]
    json_path = os.path.join(out_dir, base + ".json")
    html_path = os.path.join(out_dir, base + ".html")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
    with open(html_path, "w") as f:
        f.write(render_html(data))
    from src.desk_kit import hub
    hub.refresh_latest(out_dir, html_path)
    hub.refresh(os.path.dirname(out_dir.rstrip("/")) or "reports")
    return html_path, json_path


def build_and_write(symbol=None, out_dir="reports/research", slow=True,
                    budget_s=25.0):
    return write_desk(build(symbol=symbol, slow=slow, budget_s=budget_s),
                      out_dir=out_dir)
