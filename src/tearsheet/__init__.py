"""Self-contained HTML tearsheet for a single option contract. Display-only."""
import json
import os

# Exported as `render_html`, not `render`: a function named `render` would shadow
# the `render` submodule on the package, breaking `from src.tearsheet import render`.
from .render import render as render_html  # noqa: F401
from .collect import build  # noqa: F401


def write_tearsheet(data: dict, out_dir: str = "reports/tearsheets"):
    """Write `<base>.html` and its `<base>.json` sidecar. Returns both paths.

    The sidecar is what makes the page reproducible: `render` is pure over it,
    so `python -m src.tearsheet --from <sidecar>` rebuilds the exact bytes.
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(data["meta"]["sidecar"])[0]
    json_path = os.path.join(out_dir, base + ".json")
    html_path = os.path.join(out_dir, base + ".html")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    with open(html_path, "w") as f:
        f.write(render_html(data))
    from src.desk_kit import hub
    hub.refresh(os.path.dirname(out_dir.rstrip("/")) or "reports")
    return html_path, json_path
