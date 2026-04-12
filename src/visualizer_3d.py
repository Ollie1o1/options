"""Interactive 3D options visualization using Plotly.

Six tabbed views opened in the browser:
  1. Contract Explorer   – configurable 3-axis scatter of all screened contracts
  2. IV Surface          – SVI-fitted implied-vol surface with mispricing overlay
  3. Greek Landscape     – delta/gamma/vega/theta sensitivity surfaces
  4. P&L Scenarios       – full Black-Scholes repriced P&L surfaces
  5. Score Decomposition – parallel-coordinates of 12 score components
  6. Risk Radar          – polar spider chart comparing top contracts
"""

from __future__ import annotations

import logging
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from scipy.interpolate import griddata as _griddata
    _HAS_GRIDDATA = True
except ImportError:
    _HAS_GRIDDATA = False

from .types import ScanResult
from .visual_surface import compute_pnl_grid, compute_greek_grid

log = logging.getLogger(__name__)

# ─── Notion-inspired dark theme ──────────────────────────────────────────────

DARK_BG = "#191919"                       # page bg (Notion dark primary)
DARK_PANEL = "#202020"                    # header / tab-bar bg
DARK_HOVER = "#2a2a2a"                    # hover + active tab
DARK_BORDER = "rgba(255,255,255,0.094)"   # Notion's signature subtle border
DARK_GRID = "rgba(255,255,255,0.06)"      # subtler grid lines for plots
TEXT_PRIMARY = "#e6e6e6"                  # Notion body text
TEXT_MUTED = "#9b9a97"                    # Notion secondary text
ACCENT_BLUE = "#529cca"                   # Notion blue accent
COLOR_POS = "#4dab9a"                     # Notion green
COLOR_NEG = "#e16b6b"                     # Notion red
COLOR_WARN = "#d9a74a"                    # Notion yellow

_SCENE_AXIS = dict(
    backgroundcolor=DARK_BG, gridcolor=DARK_GRID, zerolinecolor=DARK_BORDER,
    title_font=dict(color=TEXT_MUTED, size=12),
    tickfont=dict(color=TEXT_MUTED, size=10),
)

_BASE_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=DARK_BG,
    plot_bgcolor=DARK_BG,
    font=dict(family="Inter, -apple-system, system-ui, sans-serif",
              color=TEXT_PRIMARY, size=12),
    title_font=dict(size=15, color=TEXT_PRIMARY),
    margin=dict(t=48, l=40, r=40, b=40),
    hoverlabel=dict(
        bgcolor=DARK_PANEL, bordercolor=DARK_BORDER,
        font=dict(color=TEXT_PRIMARY,
                  family="Inter, -apple-system, sans-serif", size=12),
    ),
)

# ─── Curated axis / score / color fields ─────────────────────────────────────

AXIS_FIELDS: Dict[str, str] = {
    "moneyness":           "Moneyness (K/S)",
    "DTE":                 "Days to Expiry",
    "quality_score":       "Quality Score",
    "impliedVolatility":   "Implied Vol",
    "iv_percentile_30":    "IV Percentile",
    "hv_30d":              "Historical Vol",
    "iv_vs_hv":            "IV \u2212 HV",
    "delta":               "Delta",
    "abs_delta":           "|Delta|",
    "gamma":               "Gamma",
    "vega":                "Vega",
    "theta":               "Theta",
    "prob_profit":         "Prob of Profit",
    "ev_per_contract":     "Expected Value ($)",
    "rr_ratio":            "Risk/Reward",
    "premium":             "Premium ($)",
    "volume":              "Volume",
    "openInterest":        "Open Interest",
    "spread_pct":          "Spread %",
    "iv_surface_residual": "IV Mispricing",
}

SCORE_COLUMNS: List[Tuple[str, str, float]] = [
    ("iv_rank_score",        "IV Rank",       0.18),
    ("iv_edge_score",        "IV Edge",       0.15),
    ("vrp_score",            "VRP",           0.09),
    ("pop_score",            "PoP",           0.08),
    ("rr_score",             "R/R",           0.08),
    ("liquidity_score",      "Liquidity",     0.06),
    ("momentum_score",       "Momentum",      0.05),
    ("theta_score",          "Theta",         0.05),
    ("iv_mispricing_score",  "IV Misprice",   0.05),
    ("iv_velocity_score",    "IV Velocity",   0.04),
    ("ev_score",             "EV",            0.04),
    ("term_structure_score", "Term Struct",   0.03),
]

PNL_COLORSCALE = [
    [0.0, "rgb(180,30,30)"],
    [0.4, "rgb(255,80,60)"],
    [0.5, "rgb(100,100,100)"],
    [0.6, "rgb(80,220,80)"],
    [1.0, "rgb(40,255,160)"],
]

GREEK_COLORSCALES = {"delta": "Tealgrn", "gamma": "YlOrBr", "vega": "Purp", "theta": "Reds"}

_RADAR_COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

_RADAR_AXES: List[Tuple[str, str]] = [
    ("prob_profit",     "PoP"),
    ("ev_norm",         "EV"),
    ("liquidity_score", "Liquidity"),
    ("iv_edge_score",   "IV Edge"),
    ("vrp_score",       "VRP"),
    ("momentum_score",  "Momentum"),
    ("greeks_quality",  "Greeks"),
]

_HOVER_FIELDS = [
    ("symbol", ""),  ("type", ""),  ("strike", 0.0),  ("expiration", ""),
    ("DTE", 0.0),  ("quality_score", 0.0),  ("prob_profit", 0.0),
    ("impliedVolatility", 0.0),  ("hv_30d", 0.0),  ("delta", 0.0),
    ("gamma", 0.0),  ("ev_per_contract", 0.0),  ("rr_ratio", 0.0),
    ("volume", 0),  ("openInterest", 0),  ("bid", 0.0),  ("ask", 0.0),
    ("premium", 0.0),  ("score_drivers", ""),
]

_HOVER_TEMPLATE = (
    "<b>%{customdata[0]} %{customdata[1]} $%{customdata[2]:.0f}</b><br>"
    "Exp: %{customdata[3]}  (%{customdata[4]:.0f} DTE)<br>"
    "Score: %{customdata[5]:.3f}  |  PoP: %{customdata[6]:.1%}<br>"
    "IV: %{customdata[7]:.1%}  |  HV: %{customdata[8]:.1%}<br>"
    "\u0394: %{customdata[9]:.3f}  |  \u0393: %{customdata[10]:.4f}<br>"
    "EV: $%{customdata[11]:.2f}  |  R/R: %{customdata[12]:.1f}x<br>"
    "Vol: %{customdata[13]:,.0f}  |  OI: %{customdata[14]:,.0f}<br>"
    "Bid: $%{customdata[15]:.2f}  |  Ask: $%{customdata[16]:.2f}<br>"
    "<i>%{customdata[18]}</i>"
    "<extra></extra>"
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _safe_col(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    """Safely extract a numeric column as float64 numpy array."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default).values.astype(float)
    return np.full(len(df), default, dtype=float)


def _build_customdata(df: pd.DataFrame) -> list:
    """Build list-of-lists customdata for hover tooltips."""
    # Pre-extract all columns once for speed
    col_arrays = {}
    for name, default in _HOVER_FIELDS:
        if name in df.columns:
            col_arrays[name] = df[name].values
        else:
            col_arrays[name] = np.full(len(df), default)
    rows = []
    for i in range(len(df)):
        row = []
        for name, default in _HOVER_FIELDS:
            v = col_arrays[name][i]
            if isinstance(v, float) and np.isnan(v):
                v = default
            elif pd.isna(v):
                v = default
            row.append(v)
        rows.append(row)
    return rows


# ─── Main class ──────────────────────────────────────────────────────────────

class OptionsVisualizer:
    """Interactive 3D visualization of options scan results."""

    def __init__(self, scan_result: ScanResult, config: dict | None = None):
        if not HAS_PLOTLY:
            raise ImportError("plotly is required for the 3D visualizer: pip install plotly")
        self.rfr = scan_result.rfr
        self.underlying_price = scan_result.underlying_price
        self.market_context = getattr(scan_result, "market_context", {}) or {}
        self._viz_cfg = (config or {}).get("visualizer", {})
        self._picks_raw = scan_result.picks.copy() if (scan_result.picks is not None and not scan_result.picks.empty) else pd.DataFrame()
        self.df: pd.DataFrame = pd.DataFrame()
        self.top_df: pd.DataFrame = pd.DataFrame()
        self._prepare_data()

    # ── data prep ────────────────────────────────────────────────────────

    def _prepare_data(self) -> None:
        df = self._picks_raw
        if df.empty:
            return

        # Derived columns
        und = pd.to_numeric(df.get("underlying"), errors="coerce").fillna(1).replace(0, 1)
        df["moneyness"] = pd.to_numeric(df.get("strike"), errors="coerce").fillna(0) / und
        df["log_moneyness"] = np.log(df["moneyness"].clip(lower=0.01))

        if "T_years" in df.columns:
            df["DTE"] = pd.to_numeric(df["T_years"], errors="coerce").fillna(0) * 365
        elif "expiration" in df.columns:
            exp = pd.to_datetime(df["expiration"], errors="coerce", utc=True)
            df["DTE"] = (exp - pd.Timestamp.now(tz="UTC")).dt.total_seconds() / 86400
            df["DTE"] = df["DTE"].clip(lower=0)
        else:
            df["DTE"] = 0.0

        # Fill missing score columns
        for col, _, _ in SCORE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.5

        # Normalised EV for radar chart
        ev = pd.to_numeric(df.get("ev_per_contract", 0), errors="coerce").fillna(0)
        df["ev_norm"] = ev.rank(pct=True).fillna(0.5)

        # Greeks quality composite for radar
        delta_abs = pd.to_numeric(df.get("delta", 0), errors="coerce").fillna(0).abs()
        gm = pd.to_numeric(df.get("gamma_magnitude_score", 0.5), errors="coerce").fillna(0.5)
        df["greeks_quality"] = (1.0 - (delta_abs - 0.40).abs().clip(upper=0.4) / 0.4) * 0.6 + gm * 0.4

        if "quality_score" in df.columns:
            df = df.sort_values("quality_score", ascending=False).reset_index(drop=True)

        self.df = df
        n_top = self._viz_cfg.get("top_n_annotate", 3)
        self.top_df = df.head(min(n_top, len(df))).copy()

    # ── helpers ──────────────────────────────────────────────────────────

    def _apply(self, fig: go.Figure, title: str = "") -> go.Figure:
        fig.update_layout(**_BASE_LAYOUT)
        # Let Plotly resize to fill its container instead of baking a fixed height
        fig.update_layout(autosize=True)
        if title:
            fig.update_layout(title_text=title)
        return fig

    def _empty(self, msg: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=18, color=TEXT_MUTED))
        return self._apply(fig)

    # ── View 1: Contract Explorer ────────────────────────────────────────

    def build_contract_explorer(self) -> go.Figure:
        """3D scatter of all screened contracts with axis/color remapping."""
        df, top = self.df, self.top_df
        if df.empty:
            return self._empty("No contracts to visualise")

        cfg = self._viz_cfg
        x_def = cfg.get("default_x", "moneyness")
        y_def = cfg.get("default_y", "DTE")
        z_def = cfg.get("default_z", "quality_score")
        c_def = cfg.get("default_color", "iv_percentile_30")

        cd_main = _build_customdata(df)
        cd_top = _build_customdata(top) if not top.empty else []

        sizes = np.clip(np.log1p(_safe_col(df, "volume", 1)) * 1.5, 4, 20)

        fig = go.Figure()

        # Trace 0: all contracts
        fig.add_trace(go.Scatter3d(
            x=_safe_col(df, x_def).tolist(),
            y=_safe_col(df, y_def).tolist(),
            z=_safe_col(df, z_def).tolist(),
            mode="markers",
            marker=dict(
                size=sizes,
                color=_safe_col(df, c_def).tolist(),
                colorscale="Plasma", showscale=True,
                colorbar=dict(
                    title=dict(text=AXIS_FIELDS.get(c_def, c_def),
                               font=dict(color=TEXT_MUTED, size=11)),
                    bgcolor=DARK_PANEL, bordercolor=DARK_BORDER,
                    tickfont=dict(color=TEXT_MUTED, size=10),
                ),
                opacity=0.85,
                line=dict(width=0.5, color=DARK_BORDER),
            ),
            customdata=cd_main,
            hovertemplate=_HOVER_TEMPLATE,
            name="Contracts",
        ))

        # Trace 1: top picks (annotated)
        has_top = not top.empty
        if has_top:
            labels = [f"{r.get('symbol','')} ${r.get('strike',0):.0f}"
                      for _, r in top.iterrows()]
            fig.add_trace(go.Scatter3d(
                x=_safe_col(top, x_def).tolist(),
                y=_safe_col(top, y_def).tolist(),
                z=_safe_col(top, z_def).tolist(),
                mode="markers+text",
                marker=dict(size=12, color=COLOR_WARN, symbol="diamond",
                            line=dict(width=2, color="white")),
                text=labels, textposition="top center",
                textfont=dict(color="white", size=11),
                customdata=cd_top,
                hovertemplate=_HOVER_TEMPLATE,
                name="Top Picks",
            ))

        # ── Dropdown menus ──

        field_keys = list(AXIS_FIELDS.keys())
        menus: list[dict] = []

        def _make_dropdown(name, key):
            """Generate an axis-change dropdown dict and return its active index."""
            idx = field_keys.index(key) if key in field_keys else 0
            return idx

        for axis, default_col, x_pos in [("x", x_def, 0.0),
                                          ("y", y_def, 0.28),
                                          ("z", z_def, 0.56)]:
            buttons = []
            for col in field_keys:
                main_vals = _safe_col(df, col).tolist()
                top_vals = _safe_col(top, col).tolist() if has_top else []
                trace_data = [main_vals, top_vals] if has_top else [main_vals]
                buttons.append(dict(
                    method="update", label=AXIS_FIELDS[col],
                    args=[{axis: trace_data},
                          {f"scene.{axis}axis.title.text": AXIS_FIELDS[col]}],
                ))
            active = field_keys.index(default_col) if default_col in field_keys else 0
            menus.append(dict(
                buttons=buttons, direction="down", showactive=True, active=active,
                x=x_pos, y=1.24, xanchor="left", yanchor="top",
                bgcolor=DARK_PANEL, bordercolor=DARK_BORDER,
                font=dict(color=TEXT_PRIMARY, size=11), pad=dict(r=10),
            ))

        # Color dropdown (trace 0 only)
        color_btns = []
        for col in field_keys:
            color_btns.append(dict(
                method="restyle", label=AXIS_FIELDS[col],
                args=[{"marker.color": [_safe_col(df, col).tolist()],
                       "marker.colorbar.title.text": AXIS_FIELDS[col]}, [0]],
            ))
        c_active = field_keys.index(c_def) if c_def in field_keys else 0
        menus.append(dict(
            buttons=color_btns, direction="down", showactive=True, active=c_active,
            x=0.84, y=1.24, xanchor="left", yanchor="top",
            bgcolor=DARK_PANEL, bordercolor=DARK_BORDER,
            font=dict(color=TEXT_PRIMARY, size=11),
        ))

        annotations = [
            dict(text=t, x=xp, y=1.28, xref="paper", yref="paper",
                 showarrow=False, font=dict(color=TEXT_MUTED, size=11))
            for t, xp in [("X Axis", 0.0), ("Y Axis", 0.28),
                          ("Z Axis", 0.56), ("Color", 0.84)]
        ]

        fig.update_layout(
            updatemenus=menus, annotations=annotations,
            scene=dict(
                xaxis=dict(**_SCENE_AXIS, title=AXIS_FIELDS.get(x_def, x_def)),
                yaxis=dict(**_SCENE_AXIS, title=AXIS_FIELDS.get(y_def, y_def)),
                zaxis=dict(**_SCENE_AXIS, title=AXIS_FIELDS.get(z_def, z_def)),
                camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
            ),
            margin=dict(t=150, l=40, r=40, b=40),
            legend=dict(x=0.01, y=0.01, bgcolor=DARK_PANEL, bordercolor=DARK_BORDER,
                        font=dict(color=TEXT_PRIMARY, size=11)),
        )
        return self._apply(fig, "Contract Explorer")

    # ── View 2: IV Surface ───────────────────────────────────────────────

    def build_iv_surface(self) -> go.Figure:
        """SVI-fitted IV surface with mispricing scatter overlay."""
        df = self.df
        if df.empty:
            return self._empty("No contracts for IV surface")

        fig = go.Figure()
        lm = _safe_col(df, "log_moneyness")
        dte = _safe_col(df, "DTE")
        iv = _safe_col(df, "impliedVolatility")
        residual = _safe_col(df, "iv_surface_residual")

        # Attempt surface reconstruction from residuals
        fitted_mask = df.get("iv_surface_fitted", pd.Series(False, index=df.index))
        fitted_idx = fitted_mask.astype(bool)
        n_fitted = fitted_idx.sum()

        if n_fitted >= 8 and _HAS_GRIDDATA:
            f_lm = lm[fitted_idx]
            f_dte = dte[fitted_idx]
            f_iv = iv[fitted_idx]
            f_res = residual[fitted_idx]
            denom = 1.0 + f_res
            denom[np.abs(denom) < 0.01] = 1.0
            fit_iv = f_iv / denom

            k_rng = np.linspace(f_lm.min() - 0.02, f_lm.max() + 0.02, 30)
            d_rng = np.linspace(max(f_dte.min() - 2, 0), f_dte.max() + 2, 30)
            km, dm = np.meshgrid(k_rng, d_rng)

            try:
                pts = np.column_stack([f_lm, f_dte])
                z_c = _griddata(pts, fit_iv, (km, dm), method="cubic")
                z_l = _griddata(pts, fit_iv, (km, dm), method="linear")
                z = np.where(np.isfinite(z_c), z_c, z_l)
                z = np.nan_to_num(z, nan=0.0)

                fig.add_trace(go.Surface(
                    x=k_rng.tolist(), y=d_rng.tolist(), z=z.tolist(),
                    colorscale="Viridis", opacity=0.55, showscale=False,
                    name="SVI Fitted Surface",
                    hovertemplate=("Log K/S: %{x:.3f}<br>DTE: %{y:.0f}<br>"
                                   "Fitted IV: %{z:.1%}<extra>SVI Surface</extra>"),
                ))
            except Exception:
                log.debug("IV surface griddata failed", exc_info=True)

        # Market IV scatter coloured by residual
        cd = _build_customdata(df)
        fig.add_trace(go.Scatter3d(
            x=lm.tolist(), y=dte.tolist(), z=iv.tolist(),
            mode="markers",
            marker=dict(
                size=6,
                color=residual.tolist(),
                colorscale=[[0, "#636EFA"], [0.5, "#888888"], [1, "#EF553B"]],
                cmin=-0.15, cmid=0, cmax=0.15, showscale=True,
                colorbar=dict(
                    title=dict(text="CHEAP \u2190 Residual \u2192 RICH",
                               font=dict(color=TEXT_MUTED, size=10)),
                    bgcolor=DARK_PANEL, bordercolor=DARK_BORDER,
                    tickfont=dict(color=TEXT_MUTED, size=10),
                ),
                line=dict(width=0.5, color=DARK_BORDER),
            ),
            customdata=cd, hovertemplate=_HOVER_TEMPLATE,
            name="Market IV",
        ))

        # Top picks highlighted
        if not self.top_df.empty:
            t = self.top_df
            fig.add_trace(go.Scatter3d(
                x=_safe_col(t, "log_moneyness").tolist(),
                y=_safe_col(t, "DTE").tolist(),
                z=_safe_col(t, "impliedVolatility").tolist(),
                mode="markers+text",
                marker=dict(size=10, color=COLOR_WARN, symbol="diamond",
                            line=dict(width=2, color="white")),
                text=[f"{r.get('symbol','')} ${r.get('strike',0):.0f}"
                      for _, r in t.iterrows()],
                textposition="top center",
                textfont=dict(color="white", size=10),
                customdata=_build_customdata(t), hovertemplate=_HOVER_TEMPLATE,
                name="Top Picks",
            ))

        fig.update_layout(scene=dict(
            xaxis=dict(**_SCENE_AXIS, title="Log Moneyness ln(K/S)"),
            yaxis=dict(**_SCENE_AXIS, title="Days to Expiry"),
            zaxis=dict(**_SCENE_AXIS, title="Implied Volatility"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
        ))
        return self._apply(fig, "Implied Volatility Surface")

    # ── View 3: Greek Landscape ──────────────────────────────────────────

    def build_greek_landscape(self) -> go.Figure:
        """Delta/gamma/vega/theta sensitivity surfaces for the top pick."""
        df = self.df
        if df.empty:
            return self._empty("No contracts for Greek landscape")

        row = df.iloc[0]
        opt = str(row.get("type", "call")).lower()
        S = float(row.get("underlying", 100))
        K = float(row.get("strike", 100))
        T = max(float(row.get("T_years", 0.1)), 0.002)
        sigma = max(float(row.get("impliedVolatility", 0.3)), 0.01)
        q = float(row.get("dividend_yield", 0) or 0)

        fig = go.Figure()
        greeks = ["delta", "gamma", "vega", "theta"]

        for i, gk in enumerate(greeks):
            try:
                ps, ivs, grid = compute_greek_grid(
                    gk, opt, S, K, T, self.rfr, sigma, q=q, n_price=40, n_iv=20)
                z = np.nan_to_num(grid.T, nan=0, posinf=0, neginf=0)
                fig.add_trace(go.Surface(
                    x=(ps * 100).tolist(), y=(ivs * 100).tolist(), z=z.tolist(),
                    colorscale=GREEK_COLORSCALES[gk], name=gk.capitalize(),
                    visible=(i == 0),
                    hovertemplate=(f"Price \u0394: %{{x:.1f}}%<br>IV \u0394: %{{y:.1f}}%<br>"
                                   f"{gk.capitalize()}: %{{z:.4f}}<extra></extra>"),
                ))
            except Exception:
                log.debug("Greek grid %s failed", gk, exc_info=True)
                fig.add_trace(go.Surface(z=[[0]], visible=(i == 0), name=gk.capitalize()))

        # Toggle buttons
        buttons = []
        for i, gk in enumerate(greeks):
            vis = [j == i for j in range(len(greeks))]
            buttons.append(dict(
                method="update", label=gk.capitalize(),
                args=[{"visible": vis}, {"scene.zaxis.title.text": gk.capitalize()}],
            ))

        sym = row.get("symbol", "")
        fig.update_layout(
            updatemenus=[dict(
                buttons=buttons, direction="right", showactive=True, active=0,
                x=0.0, y=1.13, xanchor="left", yanchor="top",
                bgcolor=DARK_PANEL, bordercolor=DARK_BORDER,
                font=dict(color=TEXT_PRIMARY, size=12),
            )],
            scene=dict(
                xaxis=dict(**_SCENE_AXIS, title="Underlying Price Change (%)"),
                yaxis=dict(**_SCENE_AXIS, title="IV Change (%)"),
                zaxis=dict(**_SCENE_AXIS, title="Delta"),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            ),
        )
        return self._apply(fig, f"Greek Landscape \u2014 {sym} {opt.upper()} ${K:.0f}")

    # ── View 4: P&L Scenario Surface ─────────────────────────────────────

    def build_pnl_scenario(self) -> go.Figure:
        """Full BS-repriced P&L surface for top contracts."""
        df = self.df
        if df.empty:
            return self._empty("No contracts for P&L scenarios")

        fig = go.Figure()
        n_show = min(3, len(df))
        trace_labels: list[str] = []

        for idx in range(n_show):
            row = df.iloc[idx]
            opt = str(row.get("type", "call")).lower()
            S = float(row.get("underlying", 100))
            K = float(row.get("strike", 100))
            T = max(float(row.get("T_years", 0.1)), 0.002)
            sigma = max(float(row.get("impliedVolatility", 0.3)), 0.01)
            entry = max(float(row.get("premium", 1.0) or 1.0), 0.01)
            q = float(row.get("dividend_yield", 0) or 0)
            sym = row.get("symbol", "")
            label = f"{sym} {opt.upper()} ${K:.0f}"
            trace_labels.append(label)

            try:
                ps, ivs, pnl = compute_pnl_grid(
                    opt, S, K, T, self.rfr, sigma, entry, q=q, n_price=40, n_iv=20)
                z = np.nan_to_num(pnl.T, nan=0, posinf=0, neginf=0)

                # Normalise for symmetric colorscale
                rng = max(abs(float(z.min())), abs(float(z.max())), 0.01)
                z_norm = (z + rng) / (2 * rng)

                fig.add_trace(go.Surface(
                    x=(ps * 100).tolist(), y=(ivs * 100).tolist(), z=z.tolist(),
                    surfacecolor=z_norm.tolist(), colorscale=PNL_COLORSCALE,
                    cmin=0, cmax=1, showscale=(idx == 0), opacity=0.9,
                    name=label, visible=(idx == 0),
                    colorbar=dict(
                        title=dict(text="P&L", font=dict(color=TEXT_MUTED, size=10)),
                        bgcolor=DARK_PANEL, bordercolor=DARK_BORDER,
                        tickfont=dict(color=TEXT_MUTED, size=10),
                        tickvals=[0, 0.5, 1],
                        ticktext=[f"-${rng:.2f}", "$0", f"+${rng:.2f}"],
                    ) if idx == 0 else None,
                    hovertemplate=(f"<b>{label}</b><br>Price \u0394: %{{x:.1f}}%<br>"
                                   f"IV \u0394: %{{y:.1f}}%<br>P&L: $%{{z:.2f}}<extra></extra>"),
                ))
            except Exception:
                log.debug("P&L grid row %d failed", idx, exc_info=True)
                fig.add_trace(go.Surface(z=[[0]], visible=(idx == 0), name=label))

        # Breakeven plane
        fig.add_trace(go.Surface(
            x=[-25, 25], y=[-50, 50], z=[[0, 0], [0, 0]],
            colorscale=[[0, "rgba(100,100,100,0.3)"], [1, "rgba(100,100,100,0.3)"]],
            showscale=False, name="Breakeven", hoverinfo="skip",
        ))

        # Toggle buttons
        if n_show > 1:
            buttons = []
            for i in range(n_show):
                vis = [False] * n_show + [True]
                vis[i] = True
                buttons.append(dict(method="update", label=trace_labels[i],
                                    args=[{"visible": vis}]))
            buttons.append(dict(method="update", label="All",
                                args=[{"visible": [True] * (n_show + 1)}]))
            fig.update_layout(updatemenus=[dict(
                buttons=buttons, direction="right", showactive=True, active=0,
                x=0.0, y=1.13, xanchor="left", yanchor="top",
                bgcolor=DARK_PANEL, bordercolor=DARK_BORDER,
                font=dict(color=TEXT_PRIMARY, size=12),
            )])

        fig.update_layout(scene=dict(
            xaxis=dict(**_SCENE_AXIS, title="Underlying Price Change (%)"),
            yaxis=dict(**_SCENE_AXIS, title="IV Change (%)"),
            zaxis=dict(**_SCENE_AXIS, title="P&L ($)"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ))
        return self._apply(fig, "P&L Scenario Surface")

    # ── View 5: Score Decomposition (Parallel Coordinates) ───────────────

    def build_score_decomposition(self) -> go.Figure:
        """Parallel coordinates of 12 most-weighted score components."""
        df = self.df
        if df.empty:
            return self._empty("No contracts for score decomposition")

        dims = []
        for col, label, _wt in SCORE_COLUMNS:
            vals = _safe_col(df, col, 0.5)
            dims.append(dict(label=label, values=vals.tolist(), range=[0, 1]))

        quality = _safe_col(df, "quality_score", 0.5)

        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=quality.tolist(), colorscale="Plasma", showscale=True,
                cmin=float(quality.min()), cmax=float(quality.max()),
                colorbar=dict(
                    title=dict(text="Quality Score", font=dict(color=TEXT_MUTED, size=10)),
                    bgcolor=DARK_PANEL, bordercolor=DARK_BORDER,
                    tickfont=dict(color=TEXT_MUTED, size=10),
                ),
            ),
            dimensions=dims,
            labelside="top",
            labelfont=dict(color=TEXT_PRIMARY, size=11),
            tickfont=dict(color=TEXT_MUTED, size=10),
            rangefont=dict(color=TEXT_MUTED, size=10),
        ))

        return self._apply(fig, "Score Decomposition \u2014 Parallel Coordinates")

    # ── View 6: Risk Radar ───────────────────────────────────────────────

    def build_risk_radar(self) -> go.Figure:
        """Polar spider chart comparing top contracts across risk dimensions."""
        df = self.df
        if df.empty:
            return self._empty("No contracts for risk radar")

        n_show = min(self._viz_cfg.get("top_n_radar", 5), len(df))
        fig = go.Figure()

        cats = [label for _, label in _RADAR_AXES]
        cats_closed = cats + [cats[0]]

        for i in range(n_show):
            row = df.iloc[i]
            vals = []
            for col, _ in _RADAR_AXES:
                v = row.get(col, 0.5)
                v = float(v) if pd.notna(v) else 0.5
                vals.append(float(np.clip(v, 0, 1)))
            vals_closed = vals + [vals[0]]

            sym = row.get("symbol", f"#{i+1}")
            strike = row.get("strike", 0)
            opt_type = str(row.get("type", "")).upper()
            label = f"{sym} {opt_type} ${strike:.0f}"
            c = _RADAR_COLORS[i % len(_RADAR_COLORS)]

            fig.add_trace(go.Scatterpolar(
                r=vals_closed, theta=cats_closed,
                fill="toself", fillcolor=_hex_to_rgba(c, 0.15),
                line=dict(color=c, width=2), name=label,
                hovertemplate=f"<b>{label}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>",
            ))

        fig.update_layout(
            polar=dict(
                bgcolor=DARK_BG,
                radialaxis=dict(range=[0, 1], gridcolor=DARK_GRID, linecolor=DARK_BORDER,
                                tickfont=dict(color=TEXT_MUTED, size=10)),
                angularaxis=dict(gridcolor=DARK_GRID, linecolor=DARK_BORDER,
                                 tickfont=dict(color=TEXT_PRIMARY, size=11)),
            ),
            legend=dict(x=1.05, y=1, bgcolor=DARK_PANEL, bordercolor=DARK_BORDER,
                        font=dict(color=TEXT_PRIMARY, size=11)),
        )
        return self._apply(fig, "Risk Radar \u2014 Top Contracts")

    # ── HTML assembly ────────────────────────────────────────────────────

    def to_html(self) -> str:
        """Build a self-contained HTML page with 6 tabbed views."""
        views = [
            ("Contract Explorer",   self.build_contract_explorer),
            ("IV Surface",          self.build_iv_surface),
            ("Greek Landscape",     self.build_greek_landscape),
            ("P&L Scenarios",       self.build_pnl_scenario),
            ("Score Decomposition", self.build_score_decomposition),
            ("Risk Radar",          self.build_risk_radar),
        ]

        tab_divs: list[tuple[str, str]] = []
        for name, builder in views:
            try:
                fig = builder()
                div = pio.to_html(
                    fig, include_plotlyjs=False, full_html=False,
                    config=dict(responsive=True, displayModeBar=True,
                                modeBarButtonsToAdd=["toImage"], displaylogo=False))
            except Exception as exc:
                log.warning("View %s failed: %s", name, exc)
                div = (f'<div style="color:{TEXT_MUTED};padding:60px;text-align:center;'
                       f'font-size:16px;">View unavailable: {exc}</div>')
            tab_divs.append((name, div))

        # Header metadata
        n_picks = len(self.df)
        if "symbol" in self.df.columns and not self.df.empty:
            syms = sorted(self.df["symbol"].unique())
            tickers = ", ".join(syms[:8])
            if len(syms) > 8:
                tickers += f" (+{len(syms) - 8} more)"
        else:
            tickers = "\u2014"
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Top-pick summary so the header earns its space (ranked #0 by quality)
        top_pick_html = ""
        if not self.df.empty and "symbol" in self.df.columns:
            top = self.df.iloc[0]
            sym = str(top.get("symbol", ""))
            strike_raw = top.get("strike")
            opt_type = str(top.get("type", "")).upper()[:1]  # C / P
            dte_raw = top.get("DTE")
            score_raw = top.get("quality_score")
            try:
                if sym and strike_raw is not None and score_raw is not None:
                    top_pick_html = (
                        f'<span class="top-pick">Top: <strong>{sym}</strong> '
                        f'{float(strike_raw):.0f}{opt_type} '
                        f'{int(float(dte_raw))}d \u00b7 {float(score_raw):.2f}</span>'
                    )
            except (ValueError, TypeError):
                top_pick_html = ""

        # Build HTML fragments
        btn_html = "\n".join(
            f'<button class="tab{" active" if i == 0 else ""}" '
            f'onclick="switchTab({i})" title="Press {i+1} to jump here">'
            f'<span class="tab-num">{i+1}</span>{name}</button>'
            for i, (name, _) in enumerate(tab_divs)
        )
        # Every tab gets the same .tab-content class; only tab 0 starts active.
        # Using visibility (not display:none) so all plots have real layout
        # dimensions on first render and Plotly can size them correctly.
        content_html = "\n".join(
            f'<div class="tab-content{" active" if i == 0 else ""}" id="tab-{i}">{div}</div>'
            for i, (_, div) in enumerate(tab_divs)
        )
        n_tabs = len(tab_divs)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Options 3D Visualizer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{height:100%;overflow:hidden}}
body{{
  background:{DARK_BG};color:{TEXT_PRIMARY};
  font-family:'Inter',-apple-system,BlinkMacSystemFont,system-ui,sans-serif;
  -webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;
}}
.header{{
  display:flex;justify-content:space-between;align-items:center;
  padding:0 32px;height:52px;
  background:{DARK_PANEL};border-bottom:1px solid {DARK_BORDER};
  position:relative;z-index:10;
}}
.header h1{{
  font-size:15px;color:{TEXT_PRIMARY};font-weight:600;
  letter-spacing:-0.01em;
}}
.header .meta{{
  display:flex;gap:24px;align-items:center;
  font-size:12px;color:{TEXT_MUTED};font-weight:400;
}}
.header .meta .top-pick{{color:{TEXT_PRIMARY};font-weight:500}}
.header .meta .top-pick strong{{color:{ACCENT_BLUE};font-weight:600}}
.header .meta .hint{{
  color:{TEXT_MUTED};opacity:0.7;font-size:11px;
  padding:3px 8px;border:1px solid {DARK_BORDER};border-radius:4px;
}}
.tab-bar{{
  display:flex;gap:2px;padding:0 24px;height:44px;align-items:center;
  background:{DARK_PANEL};border-bottom:1px solid {DARK_BORDER};
  position:relative;z-index:10;
}}
.tab{{
  display:flex;align-items:center;gap:8px;
  padding:6px 12px;border:none;border-radius:4px;
  background:transparent;color:{TEXT_MUTED};
  font-family:inherit;font-size:13px;font-weight:500;cursor:pointer;
  transition:all 120ms cubic-bezier(0.4,0,0.2,1);
}}
.tab:hover{{color:{TEXT_PRIMARY};background:{DARK_HOVER}}}
.tab.active{{color:{TEXT_PRIMARY};background:{DARK_HOVER}}}
.tab-num{{
  display:inline-flex;align-items:center;justify-content:center;
  min-width:18px;height:18px;padding:0 4px;
  background:rgba(255,255,255,0.06);border-radius:3px;
  font-size:10px;font-weight:500;color:{TEXT_MUTED};
  transition:all 120ms cubic-bezier(0.4,0,0.2,1);
}}
.tab.active .tab-num{{background:{ACCENT_BLUE};color:#ffffff}}
.stage{{position:absolute;top:96px;left:0;right:0;bottom:0;background:{DARK_BG}}}
.tab-content{{
  position:absolute;top:0;left:0;right:0;bottom:0;
  visibility:hidden;pointer-events:none;
  padding:8px 16px 16px 16px;
}}
.tab-content.active{{visibility:visible;pointer-events:auto}}
/* Fix: anonymous pio.to_html wrapper has no height \u2192 plot collapses.
   Flex-fill propagates full height through any number of wrappers. */
.tab-content > div{{width:100%;height:100%;display:flex;flex-direction:column}}
.tab-content .plotly-graph-div,.tab-content .js-plotly-plot{{
  width:100%!important;height:100%!important;flex:1
}}
</style>
</head>
<body>
<div class="header">
  <h1>Options 3D Visualizer</h1>
  <div class="meta">
    {top_pick_html}
    <span>{n_picks} contracts</span>
    <span>{tickers}</span>
    <span>{now}</span>
    <span class="hint">1\u2013{n_tabs} \u00b7 \u2190 \u2192</span>
  </div>
</div>
<div class="tab-bar">
{btn_html}
</div>
<div class="stage">
{content_html}
</div>
<script>
var N_TABS = {n_tabs};
var currentTab = 0;

function resizeAll(){{
  document.querySelectorAll('.js-plotly-plot').forEach(function(p){{
    try {{ Plotly.Plots.resize(p); }} catch(e) {{}}
  }});
}}

function switchTab(i){{
  if(i<0) i = N_TABS - 1;
  if(i>=N_TABS) i = 0;
  currentTab = i;
  document.querySelectorAll('.tab-content').forEach(function(d){{d.classList.remove('active')}});
  document.querySelectorAll('.tab').forEach(function(b){{b.classList.remove('active')}});
  document.getElementById('tab-'+i).classList.add('active');
  document.querySelectorAll('.tab')[i].classList.add('active');
  var p=document.getElementById('tab-'+i).querySelector('.js-plotly-plot');
  if(p) try {{ Plotly.Plots.resize(p); }} catch(e) {{}}
}}

// Resize every plot once the page is fully loaded — cheap insurance
// against CDN / layout-timing weirdness on first render.
window.addEventListener('load', function(){{
  setTimeout(resizeAll, 50);
  setTimeout(resizeAll, 300);
}});

// Re-fit plots when the browser window changes size.
window.addEventListener('resize', resizeAll);

// Keyboard shortcuts: 1..N to jump, arrows to cycle.
document.addEventListener('keydown', function(e){{
  if(e.target && (e.target.tagName==='INPUT' || e.target.tagName==='TEXTAREA')) return;
  if(e.key >= '1' && e.key <= String.fromCharCode(48+N_TABS)){{
    var idx = parseInt(e.key, 10) - 1;
    if(idx>=0 && idx<N_TABS) switchTab(idx);
  }} else if(e.key === 'ArrowRight'){{
    switchTab(currentTab + 1);
  }} else if(e.key === 'ArrowLeft'){{
    switchTab(currentTab - 1);
  }}
}});
</script>
</body>
</html>"""

    # ── Launch ───────────────────────────────────────────────────────────

    def show(self) -> None:
        """Write HTML to a temp file and open in the default browser."""
        if self.df.empty:
            print("  No scan results to visualise.")
            return
        html = self.to_html()
        path = Path(tempfile.gettempdir()) / f"options_viz_{datetime.now():%Y%m%d_%H%M%S}.html"
        path.write_text(html, encoding="utf-8")
        try:
            webbrowser.open(f"file://{path}")
        except Exception:
            print(f"  Open manually: {path}")


__all__ = ["OptionsVisualizer", "HAS_PLOTLY"]
