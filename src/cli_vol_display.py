"""CLI display logic for Volatility Analytics."""

import math
from typing import Optional
from .vol_analytics import compute_vol_cone, compute_iv_surface, classify_vol_regime

try:
    from . import formatting as fmt
    HAS_FMT = True
except ImportError:
    HAS_FMT = False
    fmt = None


def _c(text: str, color: str = "", bold: bool = False) -> str:
    """Color helper that degrades gracefully."""
    if HAS_FMT and fmt and color:
        return fmt.colorize(str(text), color, bold=bold)
    return str(text)


def _sep(width: int = 90) -> str:
    line = "  " + "\u2500" * (width - 2)
    if HAS_FMT and fmt:
        return fmt.colorize(line, fmt.Colors.DIM)
    return line


def _ordinal_suffix(n: int) -> str:
    """Return ordinal suffix for an integer: 1->st, 2->nd, 3->rd, else->th."""
    if 11 <= (n % 100) <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def print_vol_cone(ticker: str, current_iv: Optional[float] = None, width: int = 90) -> None:
    """Print a formatted vol cone table to terminal."""
    cone = compute_vol_cone(ticker)
    if cone is None:
        print(f"\n  Vol cone unavailable for {ticker}")
        return

    print()
    header = f"  VOL CONE  \u2014  {ticker}"
    if HAS_FMT and fmt:
        print(fmt.colorize(header, fmt.Colors.BRIGHT_CYAN, bold=True))
    else:
        print(header)
    print(_sep(width))

    col_hdr = f"  {'Window':<8}  {'Min':>7}  {'25th':>7}  {'Median':>7}  {'75th':>7}  {'Max':>7}  {'Current':>8}  {'Pctile':>7}"
    if HAS_FMT and fmt:
        print(fmt.colorize(col_hdr, fmt.Colors.BOLD))
    else:
        print(col_hdr)

    window_labels = {10: "10d", 21: "21d", 30: "30d", 63: "63d", 126: "126d", 252: "252d"}
    for w in sorted(cone.keys()):
        d = cone[w]
        label = window_labels.get(w, f"{w}d")
        pctile_int = int(round(d["pctile"] * 100))
        suf = _ordinal_suffix(pctile_int)
        pctile_str = f"{pctile_int}{suf}"

        # Color current by percentile
        cur_val = d["current"]
        cur_str = f"{cur_val*100:.1f}%"
        if HAS_FMT and fmt:
            if d["pctile"] >= 0.75:
                cur_colored = fmt.colorize(cur_str, fmt.Colors.RED, bold=True)
            elif d["pctile"] >= 0.50:
                cur_colored = fmt.colorize(cur_str, fmt.Colors.YELLOW)
            else:
                cur_colored = fmt.colorize(cur_str, fmt.Colors.GREEN)
        else:
            cur_colored = cur_str

        row = (
            f"  {label:<8}"
            f"  {d['min']*100:>6.1f}%"
            f"  {d['p25']*100:>6.1f}%"
            f"  {d['median']*100:>6.1f}%"
            f"  {d['p75']*100:>6.1f}%"
            f"  {d['max']*100:>6.1f}%"
            f"  {cur_colored:>8}"
            f"  {pctile_str:>7}"
        )
        print(row)

    print(_sep(width))

    # IV vs 30d median HV line
    if current_iv is not None and 30 in cone:
        hv_30_median = cone[30]["median"]
        if hv_30_median > 0:
            ratio = current_iv / hv_30_median
            iv_pct_str = f"{current_iv*100:.1f}%"
            if ratio > 1.20:
                verdict = "RICH"
                iv_color = fmt.Colors.RED if HAS_FMT and fmt else ""
            elif ratio < 0.85:
                verdict = "CHEAP"
                iv_color = fmt.Colors.GREEN if HAS_FMT and fmt else ""
            else:
                verdict = "FAIR"
                iv_color = fmt.Colors.YELLOW if HAS_FMT and fmt else ""
            iv_line = f"  Current IV: {iv_pct_str}  \u2192  {verdict} vs 30d median HV ({ratio:.2f}x)"
            if HAS_FMT and fmt and iv_color:
                print(fmt.colorize(iv_line, iv_color))
            else:
                print(iv_line)
    print()


def print_iv_surface(ticker: str, spot: Optional[float] = None, width: int = 90) -> None:
    """Print a compact IV surface table."""
    surface = compute_iv_surface(ticker)
    if surface is None or surface.empty:
        print(f"\n  IV surface unavailable for {ticker}")
        return

    spot_str = f"  (spot ${spot:.2f})" if spot else ""
    print()
    header = f"  IV SURFACE  \u2014  {ticker}{spot_str}"
    if HAS_FMT and fmt:
        print(fmt.colorize(header, fmt.Colors.BRIGHT_CYAN, bold=True))
    else:
        print(header)
    print(_sep(width))

    _skew_col = "25\u0394 Skew"
    col_hdr = f"  {'Expiry':<12}  {'DTE':>5}  {'ATM IV':>7}  {_skew_col:>9}  {'Structure'}"
    if HAS_FMT and fmt:
        print(fmt.colorize(col_hdr, fmt.Colors.BOLD))
    else:
        print(col_hdr)

    prev_atm_iv = None
    for _, row in surface.iterrows():
        atm_iv = row["atm_iv"]
        dte = int(row["dte"])
        exp_str = str(row["expiration"])[:10]

        # Term structure label
        if prev_atm_iv is None:
            structure = "FRONT"
            struct_color = fmt.Colors.DIM if HAS_FMT and fmt else ""
        else:
            diff = atm_iv - prev_atm_iv
            if diff > 0.005:
                structure = "CONTANGO"
                struct_color = fmt.Colors.GREEN if HAS_FMT and fmt else ""
            elif diff < -0.005:
                structure = "BACKWARDATION"
                struct_color = fmt.Colors.RED if HAS_FMT and fmt else ""
            else:
                structure = "FLAT"
                struct_color = fmt.Colors.YELLOW if HAS_FMT and fmt else ""

        atm_iv_str = f"{atm_iv*100:.1f}%"

        skew_25d = row.get("skew_25d")
        if skew_25d is not None and not (isinstance(skew_25d, float) and math.isnan(skew_25d)):
            skew_val = float(skew_25d)
            skew_str = f"{skew_val*100:+.1f}%"
            if HAS_FMT and fmt:
                if skew_val > 0.05:
                    skew_colored = fmt.colorize(skew_str, fmt.Colors.RED)
                elif skew_val > 0.02:
                    skew_colored = fmt.colorize(skew_str, fmt.Colors.YELLOW)
                else:
                    skew_colored = fmt.colorize(skew_str, fmt.Colors.GREEN)
            else:
                skew_colored = skew_str
        else:
            skew_colored = "   n/a"

        if HAS_FMT and fmt and struct_color:
            struct_colored = fmt.colorize(structure, struct_color)
        else:
            struct_colored = structure

        print(f"  {exp_str:<12}  {dte:>5}  {atm_iv_str:>7}  {skew_colored:>9}  {struct_colored}")
        prev_atm_iv = atm_iv

    print()


def print_regime_summary(ticker: str, current_iv: Optional[float] = None, width: int = 90) -> None:
    """
    One-line regime summary suitable for embedding in scan output.
    Format:
      Vol Regime [TICKER]: HIGH_IV (82nd pctile) | HV/IV: 0.76x RICH | CONTANGO | PUT_SKEW -> Sell premium
    """
    try:
        reg = classify_vol_regime(ticker, current_iv=current_iv)
        regime = reg.get("regime", "UNKNOWN")
        pctile = reg.get("iv_pctile_30d")
        hv_iv = reg.get("hv_iv_ratio")
        ts = reg.get("term_structure", "")
        skew = reg.get("skew_direction", "")
        rec = reg.get("recommendation", "")

        pctile_str = ""
        if pctile is not None:
            p_int = int(round(pctile * 100))
            suf = _ordinal_suffix(p_int)
            pctile_str = f" ({p_int}{suf} pctile)"

        hv_iv_str = ""
        if hv_iv is not None:
            label = "RICH" if hv_iv < 0.85 else ("FAIR" if hv_iv < 1.15 else "CHEAP")
            hv_iv_str = f" | HV/IV: {hv_iv:.2f}x {label}"

        ts_str = f" | {ts}" if ts and ts != "UNKNOWN" else ""
        skew_str = f" | {skew}" if skew and skew != "UNKNOWN" else ""
        rec_short = rec[:60] if rec else ""
        rec_str = f" \u2192 {rec_short}" if rec_short else ""

        line = f"  Vol Regime [{ticker}]: {regime}{pctile_str}{hv_iv_str}{ts_str}{skew_str}{rec_str}"

        if HAS_FMT and fmt:
            if regime in ("HIGH_IV", "EXTREME"):
                color = fmt.Colors.RED
            elif regime == "LOW_IV":
                color = fmt.Colors.GREEN
            else:
                color = fmt.Colors.YELLOW
            print(fmt.colorize(line, color))
        else:
            print(line)
    except Exception:
        pass
