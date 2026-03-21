"""3D ASCII risk surface visualization for option P&L across price and IV shocks."""

import shutil
import numpy as np

from .utils import bs_price


def compute_pnl_grid(option_type, S, K, T, r, sigma, entry_price,
                     q=0.0, n_price=40, n_iv=20,
                     price_range=(-0.25, 0.25), iv_range=(-0.50, 0.50)):
    """Compute P&L grid over price shocks × IV shocks using full BS repricing.

    Returns:
        (price_shocks_1d, iv_shocks_1d, pnl_2d) where pnl_2d[i,j] is the P&L
        at price_shocks[i] and iv_shocks[j].
    """
    price_shocks = np.linspace(price_range[0], price_range[1], n_price)
    iv_shocks = np.linspace(iv_range[0], iv_range[1], n_iv)

    price_mesh, iv_mesh = np.meshgrid(price_shocks, iv_shocks, indexing='ij')
    S_grid = S * (1.0 + price_mesh)
    sigma_grid = np.maximum(sigma * (1.0 + iv_mesh), 0.01)

    new_prices = bs_price(option_type, S_grid, K, T, r, sigma_grid, q)
    pnl = new_prices - entry_price

    return price_shocks, iv_shocks, pnl


def render_surface(price_shocks, iv_shocks, pnl, option_desc="", width=100):
    """Render an isometric ASCII 3D surface of the P&L grid to stdout."""
    try:
        from . import formatting as fmt
        has_fmt = True
        use_color = fmt.supports_color()
    except ImportError:
        has_fmt = False
        use_color = False

    n_price, n_iv = pnl.shape
    pnl_min, pnl_max = float(np.nanmin(pnl)), float(np.nanmax(pnl))

    # Graceful skip for narrow terminals
    try:
        term_w = shutil.get_terminal_size(fallback=(100, 24)).columns
    except Exception:
        term_w = 100
    if term_w < 80:
        print("  [Terminal too narrow for risk surface — need ≥80 columns]")
        return

    screen_w = min(width, term_w) - 4
    screen_h = 35

    # Shading characters from sparse to dense
    shade_chars = " .:-=+*#%@"
    n_shades = len(shade_chars)

    # Initialize screen buffer
    buf = [[' '] * (screen_w + 1) for _ in range(screen_h + 1)]
    # Depth buffer for painter's algorithm (higher = closer to viewer)
    zbuf = [[-1e9] * (screen_w + 1) for _ in range(screen_h + 1)]

    pnl_span = pnl_max - pnl_min if pnl_max != pnl_min else 1.0

    # Color buffer (store ANSI codes per cell)
    cbuf = [[''] * (screen_w + 1) for _ in range(screen_h + 1)]

    # Iterate back-to-front for depth occlusion (painter's algorithm on iy)
    for iy in range(n_iv - 1, -1, -1):
        for ix in range(n_price):
            val = pnl[ix, iy]
            if np.isnan(val):
                continue

            fx = ix / max(n_price - 1, 1)
            fy = iy / max(n_iv - 1, 1)
            fz = (val - pnl_min) / pnl_span

            # Isometric projection
            col = int(screen_w * 0.5 + (fx - 0.5) * screen_w * 0.45 - (fy - 0.5) * screen_w * 0.15)
            row = int(screen_h * 0.85 - (fx + fy) * screen_h * 0.15 - fz * screen_h * 0.55)

            if 0 <= row <= screen_h and 0 <= col <= screen_w:
                depth = fx + fy  # Simple depth ordering
                if depth >= zbuf[row][col]:
                    zbuf[row][col] = depth
                    # Map |pnl| magnitude to shade index
                    shade_idx = int(fz * (n_shades - 1))
                    shade_idx = max(0, min(n_shades - 1, shade_idx))
                    buf[row][col] = shade_chars[shade_idx]

                    if use_color:
                        if val > pnl_span * 0.02:
                            cbuf[row][col] = fmt.Colors.GREEN
                        elif val < -pnl_span * 0.02:
                            cbuf[row][col] = fmt.Colors.RED
                        else:
                            cbuf[row][col] = fmt.Colors.BRIGHT_BLACK

    # Header
    print()
    if has_fmt:
        print(fmt.draw_separator(min(width, term_w)))
    title = f"  3D P&L Risk Surface{f'  —  {option_desc}' if option_desc else ''}"
    if use_color:
        print(fmt.colorize(title, fmt.Colors.BRIGHT_CYAN, bold=True))
    else:
        print(title)

    # Axis labels
    p_lo = f"{price_shocks[0]*100:+.0f}%"
    p_hi = f"{price_shocks[-1]*100:+.0f}%"
    iv_lo = f"{iv_shocks[0]*100:+.0f}%"
    iv_hi = f"{iv_shocks[-1]*100:+.0f}%"
    print(f"  Price shock: {p_lo} ← → {p_hi}   |   IV shock: {iv_lo} ← → {iv_hi}")
    print(f"  P&L range: ${pnl_min:+,.2f}  to  ${pnl_max:+,.2f}")
    print()

    # Render buffer
    for row_idx in range(screen_h + 1):
        line_parts = []
        for col_idx in range(screen_w + 1):
            ch = buf[row_idx][col_idx]
            if use_color and ch != ' ' and cbuf[row_idx][col_idx]:
                line_parts.append(f"{cbuf[row_idx][col_idx]}{ch}{fmt.Colors.RESET}")
            else:
                line_parts.append(ch)
        line = ''.join(line_parts).rstrip()
        if line:
            print(f"  {line}")

    # Legend bar
    print()
    legend = "  Legend: "
    for i, ch in enumerate(shade_chars):
        if ch == ' ':
            continue
        frac = i / (n_shades - 1)
        val = pnl_min + frac * pnl_span
        label = f"${val:+,.0f}"
        token = f"{ch}={label} "
        if use_color:
            color = fmt.Colors.GREEN if val > 0 else (fmt.Colors.RED if val < 0 else fmt.Colors.BRIGHT_BLACK)
            token = fmt.colorize(token, color)
        legend += token
    print(legend)
    if has_fmt:
        print(fmt.draw_separator(min(width, term_w)))
    print()


def print_risk_surface(option_row, underlying_price, rfr, width=100):
    """Public entry point: extract option params and render the risk surface.

    Args:
        option_row: dict-like option record (from DataFrame row or dict)
        underlying_price: current underlying price
        rfr: risk-free rate
        width: display width
    """
    opt = option_row if isinstance(option_row, dict) else option_row.to_dict()

    option_type = "call" if opt.get("optionType", "call").lower() == "call" else "put"
    K = float(opt.get("strike", underlying_price))
    T = float(opt.get("T_years", opt.get("dte", 30) / 365.0))
    sigma = float(opt.get("impliedVolatility", 0.30))
    entry_price = opt.get("ask") if opt.get("ask") is not None else opt.get("lastPrice", 0)
    entry_price = float(entry_price) if entry_price else 0.0

    if entry_price <= 0 or sigma <= 0 or T <= 0:
        print("  [Cannot render risk surface — missing price/IV/DTE data]")
        return

    # Build description string
    symbol = opt.get("symbol", "")
    strike_str = f"${K:.0f}" if K == int(K) else f"${K:.2f}"
    desc = f"{symbol} {strike_str} {option_type.upper()}"

    price_shocks, iv_shocks, pnl_grid = compute_pnl_grid(
        option_type, underlying_price, K, T, rfr, sigma, entry_price,
    )
    render_surface(price_shocks, iv_shocks, pnl_grid, option_desc=desc, width=width)
