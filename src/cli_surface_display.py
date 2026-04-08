"""CLI display logic for 3D risk surfaces."""

import locale
import shutil
import numpy as np
from .visual_surface import compute_pnl_grid, compute_greek_grid

try:
    from . import formatting as fmt
    HAS_FMT = True
except ImportError:
    HAS_FMT = False
    fmt = None


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _pnl_to_rgb(frac):
    """Map normalized P&L fraction [0,1] to (R, G, B) tuple."""
    stops = [0.0, 0.4, 0.5, 0.6, 1.0]
    rs = [180, 255, 100, 80, 40]
    gs = [30, 80, 100, 220, 255]
    bs = [30, 60, 100, 80, 160]
    r = int(np.interp(frac, stops, rs))
    g = int(np.interp(frac, stops, gs))
    b = int(np.interp(frac, stops, bs))
    return r, g, b


def _greek_to_rgb(greek_name, frac):
    """Map normalized Greek fraction [0,1] to (R, G, B) for sequential palettes."""
    if greek_name == 'delta':
        return int(30 + 50 * (1 - frac)), int(80 + 175 * frac), int(200 * (1 - frac) + 60 * frac)
    elif greek_name == 'gamma':
        return int(60 + 195 * frac), int(50 + 200 * frac), int(20 + 20 * frac)
    elif greek_name == 'vega':
        return int(60 + 195 * frac), int(20 + 60 * frac), int(80 + 175 * frac)
    elif greek_name == 'theta':
        return int(80 + 175 * frac), int(40 + 20 * frac), int(40 + 20 * frac)
    v = int(60 + 195 * frac)
    return v, v, v


def _get_color(frac, val, pnl_span, surface_type, use_truecolor, use_color, fmt_mod):
    """Return ANSI color string for a surface point."""
    if use_truecolor:
        if surface_type == 'pnl':
            r, g, b = _pnl_to_rgb(frac)
        else:
            r, g, b = _greek_to_rgb(surface_type, frac)
        return fmt_mod.rgb_fg(r, g, b)
    elif use_color:
        if surface_type != 'pnl':
            if frac > 0.66:
                return fmt_mod.Colors.BRIGHT_YELLOW
            elif frac > 0.33:
                return fmt_mod.Colors.YELLOW
            else:
                return fmt_mod.Colors.BRIGHT_BLACK
        if val > pnl_span * 0.02:
            return fmt_mod.Colors.GREEN
        elif val < -pnl_span * 0.02:
            return fmt_mod.Colors.RED
        else:
            return fmt_mod.Colors.BRIGHT_BLACK
    return ''


# ---------------------------------------------------------------------------
# Braille canvas
# ---------------------------------------------------------------------------

class BrailleCanvas:
    """2D pixel canvas rendered to Unicode braille characters (U+2800-U+28FF)."""
    _DOT_MAP = {
        (0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (0, 3): 0x40,
        (1, 0): 0x08, (1, 1): 0x10, (1, 2): 0x20, (1, 3): 0x80,
    }

    def __init__(self, term_cols, term_rows):
        self.tw = term_cols
        self.th = term_rows
        self.pw = term_cols * 2
        self.ph = term_rows * 4
        self._grid = [[0] * term_cols for _ in range(term_rows)]
        self._color = [[''] * term_cols for _ in range(term_rows)]

    def set_pixel(self, px, py, color=''):
        if 0 <= px < self.pw and 0 <= py < self.ph:
            tc = px // 2
            tr = py // 4
            dx = px % 2
            dy = py % 4
            self._grid[tr][tc] |= self._DOT_MAP[(dx, dy)]
            if color:
                self._color[tr][tc] = color

    def render_lines(self, reset_code='\033[0m'):
        lines = []
        for tr in range(self.th):
            parts = []
            for tc in range(self.tw):
                bits = self._grid[tr][tc]
                ch = chr(0x2800 + bits)
                c = self._color[tr][tc]
                if c and bits:
                    parts.append(f"{c}{ch}{reset_code}")
                else:
                    parts.append(ch)
            lines.append(''.join(parts).rstrip())
        return lines


def _braille_line(canvas, x0, y0, x1, y1, color, zbuf, depth):
    """Draw a line on the braille canvas with depth test."""
    dx_abs = abs(x1 - x0)
    dy_abs = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx_abs - dy_abs
    while True:
        if 0 <= x0 < canvas.pw and 0 <= y0 < canvas.ph:
            if depth >= zbuf[y0, x0]:
                zbuf[y0, x0] = depth
                canvas.set_pixel(x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy_abs:
            err -= dy_abs
            x0 += sx
        if e2 < dx_abs:
            err += dx_abs
            y0 += sy


# ---------------------------------------------------------------------------
# Visualization Helpers
# ---------------------------------------------------------------------------

def _extract_contour_segments(grid_2d, level):
    """Return list of (x1,y1,x2,y2) segment endpoints in grid-fraction coords."""
    nr, nc = grid_2d.shape
    segments = []
    for i in range(nr - 1):
        for j in range(nc - 1):
            v00, v10, v01, v11 = grid_2d[i, j], grid_2d[i + 1, j], grid_2d[i, j + 1], grid_2d[i + 1, j + 1]
            case = 0
            if v00 >= level: case |= 1
            if v10 >= level: case |= 2
            if v01 >= level: case |= 4
            if v11 >= level: case |= 8
            if case == 0 or case == 15: continue

            def _interp_x(va, vb, row, col_a, col_b):
                t = (level - va) / (vb - va) if vb != va else 0.5
                return (col_a + t * (col_b - col_a)) / max(nc - 1, 1), row / max(nr - 1, 1)

            def _interp_y(va, vb, col, row_a, row_b):
                t = (level - va) / (vb - va) if vb != va else 0.5
                return col / max(nc - 1, 1), (row_a + t * (row_b - row_a)) / max(nr - 1, 1)

            top = _interp_x(v00, v10, j, i, i + 1)
            bottom = _interp_x(v01, v11, j + 1, i, i + 1)
            left = _interp_y(v00, v01, i, j, j + 1)
            right = _interp_y(v10, v11, i + 1, j, j + 1)

            edges = {
                1: [(top, left)], 2: [(top, right)], 3: [(left, right)],
                4: [(left, bottom)], 5: [(top, bottom)], 6: [(top, left), (right, bottom)],
                7: [(right, bottom)], 8: [(right, bottom)], 9: [(top, right), (left, bottom)],
                10: [(top, bottom)], 11: [(left, bottom)], 12: [(left, right)],
                13: [(top, right)], 14: [(top, left)],
            }
            for seg in edges.get(case, []):
                segments.append((seg[0][0], seg[0][1], seg[1][0], seg[1][1]))
    return segments


def _pick_contour_levels(grid_2d, surface_type, n_levels=7):
    """Choose contour levels."""
    vmin, vmax = float(np.nanmin(grid_2d)), float(np.nanmax(grid_2d))
    if vmin == vmax: return []
    levels = np.linspace(vmin, vmax, n_levels + 2)[1:-1].tolist()
    if surface_type == 'pnl' and vmin < 0 < vmax:
        levels = [l for l in levels if abs(l) > (vmax - vmin) * 0.05]
        levels.append(0.0)
        levels.sort()
    return levels


def _iso_project(fx, fy, fz, w, h):
    """Map (fx, fy, fz) in [0,1]^3 to (col, row) screen coords."""
    col = w * 0.5 + (fx - 0.5) * w * 0.45 - (fy - 0.5) * w * 0.15
    row = h * 0.85 - (fx + fy) * h * 0.15 - fz * h * 0.55
    return col, row


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render_surface(price_shocks, iv_shocks, grid_2d, option_desc="", width=100,
                   surface_type="pnl", show_contours=True):
    """Render an isometric ASCII 3D surface to stdout."""
    use_color = fmt.supports_color() if HAS_FMT else False
    use_truecolor = fmt.supports_truecolor() if HAS_FMT else False
    n_price, n_iv = grid_2d.shape
    val_min, val_max = float(np.nanmin(grid_2d)), float(np.nanmax(grid_2d))
    try:
        term_w = shutil.get_terminal_size(fallback=(100, 24)).columns
    except Exception:
        term_w = 100
    if term_w < 80:
        print("  [Terminal too narrow for risk surface]")
        return
    screen_w, screen_h = min(width, term_w) - 4, 35
    shade_chars = " .:-=+*#%@"
    n_shades = len(shade_chars)
    buf = [[' '] * (screen_w + 1) for _ in range(screen_h + 1)]
    zbuf = [[-1e9] * (screen_w + 1) for _ in range(screen_h + 1)]
    cbuf = [[''] * (screen_w + 1) for _ in range(screen_h + 1)]
    val_span = val_max - val_min if val_max != val_min else 1.0

    for iy in range(n_iv - 1, -1, -1):
        for ix in range(n_price):
            val = grid_2d[ix, iy]
            if np.isnan(val): continue
            fx, fy, fz = ix / max(n_price - 1, 1), iy / max(n_iv - 1, 1), (val - val_min) / val_span
            col, row = _iso_project(fx, fy, fz, screen_w, screen_h)
            col, row = int(col), int(row)
            if 0 <= row <= screen_h and 0 <= col <= screen_w:
                depth = fx + fy
                if depth >= zbuf[row][col]:
                    zbuf[row][col] = depth
                    shade_idx = max(0, min(n_shades - 1, int(fz * (n_shades - 1))))
                    buf[row][col] = shade_chars[shade_idx]
                    cbuf[row][col] = _get_color(fz, val, val_span, surface_type, use_truecolor, use_color, fmt)

    _print_surface_output(buf, cbuf, screen_w, screen_h, price_shocks, iv_shocks,
                          val_min, val_max, val_span, option_desc, surface_type,
                          shade_chars, width, term_w, HAS_FMT, use_color, use_truecolor)


def render_surface_braille(price_shocks, iv_shocks, grid_2d, option_desc="", width=100,
                           surface_type="pnl", show_contours=True):
    """Render an isometric 3D surface using braille characters."""
    use_color = fmt.supports_color() if HAS_FMT else False
    use_truecolor = fmt.supports_truecolor() if HAS_FMT else False
    n_price, n_iv = grid_2d.shape
    val_min, val_max = float(np.nanmin(grid_2d)), float(np.nanmax(grid_2d))
    try:
        term_w = shutil.get_terminal_size(fallback=(100, 24)).columns
    except Exception:
        term_w = 100
    if term_w < 80:
        print("  [Terminal too narrow for risk surface]")
        return
    term_cols, term_rows = min(width, term_w) - 4, 35
    canvas = BrailleCanvas(term_cols, term_rows)
    zbuf = np.full((canvas.ph, canvas.pw), -1e9)
    val_span = val_max - val_min if val_max != val_min else 1.0

    px_coords = np.zeros((n_price, n_iv), dtype=int)
    py_coords = np.zeros((n_price, n_iv), dtype=int)
    depths = np.zeros((n_price, n_iv))
    colors = [['' for _ in range(n_iv)] for _ in range(n_price)]

    for ix in range(n_price):
        for iy in range(n_iv):
            val = grid_2d[ix, iy]
            if np.isnan(val):
                px_coords[ix, iy] = -1
                continue
            fx, fy, fz = ix / max(n_price - 1, 1), iy / max(n_iv - 1, 1), (val - val_min) / val_span
            col, row = _iso_project(fx, fy, fz, canvas.pw, canvas.ph)
            px_coords[ix, iy], py_coords[ix, iy] = int(col), int(row)
            depths[ix, iy] = fx + fy
            colors[ix][iy] = _get_color(fz, val, val_span, surface_type, use_truecolor, use_color, fmt)

    for iy in range(n_iv - 1, -1, -1):
        for ix in range(n_price):
            x0, y0 = px_coords[ix, iy], py_coords[ix, iy]
            if x0 < 0: continue
            d, c = depths[ix, iy], colors[ix][iy]
            if ix + 1 < n_price and px_coords[ix + 1, iy] >= 0:
                _braille_line(canvas, x0, y0, px_coords[ix + 1, iy], py_coords[ix + 1, iy], c, zbuf, d)
            if iy - 1 >= 0 and px_coords[ix, iy - 1] >= 0:
                _braille_line(canvas, x0, y0, px_coords[ix, iy - 1], py_coords[ix, iy - 1], c, zbuf, d)
            if 0 <= x0 < canvas.pw and 0 <= y0 < canvas.ph:
                if d >= zbuf[y0, x0]:
                    zbuf[y0, x0] = d
                    canvas.set_pixel(x0, y0, c)

    if show_contours:
        levels = _pick_contour_levels(grid_2d, surface_type)
        for level in levels:
            is_be = (surface_type == 'pnl' and abs(level) < val_span * 0.01)
            if use_truecolor and HAS_FMT:
                c_color = fmt.rgb_fg(255, 255, 255) if is_be else fmt.rgb_fg(200, 200, 100)
            elif use_color and HAS_FMT:
                c_color = fmt.Colors.BRIGHT_WHITE if is_be else fmt.Colors.BRIGHT_YELLOW
            else:
                c_color = ''
            segments = _extract_contour_segments(grid_2d, level)
            fz_level = (level - val_min) / val_span
            for x1f, y1f, x2f, y2f in segments:
                c1, r1 = _iso_project(x1f, y1f, fz_level, canvas.pw, canvas.ph)
                c2, r2 = _iso_project(x2f, y2f, fz_level, canvas.pw, canvas.ph)
                _braille_line(canvas, int(c1), int(r1), int(c2), int(r2), c_color, zbuf, 100.0)

    print()
    if HAS_FMT: print(fmt.draw_separator(min(width, term_w)))
    stype_label = surface_type.upper() if surface_type != 'pnl' else 'P&L'
    desc_part = f"  \u2014  {option_desc}" if option_desc else ""
    title = f"  3D {stype_label} Risk Surface{desc_part}"
    if use_color and HAS_FMT: print(fmt.colorize(title, fmt.Colors.BRIGHT_CYAN, bold=True))
    else: print(title)
    print(f"  Price shock: {price_shocks[0]*100:+.0f}% <- -> {price_shocks[-1]*100:+.0f}%   |   IV shock: {iv_shocks[0]*100:+.0f}% <- -> {iv_shocks[-1]*100:+.0f}%")
    if surface_type == 'pnl': print(f"  P&L range: ${val_min:+,.2f}  to  ${val_max:+,.2f}")
    else: print(f"  {surface_type.capitalize()} range: {val_min:+.4f}  to  {val_max:+.4f}")
    print()
    reset = fmt.Colors.RESET if (use_color and HAS_FMT) else ''
    for line in canvas.render_lines(reset):
        if line.strip(): print(f"  {line}")
    _print_legend(val_min, val_max, val_span, surface_type, use_color, use_truecolor, HAS_FMT, width, term_w)
    print()


def _print_surface_output(buf, cbuf, screen_w, screen_h, price_shocks, iv_shocks,
                          val_min, val_max, val_span, option_desc, surface_type,
                          shade_chars, width, term_w, has_fmt, use_color, use_truecolor):
    """Print header, buffer, and legend for ASCII mode."""
    print()
    if has_fmt: print(fmt.draw_separator(min(width, term_w)))
    stype_label = surface_type.upper() if surface_type != 'pnl' else 'P&L'
    desc_part = f"  \u2014  {option_desc}" if option_desc else ""
    title = f"  3D {stype_label} Risk Surface{desc_part}"
    if use_color: print(fmt.colorize(title, fmt.Colors.BRIGHT_CYAN, bold=True))
    else: print(title)
    print(f"  Price shock: {price_shocks[0]*100:+.0f}% <- -> {price_shocks[-1]*100:+.0f}%   |   IV shock: {iv_shocks[0]*100:+.0f}% <- -> {iv_shocks[-1]*100:+.0f}%")
    if surface_type == 'pnl': print(f"  P&L range: ${val_min:+,.2f}  to  ${val_max:+,.2f}")
    else: print(f"  {surface_type.capitalize()} range: {val_min:+.4f}  to  {val_max:+.4f}")
    print()
    for r in range(screen_h + 1):
        line_parts = []
        for c in range(screen_w + 1):
            ch = buf[r][c]
            if use_color and ch != ' ' and cbuf[r][c]:
                line_parts.append(f"{cbuf[r][c]}{ch}{fmt.Colors.RESET}")
            else: line_parts.append(ch)
        line = ''.join(line_parts).rstrip()
        if line: print(f"  {line}")
    _print_legend(val_min, val_max, val_span, surface_type, use_color, use_truecolor, has_fmt, width, term_w, shade_chars=shade_chars)
    print()


def _print_legend(val_min, val_max, val_span, surface_type, use_color, use_truecolor, has_fmt, width, term_w, shade_chars=None):
    """Print gradient legend bar."""
    print()
    if use_truecolor and HAS_FMT:
        n_steps, legend = 30, "  Legend: "
        for i in range(n_steps):
            frac = i / max(n_steps - 1, 1)
            r, g, b = _pnl_to_rgb(frac) if surface_type == 'pnl' else _greek_to_rgb(surface_type, frac)
            legend += f"{fmt.rgb_fg(r, g, b)}\u2588{fmt.Colors.RESET}"
        legend += (f"  ${val_min:+,.0f} to ${val_max:+,.0f}" if surface_type == 'pnl' else f"  {val_min:+.4f} to {val_max:+.4f}")
        print(legend)
    elif shade_chars and use_color and HAS_FMT:
        legend, n_sh = "  Legend: ", len(shade_chars)
        for i, ch in enumerate(shade_chars):
            if ch == ' ': continue
            frac = i / (n_sh - 1)
            val = val_min + frac * val_span
            if surface_type == 'pnl':
                label, color = f"${val:+,.0f}", (fmt.Colors.GREEN if val > 0 else (fmt.Colors.RED if val < 0 else fmt.Colors.BRIGHT_BLACK))
            else:
                label, color = f"{val:+.3f}", (fmt.Colors.BRIGHT_YELLOW if frac > 0.5 else fmt.Colors.BRIGHT_BLACK)
            legend += fmt.colorize(f"{ch}={label} ", color)
        print(legend)
    else:
        print(f"  Range: ${val_min:+,.2f} to ${val_max:+,.2f}" if surface_type == 'pnl' else f"  Range: {val_min:+.4f} to {val_max:+.4f}")

    if use_color and HAS_FMT:
        print("  Contour lines: " + fmt.colorize("white", fmt.Colors.BRIGHT_WHITE, bold=True) + " = breakeven  |  " + fmt.colorize("yellow", fmt.Colors.BRIGHT_YELLOW) + " = iso-value")
    if has_fmt: print(fmt.draw_separator(min(width, term_w)))


def _detect_utf8():
    """Check if terminal likely supports Unicode braille characters."""
    try:
        enc = locale.getpreferredencoding(False)
        return enc.lower().replace('-', '') in ('utf8', 'utf16', 'utf32')
    except Exception: return False


def print_risk_surface(option_row, underlying_price, rfr, width=100,
                       mode="braille", surface_type="pnl", show_contours=True):
    """Public entry point: extract option params and render the risk surface."""
    opt = option_row if isinstance(option_row, dict) else option_row.to_dict()
    option_type = "call" if opt.get("optionType", "call").lower() == "call" else "put"
    K = float(opt.get("strike", underlying_price))
    T = float(opt.get("T_years", opt.get("dte", 30) / 365.0))
    sigma = float(opt.get("impliedVolatility", 0.30))
    entry_price = float(opt.get("ask") if opt.get("ask") is not None else opt.get("lastPrice", 0))

    if sigma <= 0 or T <= 0:
        print("  [Cannot render risk surface \u2014 missing IV/DTE data]")
        return
    if surface_type == 'pnl' and entry_price <= 0:
        print("  [Cannot render risk surface \u2014 missing price data]")
        return

    symbol = opt.get("symbol", "")
    strike_str = f"${K:.0f}" if K == int(K) else f"${K:.2f}"
    desc = f"{symbol} {strike_str} {option_type.upper()}"

    use_braille = (mode == "braille") and _detect_utf8()
    n_p, n_iv = (80, 40) if use_braille else (40, 20)

    if surface_type == 'pnl':
        p_shocks, iv_shocks, grid = compute_pnl_grid(option_type, underlying_price, K, T, rfr, sigma, entry_price, n_price=n_p, n_iv=n_iv)
    else:
        p_shocks, iv_shocks, grid = compute_greek_grid(surface_type, option_type, underlying_price, K, T, rfr, sigma, n_price=n_p, n_iv=n_iv)

    if use_braille:
        render_surface_braille(p_shocks, iv_shocks, grid, option_desc=desc, width=width, surface_type=surface_type, show_contours=show_contours)
    else:
        render_surface(p_shocks, iv_shocks, grid, option_desc=desc, width=width, surface_type=surface_type, show_contours=show_contours)
