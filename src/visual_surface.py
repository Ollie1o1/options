"""3D risk surface visualization for option P&L and Greeks across price and IV shocks.

Supports two rendering modes:
  - braille: Unicode braille characters (U+2800-U+28FF) for high-resolution output
  - ascii:   Classic ASCII shading characters for maximum compatibility

Features: truecolor smooth gradients, Greek sensitivity surfaces, contour lines.
"""

import locale
import shutil
import numpy as np

from .utils import bs_price, bs_delta, bs_gamma, bs_vega, bs_theta


# ---------------------------------------------------------------------------
# P&L / Greek grid computation
# ---------------------------------------------------------------------------

def compute_pnl_grid(option_type, S, K, T, r, sigma, entry_price,
                     q=0.0, n_price=40, n_iv=20,
                     price_range=(-0.25, 0.25), iv_range=(-0.50, 0.50)):
    """Compute P&L grid over price shocks x IV shocks using full BS repricing."""
    price_shocks = np.linspace(price_range[0], price_range[1], n_price)
    iv_shocks = np.linspace(iv_range[0], iv_range[1], n_iv)

    price_mesh, iv_mesh = np.meshgrid(price_shocks, iv_shocks, indexing='ij')
    S_grid = S * (1.0 + price_mesh)
    sigma_grid = np.maximum(sigma * (1.0 + iv_mesh), 0.01)

    new_prices = bs_price(option_type, S_grid, K, T, r, sigma_grid, q)
    pnl = new_prices - entry_price

    return price_shocks, iv_shocks, pnl


def compute_greek_grid(greek_name, option_type, S, K, T, r, sigma,
                       q=0.0, n_price=40, n_iv=20,
                       price_range=(-0.25, 0.25), iv_range=(-0.50, 0.50)):
    """Compute a Greek value grid over price shocks x IV shocks.

    greek_name: one of 'delta', 'gamma', 'vega', 'theta'
    """
    price_shocks = np.linspace(price_range[0], price_range[1], n_price)
    iv_shocks = np.linspace(iv_range[0], iv_range[1], n_iv)

    price_mesh, iv_mesh = np.meshgrid(price_shocks, iv_shocks, indexing='ij')
    S_grid = S * (1.0 + price_mesh)
    sigma_grid = np.maximum(sigma * (1.0 + iv_mesh), 0.01)

    # bs_gamma and bs_vega don't take option_type as first arg
    if greek_name == 'delta':
        greek_grid = bs_delta(option_type, S_grid, K, T, r, sigma_grid, q)
    elif greek_name == 'gamma':
        greek_grid = bs_gamma(S_grid, K, T, r, sigma_grid, q)
    elif greek_name == 'vega':
        greek_grid = bs_vega(S_grid, K, T, r, sigma_grid, q)
    elif greek_name == 'theta':
        greek_grid = bs_theta(option_type, S_grid, K, T, r, sigma_grid, q)
    else:
        raise ValueError(f"Unknown greek: {greek_name}")

    return price_shocks, iv_shocks, greek_grid


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _pnl_to_rgb(frac):
    """Map normalized P&L fraction [0,1] to (R, G, B) tuple.

    Gradient stops:
      0.0 -> dark red (180,30,30)
      0.4 -> bright red (255,80,60)
      0.5 -> dim gray (100,100,100)
      0.6 -> bright green (80,220,80)
      1.0 -> cyan-green (40,255,160)
    """
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
        # blue -> green
        return int(30 + 50 * (1 - frac)), int(80 + 175 * frac), int(200 * (1 - frac) + 60 * frac)
    elif greek_name == 'gamma':
        # dark -> bright yellow
        return int(60 + 195 * frac), int(50 + 200 * frac), int(20 + 20 * frac)
    elif greek_name == 'vega':
        # dark -> bright magenta
        return int(60 + 195 * frac), int(20 + 60 * frac), int(80 + 175 * frac)
    elif greek_name == 'theta':
        # dim -> bright red
        return int(80 + 175 * frac), int(40 + 20 * frac), int(40 + 20 * frac)
    # fallback: grayscale
    v = int(60 + 195 * frac)
    return v, v, v


def _get_color(frac, val, pnl_span, surface_type, use_truecolor, use_color, fmt):
    """Return ANSI color string for a surface point."""
    if use_truecolor:
        if surface_type == 'pnl':
            r, g, b = _pnl_to_rgb(frac)
        else:
            r, g, b = _greek_to_rgb(surface_type, frac)
        return fmt.rgb_fg(r, g, b)
    elif use_color:
        if surface_type != 'pnl':
            # Simple 3-level for greeks: dim/yellow/bright
            if frac > 0.66:
                return fmt.Colors.BRIGHT_YELLOW
            elif frac > 0.33:
                return fmt.Colors.YELLOW
            else:
                return fmt.Colors.BRIGHT_BLACK
        # P&L 3-color
        if val > pnl_span * 0.02:
            return fmt.Colors.GREEN
        elif val < -pnl_span * 0.02:
            return fmt.Colors.RED
        else:
            return fmt.Colors.BRIGHT_BLACK
    return ''


# ---------------------------------------------------------------------------
# Braille canvas
# ---------------------------------------------------------------------------

class BrailleCanvas:
    """2D pixel canvas rendered to Unicode braille characters (U+2800-U+28FF).

    Each terminal character encodes a 2-wide x 4-tall dot grid, giving
    2x horizontal and 4x vertical sub-character resolution.
    """
    _DOT_MAP = {
        (0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (0, 3): 0x40,
        (1, 0): 0x08, (1, 1): 0x10, (1, 2): 0x20, (1, 3): 0x80,
    }

    def __init__(self, term_cols, term_rows):
        self.tw = term_cols
        self.th = term_rows
        self.pw = term_cols * 2   # pixel width
        self.ph = term_rows * 4   # pixel height
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
    """Draw a line on the braille canvas with depth test (Bresenham)."""
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
# Contour extraction (marching squares)
# ---------------------------------------------------------------------------

def _extract_contour_segments(grid_2d, level):
    """Return list of (x1,y1,x2,y2) segment endpoints in grid-fraction coords [0,1]."""
    nr, nc = grid_2d.shape
    segments = []
    for i in range(nr - 1):
        for j in range(nc - 1):
            # Four corners of the cell
            v00 = grid_2d[i, j]
            v10 = grid_2d[i + 1, j]
            v01 = grid_2d[i, j + 1]
            v11 = grid_2d[i + 1, j + 1]

            # Classify corners as above/below level
            case = 0
            if v00 >= level:
                case |= 1
            if v10 >= level:
                case |= 2
            if v01 >= level:
                case |= 4
            if v11 >= level:
                case |= 8

            if case == 0 or case == 15:
                continue

            # Edge interpolation helpers
            def _interp_x(va, vb, row, col_a, col_b):
                t = (level - va) / (vb - va) if vb != va else 0.5
                return (col_a + t * (col_b - col_a)) / max(nc - 1, 1), row / max(nr - 1, 1)

            def _interp_y(va, vb, col, row_a, row_b):
                t = (level - va) / (vb - va) if vb != va else 0.5
                return col / max(nc - 1, 1), (row_a + t * (row_b - row_a)) / max(nr - 1, 1)

            # Edge midpoints: top(0-1), bottom(2-3), left(0-2), right(1-3)
            top = _interp_x(v00, v10, j, i, i + 1)
            bottom = _interp_x(v01, v11, j + 1, i, i + 1)
            left = _interp_y(v00, v01, i, j, j + 1)
            right = _interp_y(v10, v11, i + 1, j, j + 1)

            # Lookup table for the 16 marching-squares cases
            edges = {
                1: [(top, left)], 2: [(top, right)], 3: [(left, right)],
                4: [(left, bottom)], 5: [(top, bottom)], 6: [(top, left), (right, bottom)],
                7: [(right, bottom)], 8: [(right, bottom)], 9: [(top, right), (left, bottom)],
                10: [(top, bottom)], 11: [(left, bottom)], 12: [(left, right)],
                13: [(top, right)], 14: [(top, left)],
            }
            for seg in edges.get(case, []):
                (x1, y1), (x2, y2) = seg
                segments.append((x1, y1, x2, y2))

    return segments


def _pick_contour_levels(grid_2d, surface_type, n_levels=7):
    """Choose contour levels. For P&L, always include 0 (breakeven)."""
    vmin, vmax = float(np.nanmin(grid_2d)), float(np.nanmax(grid_2d))
    if vmin == vmax:
        return []
    levels = np.linspace(vmin, vmax, n_levels + 2)[1:-1].tolist()
    if surface_type == 'pnl' and vmin < 0 < vmax:
        # Ensure breakeven is included
        levels = [l for l in levels if abs(l) > (vmax - vmin) * 0.05]
        levels.append(0.0)
        levels.sort()
    return levels


# ---------------------------------------------------------------------------
# Isometric projection helper
# ---------------------------------------------------------------------------

def _iso_project(fx, fy, fz, w, h):
    """Map (fx, fy, fz) in [0,1]^3 to (col, row) screen coords."""
    col = w * 0.5 + (fx - 0.5) * w * 0.45 - (fy - 0.5) * w * 0.15
    row = h * 0.85 - (fx + fy) * h * 0.15 - fz * h * 0.55
    return col, row


# ---------------------------------------------------------------------------
# ASCII renderer (original, preserved as fallback)
# ---------------------------------------------------------------------------

def render_surface(price_shocks, iv_shocks, grid_2d, option_desc="", width=100,
                   surface_type="pnl", show_contours=True):
    """Render an isometric ASCII 3D surface to stdout."""
    try:
        from . import formatting as fmt
        has_fmt = True
        use_color = fmt.supports_color()
        use_truecolor = fmt.supports_truecolor()
    except ImportError:
        has_fmt = False
        use_color = False
        use_truecolor = False
        fmt = None

    n_price, n_iv = grid_2d.shape
    val_min, val_max = float(np.nanmin(grid_2d)), float(np.nanmax(grid_2d))

    try:
        term_w = shutil.get_terminal_size(fallback=(100, 24)).columns
    except Exception:
        term_w = 100
    if term_w < 80:
        print("  [Terminal too narrow for risk surface — need >=80 columns]")
        return

    screen_w = min(width, term_w) - 4
    screen_h = 35

    shade_chars = " .:-=+*#%@"
    n_shades = len(shade_chars)

    buf = [[' '] * (screen_w + 1) for _ in range(screen_h + 1)]
    zbuf = [[-1e9] * (screen_w + 1) for _ in range(screen_h + 1)]
    cbuf = [[''] * (screen_w + 1) for _ in range(screen_h + 1)]

    val_span = val_max - val_min if val_max != val_min else 1.0

    for iy in range(n_iv - 1, -1, -1):
        for ix in range(n_price):
            val = grid_2d[ix, iy]
            if np.isnan(val):
                continue
            fx = ix / max(n_price - 1, 1)
            fy = iy / max(n_iv - 1, 1)
            fz = (val - val_min) / val_span
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
                          shade_chars, width, term_w, has_fmt, use_color, use_truecolor, fmt)


# ---------------------------------------------------------------------------
# Braille renderer
# ---------------------------------------------------------------------------

def render_surface_braille(price_shocks, iv_shocks, grid_2d, option_desc="", width=100,
                           surface_type="pnl", show_contours=True):
    """Render an isometric 3D surface using braille characters for high resolution."""
    try:
        from . import formatting as fmt
        has_fmt = True
        use_color = fmt.supports_color()
        use_truecolor = fmt.supports_truecolor()
    except ImportError:
        has_fmt = False
        use_color = False
        use_truecolor = False
        fmt = None

    n_price, n_iv = grid_2d.shape
    val_min, val_max = float(np.nanmin(grid_2d)), float(np.nanmax(grid_2d))

    try:
        term_w = shutil.get_terminal_size(fallback=(100, 24)).columns
    except Exception:
        term_w = 100
    if term_w < 80:
        print("  [Terminal too narrow for risk surface — need >=80 columns]")
        return

    term_cols = min(width, term_w) - 4
    term_rows = 35
    canvas = BrailleCanvas(term_cols, term_rows)
    zbuf = np.full((canvas.ph, canvas.pw), -1e9)

    val_span = val_max - val_min if val_max != val_min else 1.0

    # Pre-compute projected pixel coords and colors for all grid points
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
            fx = ix / max(n_price - 1, 1)
            fy = iy / max(n_iv - 1, 1)
            fz = (val - val_min) / val_span
            col, row = _iso_project(fx, fy, fz, canvas.pw, canvas.ph)
            px_coords[ix, iy] = int(col)
            py_coords[ix, iy] = int(row)
            depths[ix, iy] = fx + fy
            colors[ix][iy] = _get_color(fz, val, val_span, surface_type, use_truecolor, use_color, fmt)

    # Draw surface with edges (back-to-front)
    for iy in range(n_iv - 1, -1, -1):
        for ix in range(n_price):
            x0, y0 = px_coords[ix, iy], py_coords[ix, iy]
            if x0 < 0:
                continue
            d = depths[ix, iy]
            c = colors[ix][iy]

            # Draw edges to neighbors (right and down in grid)
            if ix + 1 < n_price and px_coords[ix + 1, iy] >= 0:
                x1, y1 = px_coords[ix + 1, iy], py_coords[ix + 1, iy]
                _braille_line(canvas, x0, y0, x1, y1, c, zbuf, d)
            if iy - 1 >= 0 and px_coords[ix, iy - 1] >= 0:
                x1, y1 = px_coords[ix, iy - 1], py_coords[ix, iy - 1]
                _braille_line(canvas, x0, y0, x1, y1, c, zbuf, d)

            # Draw the point itself
            if 0 <= x0 < canvas.pw and 0 <= y0 < canvas.ph:
                if d >= zbuf[y0, x0]:
                    zbuf[y0, x0] = d
                    canvas.set_pixel(x0, y0, c)

    # Contour lines
    if show_contours:
        levels = _pick_contour_levels(grid_2d, surface_type)
        for level in levels:
            is_breakeven = (surface_type == 'pnl' and abs(level) < val_span * 0.01)
            if use_truecolor and fmt:
                contour_color = fmt.rgb_fg(255, 255, 255) if is_breakeven else fmt.rgb_fg(200, 200, 100)
            elif use_color and fmt:
                contour_color = fmt.Colors.BRIGHT_WHITE if is_breakeven else fmt.Colors.BRIGHT_YELLOW
            else:
                contour_color = ''

            segments = _extract_contour_segments(grid_2d, level)
            fz_level = (level - val_min) / val_span
            for x1f, y1f, x2f, y2f in segments:
                # x1f,y1f are in grid-fraction coords (price_frac, iv_frac)
                c1, r1 = _iso_project(x1f, y1f, fz_level, canvas.pw, canvas.ph)
                c2, r2 = _iso_project(x2f, y2f, fz_level, canvas.pw, canvas.ph)
                # Draw contour with very high depth so it sits on top
                _braille_line(canvas, int(c1), int(r1), int(c2), int(r2),
                              contour_color, zbuf, 100.0)

    # Print header
    print()
    if has_fmt:
        print(fmt.draw_separator(min(width, term_w)))

    stype_label = surface_type.upper() if surface_type != 'pnl' else 'P&L'
    title = f"  3D {stype_label} Risk Surface{f'  —  {option_desc}' if option_desc else ''}"
    if surface_type != 'pnl':
        title += f"  [{surface_type.capitalize()} sensitivity]"
    if use_color and fmt:
        print(fmt.colorize(title, fmt.Colors.BRIGHT_CYAN, bold=True))
    else:
        print(title)

    # Axis labels
    p_lo = f"{price_shocks[0]*100:+.0f}%"
    p_hi = f"{price_shocks[-1]*100:+.0f}%"
    iv_lo = f"{iv_shocks[0]*100:+.0f}%"
    iv_hi = f"{iv_shocks[-1]*100:+.0f}%"
    print(f"  Price shock: {p_lo} <- -> {p_hi}   |   IV shock: {iv_lo} <- -> {iv_hi}")

    if surface_type == 'pnl':
        print(f"  P&L range: ${val_min:+,.2f}  to  ${val_max:+,.2f}")
    else:
        print(f"  {surface_type.capitalize()} range: {val_min:+.4f}  to  {val_max:+.4f}")
    print()

    # Render canvas
    reset = fmt.Colors.RESET if (use_color and fmt) else ''
    for line in canvas.render_lines(reset):
        if line.strip():
            print(f"  {line}")

    # Legend
    _print_legend(val_min, val_max, val_span, surface_type, use_color, use_truecolor, fmt, has_fmt, width, term_w)
    print()


# ---------------------------------------------------------------------------
# Shared output helpers
# ---------------------------------------------------------------------------

def _print_surface_output(buf, cbuf, screen_w, screen_h, price_shocks, iv_shocks,
                          val_min, val_max, val_span, option_desc, surface_type,
                          shade_chars, width, term_w, has_fmt, use_color, use_truecolor, fmt):
    """Print header, buffer, and legend for ASCII mode."""
    print()
    if has_fmt:
        print(fmt.draw_separator(min(width, term_w)))

    stype_label = surface_type.upper() if surface_type != 'pnl' else 'P&L'
    title = f"  3D {stype_label} Risk Surface{f'  —  {option_desc}' if option_desc else ''}"
    if surface_type != 'pnl':
        title += f"  [{surface_type.capitalize()} sensitivity]"
    if use_color:
        print(fmt.colorize(title, fmt.Colors.BRIGHT_CYAN, bold=True))
    else:
        print(title)

    p_lo = f"{price_shocks[0]*100:+.0f}%"
    p_hi = f"{price_shocks[-1]*100:+.0f}%"
    iv_lo = f"{iv_shocks[0]*100:+.0f}%"
    iv_hi = f"{iv_shocks[-1]*100:+.0f}%"
    print(f"  Price shock: {p_lo} <- -> {p_hi}   |   IV shock: {iv_lo} <- -> {iv_hi}")

    if surface_type == 'pnl':
        print(f"  P&L range: ${val_min:+,.2f}  to  ${val_max:+,.2f}")
    else:
        print(f"  {surface_type.capitalize()} range: {val_min:+.4f}  to  {val_max:+.4f}")
    print()

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

    _print_legend(val_min, val_max, val_span, surface_type, use_color, use_truecolor, fmt, has_fmt, width, term_w,
                  shade_chars=shade_chars)
    print()


def _print_legend(val_min, val_max, val_span, surface_type, use_color, use_truecolor, fmt,
                  has_fmt, width, term_w, shade_chars=None):
    """Print gradient legend bar."""
    print()

    if use_truecolor and fmt:
        # Smooth gradient preview
        n_steps = 30
        legend = "  Legend: "
        for i in range(n_steps):
            frac = i / max(n_steps - 1, 1)
            if surface_type == 'pnl':
                r, g, b = _pnl_to_rgb(frac)
            else:
                r, g, b = _greek_to_rgb(surface_type, frac)
            legend += f"{fmt.rgb_fg(r, g, b)}\u2588{fmt.Colors.RESET}"
        val = val_min
        val_hi = val_max
        if surface_type == 'pnl':
            legend += f"  ${val:+,.0f} to ${val_hi:+,.0f}"
        else:
            legend += f"  {val:+.4f} to {val_hi:+.4f}"
        print(legend)
    elif shade_chars and use_color and fmt:
        # ASCII shade legend
        legend = "  Legend: "
        n_shades = len(shade_chars)
        for i, ch in enumerate(shade_chars):
            if ch == ' ':
                continue
            frac = i / (n_shades - 1)
            val = val_min + frac * val_span
            if surface_type == 'pnl':
                label = f"${val:+,.0f}"
                color = fmt.Colors.GREEN if val > 0 else (fmt.Colors.RED if val < 0 else fmt.Colors.BRIGHT_BLACK)
            else:
                label = f"{val:+.3f}"
                color = fmt.Colors.BRIGHT_YELLOW if frac > 0.5 else fmt.Colors.BRIGHT_BLACK
            token = f"{ch}={label} "
            legend += fmt.colorize(token, color)
        print(legend)
    else:
        if surface_type == 'pnl':
            print(f"  Range: ${val_min:+,.2f} to ${val_max:+,.2f}")
        else:
            print(f"  Range: {val_min:+.4f} to {val_max:+.4f}")

    # Contour legend hint
    if use_color and fmt:
        contour_hint = "  Contour lines: "
        contour_hint += fmt.colorize("white", fmt.Colors.BRIGHT_WHITE, bold=True) + " = breakeven"
        contour_hint += "  |  " + fmt.colorize("yellow", fmt.Colors.BRIGHT_YELLOW) + " = iso-value"
        print(contour_hint)

    if has_fmt:
        print(fmt.draw_separator(min(width, term_w)))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _detect_utf8():
    """Check if terminal likely supports Unicode braille characters."""
    try:
        enc = locale.getpreferredencoding(False)
        return enc.lower().replace('-', '') in ('utf8', 'utf16', 'utf32')
    except Exception:
        return False


def print_risk_surface(option_row, underlying_price, rfr, width=100,
                       mode="braille", surface_type="pnl", show_contours=True):
    """Public entry point: extract option params and render the risk surface.

    Args:
        option_row: dict-like option record (from DataFrame row or dict)
        underlying_price: current underlying price
        rfr: risk-free rate
        width: display width
        mode: "braille" (high-res Unicode) or "ascii" (classic shading)
        surface_type: "pnl", "delta", "gamma", "vega", "theta"
        show_contours: whether to draw contour lines (braille mode only)
    """
    opt = option_row if isinstance(option_row, dict) else option_row.to_dict()

    option_type = "call" if opt.get("optionType", "call").lower() == "call" else "put"
    K = float(opt.get("strike", underlying_price))
    T = float(opt.get("T_years", opt.get("dte", 30) / 365.0))
    sigma = float(opt.get("impliedVolatility", 0.30))
    entry_price = opt.get("ask") if opt.get("ask") is not None else opt.get("lastPrice", 0)
    entry_price = float(entry_price) if entry_price else 0.0

    if sigma <= 0 or T <= 0:
        print("  [Cannot render risk surface — missing IV/DTE data]")
        return
    if surface_type == 'pnl' and entry_price <= 0:
        print("  [Cannot render risk surface — missing price data]")
        return

    # Build description string
    symbol = opt.get("symbol", "")
    strike_str = f"${K:.0f}" if K == int(K) else f"${K:.2f}"
    desc = f"{symbol} {strike_str} {option_type.upper()}"

    # Higher resolution grid for braille
    use_braille = (mode == "braille") and _detect_utf8()
    n_price = 80 if use_braille else 40
    n_iv = 40 if use_braille else 20

    if surface_type == 'pnl':
        price_shocks, iv_shocks, grid = compute_pnl_grid(
            option_type, underlying_price, K, T, rfr, sigma, entry_price,
            n_price=n_price, n_iv=n_iv,
        )
    else:
        price_shocks, iv_shocks, grid = compute_greek_grid(
            surface_type, option_type, underlying_price, K, T, rfr, sigma,
            n_price=n_price, n_iv=n_iv,
        )

    if use_braille:
        render_surface_braille(price_shocks, iv_shocks, grid, option_desc=desc,
                               width=width, surface_type=surface_type,
                               show_contours=show_contours)
    else:
        render_surface(price_shocks, iv_shocks, grid, option_desc=desc,
                       width=width, surface_type=surface_type,
                       show_contours=show_contours)
