import sounddevice as sd
import numpy as np
import sys
import shutil
import termios
import tty
import select
import math
import os
import threading

# ── Braille primitives ──────────────────────────────────────────────

BRAILLE_BASE = 0x2800
DOT_MAP = {
    (0, 0): 0x01, (1, 0): 0x02, (2, 0): 0x04, (3, 0): 0x40,
    (0, 1): 0x08, (1, 1): 0x10, (2, 1): 0x20, (3, 1): 0x80,
}


def plot_point(cells, colors, px, py, rows, cols, color):
    cell_r, cell_c = py // 4, px // 2
    dot_r, dot_c = py % 4, px % 2
    if 0 <= cell_r < rows and 0 <= cell_c < cols:
        cells[cell_r][cell_c] |= DOT_MAP[(dot_r, dot_c)]
        colors[cell_r][cell_c] = color


def plot_line(cells, colors, px, py0, py1, rows, cols, color_fn):
    lo, hi = min(py0, py1), max(py0, py1)
    for y in range(lo, hi + 1):
        plot_point(cells, colors, px, y, rows, cols, color_fn(y))


def render_braille(cells, colors, rows, cols):
    lines = []
    for r in range(rows):
        line = ''
        for c in range(cols):
            if cells[r][c]:
                line += colors[r][c] + chr(BRAILLE_BASE + cells[r][c])
            else:
                line += ' '
        lines.append(line)
    return lines


def render_char_grid(grid, colors, rows, cols):
    lines = []
    for r in range(rows):
        line = ''
        for c in range(cols):
            if grid[r][c] != ' ':
                line += colors[r][c] + grid[r][c]
            else:
                line += ' '
        lines.append(line)
    return lines


# ── Color helpers ───────────────────────────────────────────────────

THEMES = {
    'cyan-purple-magenta': [
        (0.0, (0, 255, 255)), (0.3, (0, 255, 255)),
        (0.6, (140, 0, 255)), (1.0, (255, 0, 200)),
    ],
    'green-yellow-red': [
        (0.0, (0, 255, 80)), (0.4, (255, 255, 0)), (1.0, (255, 0, 0)),
    ],
    'blue-white-red': [
        (0.0, (30, 100, 255)), (0.5, (255, 255, 255)), (1.0, (255, 40, 40)),
    ],
    'rainbow': [
        (0.0, (255, 0, 0)), (0.2, (255, 165, 0)), (0.4, (255, 255, 0)),
        (0.6, (0, 255, 0)), (0.8, (0, 100, 255)), (1.0, (180, 0, 255)),
    ],
}

THEME_NAMES = list(THEMES.keys())


def lerp_rgb(stops, t):
    t = max(0.0, min(1.0, t))
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            f = (t - t0) / max(t1 - t0, 1e-9)
            return (int(c0[0] + (c1[0] - c0[0]) * f),
                    int(c0[1] + (c1[1] - c0[1]) * f),
                    int(c0[2] + (c1[2] - c0[2]) * f))
    return stops[-1][1]


def ansi(r, g, b):
    return f'\033[38;2;{r};{g};{b}m'


def lerp_color(stops, t):
    return ansi(*lerp_rgb(stops, t))


# ── Braille visualizations ─────────────────────────────────────────

def viz_waveform_braille(wave, rows, cols, theme_stops):
    res_y, res_x, mid = rows * 4, cols * 2, rows * 2
    resampled = np.interp(np.linspace(0, len(wave) - 1, res_x),
                          np.arange(len(wave)), wave)
    resampled = np.clip(resampled, -1, 1)
    cells = [[0] * cols for _ in range(rows)]
    colors = [[''] * cols for _ in range(rows)]
    color_at = lambda y: lerp_color(theme_stops, abs(y - mid) / max(mid, 1))
    prev = None
    for px in range(res_x):
        py = int(np.clip(mid - resampled[px] * (mid - 2), 0, res_y - 1))
        if prev is not None:
            plot_line(cells, colors, px, prev, py, rows, cols, color_at)
        plot_point(cells, colors, px, py, rows, cols, color_at(py))
        prev = py
    return render_braille(cells, colors, rows, cols)


def viz_mirror_braille(wave, rows, cols, theme_stops):
    res_y, res_x, mid = rows * 4, cols * 2, rows * 2
    resampled = np.interp(np.linspace(0, len(wave) - 1, res_x),
                          np.arange(len(wave)), wave)
    resampled = np.clip(resampled, -1, 1)
    abs_w = np.abs(resampled)
    cells = [[0] * cols for _ in range(rows)]
    colors = [[''] * cols for _ in range(rows)]
    color_at = lambda y: lerp_color(theme_stops, abs(y - mid) / max(mid, 1))
    prev_up = prev_down = None
    for px in range(res_x):
        extent = abs_w[px] * (mid - 2)
        py_up = int(np.clip(mid - extent, 0, res_y - 1))
        py_down = int(np.clip(mid + extent, 0, res_y - 1))
        if prev_up is not None:
            plot_line(cells, colors, px, prev_up, py_up, rows, cols, color_at)
            plot_line(cells, colors, px, prev_down, py_down, rows, cols, color_at)
        plot_point(cells, colors, px, py_up, rows, cols, color_at(py_up))
        plot_point(cells, colors, px, py_down, rows, cols, color_at(py_down))
        prev_up, prev_down = py_up, py_down
    return render_braille(cells, colors, rows, cols)


def viz_spectrum_braille(wave, rows, cols, theme_stops):
    res_y = rows * 4
    fft = np.abs(np.fft.rfft(wave))[:len(wave) // 2]
    if len(fft) < 2:
        return [' ' * cols for _ in range(rows)]
    num_bars = cols * 2
    log_idx = np.logspace(np.log10(1), np.log10(len(fft) - 1), num_bars).astype(int)
    log_idx = np.clip(log_idx, 0, len(fft) - 1)
    bars = np.zeros(num_bars)
    for i in range(num_bars - 1):
        lo, hi = log_idx[i], max(log_idx[i + 1], log_idx[i] + 1)
        bars[i] = np.mean(fft[lo:hi])
    bars[-1] = fft[log_idx[-1]]
    bars = bars / max(np.max(bars), 1e-6)
    cells = [[0] * cols for _ in range(rows)]
    colors = [[''] * cols for _ in range(rows)]
    for px in range(num_bars):
        h = int(bars[px] * (res_y - 1))
        color = lerp_color(theme_stops, px / max(num_bars - 1, 1))
        for dy in range(h + 1):
            plot_point(cells, colors, px, res_y - 1 - dy, rows, cols, color)
    return render_braille(cells, colors, rows, cols)


def viz_oscilloscope_braille(wave, rows, cols, theme_stops):
    res_y, res_x = rows * 4, cols * 2
    cx, cy = res_x / 2, res_y / 2
    radius = min(cx, cy) * 0.8
    n_pts = max(len(wave), 256)
    resampled = np.interp(np.linspace(0, len(wave) - 1, n_pts),
                          np.arange(len(wave)), wave)
    amp = max(np.max(np.abs(resampled)), 0.001)
    resampled = resampled / amp
    cells = [[0] * cols for _ in range(rows)]
    colors = [[''] * cols for _ in range(rows)]
    prev_px, prev_py = None, None
    for i in range(n_pts):
        ang = 2 * math.pi * i / n_pts
        r = radius * (0.5 + 0.5 * resampled[i])
        px = int(cx + r * math.cos(ang) * 2.0)
        py = int(cy + r * math.sin(ang))
        color = lerp_color(theme_stops, i / max(n_pts - 1, 1))
        if 0 <= px < res_x and 0 <= py < res_y:
            plot_point(cells, colors, px, py, rows, cols, color)
        if prev_px is not None:
            dx, dy = px - prev_px, py - prev_py
            steps = max(abs(dx), abs(dy), 1)
            if steps <= 4:
                for s in range(1, steps):
                    ix = prev_px + dx * s // steps
                    iy = prev_py + dy * s // steps
                    if 0 <= ix < res_x and 0 <= iy < res_y:
                        plot_point(cells, colors, ix, iy, rows, cols, color)
        prev_px, prev_py = px, py
    return render_braille(cells, colors, rows, cols)


# ── Block visualizations ───────────────────────────────────────────

LOWER_BLOCKS = ' \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588'


def viz_waveform_blocks(wave, rows, cols, theme_stops):
    mid = rows // 2
    resampled = np.interp(np.linspace(0, len(wave) - 1, cols),
                          np.arange(len(wave)), wave)
    resampled = np.clip(resampled, -1, 1)
    grid = [[' '] * cols for _ in range(rows)]
    colors = [[''] * cols for _ in range(rows)]
    for c in range(cols):
        target = max(0, min(rows - 1, int(mid - resampled[c] * mid)))
        lo, hi = min(mid, target), max(mid, target)
        for r in range(lo, hi + 1):
            colors[r][c] = lerp_color(theme_stops, abs(r - mid) / max(mid, 1))
            grid[r][c] = '\u2588'
    return render_char_grid(grid, colors, rows, cols)


def viz_mirror_blocks(wave, rows, cols, theme_stops):
    mid = rows // 2
    resampled = np.interp(np.linspace(0, len(wave) - 1, cols),
                          np.arange(len(wave)), wave)
    resampled = np.clip(resampled, -1, 1)
    abs_w = np.abs(resampled)
    grid = [[' '] * cols for _ in range(rows)]
    colors = [[''] * cols for _ in range(rows)]
    for c in range(cols):
        extent = int(abs_w[c] * mid)
        for r in range(mid - extent, mid + extent + 1):
            if 0 <= r < rows:
                colors[r][c] = lerp_color(theme_stops, abs(r - mid) / max(mid, 1))
                grid[r][c] = '\u2588'
    return render_char_grid(grid, colors, rows, cols)


def viz_spectrum_blocks(wave, rows, cols, theme_stops):
    fft = np.abs(np.fft.rfft(wave))[:len(wave) // 2]
    if len(fft) < 2:
        return [' ' * cols for _ in range(rows)]
    log_idx = np.logspace(np.log10(1), np.log10(len(fft) - 1), cols).astype(int)
    log_idx = np.clip(log_idx, 0, len(fft) - 1)
    bars = np.zeros(cols)
    for i in range(cols - 1):
        lo, hi = log_idx[i], max(log_idx[i + 1], log_idx[i] + 1)
        bars[i] = np.mean(fft[lo:hi])
    bars[-1] = fft[log_idx[-1]]
    bars = bars / max(np.max(bars), 1e-6)
    grid = [[' '] * cols for _ in range(rows)]
    colors = [[''] * cols for _ in range(rows)]
    for c in range(cols):
        height = bars[c] * rows
        full = int(height)
        frac = int((height - full) * 8)
        color = lerp_color(theme_stops, c / max(cols - 1, 1))
        for r in range(rows - full, rows):
            grid[r][c] = '\u2588'
            colors[r][c] = color
        if frac > 0 and rows - full - 1 >= 0:
            grid[rows - full - 1][c] = LOWER_BLOCKS[frac]
            colors[rows - full - 1][c] = color
    return render_char_grid(grid, colors, rows, cols)


def viz_oscilloscope_blocks(wave, rows, cols, theme_stops):
    cx, cy = cols / 2, rows / 2
    radius = min(cx, cy) * 0.8
    n_pts = max(len(wave), 256)
    resampled = np.interp(np.linspace(0, len(wave) - 1, n_pts),
                          np.arange(len(wave)), wave)
    amp = max(np.max(np.abs(resampled)), 0.001)
    resampled = resampled / amp
    grid = [[' '] * cols for _ in range(rows)]
    colors = [[''] * cols for _ in range(rows)]
    prev_c, prev_r = None, None
    for i in range(n_pts):
        ang = 2 * math.pi * i / n_pts
        r = radius * (0.5 + 0.5 * resampled[i])
        pc = int(cx + r * math.cos(ang) * 2.0)
        pr = int(cy + r * math.sin(ang))
        color = lerp_color(theme_stops, i / max(n_pts - 1, 1))
        if 0 <= pr < rows and 0 <= pc < cols:
            grid[pr][pc] = '\u2588'
            colors[pr][pc] = color
        if prev_c is not None:
            dc, dr = pc - prev_c, pr - prev_r
            steps = max(abs(dc), abs(dr), 1)
            if steps <= 3:
                for s in range(1, steps):
                    ic = prev_c + dc * s // steps
                    ir = prev_r + dr * s // steps
                    if 0 <= ir < rows and 0 <= ic < cols:
                        grid[ir][ic] = '\u2588'
                        colors[ir][ic] = color
        prev_c, prev_r = pc, pr
    return render_char_grid(grid, colors, rows, cols)


# ── Shared visualizations (same for both styles) ───────────────────

_spec_history = []
_spec_peak = 1e-6


def viz_bands(wave, rows, cols, theme_stops):
    """Graphic equalizer — 16 wide frequency bars."""
    fft = np.abs(np.fft.rfft(wave))[:len(wave) // 2]
    if len(fft) < 2:
        return [' ' * cols for _ in range(rows)]
    num_bands = 16
    log_idx = np.logspace(np.log10(1), np.log10(len(fft) - 1), num_bands + 1).astype(int)
    log_idx = np.clip(log_idx, 0, len(fft) - 1)
    bands = np.zeros(num_bands)
    for i in range(num_bands):
        lo, hi = log_idx[i], max(log_idx[i + 1], log_idx[i] + 1)
        bands[i] = np.mean(fft[lo:min(hi, len(fft))])
    bands = bands / max(np.max(bands), 1e-6)

    bar_w = max((cols - num_bands + 1) // num_bands, 1)
    total_w = num_bands * bar_w + (num_bands - 1)
    offset = (cols - total_w) // 2

    grid = [[' '] * cols for _ in range(rows)]
    colors = [[''] * cols for _ in range(rows)]
    for bi in range(num_bands):
        h = bands[bi] * rows
        full = int(h)
        frac = int((h - full) * 8)
        t = bi / max(num_bands - 1, 1)
        color = lerp_color(theme_stops, t)
        bar_start = offset + bi * (bar_w + 1)
        for c in range(bar_start, min(bar_start + bar_w, cols)):
            for r in range(rows - full, rows):
                grid[r][c] = '\u2588'
                colors[r][c] = color
            if frac > 0 and rows - full - 1 >= 0:
                grid[rows - full - 1][c] = LOWER_BLOCKS[frac]
                colors[rows - full - 1][c] = color
    return render_char_grid(grid, colors, rows, cols)


def viz_spectrogram(wave, rows, cols, theme_stops):
    """Scrolling frequency waterfall."""
    global _spec_history, _spec_peak
    fft = np.abs(np.fft.rfft(wave))[:len(wave) // 2]
    if len(fft) < 2:
        _spec_history.append(np.zeros(rows))
    else:
        num_bands = rows
        log_idx = np.logspace(np.log10(1), np.log10(len(fft) - 1),
                              num_bands + 1).astype(int)
        log_idx = np.clip(log_idx, 0, len(fft) - 1)
        bands = np.zeros(num_bands)
        for i in range(num_bands):
            lo, hi = log_idx[i], max(log_idx[i + 1], log_idx[i] + 1)
            bands[i] = np.mean(fft[lo:min(hi, len(fft))])
        current_peak = np.max(bands)
        _spec_peak = max(_spec_peak * 0.995, current_peak, 1e-6)
        _spec_history.append(bands)

    # trim history
    if len(_spec_history) > cols + 50:
        _spec_history = _spec_history[-(cols + 50):]

    grid = [[' '] * cols for _ in range(rows)]
    colors = [[''] * cols for _ in range(rows)]

    visible = _spec_history[-cols:]
    start_col = cols - len(visible)

    for ci, frame in enumerate(visible):
        col = start_col + ci
        for ri in range(rows):
            band_idx = rows - 1 - ri  # low freq at bottom
            if band_idx < len(frame):
                intensity = min(frame[band_idx] / _spec_peak, 1.0)
            else:
                intensity = 0
            if intensity > 0.03:
                r, g, b = lerp_rgb(theme_stops, ri / max(rows - 1, 1))
                r = int(r * intensity)
                g = int(g * intensity)
                b = int(b * intensity)
                colors[ri][col] = ansi(r, g, b)
                grid[ri][col] = '\u2588'

    return render_char_grid(grid, colors, rows, cols)


# ── Input handling ──────────────────────────────────────────────────

def read_key(fd):
    if not select.select([fd], [], [], 0)[0]:
        return None
    ch = os.read(fd, 1)
    if ch == b'\x1b':
        if select.select([fd], [], [], 0.02)[0]:
            ch2 = os.read(fd, 1)
            if ch2 == b'[':
                ch3 = os.read(fd, 1)
                return {b'A': 'up', b'B': 'down',
                        b'C': 'right', b'D': 'left'}.get(ch3)
        return 'esc'
    if ch in (b'q', b'Q'):
        return 'quit'
    return None


# ── Parameters ──────────────────────────────────────────────────────

MODE_NAMES = ['Waveform', 'Mirror', 'Spectrum', 'Bands',
              'Spectrogram', 'Oscilloscope']
STYLE_NAMES = ['Braille', 'Blocks']
GAIN_VALUES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

VIZ_FUNCS = {
    'Braille': [viz_waveform_braille, viz_mirror_braille, viz_spectrum_braille,
                viz_bands, viz_spectrogram, viz_oscilloscope_braille],
    'Blocks':  [viz_waveform_blocks, viz_mirror_blocks, viz_spectrum_blocks,
                viz_bands, viz_spectrogram, viz_oscilloscope_blocks],
}

PARAM_NAMES = ['Mode', 'Style', 'Gain', 'Color']


def format_param(idx, mode_idx, style_idx, gain_idx, theme_idx):
    if idx == 0: return MODE_NAMES[mode_idx]
    if idx == 1: return STYLE_NAMES[style_idx]
    if idx == 2: return f'{GAIN_VALUES[gain_idx]:.1f}x'
    return THEME_NAMES[theme_idx]


# ── Status bar ──────────────────────────────────────────────────────

def render_status(cols, param_idx, mode_idx, style_idx, gain_idx, theme_idx):
    parts = []
    for i, name in enumerate(PARAM_NAMES):
        val = format_param(i, mode_idx, style_idx, gain_idx, theme_idx)
        if i == param_idx:
            parts.append(f'\033[7m {name}: {val} \033[0m')
        else:
            parts.append(f' {name}: {val} ')
    left = '  '.join(parts)
    hint = '\033[2m  \u2190\u2192 select  \u2191\u2193 adjust  q quit\033[0m'
    return left + hint + '\033[K'  # clear to end of line


# ── Main loop ───────────────────────────────────────────────────────

def main():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    mode_idx = 0
    style_idx = 0
    gain_idx = 2         # 2.0x default
    theme_idx = 0
    param_idx = 0
    smooth_wave = None

    audio_lock = threading.Lock()
    audio_buf = {'data': np.zeros(4096)}

    def audio_callback(indata, frames, time_info, status):
        with audio_lock:
            audio_buf['data'] = indata[:, 0].copy()

    stream = sd.InputStream(device="BlackHole 2ch", channels=1,
                            samplerate=48000, blocksize=4096,
                            callback=audio_callback)

    try:
        tty.setraw(fd)
        stream.start()
        sys.stdout.write('\033[2J\033[?25l')
        sys.stdout.flush()

        while True:
            select.select([fd], [], [], 0.033)

            quit_pressed = False
            while True:
                key = read_key(fd)
                if key is None:
                    break
                if key == 'quit':
                    quit_pressed = True
                    break
                elif key == 'left':
                    param_idx = (param_idx - 1) % len(PARAM_NAMES)
                elif key == 'right':
                    param_idx = (param_idx + 1) % len(PARAM_NAMES)
                elif key == 'up':
                    if param_idx == 0:
                        mode_idx = (mode_idx + 1) % len(MODE_NAMES)
                    elif param_idx == 1:
                        style_idx = (style_idx + 1) % len(STYLE_NAMES)
                    elif param_idx == 2:
                        gain_idx = min(gain_idx + 1, len(GAIN_VALUES) - 1)
                    elif param_idx == 3:
                        theme_idx = (theme_idx + 1) % len(THEME_NAMES)
                elif key == 'down':
                    if param_idx == 0:
                        mode_idx = (mode_idx - 1) % len(MODE_NAMES)
                    elif param_idx == 1:
                        style_idx = (style_idx - 1) % len(STYLE_NAMES)
                    elif param_idx == 2:
                        gain_idx = max(gain_idx - 1, 0)
                    elif param_idx == 3:
                        theme_idx = (theme_idx - 1) % len(THEME_NAMES)
            if quit_pressed:
                break

            with audio_lock:
                raw = audio_buf['data'] * GAIN_VALUES[gain_idx]

            alpha = 0.3
            if smooth_wave is None or len(smooth_wave) != len(raw):
                smooth_wave = raw.copy()
            else:
                smooth_wave = alpha * smooth_wave + (1 - alpha) * raw

            kernel = np.ones(7) / 7
            wave = np.convolve(smooth_wave, kernel, mode='same')

            ts = shutil.get_terminal_size()
            cols = ts.columns
            rows = ts.lines - 2

            theme_stops = THEMES[THEME_NAMES[theme_idx]]
            style = STYLE_NAMES[style_idx]
            lines = VIZ_FUNCS[style][mode_idx](wave, rows, cols, theme_stops)

            status = render_status(cols, param_idx, mode_idx, style_idx,
                                   gain_idx, theme_idx)

            frame = '\033[H' + status + '\033[0m\r\n' + '\033[0m\r\n'.join(lines)
            sys.stdout.write(frame)
            sys.stdout.flush()

    finally:
        stream.stop()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sys.stdout.write('\033[?25h\033[0m\033[2J\033[H')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
