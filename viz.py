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


def braille_cell(dots):
    """Convert a dot bitmask to a braille character."""
    return chr(BRAILLE_BASE + dots)


def plot_point(cells, colors, px, py, rows, cols, color):
    """Set a braille dot and its color in the grid."""
    cell_r, cell_c = py // 4, px // 2
    dot_r, dot_c = py % 4, px % 2
    if 0 <= cell_r < rows and 0 <= cell_c < cols:
        cells[cell_r][cell_c] |= DOT_MAP[(dot_r, dot_c)]
        colors[cell_r][cell_c] = color


def plot_line(cells, colors, px, py0, py1, rows, cols, color_fn):
    """Draw a vertical braille line segment between two y positions."""
    lo, hi = min(py0, py1), max(py0, py1)
    for y in range(lo, hi + 1):
        plot_point(cells, colors, px, y, rows, cols, color_fn(y))


def render_grid(cells, colors, rows, cols):
    """Convert braille cell/color grids into a list of ANSI-colored strings."""
    lines = []
    for r in range(rows):
        line = ''
        for c in range(cols):
            if cells[r][c]:
                line += colors[r][c] + braille_cell(cells[r][c])
            else:
                line += ' '
        lines.append(line)
    return lines


# ── Color themes ────────────────────────────────────────────────────

THEMES = {
    'cyan-purple-magenta': [
        (0.0, (0, 255, 255)),
        (0.3, (0, 255, 255)),
        (0.6, (140, 0, 255)),
        (1.0, (255, 0, 200)),
    ],
    'green-yellow-red': [
        (0.0, (0, 255, 80)),
        (0.4, (255, 255, 0)),
        (1.0, (255, 0, 0)),
    ],
    'blue-white-red': [
        (0.0, (30, 100, 255)),
        (0.5, (255, 255, 255)),
        (1.0, (255, 40, 40)),
    ],
    'rainbow': [
        (0.0, (255, 0, 0)),
        (0.2, (255, 165, 0)),
        (0.4, (255, 255, 0)),
        (0.6, (0, 255, 0)),
        (0.8, (0, 100, 255)),
        (1.0, (180, 0, 255)),
    ],
}

THEME_NAMES = list(THEMES.keys())


def lerp_color(stops, t):
    """Interpolate between color stops. t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            f = (t - t0) / max(t1 - t0, 1e-9)
            r = int(c0[0] + (c1[0] - c0[0]) * f)
            g = int(c0[1] + (c1[1] - c0[1]) * f)
            b = int(c0[2] + (c1[2] - c0[2]) * f)
            return f'\033[38;2;{r};{g};{b}m'
    _, c = stops[-1]
    return f'\033[38;2;{c[0]};{c[1]};{c[2]}m'


# ── Visualization modes ─────────────────────────────────────────────

def viz_waveform(wave, rows, cols, theme_stops):
    """Classic braille waveform — the original renderer."""
    res_y = rows * 4
    res_x = cols * 2
    mid = res_y // 2

    resampled = np.interp(np.linspace(0, len(wave) - 1, res_x),
                          np.arange(len(wave)), wave)
    amp = max(np.max(np.abs(resampled)), 0.001)
    resampled = resampled / amp

    cells = [[0] * cols for _ in range(rows)]
    colors = [['' ] * cols for _ in range(rows)]

    def color_at(y):
        return lerp_color(theme_stops, abs(y - mid) / max(mid, 1))

    prev_py = None
    for px in range(res_x):
        py = int(np.clip(mid - resampled[px] * (mid - 2), 0, res_y - 1))
        if prev_py is not None:
            plot_line(cells, colors, px, prev_py, py, rows, cols, color_at)
        plot_point(cells, colors, px, py, rows, cols, color_at(py))
        prev_py = py

    return render_grid(cells, colors, rows, cols)


def viz_spectrum(wave, rows, cols, theme_stops, samplerate=48000):
    """FFT frequency bars, log-scaled."""
    res_y = rows * 4
    n = len(wave)

    fft = np.abs(np.fft.rfft(wave))[:n // 2]
    if len(fft) < 2:
        return [' ' * cols for _ in range(rows)]

    # log-spaced frequency bins mapped to columns
    num_bars = cols * 2
    log_indices = np.logspace(np.log10(1), np.log10(len(fft) - 1),
                              num_bars).astype(int)
    log_indices = np.clip(log_indices, 0, len(fft) - 1)

    bars = np.zeros(num_bars)
    for i in range(num_bars - 1):
        lo, hi = log_indices[i], log_indices[i + 1]
        if hi <= lo:
            hi = lo + 1
        bars[i] = np.mean(fft[lo:hi])
    bars[-1] = fft[log_indices[-1]]

    peak = max(np.max(bars), 1e-6)
    bars = bars / peak

    cells = [[0] * cols for _ in range(rows)]
    colors = [['' ] * cols for _ in range(rows)]

    for px in range(num_bars):
        bar_h = int(bars[px] * (res_y - 1))
        t = px / max(num_bars - 1, 1)
        color = lerp_color(theme_stops, t)
        for dy in range(bar_h + 1):
            py = res_y - 1 - dy
            plot_point(cells, colors, px, py, rows, cols, color)

    return render_grid(cells, colors, rows, cols)


def viz_oscilloscope(wave, rows, cols, theme_stops):
    """Circular radial waveform — oscilloscope style."""
    res_y = rows * 4
    res_x = cols * 2
    cx, cy = res_x / 2, res_y / 2
    radius = min(cx, cy) * 0.8

    n_points = max(len(wave), 256)
    resampled = np.interp(np.linspace(0, len(wave) - 1, n_points),
                          np.arange(len(wave)), wave)
    amp = max(np.max(np.abs(resampled)), 0.001)
    resampled = resampled / amp

    cells = [[0] * cols for _ in range(rows)]
    colors = [['' ] * cols for _ in range(rows)]

    aspect = 2.0  # terminal chars are ~2x taller than wide
    prev_px, prev_py = None, None

    for i in range(n_points):
        angle = 2 * math.pi * i / n_points
        r = radius * (0.5 + 0.5 * resampled[i])
        px = int(cx + r * math.cos(angle) * aspect)
        py = int(cy + r * math.sin(angle))
        t = i / max(n_points - 1, 1)
        color = lerp_color(theme_stops, t)

        if 0 <= px < res_x and 0 <= py < res_y:
            plot_point(cells, colors, px, py, rows, cols, color)

        # simple connect to previous
        if prev_px is not None:
            dx = px - prev_px
            dy = py - prev_py
            steps = max(abs(dx), abs(dy), 1)
            if steps <= 4:  # only fill short gaps
                for s in range(1, steps):
                    ix = prev_px + dx * s // steps
                    iy = prev_py + dy * s // steps
                    if 0 <= ix < res_x and 0 <= iy < res_y:
                        plot_point(cells, colors, ix, iy, rows, cols, color)

        prev_px, prev_py = px, py

    return render_grid(cells, colors, rows, cols)


# ── Input handling ──────────────────────────────────────────────────

def read_key(fd):
    """Non-blocking read of a keypress. Returns key string or None."""
    if not select.select([fd], [], [], 0)[0]:
        return None
    ch = os.read(fd, 1)
    if ch == b'\x1b':
        if select.select([fd], [], [], 0.02)[0]:
            ch2 = os.read(fd, 1)
            if ch2 == b'[':
                ch3 = os.read(fd, 1)
                return {b'A': 'up', b'B': 'down', b'C': 'right', b'D': 'left'}.get(ch3)
        return 'esc'
    if ch in (b'q', b'Q'):
        return 'quit'
    return None


# ── Parameters ──────────────────────────────────────────────────────

MODE_NAMES = ['Waveform', 'Spectrum', 'Oscilloscope']
MODE_FUNCS = [viz_waveform, viz_spectrum, viz_oscilloscope]

SMOOTHING_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
GAIN_VALUES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

PARAM_NAMES = ['Mode', 'Smoothing', 'Gain', 'Color']


def format_param(idx, mode_idx, smooth_idx, gain_idx, theme_idx):
    """Format the current value of a parameter for display."""
    if idx == 0:
        return MODE_NAMES[mode_idx]
    if idx == 1:
        return f'{SMOOTHING_VALUES[smooth_idx]:.1f}'
    if idx == 2:
        return f'{GAIN_VALUES[gain_idx]:.1f}x'
    return THEME_NAMES[theme_idx]


# ── Status bar ──────────────────────────────────────────────────────

def render_status(cols, param_idx, mode_idx, smooth_idx, gain_idx, theme_idx):
    """Render the bottom status bar."""
    parts = []
    for i, name in enumerate(PARAM_NAMES):
        val = format_param(i, mode_idx, smooth_idx, gain_idx, theme_idx)
        if i == param_idx:
            parts.append(f'\033[7m {name}: {val} \033[0m')
        else:
            parts.append(f' {name}: {val} ')
    left = '  '.join(parts)
    hint = '\033[2m  \u2190\u2192 select  \u2191\u2193 adjust  q quit\033[0m'
    bar = left + hint
    # truncate to terminal width (accounting for ANSI codes)
    return bar[:cols + 200]  # generous allowance for escape sequences


# ── Main loop ───────────────────────────────────────────────────────

def main():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    # state
    mode_idx = 0
    smooth_idx = 3       # 0.3 default
    gain_idx = 1         # 1.0x default
    theme_idx = 0
    param_idx = 0
    smooth_wave = None

    # audio buffer — callback writes, main loop reads
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
        sys.stdout.write('\033[2J\033[?25l')  # clear screen, hide cursor
        sys.stdout.flush()

        while True:
            # wait up to 33ms for input — doubles as frame pacer
            select.select([fd], [], [], 0.033)

            # handle input — drain all pending keys
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
                        smooth_idx = min(smooth_idx + 1, len(SMOOTHING_VALUES) - 1)
                    elif param_idx == 2:
                        gain_idx = min(gain_idx + 1, len(GAIN_VALUES) - 1)
                    elif param_idx == 3:
                        theme_idx = (theme_idx + 1) % len(THEME_NAMES)
                elif key == 'down':
                    if param_idx == 0:
                        mode_idx = (mode_idx - 1) % len(MODE_NAMES)
                    elif param_idx == 1:
                        smooth_idx = max(smooth_idx - 1, 0)
                    elif param_idx == 2:
                        gain_idx = max(gain_idx - 1, 0)
                    elif param_idx == 3:
                        theme_idx = (theme_idx - 1) % len(THEME_NAMES)
            if quit_pressed:
                break

            # grab latest audio (non-blocking)
            with audio_lock:
                raw = audio_buf['data'] * GAIN_VALUES[gain_idx]

            # temporal smoothing
            alpha = SMOOTHING_VALUES[smooth_idx]
            if smooth_wave is None or len(smooth_wave) != len(raw):
                smooth_wave = raw.copy()
            else:
                smooth_wave = alpha * smooth_wave + (1 - alpha) * raw

            # spatial smoothing
            kernel = np.ones(7) / 7
            wave = np.convolve(smooth_wave, kernel, mode='same')

            # terminal size
            ts = shutil.get_terminal_size()
            cols = ts.columns
            rows = ts.lines - 2  # reserve for status bar

            # render viz
            theme_stops = THEMES[THEME_NAMES[theme_idx]]
            lines = MODE_FUNCS[mode_idx](wave, rows, cols, theme_stops)

            # render status bar
            status = render_status(cols, param_idx, mode_idx, smooth_idx,
                                   gain_idx, theme_idx)

            # draw
            frame = '\033[H' + '\033[0m\n'.join(lines) + '\033[0m\n' + status
            sys.stdout.write(frame)
            sys.stdout.flush()

    finally:
        stream.stop()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sys.stdout.write('\033[?25h\033[0m\033[2J\033[H')  # restore
        sys.stdout.flush()


if __name__ == '__main__':
    main()
