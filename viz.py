import sounddevice as sd
import numpy as np
import sys, shutil

cols = shutil.get_terminal_size().columns
rows = shutil.get_terminal_size().lines - 2
# braille gives 4x vertical, 2x horizontal resolution
res_y = rows * 4
res_x = cols * 2

print('\033[2J\033[?25l')

stream = sd.InputStream(device="BlackHole 2ch", channels=1, samplerate=48000, blocksize=4096)
stream.start()

mid = res_y // 2

def color_for_dist(d, max_d):
    t = min(d / max(max_d, 1), 1.0)
    if t < 0.3:
        r, g, b = 0, 255, 255       # cyan
    elif t < 0.6:
        r, g, b = 140, 0, 255       # purple
    else:
        r, g, b = 255, 0, 200       # magenta
    return f'\033[38;2;{r};{g};{b}m'

BRAILLE_BASE = 0x2800
# braille dot positions: each cell is 2 wide x 4 tall
# dots numbered:  0  3
#                 1  4
#                 2  5
#                 6  7
dot_map = {
    (0, 0): 0x01, (1, 0): 0x02, (2, 0): 0x04, (3, 0): 0x40,
    (0, 1): 0x08, (1, 1): 0x10, (2, 1): 0x20, (3, 1): 0x80,
}

smooth_wave = None
alpha = 0.3

try:
    while True:
        data, _ = stream.read(4096)
        raw = np.interp(np.linspace(0, len(data)-1, res_x), np.arange(len(data)), data[:, 0])
        
        # smooth over time
        if smooth_wave is None:
            smooth_wave = raw.copy()
        else:
            smooth_wave = alpha * raw + (1 - alpha) * smooth_wave
        
        # spatial smoothing
        kernel = np.ones(7) / 7
        wave = np.convolve(smooth_wave, kernel, mode='same')
        
        amp = max(np.max(np.abs(wave)), 0.001)
        wave = wave / amp
        
        # build braille grid
        cells = [[0] * cols for _ in range(rows)]
        colors = [['' ] * cols for _ in range(rows)]
        
        for px in range(res_x):
            py = int(np.clip(mid - wave[px] * (mid - 2), 0, res_y - 1))
            
            # connect to previous point
            if px > 0:
                py_prev = int(np.clip(mid - wave[px-1] * (mid - 2), 0, res_y - 1))
                lo, hi = min(py, py_prev), max(py, py_prev)
                for fill_py in range(lo, hi + 1):
                    cell_r, cell_c = fill_py // 4, px // 2
                    dot_r, dot_c = fill_py % 4, px % 2
                    if cell_r < rows and cell_c < cols:
                        cells[cell_r][cell_c] |= dot_map[(dot_r, dot_c)]
                        dist = abs(fill_py - mid)
                        colors[cell_r][cell_c] = color_for_dist(dist, mid)
            
            cell_r, cell_c = py // 4, px // 2
            dot_r, dot_c = py % 4, px % 2
            if cell_r < rows and cell_c < cols:
                cells[cell_r][cell_c] |= dot_map[(dot_r, dot_c)]
                dist = abs(py - mid)
                colors[cell_r][cell_c] = color_for_dist(dist, mid)
        
        lines = []
        for r in range(rows):
            line = ''
            for c in range(cols):
                if cells[r][c]:
                    line += colors[r][c] + chr(BRAILLE_BASE + cells[r][c])
                else:
                    line += ' '
            lines.append(line)
        
        sys.stdout.write('\033[H' + '\033[0m\n'.join(lines) + '\033[0m')
        sys.stdout.flush()

except KeyboardInterrupt:
    print('\033[?25h\033[0m')
    stream.stop()
