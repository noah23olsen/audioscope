# audioscope

Terminal audio visualizer with 6 modes, 2 render styles, and 4 color themes. Tested on macOS only.

![screenshot](screenshot.png)

## Modes

- **Waveform** — amplitude over time
- **Mirror** — symmetric center-line waveform
- **Spectrum** — FFT frequency bars (log-scaled)
- **Bands** — 16-bar graphic equalizer
- **Spectrogram** — scrolling frequency waterfall
- **Oscilloscope** — circular radial waveform

## Install

```bash
brew install blackhole-2ch
pip install -r requirements.txt
```

## Setup

1. Open **Audio MIDI Setup** (Cmd+Space → "Audio MIDI Setup")
2. Create **Multi-Output Device** → check **BlackHole 2ch** + your speakers
3. Set system output to the Multi-Output Device

## Run

```bash
python viz.py
```

## Controls

```
←→  select parameter
↑↓  adjust value
q   quit
```

Parameters: Mode, Style (Braille/Blocks), Gain, Color Theme
