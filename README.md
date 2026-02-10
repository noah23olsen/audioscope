# audioscope

Visualize audio inside your terminal.


https://github.com/user-attachments/assets/088d7c8d-8f78-480f-8979-144486f40272

## Install

```bash
git clone https://github.com/noah23olsen/audioscope.git
cd audioscope
brew install blackhole-2ch
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Setup (MacOS only)

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
