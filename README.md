# Terminal Audio Visualizer

Real-time audio waveform visualizer rendered with braille unicode characters in your terminal.

## Setup (macOS)

### 1. Install BlackHole

BlackHole is a virtual audio driver that creates a "fake" audio device on your Mac. This is what lets the visualizer tap into whatever audio your system is playing.

```bash
brew install blackhole-2ch
```

If you don't have Homebrew, install it first: https://brew.sh

After installing, you may need to allow the BlackHole system extension. macOS will prompt you, or you can find it in **System Settings → Privacy & Security** — scroll down and click **Allow**.

### 2. Create a Multi-Output Device

Without this step, you'd have to choose between hearing your audio OR sending it to the visualizer. A multi-output device sends audio to both at the same time.

1. Open **Audio MIDI Setup** — press `Cmd + Space`, type "Audio MIDI Setup", hit Enter
2. In the bottom left, click the **+** button and select **Create Multi-Output Device**
3. In the right panel, check the boxes for:
   - **BlackHole 2ch**
   - Your normal output (whatever you listen through — speakers, headphones, etc.)
4. Right-click your normal output in the list and select **Use This Device As Clock Source** (this prevents audio glitches)

### 3. Switch your audio output

Go to **System Settings → Sound → Output** and select the **Multi-Output Device** you just created.

You won't be able to change volume with the menu bar slider while using a multi-output device — this is a macOS limitation. Control volume from the app playing audio instead, or from your speaker/headphone hardware.

### 4. Install Python dependencies

```bash
pip install sounddevice numpy
```

If you get a permissions error, try `pip install --user sounddevice numpy` or use a virtual environment.

### 5. Run

```bash
python viz.py
```

Play some audio and you should see the waveform. Press `Ctrl+C` to quit.

## Troubleshooting

**No waveform / flat line:**
- Make sure your system output is set to the Multi-Output Device (step 3)
- Make sure audio is actually playing
- Run `python -c "import sounddevice; print(sounddevice.query_devices())"` and verify "BlackHole 2ch" shows up in the list

**`PortAudio` or device errors:**
- You may need to install PortAudio: `brew install portaudio`
- Then reinstall sounddevice: `pip install --force-reinstall sounddevice`

**No sound coming out:**
- Check that your real speakers/headphones are checked in the Multi-Output Device (step 2)
- Check that they are set as the clock source

**Want to stop using the multi-output device?**
- Just switch your output back to your normal device in **System Settings → Sound → Output**
