# Visual Filter Track

Apply motion-tracked networks, glow, scanlines, and glitch effects to video with OpenCV. Includes `cyber`, `ghost`, and `overdrive` presets.

## Setup

Use Python 3.10 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Render

```sh
python main.py
python main.py Videos/Input/tau.mp4 -o Videos/Output/custom.mp4 --preset ghost
python main.py --preset overdrive --scale 0.5 --limit-frames 120
```

Videos are kept locally and excluded from Git. Supply your own input video or place one at the default path.

Defaults are relative to the script: input `Videos/Input/tau.mp4`, output `Videos/Output/rendered.mp4`. Existing output files are overwritten. Input and output must be different files. Output dimensions after scaling must be even and at least two pixels in each direction for MP4 encoding.

Use `--preview` for a live window (press **q** to stop) and `--seed` for repeatable effect randomness. Preview requires a graphical desktop. Run `python main.py --help` for all options.

Output uses the MPEG-4 video codec and contains no audio; source audio is not copied.

## Verify

```sh
python -m unittest discover -s tests -v
```

Tests cover tracking loss, invalid inputs, resource cleanup, and encoding/decoding all three presets.
