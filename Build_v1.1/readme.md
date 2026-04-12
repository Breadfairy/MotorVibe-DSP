# Motor Signal Pipeline

Simple Python scripts for motor vibration capture, analysis, training, and
live monitoring.

The active refactor path is now in `simpleSrc/`.

## How To Use

Install dependencies:

```bash
python3 -m pip install numpy pandas pyserial matplotlib scipy torch
```

Run visual CSV analysis:

```bash
python3 simpleSrc/analysis.py
```

Run visual CSV analysis on a chosen file:

```bash
python3 simpleSrc/analysis.py simplified/Adaption/motordata_20260408_131505.csv
```

Run capture:

```bash
python3 simpleSrc/capture.py test1
```

Run training:

```bash
python3 simpleSrc/training.py
```

Run live monitoring:

```bash
python3 simpleSrc/live.py
```

Run live visualisation:

```bash
python3 simpleSrc/visualiser.py
```

Run offline serial preflight smoke test:

```bash
python3 simpleSrc/offline_smoke.py
```

## Current Direction

The repo is moving away from the old `src/main.py` CLI router and toward
separate top-down scripts:

- `python3 simpleSrc/capture.py`
- `python3 simpleSrc/analysis.py`
- `python3 simpleSrc/training.py`
- `python3 simpleSrc/live.py`
- `python3 simpleSrc/visualiser.py`

The goal is a more direct Python workflow with one file per main task.

## Current Data Shape

The current shared CSV and live row layout is:

- `t_us`
- `t_s`
- `ax1`, `ay1`, `az1`
- `gx1`, `gy1`, `gz1`
- `ax2`, `ay2`, `az2`
- `gx2`, `gy2`, `gz2`
- `tempC`

This matches the adaption work more closely than the older `src/` pipeline.

## simpleSrc Layout

- `simpleSrc/capture.py`
  Serial capture to flat CSV files.
- `simpleSrc/analysis.py`
  Read one CSV and save the visual plots only.
- `simpleSrc/training.py`
  Read labelled CSV folders, build features, train, and save the model.
- `simpleSrc/live.py`
  Read live serial data, build the current feature vector, run ML, and print
  a simple live monitor.
- `simpleSrc/visualiser.py`
  Read live serial data and draw a smooth live sensor view with no ML.
- `simpleSrc/signals_core.py`
  Shared signal maths.
- `simpleSrc/ml_core.py`
  Shared feature extraction and model logic.
- `simpleSrc/charting_core.py`
  Shared chart styling and saved chart output.

## Current Flow

Analysis flow:

- read one CSV
- convert columns to arrays
- build magnitudes and FFTs
- save plots

Training flow:

- read labelled CSV windows
- build the shared `1s` feature vector
- train the model
- save model weights

Live flow:

- read binary serial packets
- convert packets into the shared row format
- keep the latest `1s` live window
- build the same feature vector as training
- run ML
- print a live status block

## Current Timing

- nominal sample rate: `1000 Hz`
- shared feature window: `1.0 s`
- training window rows: `1000`
- live window rows: `1000`
- serial write batch from current firmware direction: `32` samples

The code currently uses the nominal locked sample rate rather than trusting
incoming timestamps. This is intentional for now because the sample adaption
CSV timestamps are not stable yet.

## Current Charts

`simpleSrc/analysis.py` saves:

- `outputs/simpleCharts/sensors1/sensor1.png`
- `outputs/simpleCharts/sensors2/sensor2.png`
- `outputs/simpleCharts/allFFT/fftGrid.png`

The chart style is carried over from the earlier charting code with minimal
visual change.

## Current ML Features

The shared model uses `18` scalar features:

- accel magnitude mean for MPU1
- gyro magnitude mean for MPU1
- temperature average mean
- temperature gradient mean
- accel FFT peak `Hz` and `Mag` for `X`, `Y`, `Z` on MPU1
- accel magnitude mean for MPU2
- gyro magnitude mean for MPU2
- accel FFT peak `Hz` and `Mag` for `X`, `Y`, `Z` on MPU2

`training.py` and `live.py` use the same feature vector path.

## Serial Handshake

The current handshake between `simpleSrc/` and
`simplified/Adaption/rolhiltcode.ino` is:

- baud rate: `1000000`
- nominal sample rate: `1000 Hz`
- host waits for: `Sample struct size (bytes): 32`
- host sends: `START\n`
- binary stream starts only after that `START` command
- packet format: `"<I12hf"`
- packet contents:
  `t_us`,
  `ax1 ay1 az1`,
  `gx1 gy1 gz1`,
  `ax2 ay2 az2`,
  `gx2 gy2 gz2`,
  `tempC`
- firmware write batch: `32` samples

Current scaling inside Python:

- accel raw to `g`: divide by `16384.0`
- gyro raw to `deg/s`: divide by `131.0`
- temp is read directly as the packed float
- Python zero-bases `t_us` with wrap-safe `uint32` arithmetic

## Handshake Adaptation

`capture.py`, `live.py`, and `visualiser.py` are intentionally standalone and
each script carries its own handshake constants and packet conversion logic.

If the firmware changes again, patch the handshake sections in all three
scripts:

1. `baudRate`
2. `readyPrefix`
3. `startCommand`
4. `recordFormat`
5. `bufferSampleCount`
6. accel and gyro scaling values
7. packet-to-row conversion block

## Training Data Layout

Training currently expects one folder per label:

- `data/train/healthy/*.csv`
- `data/train/looseMounting/*.csv`
- `data/train/maxLoading/*.csv`
- `data/train/minLoading/*.csv`
- `data/train/offAxis/*.csv`
- `data/train/multipleFailures/*.csv`

Capture currently writes to a flat folder:

- `ML_data/*.csv`

You can sort those files into labelled folders later.

## Current Limits

- `simpleSrc/live.py` still needs real hardware verification on the updated
  handshake
- the firmware-side `1 kHz` lock needs to be confirmed after flashing the new
  non-blocking batch sender
- the live script currently updates once per full `1s` feature window
- real model quality still depends on correctly captured and labelled data
- the older `src/` pipeline still exists, but `simpleSrc/` is now the active
  simplification path
