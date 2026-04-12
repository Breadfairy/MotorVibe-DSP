# Fan Test Pipeline

This repo is the reduced fan validation build for the motor pipeline.

The active direction is:

- `1 x MPU6050`
- packet timestamp only
- controlled fan failure states
- separate Python scripts for capture, analysis, training, and live inference

The older two-MPU motor captures remain in `data/Motordata` as reference only.
The active `src/` code now targets the reduced one-MPU packet.

## Active Python Files

- `src/capture.py`
  capture one-MPU binary serial data into CSV
- `src/analysis.py`
  read one CSV and save one-sensor plots
- `src/training.py`
  read labelled CSV folders, build features, train, and save the model
- `src/live.py`
  read live binary packets, build one live window, and run inference
- `src/visualiser.py`
  read live binary packets and draw the one-sensor live view
- `src/signals_core.py`
  shared signal maths
- `src/ml_core.py`
  shared feature extraction and model logic
- `src/charting_core.py`
  shared chart output

## Active Packet Contract

The Python `src/` pipeline expects the same packet from the one-MPU legacy fan
firmware:

- baud: `1000000`
- host waits for:
  `Sample struct size (bytes): 16`
- host sends:
  `START\n`
- packet format:
  `"<I6h"`
- packet fields:
  `t_us ax ay az gx gy gz`
- firmware send block:
  `32` samples

Python scaling:

- accel raw to `g`: divide by `16384.0`
- gyro raw to `deg/s`: divide by `131.0`

CSV columns written by `src/capture.py`:

- `t_us`
- `t_s`
- `ax`
- `ay`
- `az`
- `gx`
- `gy`
- `gz`

## Active Feature Set

The current model uses `12` scalar features:

- accel magnitude mean
- gyro magnitude mean
- accel X fundamental `Hz`
- accel X fundamental magnitude
- accel Y fundamental `Hz`
- accel Y fundamental magnitude
- accel Z fundamental `Hz`
- accel Z fundamental magnitude
- combined accel fundamental `Hz`
- combined accel fundamental magnitude
- combined accel BPFO band RMS
- combined accel BPFI band RMS

## Legacy Firmware Path

The isolated one-MPU legacy firmware path is:

- `MCU/LegacyHardware/legacyFanStream/legacyFanStream.ino`

That sketch keeps `MCU/newHardware` untouched and mirrors its host handshake
model on the older MCU using one MPU only.

## Training Layout

Training expects one folder per label:

- `data/train/healthy/*.csv`
- `data/train/tiltX/*.csv`
- `data/train/tiltY/*.csv`
- `data/train/looseScrew1/*.csv`
- `data/train/looseScrew2/*.csv`
- `data/train/looseScrew3/*.csv`
- `data/train/looseScrew4/*.csv`
- `data/train/imbalanceAdded/*.csv`
- `data/train/rubbingContact/*.csv`

Each CSV is expected to match the one-MPU column layout listed above.

## Commands

Install dependencies:

```bash
python3 -m pip install numpy pandas pyserial matplotlib scipy torch
```

Run capture:

```bash
python3 src/capture.py test1
```

Run analysis:

```bash
python3 src/analysis.py ML_data/test1.csv
```

Run training:

```bash
python3 src/training.py
```

Run live inference:

```bash
python3 src/live.py
```

Run live visualisation:

```bash
python3 src/visualiser.py
```

Run the packet smoke test:

```bash
python3 src/offline_smoke.py
```
