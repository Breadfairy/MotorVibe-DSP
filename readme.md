# Motor Signal Pipeline

Direct Python pipeline for early motor sensing analysis from CSV or live
binary serial data.

Current flow:
- read data
- build `2s` buffers
- analyse the latest `1s` period inside the buffer
- build signals
- build ML features / inference
- save exploration charts

## Startup

1. Flash MCU firmware that matches the reduced binary schema in
   `MCU/dataStreamBinary/README.md`.
2. Install Python dependencies.
3. Plug in the MCU over USB.
4. Run one of the modes below from the repo root.

Live mode will auto-pick the first likely serial device. You can also pass a
port explicitly.

## Hardware
- ESP32 CP2102 38 Pin Development Board USB-C
- `2 x MPU6050`
- `1 x DS18B20`

## Raw Field Order
- `sample`
- `mpu1AccX`, `mpu1AccY`, `mpu1AccZ`
- `mpu1GyrX`, `mpu1GyrY`, `mpu1GyrZ`
- `mpu2AccX`, `mpu2AccY`, `mpu2AccZ`
- `mpu2GyrX`, `mpu2GyrY`, `mpu2GyrZ`
- `ds18b20`

CSV and live binary data need to follow this exact order.

## Current Program State
- CSV replay is the main ready path for first captured data.
- Buffers are `2s` wide across CSV, live, and training preparation.
- Analysis period is the latest `1s` inside the current `2s` buffer.
- CSV mode generates three short-period plot groups plus long exploratory
  plots.
- Live mode opens one simple live `mpu1AccMag` plot and prints one refreshing
  terminal monitor block.
- Live mode prints the measured sample rate from the incoming MCU stream.
- Live mode expects the reduced binary packet layout from the flashed firmware.
- Training code is wired for labelled CSV folders and the current feature
  shape.

## File Layout
- `src/main.py`
  Top-down orchestration for CSV, live, and training modes.
- `src/buffer.py`
  Fixed-size and rolling buffer construction.
- `src/signals.py`
  Time-domain signals, per-axis FFT summaries, combined acceleration FFT, and
  bearing band RMS.
- `src/ml.py`
  Feature extraction, PyTorch classifier, training, and inference.
- `src/charting.py`
  Shared chart styling and saved chart output.
- `src/gui.py`
  Terminal summaries and the simple live accelerometer monitor.
- `Tests/serial_harness/serialPrint.py`
  Bare serial line print harness.
- `Tests/fft_axis_harness.py`
  Standalone FFT harness for quick CSV FFT checks.

## Buffer and Period Model
- `sampleRate = 800`
- `bufferSeconds = 2`
- `periodSeconds = 1`
- `liveStepSeconds = 1`

Meaning:
- one `2s` buffer is built
- the latest `1s` inside that buffer is analysed
- live mode advances by one incoming `1s` MCU batch
- training also steps by `1s` while using full `2s` buffers

## Current Signal Outputs
Time-domain:
- `mpu1AccMag`, `mpu1GyrMag`
- `mpu2AccMag`, `mpu2GyrMag`
- `ds18b20Avg`, `ds18b20Grad`

Frequency-domain:
- per-axis FFT spectra and peak metrics for:
  `mpu1AccX/Y/Z`, `mpu2AccX/Y/Z`, `mpu1GyrX/Y/Z`, `mpu2GyrX/Y/Z`
- combined acceleration FFT per MPU:
  `mpu1AccSpectrum`, `mpu2AccSpectrum`
- combined acceleration FFT peak metrics:
  `mpu1AccFundamentalHz`, `mpu1AccFundamentalMag`
  `mpu2AccFundamentalHz`, `mpu2AccFundamentalMag`
- bearing metrics from combined acceleration FFT:
  `BpfoBand`, `BpfiBand`, `BpfoBandRms`, `BpfiBandRms`

## Current ML Features
The current model uses `18` scalar features:
- accel magnitude mean for each MPU
- gyro magnitude mean for each MPU
- one DS18B20 average mean and temp gradient mean
- accel FFT peak `Hz` and `Mag` for `X`, `Y`, `Z` on each MPU

The model does not currently use:
- full FFT bins
- bearing RMS features
- gyro FFT features

## CSV Plot Outputs
CSV chart output goes to:
- `outputs/csvCharts/sensors1/`
- `outputs/csvCharts/sensors2/`
- `outputs/csvCharts/allFFT/`
- `outputs/csvCharts/longs/`

Regular short-period outputs:
- sensor 1 plots: `s1p1.png`, `s1p2.png`, `s1p3.png`
- sensor 2 plots: `s2p1.png`, `s2p2.png`, `s2p3.png`
- axis FFT grids: `fftp1.png`, `fftp2.png`, `fftp3.png`

Long exploratory outputs:
- `s1l.png`
- `s2l.png`
- `fftl.png`

Current chart types:
- one `3 x 2` sensor chart per MPU:
  accMag, gyroMag, DS18B20 temp, combined accel FFT, bearing RMS
- one `12` panel axis FFT grid:
  all accel and gyro axes for both MPUs

## Serial Agreement
Firmware and Python need one agreed serial contract before live sensing starts.

Agree on:
- baud rate
- field order
- units / scaling
- sample rate
- batch header
- packet format

Recommended first pass:
- binary payload
- fixed batch header
- fixed packet format
- fixed field order matching this README

Current Python live settings:
- `2000000` baud
- batch header `AABBCCDD`
- packet format `<I13h`
- nominal sample rate `800 Hz`

## Training Data Layout
Use one folder per label:
- `data/train/healthy/*.csv`
- `data/train/looseMounting/*.csv`
- `data/train/maxLoading/*.csv`
- `data/train/minLoading/*.csv`
- `data/train/offAxis/*.csv`
- `data/train/multipleFailures/*.csv`

Each CSV should follow the raw field order listed above.

## Dependencies
Install:

```bash
python3 -m pip install numpy pandas pyserial matplotlib scipy torch
```

## Commands
CSV replay:

```bash
python3 src/main.py csv
```

Offline training:

```bash
python3 src/main.py train
```

Live processing:

```bash
python3 src/main.py live
```

Live processing with an explicit port:

```bash
python3 src/main.py live COM3
```

or:

```bash
python3 src/main.py live /dev/cu.usbmodem5A360305921
```

FFT harness:

```bash
python3 Tests/fft_axis_harness.py
```

Serial raw print harness:

```bash
python3 Tests/serial_harness/serialPrint.py
```

## Current Limits
- default CSV path uses `data/testdata/reduced_sample.csv`
- live plot cadence is currently limited by the firmware batch size of `1s`
- real model quality depends entirely on the first correctly structured capture
  set
- current bearing metrics are exploratory features, not trained labels
