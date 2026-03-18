# Motor Signal Pipeline

Direct Python pipeline for early motor sensing analysis from CSV or live
serial data.

Current flow:
- read data
- build `2s` buffers
- analyse the latest `1s` period inside the buffer
- build signals
- build ML features / inference
- save exploration charts

## Hardware
- ESP32 CP2102 38 Pin Development Board USB-C
- `2 x MPU6050`
- `2 x DS18B20`

## Raw Field Order
- `sample`
- `mpu1AccX`, `mpu1AccY`, `mpu1AccZ`
- `mpu1GyrX`, `mpu1GyrY`, `mpu1GyrZ`
- `mpu1Temp`
- `mpu2AccX`, `mpu2AccY`, `mpu2AccZ`
- `mpu2GyrX`, `mpu2GyrY`, `mpu2GyrZ`
- `mpu2Temp`
- `ds18b20One`
- `ds18b20Two`

CSV and live serial data need to follow this exact order.

## Current Program State
- CSV replay is the main ready path for first captured data.
- Buffers are `2s` wide across CSV, live, and training preparation.
- Analysis period is the latest `1s` inside the current `2s` buffer.
- CSV mode generates three short-period plot groups plus long exploratory
  plots.
- Live mode builds the same signals and ML features, but currently only prints
  summaries.
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
  Saved chart output.
- `src/gui.py`
  Terminal summaries and chart call wrappers.
- `Tests/serial_harness/serialPrint.py`
  Bare serial line print harness.
- `Tests/fft_axis_harness.py`
  Standalone FFT harness for quick CSV FFT checks.

## Buffer and Period Model
- `sampleRate = 1000`
- `bufferSeconds = 2`
- `periodSeconds = 1`
- `liveStepSeconds = 1`

Meaning:
- one `2s` buffer is built
- the latest `1s` inside that buffer is analysed
- live mode advances by `1s`
- training also steps by `1s` while using full `2s` buffers

## Current Signal Outputs
Time-domain:
- `mpu1AccMag`, `mpu1GyrMag`
- `mpu2AccMag`, `mpu2GyrMag`
- `mpu1TempAvg`, `mpu1TempGrad`
- `mpu2TempAvg`, `mpu2TempGrad`
- `ds18b20OneAvg`, `ds18b20OneGrad`
- `ds18b20TwoAvg`, `ds18b20TwoGrad`

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
The current model uses `24` scalar features:
- accel magnitude mean for each MPU
- gyro magnitude mean for each MPU
- MPU temp average mean and temp gradient mean for each MPU
- DS18B20 temp average mean and temp gradient mean for each sensor
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
  accMag, gyroMag, MPU temp, DS18B20 temp, combined accel FFT, bearing RMS
- one `12` panel axis FFT grid:
  all accel and gyro axes for both MPUs

## Serial Agreement
Firmware and Python need one agreed serial contract before live sensing starts.

Agree on:
- baud rate
- delimiter
- line ending
- field order
- units / scaling
- sample rate
- whether a header row is sent

Recommended first pass:
- text payload
- UTF-8
- one CSV row per sample
- comma-delimited
- fixed field order matching this README

Example:
- `124,0.12,-0.03,9.81,0.10,0.02,-0.01,31.5,0.11,-0.02,9.79,0.09,0.01,-0.02,31.4,28.2,28.1`

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
python3 -m pip install numpy pandas pyserial matplotlib torch scipy
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

FFT harness:

```bash
python3 Tests/fft_axis_harness.py
```

Serial raw print harness:

```bash
python3 Tests/serial_harness/serialPrint.py
```

## Current Limits
- default CSV path still assumes you place `test.csv` in the expected location
- live chart streaming is not implemented yet
- real model quality depends entirely on the first correctly structured capture
  set
- current bearing metrics are exploratory features, not trained labels
