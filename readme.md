# Motor Signal Pipeline

This repo builds a direct motor sensing pipeline:
- serial MCU data -> arrays
- CSV replay data -> arrays
- DSP and ML consume those arrays
- charting writes visual output

The current build order is:
- buffer
- signals
- ML
- gui / visualisation

## Sensors
Current MCU and link target:
- ESP32 CP2102 38 Pin Development Board USB-C

Target hardware uses:
- 2 x MPU6050
- 2 x DS18B20

Current raw field order:
- `sample`
- `mpu1AccX`, `mpu1AccY`, `mpu1AccZ`
- `mpu1GyrX`, `mpu1GyrY`, `mpu1GyrZ`
- `mpu1Temp`
- `mpu2AccX`, `mpu2AccY`, `mpu2AccZ`
- `mpu2GyrX`, `mpu2GyrY`, `mpu2GyrZ`
- `mpu2Temp`
- `ds18b20One`
- `ds18b20Two`

## File Layout
- `src/main.py`: top-down orchestration for csv, live, and train modes
- `src/buffer.py`: fixed-size and rolling live buffer building
- `src/signals.py`: raw, time-domain, and frequency-domain signals
- `src/ml.py`: feature extraction, model building, training, inference
- `src/gui.py`: terminal summaries and output orchestration
- `src/charting.py`: offline chart image output
- `serial_harness/serialPrint.py`: basic raw serial line monitor
- `data/testData/`: scaffold CSV replay data
- `data/train/`: labelled training captures by condition
- `outputs/csv/`: saved charts
- `outputs/models/`: saved model weights

## Serial Agreement
The firmware side and Python side need one agreed serial contract before
live capture starts.

Current live transport target:
- ESP32 MCU over onboard CP2102 USB serial bridge
- host connection over USB-C

Agree on:
- baud rate
- line ending
- text or binary payload
- payload delimiter if text is used
- payload field order
- sample rate target
- whether a header row is sent
- whether sample index or timestamp is sent
- units and scaling for every field
- whether the ESP32 waits for a host start token
- whether the board auto-resets when the serial port opens

Recommended first protocol for this repo:
- text payload
- UTF-8 encoding
- one newline-terminated CSV row per sample
- comma-delimited fields
- fixed field order matching the raw field list above

Example row:
- `124,0.12,-0.03,9.81,0.10,0.02,-0.01,31.5,0.11,-0.02,9.79,0.09,0.01,-0.02,31.4,28.2,28.1`

## Live Buffering
Live buffering is rolling rather than disjoint block reads.

Current live flow:
- fill one 2-second live buffer
- read the next live update chunk
- drop the oldest rows
- append the newest rows
- rebuild signals and ML output on the updated window
- repeat

This keeps a fixed live analysis window while allowing overlapped updates.

## Training Data Layout
Use one folder per condition label:
- `data/train/healthy/*.csv`
- `data/train/looseMounting/*.csv`
- `data/train/maxLoading/*.csv`
- `data/train/minLoading/*.csv`
- `data/train/offAxis/*.csv`
- `data/train/multipleFailures/*.csv`

Training flow:
- scan label folders
- slice each CSV into 1-second windows
- build one feature vector per window
- stack all windows into one training set
- train the model
- save weights for later inference

## Generated Signals
Time-domain signals:
- `mpu1AccMag`
- `mpu1GyrMag`
- `mpu2AccMag`
- `mpu2GyrMag`
- `mpu1TempAvg`, `mpu1TempGrad`
- `mpu2TempAvg`, `mpu2TempGrad`
- `ds18b20OneAvg`, `ds18b20OneGrad`
- `ds18b20TwoAvg`, `ds18b20TwoGrad`

Frequency-domain signals:
- `mpu1AccMagDomFreq`, `mpu1AccMagDomMag`
- `mpu1AccAxisPowerSpectrum`
- `mpu1AccAxisPowerEnergy1_A`, `mpu1AccAxisPowerEnergy2_A`
- `mpu1AccAxisPowerEnergy3_A`
- `mpu1GyrMagDomFreq`, `mpu1GyrMagDomMag`
- `mpu1GyrAxisPowerSpectrum`
- `mpu1GyrAxisPowerEnergy1_A`, `mpu1GyrAxisPowerEnergy2_A`
- `mpu1GyrAxisPowerEnergy3_A`
- `mpu2AccMagDomFreq`, `mpu2AccMagDomMag`
- `mpu2AccAxisPowerSpectrum`
- `mpu2AccAxisPowerEnergy1_A`, `mpu2AccAxisPowerEnergy2_A`
- `mpu2AccAxisPowerEnergy3_A`
- `mpu2GyrMagDomFreq`, `mpu2GyrMagDomMag`
- `mpu2GyrAxisPowerSpectrum`
- `mpu2GyrAxisPowerEnergy1_A`, `mpu2GyrAxisPowerEnergy2_A`
- `mpu2GyrAxisPowerEnergy3_A`

Signals used by ML:
- all `MagDomFreq` and `MagDomMag` values
- all `AxisPowerEnergy1_A`, `AxisPowerEnergy2_A`, `AxisPowerEnergy3_A`
- resampled `AxisPowerSpectrum` bins from all 4 motion signals
- mean of `mpu1TempAvg`, `mpu1TempGrad`
- mean of `mpu2TempAvg`, `mpu2TempGrad`
- mean of `ds18b20OneAvg`, `ds18b20OneGrad`
- mean of `ds18b20TwoAvg`, `ds18b20TwoGrad`

Signals for non-ML:
- `ds18b20OneAvg`, `ds18b20TwoAvg`
- `ds18b20OneGrad`, `ds18b20TwoGrad`
- `DomFreq`, `DomMag`
- `AxisPowerEnergy1_A`, `AxisPowerEnergy2_A`, `AxisPowerEnergy3_A`

## ML Program Flow
There are really two ML paths in this repo:
- training from labelled CSV files
- inference from one current signal block

Training path:

```text
label folders
-> buildLabelledCsvPaths
-> ml.buildTrainingSet
-> ml.buildLabelledFeatureRows
-> ml.buildWindowSignalData
-> buffer.buildBuffer
-> signals.buildSignals
-> ml.buildFeatureVector
-> ml.buildFeatureTensor + ml.buildLabelTensor
-> buildClassifierModel
-> ml.trainModel
-> ml.saveModel
```

In plain terms:
- `main.runTraining(...)` starts the training flow
- each labelled CSV is cut into 1-second windows
- each window becomes one `signalData` block
- each `signalData` block becomes one feature vector
- all feature vectors are stacked into one training matrix
- the model is trained on that matrix and the labels
- the trained weights are saved to `modelPath`

Inference path:

```text
current CSV or live buffer
-> signals.readCSV or buffer.liveRollingBuffer
-> signals.buildSignals
-> buildMLData
-> ml.buildFeatureVector
-> buildClassifierModel
-> ml.loadModel
-> ml.buildFeatureTensor
-> ml.runInference
-> ml.buildProbabilityDict
-> ml.buildPredictedLabel
```

In plain terms:
- `main.runCSV(...)` or `main.runLive(...)` builds one current signal block
- `buildMLData(...)` turns that block into the same features used in
  training
- the saved model weights are loaded
- the model returns class probabilities
- the top probability becomes the predicted label

## Why These ML Settings Exist
Main ML config in `main.py`:
- `hiddenSize = 32`
- `epochCount = 200`
- `learningRate = 0.001`

What they mean:
- `hiddenSize`
  How wide the hidden layers are. Bigger can learn more, but also adds
  weight and can overfit faster.
- `epochCount`
  How many times the model sees the full training set.
- `learningRate`
  How big each optimiser update step is during training.

Why they are needed:
- without `hiddenSize`, the network shape is incomplete
- without `epochCount`, training has no stopping point
- without `learningRate`, the optimiser has no step size

Why these values are a reasonable start:
- `hiddenSize = 32`
  Small model, enough room to learn, still light
- `epochCount = 200`
  Gives the small dense model time to settle
- `learningRate = 0.001`
  Common safe starting point for Adam

## Labelling And Capture
Labelling recommendations:
- keep one physical condition per CSV file
- use the folder name as the ground-truth label
- keep raw schema identical across all captures
- keep sample rate identical across all captures
- split train, validation, and test by capture file, not by window
- do not let windows from one CSV land in multiple dataset splits
- record `multipleFailures` only after single-fault captures exist

Capture recommendations:
- record a healthy baseline at the start of each session
- capture each fault condition in separate files
- hold rpm and load steady inside a labelled file when possible
- record several takes per condition across different sessions
- capture startup, steady-state, and shutdown as separate files
- keep enough duration per file to create many 1-second windows
- store notes for motor, rig, mount, load, rpm, ambient temperature,
  and induced fault

## Dependencies
- `numpy`
- `pandas`
- `pyserial`
- `matplotlib`
- `torch`

Install:

```bash
python3 -m pip install numpy pandas pyserial matplotlib torch
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

Raw serial monitor:

```bash
python3 serial_harness/serialPrint.py
```

## Current Limits
- current sample CSVs are scaffold data, not validated real fault captures
- current ML feature count is 232, so saved models must be retrained when
  the feature set changes
- model quality will depend more on capture discipline than model depth
- live charting and terminal redraw are still unfinished
