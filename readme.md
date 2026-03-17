# Motor Signal Pipeline

This repo builds a direct motor sensing pipeline. runs live or via csv
input. The current program flow  is:
- read
- buffer
- build signals
- ML
- gui / visualisation

## Hardware
- ESP32 CP2102 38 Pin Development Board USB-C
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

Agree on:
- baud rate
- line ending
- payload delimiter. eg ",".
- payload field order
- sample rate
- whether a header row is sent
- units and scaling for every field. eg rounding

Recommended first protocol for this repo:
- text payload
- UTF-8 encoding
- one CSV row per sample
- comma-delimited fields
- fixed field order matching the raw field list above

Example:
- `124,0.12,-0.03,9.81,0.10,0.02,-0.01,31.5,0.11,-0.02,9.79,0.09,0.01,-0.02,31.4,28.2,28.1`

## Live Buffering
Live buffering is rolling rather than disjoint block reads.

Current live flow:
- fill one 2-second live buffer array
- read the next live update chunk. array updated every 0.25s
- drop the oldest rows
- append the newest rows
- rebuild signals and ML output on the updated window. (1s window analysis and
  2s buffer for processing time provisions)
- repeat

This keeps a fixed live analysis window while allowing overlapped updates.

## Training Data Layout

Use one folder per condition label:
- `data/train/healthy/*.csv`
- `data/train/looseMounting/*.csv`
- `data/train/maxLoading/*.csv`
- `data/train/minLoading/*.csv`
- `data/train/offAxis/*.csv`
- `data/train/multipleFailures/*.csv` (eg, 1 screw loose and off axis)

each failure mode should be run for a minimum 1m. 
More minutes = more training samples = stronger ML.


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

## Frequency Analysis
Current prototype frequency analysis follows this flow:
- build one time-domain window
- build one real FFT magnitude spectrum
- detect the dominant fundamental frequency
- derive shaft speed from that fundamental:
  `shaftRpm = 60 * fundamentalHz`
- derive bearing order bands from the fundamental
- track band RMS over time rather than one single FFT bin peak

Current bearing proof-of-concept uses:
- `BPFO` on the raw FFT
- `BPFI` on the raw FFT order band

Current prototype order bands are:
- `BPFO`: `3x` to `5x` shaft frequency with `+/- 10%` tolerance
- `BPFI`: `5x` to `7x` shaft frequency with `+/- 10%` tolerance

The main levers are:
- sample rate:
  sets the max observable frequency and the frequency axis scaling
- sample length:
  sets frequency resolution using
  `frequencyResolutionHz = sampleRate / sampleCount`

At `1000 Hz` sample rate:
- Nyquist is `500 Hz`
- low-frequency order tracking is still usable
- fundamental, BPFO, and BPFI order bands can still be tracked with FFT if
  they fall below `500 Hz`

For the current planned build, this means:
- shaft frequency tracking is fine
- BPFO FFT band tracking is fine
- BPFI FFT band tracking is fine as a prototype feature

## ML Program Flow
There are really two ML paths in this repo:
- training from labelled CSV files
- live from one current signal block

Training flow:
- scan label folders
- slice each CSV into 1-second windows
- build one feature vector per window
- stack all windows into one training set
- train the model
- save weights for later inference

Live flow:

- `main.runLive(...)` builds one current signal block
- `buildMLData(...)` turns that block into the same features used in
  training
- the saved model weights are loaded
- the model returns class probabilities
- the top probability becomes the predicted label

## ML Settings
Main ML config in `main.py`:
- `hiddenSize = 32`
    How wide the hidden layers are. Bigger can learn more, but also adds
    weight and can overfit faster. Small model, enough room to learn, still light
- `epochCount = 200`
    How many times the model sees the full training set.
    Gives the small dense model time to settle
- `learningRate = 0.001`
    How big each optimiser update step is during training.
    Common safe starting point for Adam

## Dependencies
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

## FFT Testing Notes
When testing FFT or DFFT plots, the two main levers are:
- sample rate
- sample length

Sample rate controls the frequency axis scaling.
If the sample rate is wrong or unknown, the plotted peak positions in Hz
will also be wrong.

Sample length controls frequency resolution.
Resolution is:
- `frequencyResolutionHz = sampleRate / sampleCount`

Longer captures give tighter peak separation and more reliable
fundamental estimation.
Short captures make peaks broader and less stable.

For the current web CSV test data, FFT output should only be treated as a
pipeline and plotting test.
It should not be treated as physically correct motor frequency truth,
because the real sample rate is unknown.

Use the real CSV FFT harness here for quick testing:
```bash
python3 Tests/fft_axis_harness.py
```

That harness is useful for checking:
- CSV read path
- time-domain plotting
- frequency-domain plotting
- how peak placement changes when `sampleRate` and `bufferSeconds` are
  changed

For real-world motor interpretation, the minimum useful conditions are:
- known sample rate
- longer stable capture duration
- one steady motor speed during the capture


## Current Limits
- current sample CSVs are test data from the web and there is no known sample rate.
- model quality will depend more on capture discipline than model depth
- live charting and terminal redraw are still unfinished
