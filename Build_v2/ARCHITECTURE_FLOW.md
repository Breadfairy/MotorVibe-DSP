# Build V2 Architecture Flow

This document describes the whole induction motor pump vibration sensing system from top to bottom.

The basic procedure is the same as the fan-test build:

1. The ESP32 firmware samples the sensors.
2. Python captures labelled CSV recordings.
3. Python trains a model from those CSV recordings.
4. Python runs live inference using the same serial stream and the same DSP feature code.

The main difference from the fan-test build is the motor feature set. The induction motor build keeps the same architecture, but the DSP/ML input uses a compact set of simple magnitude and bearing/order frequency features.

## Top-Level Block Diagram

```text
Induction motor pump
        |
        | mechanical vibration and temperature
        v
Two MPU6050 IMUs + one DS18B20 temperature sensor
        |
        | I2C for the MPUs, OneWire for temperature
        v
ESP32 firmware
        |
        | 1 kHz sample loop
        | binary packed records
        | double-buffered serial output
        v
USB serial link at 1,000,000 baud
        |
        +------------------------------+
        |                              |
        v                              v
capture.py                        live.py
records labelled CSVs             reads live stream
        |                              |
        v                              v
data/training/main/*.csv          same 1 second DSP feature row
        |                              |
        v                              v
train.py                          trained sklearn model
builds windows                    predicts live state
extracts features                         |
trains classifier                         v
        |                          terminal state + live plots
        v
outputs/models/motorPumpClassifier.joblib
        |
        +------------------------------+
        |                              |
        v                              v
infer.py                       ml_charts.py
offline CSV inference          validation confusion chart
```

## Main Files

```text
Build_v2/MCU/firmware_1.1.ino
    ESP32 firmware. Samples the sensors and streams binary packets.

Build_v2/src/capture.py
    Reads the binary serial stream and saves labelled CSV recordings.

Build_v2/src/data.py
    Loads CSV data, cleans basic numeric values, and splits recordings into windows.

Build_v2/src/signals.py
    Builds the time-domain, FFT, and ML feature vectors.

Build_v2/src/train.py
    Finds labelled CSV files, trains the classifier, and saves the model bundle.

Build_v2/src/infer.py
    Runs offline inference on one CSV file or a folder of CSV files.

Build_v2/src/live.py
    Reads the live serial stream, calculates the same features, and displays live predictions.

Build_v2/src/ml_charts.py
    Builds a validation confusion matrix chart from the held-out CSV files.

Build_v2/src/visualise.py
    Saves simple time and FFT charts for one captured CSV.
```

## Hardware Setup

The hardware side is built around one ESP32 board.

The ESP32 talks to:

```text
MPU6050 number 1
    I2C address: 0x68

MPU6050 number 2
    I2C address: 0x69

DS18B20 temperature sensor
    OneWire data pin: GPIO 4
```

The I2C bus uses:

```text
SDA: GPIO 21
SCL: GPIO 22
I2C clock: 1,000,000 Hz
```

Both MPU6050 sensors share the same SDA and SCL wires. That is normal for I2C. The reason the ESP32 can talk to both separately is that they have different addresses. The first MPU is addressed as `0x68`, and the second MPU is addressed as `0x69`.

The firmware does not use a separate bus for each sensor. It uses the same I2C bus and selects the sensor by sending the target address with each I2C transaction.

## Firmware Startup Flow

At startup, the firmware does this:

```text
Start serial at 1,000,000 baud
        |
Start DS18B20 temperature sensor
        |
Start background temperature task
        |
Start I2C on GPIO 21 / GPIO 22
        |
Set I2C speed to 1 MHz
        |
Configure MPU at address 0x68
        |
Configure MPU at address 0x69
        |
Print ready message and sample struct size
        |
Wait until Python sends START
        |
Begin 1 kHz sampling loop
```

The firmware waits for Python before it starts streaming. This matters because the Python code needs a clean start point. It watches the serial text until it sees:

```text
Sample struct size (bytes):
```

Then Python sends:

```text
START
```

After that, the firmware stops acting like a text logger and starts sending raw binary sample records.

## MPU6050 Sensor Handling

Each MPU6050 is configured the same way.

The setup writes to the MPU registers to:

```text
Reset the sensor
Wake it back up
Enable all axes
Set digital low pass filtering
Set sample divider
Set gyro full-scale range
Set accelerometer full-scale range
Enable data ready interrupt
```

The actual sample read is simple:

```text
Start I2C transmission to one MPU address
        |
Set register pointer to ACCEL_XOUT_H
        |
Request 14 bytes
        |
Read accelerometer X, Y, Z
        |
Skip the MPU internal temperature bytes
        |
Read gyroscope X, Y, Z
```

That gives six useful values from each MPU:

```text
ax, ay, az
gx, gy, gz
```

Because there are two MPUs, each complete sample contains twelve sensor axes:

```text
ax1 ay1 az1 gx1 gy1 gz1 ax2 ay2 az2 gx2 gy2 gz2
```

The firmware only keeps a sample if both MPU reads succeed. If either sensor read fails, that sample is skipped instead of sending a partial record.

## Temperature Handling

The DS18B20 temperature read is slower than the vibration sampling. A DS18B20 conversion takes much longer than 1 ms, so it cannot be read directly inside the 1 kHz sample loop.

The firmware handles that by running a separate temperature task:

```text
Request DS18B20 conversion
        |
Wait about 750 ms
        |
Read temperature
        |
If temperature is in a sane range, store it in temperatureC
        |
Request the next conversion
        |
Repeat forever
```

The main vibration sampling loop just copies the latest `temperatureC` value into each sample record. This keeps temperature available without slowing down vibration capture.

## Firmware Sample Record

Each firmware sample is packed into this structure:

```text
t_us        uint32 timestamp from micros()
ax1         int16
ay1         int16
az1         int16
gx1         int16
gy1         int16
gz1         int16
ax2         int16
ay2         int16
az2         int16
gx2         int16
gy2         int16
gz2         int16
tempC       float
```

The Python binary format string for this is:

```text
<I12hf
```

That means:

```text
<       little-endian byte order
I       one unsigned 32-bit timestamp
12h     twelve signed 16-bit sensor axis values
f       one 32-bit float temperature value
```

The struct is 32 bytes:

```text
4 bytes timestamp
24 bytes sensor axes
4 bytes temperature
```

## Firmware Timing

The firmware aims for:

```text
1,000 samples per second
1 sample every 1,000 microseconds
```

The loop uses `micros()` and a `nextTick` timestamp. Every time a sample is due, `nextTick` is advanced by `SAMPLE_PERIOD_US`.

That means the firmware is trying to stay on a regular time grid instead of just sampling whenever the loop happens to come around.

## Firmware Buffering

The firmware uses double buffering.

There are two buffers:

```text
bufferA
bufferB
```

Each buffer holds:

```text
32 samples
```

Since each sample is 32 bytes, one full buffer is:

```text
32 samples * 32 bytes = 1024 bytes
```

The firmware keeps two buffer pointers:

```text
fillBuffer
    The buffer currently being filled by new sensor samples.

sendBuffer
    The buffer currently being sent over serial.
```

The flow is:

```text
Fill sample into fillBuffer
        |
When fillBuffer reaches 32 samples
        |
If old sendBuffer is still sending, service serial and wait
        |
Swap fillBuffer and sendBuffer
        |
Mark sendPending true
        |
Start filling the other buffer
```

The reason for this is simple: sensor sampling and serial sending are different jobs. Double buffering lets the firmware collect the next batch of samples while the previous batch is being written to USB serial.

## Serial Sending

The firmware does not assume the whole 1024 byte buffer can be written at once.

It checks:

```text
Serial.availableForWrite()
```

Then it writes only as many bytes as the serial driver says it can accept. If only part of the buffer is written, the firmware remembers the byte offset in:

```text
sendOffset
```

On the next pass, it continues from that offset.

This avoids blocking too hard on serial output and makes the stream more stable.

## Python Binary Stream Handling

Both `capture.py` and `live.py` decode the same stream.

They use:

```text
recordFormat = "<I12hf"
recordSize = struct.calcsize(recordFormat)
bufferSampleCount = 32
bufferSize = recordSize * bufferSampleCount
```

The Python code reads chunks of bytes from serial. Serial data does not always arrive exactly aligned to one record. Because of that, Python keeps leftover bytes:

```text
leftOverBytes in capture.py
remBytes in live.py
```

The logic is:

```text
Read new bytes from serial
        |
Append them to leftover bytes
        |
While there is at least one full 32 byte record
        |
Take first 32 bytes
        |
Unpack with struct.unpack("<I12hf", packetBytes)
        |
Convert packet into one row
        |
Keep any extra partial bytes for the next read
```

This is important because USB serial is a byte stream, not a packet protocol. The firmware sends fixed-size records, but Python still has to rebuild the record boundaries.

## Scaling Raw Sensor Values

The firmware sends raw MPU integer counts.

Python converts them to physical-ish units:

```text
Accelerometer scale: 16384.0
Gyroscope scale: 131.0
```

So:

```text
accelerometer_g = raw_accel / 16384.0
gyro_deg_per_sec = raw_gyro / 131.0
```

The CSV stores the scaled values, not the raw integer counts.

## Capture Procedure

`capture.py` is used for recording labelled training data.

Accepted labels are:

```text
good
bad_leak
```

Example:

```bash
PYTHONPATH=Build_v2/src python3 Build_v2/src/capture.py good good_001 180 /dev/cu.usbserial-0001
```

Argument order:

```text
label
file name
capture seconds
serial port
```

The output goes here:

```text
Build_v2/data/training/main/
```

The file naming matters because training discovers files by prefix:

```text
good*.csv
bad_leak*.csv
```

If a file is named `leak1.csv`, the current training code will ignore it. It must be named something like:

```text
bad_leak_001.csv
```

## CSV Layout

Every captured CSV has these columns:

```text
t_us
t_s
ax1
ay1
az1
gx1
gy1
gz1
ax2
ay2
az2
gx2
gy2
gz2
tempC
```

`t_us` is the timestamp in microseconds, normalized so the first sample starts at zero.

`t_s` is the same time in seconds.

## Data Cleaning

`data.py` does the basic cleaning before training or inference.

The flow is:

```text
Read CSV
        |
Keep only the expected signal columns
        |
Convert everything to numeric
        |
Replace infinity with NaN
        |
Fill missing or bad values with zero
        |
Clip accelerometer values to the sensor range
        |
Clip gyroscope values to the sensor range
        |
Clip temperature values to the sensor range
        |
Reset time so first sample is t = 0
```

This is intentionally basic. The code is not trying to rebuild missing data or do advanced statistical filtering. Missing values are set to zero, and outliers are clipped to the physical ranges expected from the sensors:

```text
accelerometer: -2 g to +2 g
gyroscope: -250 deg/s to +250 deg/s
temperature: -40 C to +125 C
```

That handles obvious impossible values without hiding a bad recording. If a capture has many missing values, flat sections, or stream faults, the better approach for this build is to re-record that test run.

## Window Splitting

Training and inference do not classify the whole recording in one go.

They split recordings into fixed windows:

```text
sample rate: 1000 Hz
window length: 1.0 second
window rows: 1000 samples
step length: 0.25 seconds
step rows: 250 samples
```

That means each training window is one second long, and the windows overlap by 0.75 seconds.

Example:

```text
window 1: samples 0 to 999
window 2: samples 250 to 1249
window 3: samples 500 to 1499
window 4: samples 750 to 1749
```

The reason for this is that one second is long enough to contain useful vibration frequency content, while the 0.25 second step gives more training examples and smoother live updates.

## DSP Feature Flow

The DSP feature code is in `signals.py`.

For every one-second window, the order is:

```text
CSV window
        |
timeSignals()
        |
fftSignals()
        |
modelInput()
        |
one numeric feature row
```

## Time Signal Setup

`timeSignals()` copies the raw axis columns into a plain dictionary.

It keeps these raw axes:

```text
ax1 ay1 az1
gx1 gy1 gz1
ax2 ay2 az2
gx2 gy2 gz2
```

Then it builds four magnitude time signals:

```text
acc1Mag = sqrt(ax1^2 + ay1^2 + az1^2)
gyr1Mag = sqrt(gx1^2 + gy1^2 + gz1^2)
acc2Mag = sqrt(ax2^2 + ay2^2 + az2^2)
gyr2Mag = sqrt(gx2^2 + gy2^2 + gz2^2)
```

These magnitude signals are useful because they do not depend as strongly on the exact physical orientation of each MPU.

## FFT Setup

`fftSignals()` builds an FFT magnitude spectrum for every raw axis.

For each signal:

```text
Convert to float
        |
Remove DC offset by subtracting the mean
        |
Run real FFT
        |
Build frequency axis
        |
Scale magnitude
```

Removing the DC offset matters because accelerometers include gravity and fixed mounting offsets. Without subtracting the mean, the zero-frequency part can dominate the spectrum.

## Fundamental Frequency

The code looks for the biggest FFT peak in this range:

```text
1 Hz to 120 Hz
```

That peak is treated as the current running fundamental for that signal or signal group.

The output is:

```text
fundHz
fundMag
```

`fundHz` is the frequency of the strongest peak.

`fundMag` is the magnitude of that peak.

## Order Bands

The code calculates three order-based frequency bands from the detected fundamental:

```text
Low-order rumble band: 20 Hz to 3x fundamental
BPFO-style band: 3x to 5x fundamental
BPFI-style band: 5x to 7x fundamental
Tolerance: 10 percent
```

The BPFO and BPFI names came from bearing-style vibration feature logic in the fan-test system. In this motor build they are still useful as broad order-band vibration features. They are not a full mechanical diagnosis by themselves. They are engineered frequency regions that the model can learn from.

The low-order band is the band below BPFO. It starts at 20 Hz instead of 1x fundamental so it can include sub-fundamental motor/pump rumble. It is included because the motor/pump can rumble or load up in the lower orders even when the main fundamental frequency does not move very much.

For each band, the code calculates RMS FFT magnitude inside the band.

## Axis Groups

The feature code works with four axis groups:

```text
acc1: ax1 ay1 az1
gyr1: gx1 gy1 gz1
acc2: ax2 ay2 az2
gyr2: gx2 gy2 gz2
```

For each group, the FFT magnitudes from X, Y, and Z are combined:

```text
combinedSpectrum = sqrt(xSpectrum^2 + ySpectrum^2 + zSpectrum^2)
```

Then the group-level fundamental, low-order RMS, BPFO RMS, and BPFI RMS values are calculated from that combined spectrum.

## Final ML Feature Vector

`modelInput()` builds the final feature row in a fixed order.

It does not feed the full one-second time trace into the model.

That was deliberately avoided because four one-second magnitude traces would add:

```text
4 groups * 1000 samples = 4000 raw time-domain features
```

Those raw time samples can give the classifier too many ways to fit the shape of a specific recording. Even with scaling, 4000 time samples can outweigh a small set of frequency features just because there are so many of them.

Instead, the model uses compact summary features for each of the four groups:

```text
rms
min
max
```

Then it adds the frequency features for each group:

```text
fundHz
fundMag
lowOrderRms
bpfoRms
bpfiRms
```

That gives:

```text
4 groups * 8 summary features = 32 summary features
```

So the full model input size is:

```text
32 features
```

This keeps the feature set small and focused on the bearing/order profile instead of adding broad raw FFT bins or extra fixed energy bands.

## Training Flow

`train.py` follows the fan-test procedure.

The training script has two classes:

```text
good
bad_leak
```

For each class:

```text
Find CSV files matching the class prefix
        |
Read only files from Build_v2/data/training/main
        |
Clean each CSV
        |
Split each CSV by time
        |
First 35 percent goes to training
        |
Next 25 percent goes to validation
        |
Remaining 40 percent goes to holdout
        |
Split each time block into 1 second windows
        |
Convert every window into one feature row
        |
Add the class index as the label
```

The split is time-based inside each trusted main CSV. The `unused` folder is not used for training, validation, or holdout because those files are not trusted labels.

The model is:

```text
StandardScaler
        |
LogisticRegression
```

`StandardScaler` normalizes each feature column so the model is not dominated by features with large numeric ranges.

`LogisticRegression` is the classifier. It is simple, fast, and easy to run live. The code uses:

```text
class_weight="balanced"
```

That helps if one class has more training windows than another.

After training, the script prints:

```text
training shape
training accuracy
training confusion matrix
training classification report
validation shape
validation accuracy
validation confusion matrix
validation classification report
```

Then it saves the model bundle here:

```text
Build_v2/outputs/models/motorPumpClassifier.joblib
```

The saved bundle includes:

```text
trained sklearn model
label names
sample rate
window length
step length
feature names
feature mode name
training accuracy
validation accuracy
```

## Offline Inference Flow

`infer.py` uses the saved model on recorded CSVs.

The flow is:

```text
Load model bundle
        |
Read one CSV or every CSV in a folder
        |
Clean the data
        |
Split into 1 second windows
        |
Build the same 32 feature row for each window
        |
Run model.predict_proba()
        |
Average probabilities across all windows in the file
        |
Print strongest state and confidence
```

The average probability is useful because one bad window should not decide the whole recording. The file-level answer is the average of all window-level probabilities.

## Live Inference Flow

`live.py` uses the same serial stream as `capture.py`, but instead of saving CSV rows, it keeps a rolling buffer in memory and runs inference.

Startup:

```text
Load motorPumpClassifier.joblib
        |
Open serial port
        |
Reset serial link
        |
Wait for firmware ready message
        |
Send START
        |
Begin reading binary stream
```

Live row handling:

```text
Read serial bytes
        |
Append to remaining byte buffer
        |
Unpack every full 32 byte sample
        |
Convert raw values to scaled row values
        |
Append row to live row list
        |
Keep only the rows needed for plots and inference
```

The live code keeps enough rows for:

```text
1 second inference window
2 second visual plot window
```

The plot window is longer, so it normally keeps about:

```text
2000 rows
```

Live inference runs every:

```text
0.5 seconds
```

For each inference update:

```text
Take the newest 1 second of rows
        |
Build one feature row using the same signals.py code
        |
Run model.predict_proba()
        |
Add raw probabilities to history
        |
Smooth probabilities over the last 10 seconds
        |
Print current state and confidence
        |
Update probability plot
```

The terminal state is smoothed over 10 seconds. This stops the displayed state from flickering too much if one inference window jumps around.

Raw-axis live mode can be run with:

```bash
PYTHONPATH=Build_v2/src python3 Build_v2/src/live.py raw
```

Normal compact live mode is:

```bash
PYTHONPATH=Build_v2/src python3 Build_v2/src/live.py
```

The raw probability plot keeps about:

```text
60 seconds of probability history
```

## Live Plots

`live.py` shows three plots:

```text
MPU1 acceleration magnitude
MPU1 summed acceleration FFT
Raw model probability history
```

The FFT plot also marks:

```text
detected fundamental frequency
BPFO-style order band
BPFI-style order band
```

These plots are not separate from the model path. They are there so the operator can see what the signal and model are doing while the pump is running.

## Validation Chart Flow

`ml_charts.py` builds validation and holdout confusion matrix charts.

It uses the same row split rule as `train.py`:

```text
For each trusted main CSV:
        |
First 35 percent is training
        |
Next 25 percent is validation
        |
Remaining 40 percent is holdout
```

For each validation or holdout block:

```text
Clean data
        |
Split into 1 second windows
        |
Build feature rows
        |
Predict each window
        |
Add result to confusion matrix
```

It saves:

```text
Build_v2/outputs/ML-charts/confusion_matrix.png
Build_v2/outputs/ML-charts/confusion_matrix_counts.csv
```

## Normal Operating Procedure

### 1. Flash Firmware

From the MCU folder:

```bash
cd Build_v2/MCU
make flash FLASH_PORT=/dev/cu.usbserial-0001
```

The firmware expects:

```text
NodeMCU-32S ESP32 board
1,000,000 baud serial
two MPU6050 sensors
one DS18B20 sensor
```

### 2. Capture Labelled Data

Examples:

```bash
PYTHONPATH=Build_v2/src python3 Build_v2/src/capture.py good good_001 180 /dev/cu.usbserial-0001
PYTHONPATH=Build_v2/src python3 Build_v2/src/capture.py bad_leak bad_leak_001 180 /dev/cu.usbserial-0001
```

Capture more than one file per class if possible. The code can now still validate single-file classes because it splits each trusted main CSV by time.

### 3. Train

```bash
PYTHONPATH=Build_v2/src python3 Build_v2/src/train.py
```

This writes:

```text
Build_v2/outputs/models/motorPumpClassifier.joblib
```

Raw-axis test mode can also be trained:

```bash
PYTHONPATH=Build_v2/src python3 Build_v2/src/train.py raw
```

This writes:

```text
Build_v2/outputs/models/motorPumpRawAxisClassifier.joblib
```

The compact model uses the 32 RMS/frequency features. The raw model uses the full one-second 12-axis sensor window, so it has 12000 input values.

### 4. Generate Validation Chart

```bash
PYTHONPATH=Build_v2/src MPLBACKEND=Agg python3 Build_v2/src/ml_charts.py
```

This writes:

```text
Build_v2/outputs/ML-charts/confusion_matrix.png
```

### 5. Run Live Inference

Set the serial port at the top of `Build_v2/src/live.py`, then run:

```bash
PYTHONPATH=Build_v2/src python3 Build_v2/src/live.py
```

The live script loads the latest saved model. If you capture new data, train again before running live.

## Important Consistency Rules

The system works cleanly only if these stay aligned:

```text
Firmware Sample struct
Python recordFormat "<I12hf"
CSV column order
signals.py feature order
trained model feature order
live.py feature extraction
```

If any of those change, the model must be retrained, and every stream decoder must be checked.

The capture, training, offline inference, charting, and live inference all rely on the same `data.py` and `signals.py` path. That is deliberate. It means the features used in live inference are the same features used during training.

## What To Watch During Testing

Before trusting a live prediction, check these basics:

```text
Serial stream starts only after START
Rows logged are near 1000 rows per second
CSV files use the correct class prefix
Every class has at least one recording
Preferably every class has at least two recordings
Training prints a sensible validation matrix
Live plot shows non-flat vibration signals
Live probabilities are not stuck at one class for every condition
```

The most important practical rule is file naming. Training does not read labels from inside the CSV. It reads the label from the file prefix.

Correct:

```text
good_001.csv
bad_leak_001.csv
```

Incorrect:

```text
outflow1.csv
leak1.csv
```

Those older names do not match the current motor training labels.
