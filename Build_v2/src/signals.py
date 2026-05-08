################################################################################
# Imports                                                                      #
################################################################################
import numpy as np
from scipy import fft

################################################################################
# variables/constants                                                          #
################################################################################
# Raw sensor columns from the two MPU devices.
# These names match the CSV headers written by capture.py and live.py.
axisCols = [
    "ax1",
    "ay1",
    "az1",
    "gx1",
    "gy1",
    "gz1",
    "ax2",
    "ay2",
    "az2",
    "gx2",
    "gy2",
    "gz2",
]

# Human-readable names used by visualise.py when it saves the signal charts.
axisLabels = [
    ("ax1", "MPU1 Acc X"),
    ("ay1", "MPU1 Acc Y"),
    ("az1", "MPU1 Acc Z"),
    ("gx1", "MPU1 Gyr X"),
    ("gy1", "MPU1 Gyr Y"),
    ("gz1", "MPU1 Gyr Z"),
    ("ax2", "MPU2 Acc X"),
    ("ay2", "MPU2 Acc Y"),
    ("az2", "MPU2 Acc Z"),
    ("gx2", "MPU2 Gyr X"),
    ("gy2", "MPU2 Gyr Y"),
    ("gz2", "MPU2 Gyr Z"),
]

# FFT settings used by training, inference, live display, and charts.
# The model keeps one low-order rumble band plus BPFO/BPFI-style order bands.
fftConfig = {
    "minHz": 1.0,
    "maxHz": 70.0,
    "rumbleLowHz": 20.0,
    "lowOrderHighOrder": 3.0,
    "bpfoLowOrder": 3.0,
    "bpfoHighOrder": 5.0,
    "bpfiLowOrder": 5.0,
    "bpfiHighOrder": 7.0,
    "tolFraction": 0.10,
}

# Each group combines X/Y/Z into one magnitude signal.
# This keeps the model feature count smaller than treating every axis separately.
axisGroups = [
    ("acc1", "ax1", "ay1", "az1"),
    ("gyr1", "gx1", "gy1", "gz1"),
    ("acc2", "ax2", "ay2", "az2"),
    ("gyr2", "gx2", "gy2", "gz2"),
]

################################################################################
# helpers                                                                      #
################################################################################


# Builds one FFT magnitude spectrum after removing the DC offset.
# Removing the average value stops gravity or gyro bias from becoming the
# strongest "frequency" in the FFT.
def buildSpectrum(signal, sampleRate):
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)
    y = fft.rfft(x)
    n = x.size
    hz = fft.rfftfreq(n, d=1.0 / sampleRate)
    mag = (2.0 / n) * np.abs(y)
    return hz, mag


# Finds the largest FFT peak inside the expected running range.
# This is used as the basic motor/pump running frequency for the order bands.
def fundamentalPeak(freqAxis, magnitude, minHz, maxHz):
    mask = (freqAxis >= minHz) & (freqAxis <= maxHz)
    hz = freqAxis[mask]
    mag = magnitude[mask]
    peakIndex = int(np.argmax(mag))
    return float(hz[peakIndex]), float(mag[peakIndex])


# Converts a fundamental frequency into one order-based band.
# Example: if the fundamental is 50 Hz, then 3x to 5x is roughly 150 to 250 Hz,
# with a small tolerance added at both ends.
def orderBand(fundHz, lowOrder, highOrder, tolFraction):
    lowHz = (fundHz * lowOrder) * (1.0 - tolFraction)
    highHz = (fundHz * highOrder) * (1.0 + tolFraction)
    return lowHz, highHz


# Calculates RMS magnitude inside one FFT band.
# This gives one number for how much vibration energy sits inside that band.
def bandRms(freqAxis, magnitude, lowHz, highHz):
    mask = (freqAxis >= lowHz) & (freqAxis <= highHz)
    mag = magnitude[mask]
    if mag.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(mag * mag)))


# Returns the three acceleration columns for one MPU number.
# Used by live plotting so the sensor number can be changed at the top of live.py.
def accelCols(sensorNumber):
    return [
        f"ax{sensorNumber}",
        f"ay{sensorNumber}",
        f"az{sensorNumber}",
    ]


# Combines X, Y, and Z time signals into one magnitude signal.
# Magnitude is used because the fault may show on any axis depending on mounting.
def combinedTimeMagnitude(timeData, colX, colY, colZ):
    x = timeData[colX]
    y = timeData[colY]
    z = timeData[colZ]
    return np.sqrt((x * x) + (y * y) + (z * z))


# Calculates acceleration magnitude for one MPU in a DataFrame.
# This is the live-plot version that works directly from a pandas DataFrame.
def accelMagnitude(dataFrame, sensorNumber):
    cols = accelCols(sensorNumber)
    x = dataFrame[cols[0]].to_numpy(dtype=np.float64)
    y = dataFrame[cols[1]].to_numpy(dtype=np.float64)
    z = dataFrame[cols[2]].to_numpy(dtype=np.float64)
    return np.sqrt((x * x) + (y * y) + (z * z))


# Combines X, Y, and Z FFT magnitudes into one magnitude spectrum.
# The combined spectrum is easier to chart and gives the model one band value
# per sensor group instead of three separate band values per axis.
def combinedSpectrum(freqData, colX, colY, colZ):
    x = freqData[f"{colX}Spectrum"]
    y = freqData[f"{colY}Spectrum"]
    z = freqData[f"{colZ}Spectrum"]
    return np.sqrt((x * x) + (y * y) + (z * z))


# Builds the summed acceleration FFT for one MPU in a DataFrame.
# This is only for live display. The model features still come from the same
# signal functions used during training.
def accelFftSum(dataFrame, sensorNumber, sampleRate):
    cols = accelCols(sensorNumber)
    sumParts = []
    hz = None

    # Build one FFT per acceleration axis, then combine the magnitudes.
    for col in cols:
        hz, mag = buildSpectrum(dataFrame[col], sampleRate)
        sumParts.append(mag)
    stacked = np.vstack(sumParts)
    total = np.sqrt(np.sum(stacked * stacked, axis=0))
    return hz, total

################################################################################
# main functions                                                               #
################################################################################


# Copies one window of CSV data into a plain signal dictionary.
# The dictionary keeps time-domain data in one place before the FFT functions
# add frequency-domain data.
def timeSignals(windowFrame, sampleRate):
    if "t_s" in windowFrame.columns:
        t = windowFrame["t_s"].to_numpy(dtype=np.float64)
    else:
        t = np.arange(windowFrame.shape[0], dtype=np.float64) / sampleRate
    out = {"t_s": t}

    # Copy each raw axis into numpy arrays so the math code is consistent for
    # CSV files and live data.
    for col in axisCols:
        out[col] = windowFrame[col].to_numpy(dtype=np.float64)

    # Add combined magnitude signals for acc1, gyr1, acc2, and gyr2.
    for groupName, colX, colY, colZ in axisGroups:
        out[f"{groupName}Mag"] = combinedTimeMagnitude(
            out,
            colX,
            colY,
            colZ,
        )
    return out


# Builds FFT data and the engineered frequency features for one window.
# First it calculates per-axis FFTs for charting, then it calculates the compact
# grouped features that are actually fed into the model.
def fftSignals(timeData, sampleRate, fftConfigValue):
    out = {}

    # Per-axis spectra are stored so visualise.py can show each raw channel.
    for col in axisCols:
        hz, mag = buildSpectrum(timeData[col], sampleRate)
        fundHz, fundMag = fundamentalPeak(
            hz,
            mag,
            fftConfigValue["minHz"],
            fftConfigValue["maxHz"],
        )
        bpfo = orderBand(
            fundHz,
            fftConfigValue["bpfoLowOrder"],
            fftConfigValue["bpfoHighOrder"],
            fftConfigValue["tolFraction"],
        )
        bpfi = orderBand(
            fundHz,
            fftConfigValue["bpfiLowOrder"],
            fftConfigValue["bpfiHighOrder"],
            fftConfigValue["tolFraction"],
        )
        out["freqAxis"] = hz
        out[f"{col}Spectrum"] = mag
        out[f"{col}FundHz"] = fundHz
        out[f"{col}FundMag"] = fundMag
        out[f"{col}BpfoBand"] = bpfo
        out[f"{col}BpfiBand"] = bpfi

    # Grouped spectra are used for the ML feature set.
    # Each group gets fundamental frequency, fundamental magnitude, low-order
    # rumble RMS, BPFO RMS, and BPFI RMS.
    for groupName, colX, colY, colZ in axisGroups:
        hz = out["freqAxis"]
        mag = combinedSpectrum(out, colX, colY, colZ)
        fundHz, fundMag = fundamentalPeak(
            hz,
            mag,
            fftConfigValue["minHz"],
            fftConfigValue["maxHz"],
        )
        lowOrder = orderBand(
            fundHz,
            1.0,
            fftConfigValue["lowOrderHighOrder"],
            fftConfigValue["tolFraction"],
        )
        lowOrder = (fftConfigValue["rumbleLowHz"], lowOrder[1])
        bpfo = orderBand(
            fundHz,
            fftConfigValue["bpfoLowOrder"],
            fftConfigValue["bpfoHighOrder"],
            fftConfigValue["tolFraction"],
        )
        bpfi = orderBand(
            fundHz,
            fftConfigValue["bpfiLowOrder"],
            fftConfigValue["bpfiHighOrder"],
            fftConfigValue["tolFraction"],
        )
        out[f"{groupName}FundHz"] = fundHz
        out[f"{groupName}FundMag"] = fundMag
        out[f"{groupName}LowOrderRms"] = bandRms(
            hz,
            mag,
            lowOrder[0],
            lowOrder[1],
        )
        out[f"{groupName}BpfoRms"] = bandRms(hz, mag, bpfo[0], bpfo[1])
        out[f"{groupName}BpfiRms"] = bandRms(hz, mag, bpfi[0], bpfi[1])
    return out


# Builds compact magnitude/frequency features for the model.
# For each grouped signal the model gets time-domain size values plus
# frequency-domain order-band values. No raw FFT bins are included.
def modelInput(timeData, freqData):
    parts = []
    for groupName, colX, colY, colZ in axisGroups:
        mag = timeData[f"{groupName}Mag"]

        # These eight numbers are the feature order for one group.
        # featureNames() below must stay in the same order.
        stats = [
            float(np.sqrt(np.mean(mag * mag))),
            float(np.min(mag)),
            float(np.max(mag)),
            freqData[f"{groupName}FundHz"],
            freqData[f"{groupName}FundMag"],
            freqData[f"{groupName}LowOrderRms"],
            freqData[f"{groupName}BpfoRms"],
            freqData[f"{groupName}BpfiRms"],
        ]
        parts.append(np.array(stats, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


# Builds one raw-axis model row from a one-second window.
# This keeps the actual 12 sensor waveforms instead of reducing them to the
# compact RMS/frequency features.
def rawModelInput(windowFrame):
    parts = []
    for col in axisCols:
        x = windowFrame[col].to_numpy(dtype=np.float32)
        parts.append(x)
    return np.concatenate(parts).astype(np.float32)


# Lists feature names in the same order as modelInput().
# The saved model bundle includes these names so the feature set is documented
# beside the trained classifier.
def featureNames():
    names = []
    for groupName, colX, colY, colZ in axisGroups:
        names.append(f"{groupName}_rms")
        names.append(f"{groupName}_min")
        names.append(f"{groupName}_max")
        names.append(f"{groupName}_fundHz")
        names.append(f"{groupName}_fundMag")
        names.append(f"{groupName}_lowOrderRms")
        names.append(f"{groupName}_bpfoRms")
        names.append(f"{groupName}_bpfiRms")
    return names


# Lists raw-axis feature names in the same order as rawModelInput().
def rawFeatureNames(rowCount):
    names = []
    for col in axisCols:
        for row in range(rowCount):
            names.append(f"{col}_{row:04d}")
    return names
