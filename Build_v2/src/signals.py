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
def buildSpectrum(signal, sampleRate):
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)
    y = fft.rfft(x)
    n = x.size
    hz = fft.rfftfreq(n, d=1.0 / sampleRate)
    mag = (2.0 / n) * np.abs(y)
    return hz, mag


# Finds the largest FFT peak inside the expected running range.
def fundamentalPeak(freqAxis, magnitude, minHz, maxHz):
    mask = (freqAxis >= minHz) & (freqAxis <= maxHz)
    hz = freqAxis[mask]
    mag = magnitude[mask]
    peakIndex = int(np.argmax(mag))
    return float(hz[peakIndex]), float(mag[peakIndex])


# Converts a fundamental frequency into one order band.
def _orderBand(fundHz, lowOrder, highOrder, tolFraction):
    lowHz = (fundHz * lowOrder) * (1.0 - tolFraction)
    highHz = (fundHz * highOrder) * (1.0 + tolFraction)
    return lowHz, highHz


# Low-order rumble band used by the compact ML feature set.
def lowBand(fundHz, fftConfigValue):
    highBand = _orderBand(
        fundHz,
        1.0,
        fftConfigValue["lowOrderHighOrder"],
        fftConfigValue["tolFraction"],
    )
    return fftConfigValue["rumbleLowHz"], highBand[1]


# BPFO and BPFI style bearing/order bands.
def bearingBands(fundHz, fftConfigValue):
    bpfo = _orderBand(
        fundHz,
        fftConfigValue["bpfoLowOrder"],
        fftConfigValue["bpfoHighOrder"],
        fftConfigValue["tolFraction"],
    )
    bpfi = _orderBand(
        fundHz,
        fftConfigValue["bpfiLowOrder"],
        fftConfigValue["bpfiHighOrder"],
        fftConfigValue["tolFraction"],
    )
    return bpfo, bpfi


# Calculates RMS magnitude inside one FFT band.
def bandRms(freqAxis, magnitude, lowHz, highHz):
    mask = (freqAxis >= lowHz) & (freqAxis <= highHz)
    mag = magnitude[mask]
    if mag.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(mag * mag)))


# Returns the three acceleration columns for one MPU number.
def accelCols(sensorNumber):
    return [
        f"ax{sensorNumber}",
        f"ay{sensorNumber}",
        f"az{sensorNumber}",
    ]

################################################################################
# main functions                                                               #
################################################################################


# Builds raw time signals for plotting.
def plotTime(windowFrame, sampleRate, columns=None):
    if "t_s" in windowFrame.columns:
        t = windowFrame["t_s"].to_numpy(dtype=np.float64)
    else:
        t = np.arange(windowFrame.shape[0], dtype=np.float64) / sampleRate
    out = {"t_s": t}

    if columns is None:
        columns = axisCols
    for col in columns:
        out[col] = windowFrame[col].to_numpy(dtype=np.float64)
    return out


# Builds raw axis signals plus grouped magnitudes for the ML model.
def mlTimeSignals(windowFrame, sampleRate):
    out = plotTime(windowFrame, sampleRate)

    for groupName, colX, colY, colZ in axisGroups:
        x = out[colX]
        y = out[colY]
        z = out[colZ]
        out[f"{groupName}Mag"] = np.sqrt((x * x) + (y * y) + (z * z))
    return out


# Builds per-axis FFT data for plotting.
def plotFreq(timeData, sampleRate, fftConfigValue):
    out = {}
    plotCols = [col for col in timeData.keys() if col != "t_s"]
    for col in plotCols:
        hz, mag = buildSpectrum(timeData[col], sampleRate)
        fundHz, fundMag = fundamentalPeak(
            hz,
            mag,
            fftConfigValue["minHz"],
            fftConfigValue["maxHz"],
        )
        bpfo, bpfi = bearingBands(fundHz, fftConfigValue)
        out["freqAxis"] = hz
        out[f"{col}Spectrum"] = mag
        out[f"{col}FundHz"] = fundHz
        out[f"{col}FundMag"] = fundMag
        out[f"{col}BpfoBand"] = bpfo
        out[f"{col}BpfiBand"] = bpfi
    return out


# Builds grouped frequency features for the compact ML model.
def mlFreqSignals(timeData, sampleRate, fftConfigValue):
    out = {}
    axisSpectra = {}
    hz = None

    for col in axisCols:
        hz, axisSpectra[col] = buildSpectrum(timeData[col], sampleRate)

    for groupName, colX, colY, colZ in axisGroups:
        x = axisSpectra[colX]
        y = axisSpectra[colY]
        z = axisSpectra[colZ]
        mag = np.sqrt((x * x) + (y * y) + (z * z))
        fundHz, fundMag = fundamentalPeak(
            hz,
            mag,
            fftConfigValue["minHz"],
            fftConfigValue["maxHz"],
        )
        lowOrder = lowBand(fundHz, fftConfigValue)
        bpfo, bpfi = bearingBands(fundHz, fftConfigValue)
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
