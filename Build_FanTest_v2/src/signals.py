################################################################################
# Imports                                                                      #
################################################################################
import numpy as np
from scipy import fft

################################################################################
# variables/constants                                                          #
################################################################################
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
fftConfig = {
    "minHz": 1.0,
    "maxHz": 120.0,
    "bpfoLowOrder": 3.0,
    "bpfoHighOrder": 5.0,
    "bpfiLowOrder": 5.0,
    "bpfiHighOrder": 7.0,
    "tolFraction": 0.10,
}
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


# Finds the largest FFT peak inside the expected fan running range.
def fundamentalPeak(freqAxis, magnitude, minHz, maxHz):
    mask = (freqAxis >= minHz) & (freqAxis <= maxHz)
    hz = freqAxis[mask]
    mag = magnitude[mask]
    peakIndex = int(np.argmax(mag))
    return float(hz[peakIndex]), float(mag[peakIndex])


# Converts a fundamental frequency into one order-based frequency band.
def orderBand(fundHz, lowOrder, highOrder, tolFraction):
    lowHz = (fundHz * lowOrder) * (1.0 - tolFraction)
    highHz = (fundHz * highOrder) * (1.0 + tolFraction)
    return lowHz, highHz


# Calculates the RMS level inside one FFT frequency band.
def bandRms(freqAxis, magnitude, lowHz, highHz):
    mask = (freqAxis >= lowHz) & (freqAxis <= highHz)
    mag = magnitude[mask]
    if mag.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(mag * mag)))


# Combines X, Y, and Z FFT magnitudes into one vector magnitude spectrum.
def combinedSpectrum(freqData, colX, colY, colZ):
    x = freqData[f"{colX}Spectrum"]
    y = freqData[f"{colY}Spectrum"]
    z = freqData[f"{colZ}Spectrum"]
    return np.sqrt((x * x) + (y * y) + (z * z))


# Combines X, Y, and Z time signals into one magnitude signal.
def combinedTimeMagnitude(timeData, colX, colY, colZ):
    x = timeData[colX]
    y = timeData[colY]
    z = timeData[colZ]
    return np.sqrt((x * x) + (y * y) + (z * z))

################################################################################
# main functions                                                               #
################################################################################


# Copies one window of CSV data into a plain signal dictionary.
def timeSignals(windowFrame, sampleRate):
    if "t_s" in windowFrame.columns:
        t = windowFrame["t_s"].to_numpy(dtype=np.float64)
    else:
        t = np.arange(windowFrame.shape[0], dtype=np.float64) / sampleRate
    out = {"t_s": t}
    for col in axisCols:
        out[col] = windowFrame[col].to_numpy(dtype=np.float64)
    return out


# Builds FFT data and the bearing-order features for one window.
def fftSignals(timeData, sampleRate, fftConfigValue):
    out = {}
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

    for groupName, colX, colY, colZ in axisGroups:
        hz = out["freqAxis"]
        mag = combinedSpectrum(out, colX, colY, colZ)
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
        out[f"{groupName}FundHz"] = fundHz
        out[f"{groupName}FundMag"] = fundMag
        out[f"{groupName}BpfoRms"] = bandRms(hz, mag, bpfo[0], bpfo[1])
        out[f"{groupName}BpfiRms"] = bandRms(hz, mag, bpfi[0], bpfi[1])
    return out


# Builds magnitude time signals plus the engineered frequency features.
def modelInput(timeData, freqData):
    parts = []
    for groupName, colX, colY, colZ in axisGroups:
        mag = combinedTimeMagnitude(timeData, colX, colY, colZ)
        parts.append(mag.astype(np.float32))
    for groupName, colX, colY, colZ in axisGroups:
        parts.append(
            np.array(
                [
                    freqData[f"{groupName}FundHz"],
                    freqData[f"{groupName}FundMag"],
                    freqData[f"{groupName}BpfoRms"],
                    freqData[f"{groupName}BpfiRms"],
                ],
                dtype=np.float32,
            )
        )
    return np.concatenate(parts).astype(np.float32)


# Lists the feature names in the same order as modelInput().
def featureNames(rowCount):
    names = []
    for groupName, colX, colY, colZ in axisGroups:
        for index in range(rowCount):
            names.append(f"{groupName}_mag_t{index}")
    for groupName, colX, colY, colZ in axisGroups:
        names.append(f"{groupName}_fundHz")
        names.append(f"{groupName}_fundMag")
        names.append(f"{groupName}_bpfoRms")
        names.append(f"{groupName}_bpfiRms")
    return names
