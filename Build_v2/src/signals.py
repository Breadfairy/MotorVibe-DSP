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
    "maxHz": 60.0,
    "bpfoLowOrder": 3.0,
    "bpfoHighOrder": 5.0,
    "bpfiLowOrder": 5.0,
    "bpfiHighOrder": 7.0,
    "tolFraction": 0.10,
}

################################################################################
# helpers                                                                      #
################################################################################


def buildSpectrum(signal, sampleRate):
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)
    y = fft.rfft(x)
    n = x.size
    hz = fft.rfftfreq(n, d=1.0 / sampleRate)
    mag = (2.0 / n) * np.abs(y)
    return hz, mag


def fundamentalPeak(freqAxis, magnitude, minHz, maxHz):
    mask = (freqAxis >= minHz) & (freqAxis <= maxHz)
    hz = freqAxis[mask]
    mag = magnitude[mask]
    peakIndex = int(np.argmax(mag))
    return float(hz[peakIndex]), float(mag[peakIndex])


def orderBand(fundHz, lowOrder, highOrder, tolFraction):
    lowHz = (fundHz * lowOrder) * (1.0 - tolFraction)
    highHz = (fundHz * highOrder) * (1.0 + tolFraction)
    return lowHz, highHz

################################################################################
# main functions                                                               #
################################################################################


def timeSignals(windowFrame, sampleRate):
    if "t_s" in windowFrame.columns:
        t = windowFrame["t_s"].to_numpy(dtype=np.float64)
    else:
        t = np.arange(windowFrame.shape[0], dtype=np.float64) / sampleRate
    out = {"t_s": t}
    for col in axisCols:
        out[col] = windowFrame[col].to_numpy(dtype=np.float64)
    return out


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
    return out


def modelInput(timeData, freqData):
    parts = []
    for col in axisCols:
        parts.append(timeData[col].astype(np.float32))
    for col in axisCols:
        parts.append(freqData[f"{col}Spectrum"].astype(np.float32))
    return np.concatenate(parts).astype(np.float32)
