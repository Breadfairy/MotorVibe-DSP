################################################################################
# Imports                                                                      #
################################################################################
import numpy as np

import buffer

################################################################################
# variables/constants                                                          #
################################################################################
fftConfig = {
    "minHz": 1.0,
    "maxHz": 1200.0,
    "bpfoLowOrder": 3.0,
    "bpfoHighOrder": 5.0,
    "bpfiLowOrder": 5.0,
    "bpfiHighOrder": 7.0,
    "tolFraction": 0.10,
}
axisLabel = {
    "ax": "Acc X",
    "ay": "Acc Y",
    "az": "Acc Z",
    "gx": "Gyro X",
    "gy": "Gyro Y",
    "gz": "Gyro Z",
}

################################################################################
# helpers                                                                      #
################################################################################


# Reads one CSV file for the current signal pipeline.
def readCSV(csvPath):
    return buffer.readCsv(csvPath)


# Reads one live framed batch and converts it into rows.
def liveSense(link, mode, firstFrame):
    head, payload = buffer.readFrame(link, mode, firstFrame)
    rows = buffer.decodeRows(head, payload)
    return head, rows


# Builds one generic vector magnitude from the selected raw axes.
def vectorMag(arrs):
    stack = np.vstack(arrs)
    return np.sqrt(np.sum(stack**2, axis=0))


# Builds one DC-removed FFT magnitude spectrum.
def buildSpectrum(x, rate):
    y = np.asarray(x, dtype=np.float64)
    y = y - np.mean(y)
    z = np.fft.rfft(y)
    hz = np.fft.rfftfreq(y.size, d=1.0 / rate)
    mag = (2.0 / y.size) * np.abs(z)
    return hz, mag


# Finds one dominant peak inside the selected frequency window.
def fundamentalPeak(hz, mag, lowHz, highHz):
    i = (hz >= lowHz) & (hz <= highHz)
    x = hz[i]
    y = mag[i]
    j = int(np.argmax(y))
    return float(x[j]), float(y[j])


# Builds one order band with a symmetric tolerance.
def orderBand(fundHz, lowOrder, highOrder, tolFraction):
    lowHz = (fundHz * lowOrder) * (1.0 - tolFraction)
    highHz = (fundHz * highOrder) * (1.0 + tolFraction)
    return lowHz, highHz


# Builds one RMS value for the selected FFT band.
def bandRms(hz, mag, lowHz, highHz):
    i = (hz >= lowHz) & (hz <= highHz)
    x = mag[i]
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x**2)))

################################################################################
# main functions                                                               #
################################################################################


# Builds the full raw and derived signal dictionary for one dataframe.
def buildSignals(dataFrame, mode):
    rate = buffer.sampleRate(mode)
    cols = buffer.csvCols(mode)
    axes = buffer.timeCols(mode)
    vis = buffer.visualCols(mode)
    sig = {
        "mode": mode,
        "sampleRate": rate,
        "timeCols": axes,
        "fftCols": axes,
        "visualCols": vis,
    }
    for i in cols:
        sig[i] = dataFrame[i].to_numpy(dtype=np.float64)
    if "t_s" not in sig:
        n = dataFrame.shape[0]
        sig["t_s"] = np.arange(n, dtype=np.float64) / rate
    magArrs = [sig[i] for i in vis]
    sig["timeMag"] = vectorMag(magArrs)
    hz = None
    sumParts = []
    for i in axes:
        x, y = buildSpectrum(sig[i], rate)
        hz = x
        sig[f"{i}Fft"] = y
        sumParts.append(y)
    sig["freqHz"] = hz
    sig["sumFft"] = vectorMag(sumParts)
    fundHz, fundMag = fundamentalPeak(
        sig["freqHz"],
        sig["sumFft"],
        fftConfig["minHz"],
        fftConfig["maxHz"],
    )
    sig["fundHz"] = fundHz
    sig["fundMag"] = fundMag
    sig["bpfoBand"] = orderBand(
        fundHz,
        fftConfig["bpfoLowOrder"],
        fftConfig["bpfoHighOrder"],
        fftConfig["tolFraction"],
    )
    sig["bpfiBand"] = orderBand(
        fundHz,
        fftConfig["bpfiLowOrder"],
        fftConfig["bpfiHighOrder"],
        fftConfig["tolFraction"],
    )
    sig["bpfoRms"] = bandRms(
        sig["freqHz"],
        sig["sumFft"],
        sig["bpfoBand"][0],
        sig["bpfoBand"][1],
    )
    sig["bpfiRms"] = bandRms(
        sig["freqHz"],
        sig["sumFft"],
        sig["bpfiBand"][0],
        sig["bpfiBand"][1],
    )
    return sig

