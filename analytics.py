import numpy as np


# Calculates a simple moving average on a 1D array.
def movingAverage(signal, windowSize):
    kernel = np.ones(windowSize, dtype=np.float64) / windowSize
    return np.convolve(signal, kernel, mode="valid")


# Keeps every nth sample from a 1D array.
def nthSample(signal, n):
    return signal[::n]


# Builds a small named DSP view from the current raw signal set.
def buildDSPSignals(rawSignals, tempTrendNth):
    sample = rawSignals["sample"]
    mpu1AccX = rawSignals["mpu1AccX"]
    mpu1AccY = rawSignals["mpu1AccY"]
    mpu1AccZ = rawSignals["mpu1AccZ"]
    mpu1GyrX = rawSignals["mpu1GyrX"]
    mpu1GyrY = rawSignals["mpu1GyrY"]
    mpu1GyrZ = rawSignals["mpu1GyrZ"]
    ds18b20One = rawSignals["ds18b20One"]
    tempTrendSample = nthSample(sample, tempTrendNth)
    tempTrend = nthSample(ds18b20One, tempTrendNth)
    dspSignals = {
        "sample": sample,
        "mpu1AccX": mpu1AccX,
        "mpu1AccY": mpu1AccY,
        "mpu1AccZ": mpu1AccZ,
        "mpu1GyrX": mpu1GyrX,
        "mpu1GyrY": mpu1GyrY,
        "mpu1GyrZ": mpu1GyrZ,
        "ds18b20One": ds18b20One,
        "tempTrendSample": tempTrendSample,
        "tempTrend": tempTrend,
    }
    return dspSignals
