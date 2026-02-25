import numpy as np


# Calculates vector magnitude for 3-axis signals.
def axisMagnitude(axisX, axisY, axisZ):
    return np.sqrt((axisX ** 2) + (axisY ** 2) + (axisZ ** 2))


# Calculates a simple moving average on a 1D array.
def movingAverage(signal, windowSize):
    kernel = np.ones(windowSize, dtype=np.float64) / windowSize
    return np.convolve(signal, kernel, mode="valid")
