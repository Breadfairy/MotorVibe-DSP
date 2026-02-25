import numpy as np

# Calculates a simple moving average on a 1D array.
def movingAverage(signal, windowSize):
    kernel = np.ones(windowSize, dtype=np.float64) / windowSize
    return np.convolve(signal, kernel, mode="valid")


# Keeps every nth sample from a 1D array.
def nthSample(signal, n):
    return signal[::n]


# Builds dspSignals from rawSignals and appends filtered temperature trend.
def buildDSPSignals(rawSignals, tempTrendNth):
    sample = rawSignals[0]
    Acel_X = rawSignals[1]
    Acel_Y = rawSignals[2]
    Acel_Z = rawSignals[3]
    Giro_X = rawSignals[4]
    Giro_Y = rawSignals[5]
    Giro_Z = rawSignals[6]
    Temperatura = rawSignals[7]

    tempTrendSample = nthSample(sample, tempTrendNth)
    tempTrend = nthSample(Temperatura, tempTrendNth)

    dspSignals = [
        sample,
        Acel_X,
        Acel_Y,
        Acel_Z,
        Giro_X,
        Giro_Y,
        Giro_Z,
        Temperatura,
        tempTrendSample,
        tempTrend,
    ]
    return dspSignals
