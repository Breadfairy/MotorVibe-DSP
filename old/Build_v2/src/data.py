################################################################################
# Imports                                                                      #
################################################################################
import pandas as pd

################################################################################
# variables/constants                                                          #
################################################################################
# Shared CSV column order.
signalCols = [
    "t_us",
    "t_s",
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
    "tempC",
]

# Columns clipped by sensor type.
accelCols = [
    "ax1",
    "ay1",
    "az1",
    "ax2",
    "ay2",
    "az2",
]
gyroCols = [
    "gx1",
    "gy1",
    "gz1",
    "gx2",
    "gy2",
    "gz2",
]

# Outlier limits.
accelMin = -2.0
accelMax = 2.0
gyroMin = -250.0
gyroMax = 250.0
tempMin = -40.0
tempMax = 125.0

################################################################################
# main functions                                                               #
################################################################################


# Reads one CSV and keeps the known signal columns.
def readCsv(csvPath):
    capture = pd.read_csv(csvPath)
    capture = capture.loc[:, signalCols].copy()
    return capture


# Cleans one CSV or live DataFrame before windowing.
def cleanCapture(capture):
    cleanedCapture = capture.copy()

    for col in signalCols:
        cleanedCapture[col] = pd.to_numeric(cleanedCapture[col], errors="coerce")

    cleanedCapture = cleanedCapture.replace(
        [float("inf"), float("-inf")],
        float("nan"),
    )
    cleanedCapture = cleanedCapture.fillna(0.0)

    for col in accelCols:
        cleanedCapture[col] = cleanedCapture[col].clip(
            lower=accelMin,
            upper=accelMax,
        )
    for col in gyroCols:
        cleanedCapture[col] = cleanedCapture[col].clip(
            lower=gyroMin,
            upper=gyroMax,
        )
    cleanedCapture["tempC"] = cleanedCapture["tempC"].clip(
        lower=tempMin,
        upper=tempMax,
    )

    x0 = float(cleanedCapture["t_us"].iloc[0])
    cleanedCapture["t_us"] = cleanedCapture["t_us"] - x0
    cleanedCapture["t_s"] = cleanedCapture["t_us"] / 1e6
    cleanedCapture = cleanedCapture.reset_index(drop=True)
    return cleanedCapture


# Splits a cleaned recording into overlapping windows.
def windowsFromCapture(capture, sampleRate, winSecs, stepSecs):
    winRows = int(sampleRate * winSecs)
    stepRows = int(sampleRate * stepSecs)
    lastSample = capture.shape[0] - winRows
    windows = []

    for sampleIndex in range(0, lastSample + 1, stepRows):
        oneSecondWindow = capture.iloc[
            sampleIndex : sampleIndex + winRows
        ].reset_index(drop=True)
        windows.append(oneSecondWindow)
    return windows


# Backwards-compatible names used by older scripts in this build.
cleanFrame = cleanCapture
windowFrames = windowsFromCapture
