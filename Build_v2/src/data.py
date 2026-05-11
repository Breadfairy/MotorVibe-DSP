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
    df = pd.read_csv(csvPath)
    df = df.loc[:, signalCols].copy()
    return df


# Cleans one CSV or live DataFrame before windowing.
def cleanFrame(dataFrame):
    df = dataFrame.copy()

    for col in signalCols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([float("inf"), float("-inf")], float("nan"))
    df = df.fillna(0.0)

    for col in accelCols:
        df[col] = df[col].clip(lower=accelMin, upper=accelMax)
    for col in gyroCols:
        df[col] = df[col].clip(lower=gyroMin, upper=gyroMax)
    df["tempC"] = df["tempC"].clip(lower=tempMin, upper=tempMax)

    x0 = float(df["t_us"].iloc[0])
    df["t_us"] = df["t_us"] - x0
    df["t_s"] = df["t_us"] / 1e6
    df = df.reset_index(drop=True)
    return df


# Splits a cleaned recording into overlapping windows.
def windowFrames(dataFrame, sampleRate, winSecs, stepSecs):
    winRows = int(sampleRate * winSecs)
    stepRows = int(sampleRate * stepSecs)
    lastRow = dataFrame.shape[0] - winRows
    windows = []

    for row in range(0, lastRow + 1, stepRows):
        df = dataFrame.iloc[row : row + winRows].reset_index(drop=True)
        windows.append(df)
    return windows
