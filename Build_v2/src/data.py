# Loads, cleans, and windows recorded CSV data for training and live inference.
import numpy as np
import pandas as pd

# Defines the required CSV column order for captured recordings.
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

# Defines the motion channels used for model features.
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

# Limits outlier clipping to sensor and temperature channels.
cleanCols = axisCols + ["tempC"]

# Sets the median absolute deviation threshold for outlier clipping.
outlierMadScale = 8.0


# Reads one captured CSV and keeps the expected signal columns.
def readCsv(csvPath):
    df = pd.read_csv(csvPath)
    df = df.loc[:, signalCols].copy()
    return df


# Cleans one full recording before it is split into training windows.
def cleanFrame(dataFrame):
    df = dataFrame.copy()

    # Converts every expected column to numeric values before interpolation.
    for col in signalCols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.interpolate(method="linear", limit_direction="both")
    df = df.fillna(0.0)

    # Clips large spikes using a robust median absolute deviation limit.
    for col in cleanCols:
        x = df[col].to_numpy(dtype=np.float64)
        median = float(np.median(x))
        mad = float(np.median(np.abs(x - median)))
        if mad == 0.0:
            continue
        scale = mad * 1.4826
        lo = median - (outlierMadScale * scale)
        hi = median + (outlierMadScale * scale)
        df[col] = df[col].clip(lower=lo, upper=hi)

    # Normalises each recording to start at t=0 seconds.
    x0 = float(df["t_us"].iloc[0])
    df["t_us"] = df["t_us"] - x0
    df["t_s"] = df["t_us"] / 1e6
    df = df.reset_index(drop=True)
    return df


# Splits one recording into overlapping fixed-length windows.
def windowFrames(dataFrame, sampleRate, winSecs, stepSecs):
    winRows = int(sampleRate * winSecs)
    stepRows = int(sampleRate * stepSecs)
    lastRow = dataFrame.shape[0] - winRows
    windows = []
    for row in range(0, lastRow + 1, stepRows):
        df = dataFrame.iloc[row : row + winRows].reset_index(drop=True)
        windows.append(df)
    return windows
