################################################################################
# Imports                                                                      #
################################################################################
import pandas as pd

################################################################################
# variables/constants                                                          #
################################################################################
# These are the exact columns that the rest of the program expects.
# Keeping one fixed list means capture, training, live, and charts all use
# the same signal order.
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

# Acceleration and gyro columns are separated because they use different
# physical units and different sensible clipping ranges.
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

# These limits are a simple outlier guard. They are not trying to repair data;
# they just stop one bad serial value from making a huge training feature.
accelMin = -2.0
accelMax = 2.0
gyroMin = -250.0
gyroMax = 250.0
tempMin = -40.0
tempMax = 125.0

################################################################################
# main functions                                                               #
################################################################################


# Reads one CSV and keeps only the known signal columns.
# Extra columns are ignored so older notes or exported debug columns do not
# change the model input.
def readCsv(csvPath):
    df = pd.read_csv(csvPath)
    df = df.loc[:, signalCols].copy()
    return df


# Cleans one loaded CSV or live DataFrame before windowing.
# The cleaning is deliberately basic: convert to numbers, replace missing/bad
# values with zero, clip impossible sensor values, then reset time to start at 0.
def cleanFrame(dataFrame):
    df = dataFrame.copy()

    # Convert every expected column to a numeric value.
    # Anything that cannot be parsed becomes NaN and is handled below.
    for col in signalCols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Infinity and NaN values are not useful to the model.
    # Filling with zero is simple and visible; it avoids hidden interpolation.
    df = df.replace([float("inf"), float("-inf")], float("nan"))
    df = df.fillna(0.0)

    # Clip each sensor type to its expected hardware range.
    # This handles outliers without trying to guess what the missing signal was.
    for col in accelCols:
        df[col] = df[col].clip(lower=accelMin, upper=accelMax)
    for col in gyroCols:
        df[col] = df[col].clip(lower=gyroMin, upper=gyroMax)
    df["tempC"] = df["tempC"].clip(lower=tempMin, upper=tempMax)

    # Make each file start at zero time so old absolute MCU timestamps do not
    # become part of the learning problem.
    x0 = float(df["t_us"].iloc[0])
    df["t_us"] = df["t_us"] - x0
    df["t_s"] = df["t_us"] / 1e6
    df = df.reset_index(drop=True)
    return df


# Splits a cleaned recording into overlapping windows.
# Training and live inference both use the same one-second window shape so the
# model sees the same type of input in both places.
def windowFrames(dataFrame, sampleRate, winSecs, stepSecs):
    winRows = int(sampleRate * winSecs)
    stepRows = int(sampleRate * stepSecs)
    lastRow = dataFrame.shape[0] - winRows
    windows = []

    # Move through the file in fixed row steps and keep full windows only.
    # Partial windows are skipped because their FFT bins would not match.
    for row in range(0, lastRow + 1, stepRows):
        df = dataFrame.iloc[row : row + winRows].reset_index(drop=True)
        windows.append(df)
    return windows
