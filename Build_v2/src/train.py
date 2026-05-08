################################################################################
# Imports                                                                      #
################################################################################
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data
import signals

################################################################################
# variables/constants                                                          #
################################################################################
# Main project paths. Training reads labelled CSVs and writes one joblib model
# bundle that live.py and infer.py can load later.
buildDir = Path(__file__).resolve().parents[1]
trainDir = buildDir / "data" / "training" / "main"
compactModelPath = buildDir / "outputs" / "models" / "motorPumpClassifier.joblib"
rawModelPath = buildDir / "outputs" / "models" / "motorPumpRawAxisClassifier.joblib"
classNames = [
    "good",
    "bad_leak",
]

# Default mode is the compact DSP feature model.
# Run `python3 Build_v2/src/train.py raw` to train the raw-axis model instead.
modelMode = "compact"
if len(sys.argv) > 1:
    modelMode = sys.argv[1]
if modelMode not in ["compact", "raw"]:
    raise SystemExit("mode must be compact or raw")
if modelMode == "raw":
    modelPath = rawModelPath
else:
    modelPath = compactModelPath

# These values define the ML procedure.
# One-second windows match the fan test, and 0.25 seconds gives overlap so each
# recording produces more training examples.
sampleRate = 1000.0
winSecs = 1.0
stepSecs = 0.25

# Split each trusted main CSV by time order.
# The unused folder is not read at all.
trainFraction = 0.35
valFraction = 0.25
holdoutFraction = 1.0 - trainFraction - valFraction

################################################################################
# helpers                                                                      #
################################################################################


# Finds the CSV recordings for one fixed class label.
# Files can either be in a class folder or in one flat folder using the label
# prefix, such as good_001.csv or bad_leak_001.csv.
def classCsvs(className):
    classDir = trainDir / className
    if classDir.exists():
        return sorted(classDir.glob("*.csv"))
    return sorted(trainDir.glob(f"{className}*.csv"))


# Splits one cleaned CSV into train, validation, and holdout time blocks.
# The blocks do not overlap. This keeps validation and holdout from using
# exactly the same rows as training.
def splitFrame(df):
    rowCount = df.shape[0]
    trainEnd = int(rowCount * trainFraction)
    valEnd = trainEnd + int(rowCount * valFraction)

    trainDf = df.iloc[:trainEnd].reset_index(drop=True)
    valDf = df.iloc[trainEnd:valEnd].reset_index(drop=True)
    holdoutDf = df.iloc[valEnd:].reset_index(drop=True)
    return trainDf, valDf, holdoutDf


# Adds every window from one DataFrame segment into the selected feature lists.
# This is the repeated signal flow:
# segment -> one-second windows -> time signals -> FFT signals -> ML row.
def addFrameFeatures(df, classIndex, featureRows, classRows):
    windows = data.windowFrames(df, sampleRate, winSecs, stepSecs)

    # Each window becomes one ML row with the same class label as the source file.
    for windowFrame in windows:
        featureRows.append(featureRowFromWindow(windowFrame))
        classRows.append(classIndex)


# Builds one feature row using the selected model mode.
def featureRowFromWindow(windowFrame):
    if modelMode == "raw":
        return signals.rawModelInput(windowFrame)

    timeData = signals.timeSignals(windowFrame, sampleRate)
    freqData = signals.fftSignals(
        timeData,
        sampleRate,
        signals.fftConfig,
    )
    return signals.modelInput(timeData, freqData)


# Reads labelled CSV recordings and sends each time block to the right split.
def addCsvFeatures(
    csvPaths,
    classIndex,
    featureRows,
    classRows,
    valFeatureRows,
    valClassRows,
    holdoutFeatureRows,
    holdoutClassRows,
):
    for csvPath in csvPaths:
        df = data.readCsv(csvPath)
        df = data.cleanFrame(df)
        trainDf, valDf, holdoutDf = splitFrame(df)
        addFrameFeatures(trainDf, classIndex, featureRows, classRows)
        addFrameFeatures(valDf, classIndex, valFeatureRows, valClassRows)
        addFrameFeatures(
            holdoutDf,
            classIndex,
            holdoutFeatureRows,
            holdoutClassRows,
        )


# Converts Python lists into numpy arrays for sklearn.
# sklearn expects a 2D feature matrix and a 1D class vector.
def matrixFromRows(featureRows, classRows):
    featureMatrix = np.vstack(featureRows).astype(np.float32)
    classVector = np.array(classRows, dtype=np.int64)
    return featureMatrix, classVector


# Prints accuracy, confusion matrix, and classification report for one split.
def printScores(name, modelValue, featureRows, classRows):
    if len(featureRows) == 0:
        print(f"{name}Shape: none")
        print(f"{name}Acc: no rows")
        return None

    featureMatrix, classVector = matrixFromRows(featureRows, classRows)
    acc = modelValue.score(featureMatrix, classVector)
    pred = modelValue.predict(featureMatrix)

    print(f"{name}Shape:", featureMatrix.shape)
    print(f"{name}Acc:", f"{acc * 100.0:.2f}%")
    print(f"{name}Confusion:")
    print(
        confusion_matrix(
            classVector,
            pred,
            labels=list(range(len(classNames))),
        )
    )
    print(f"{name}Report:")
    print(
        classification_report(
            classVector,
            pred,
            labels=list(range(len(classNames))),
            target_names=classNames,
            zero_division=0,
        )
    )
    return acc

################################################################################
# main functions                                                               #
################################################################################


# Trains the sklearn model and saves the model bundle.
def main():
    # These lists collect the window-level feature rows before they are converted
    # into numpy arrays for sklearn.
    print("modelMode:", modelMode)
    featureRows = []
    classRows = []
    valFeatureRows = []
    valClassRows = []
    holdoutFeatureRows = []
    holdoutClassRows = []

    # Loop through each fixed class name and split each trusted main file by
    # time range.
    for classIndex, className in enumerate(classNames):
        csvPaths = classCsvs(className)
        if len(csvPaths) == 0:
            raise SystemExit(
                f"no training CSVs found for {className} in {trainDir}"
            )
        addCsvFeatures(
            csvPaths,
            classIndex,
            featureRows,
            classRows,
            valFeatureRows,
            valClassRows,
            holdoutFeatureRows,
            holdoutClassRows,
        )

    featureMatrix, classVector = matrixFromRows(featureRows, classRows)

    # StandardScaler keeps large-valued features from overpowering smaller ones.
    # LogisticRegression is simple, fast, and matches the fan-test style.
    modelValue = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
    )
    modelValue.fit(featureMatrix, classVector)

    # Print all split results so the user can see train, validation, and holdout
    # behaviour from one run.
    trainAcc = printScores("train", modelValue, featureRows, classRows)
    valAcc = printScores("val", modelValue, valFeatureRows, valClassRows)
    holdoutAcc = printScores(
        "holdout",
        modelValue,
        holdoutFeatureRows,
        holdoutClassRows,
    )

    # Save the trained model and the settings needed to rebuild features later.
    # live.py and infer.py rely on this bundle to use the same window size and
    # label order as training.
    if modelMode == "raw":
        featureNames = signals.rawFeatureNames(int(sampleRate * winSecs))
        featureMode = "raw_axes_1s"
    else:
        featureNames = signals.featureNames()
        featureMode = "motor_compact_low_order_bearing_orders"

    bundle = {
        "model": modelValue,
        "labelNames": classNames,
        "sampleRate": sampleRate,
        "winSecs": winSecs,
        "stepSecs": stepSecs,
        "featureNames": featureNames,
        "featureMode": featureMode,
        "modelMode": modelMode,
        "trainAcc": trainAcc,
        "valAcc": valAcc,
        "holdoutAcc": holdoutAcc,
        "trainFraction": trainFraction,
        "valFraction": valFraction,
        "holdoutFraction": holdoutFraction,
    }

    modelPath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, modelPath)
    print("modelPath:", modelPath)


if __name__ == "__main__":
    main()
