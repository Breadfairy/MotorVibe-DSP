################################################################################
# Imports                                                                      #
################################################################################
from pathlib import Path
import sys

import joblib
import numpy as np

import data
import signals

################################################################################
# variables/constants                                                          #
################################################################################
# Offline inference loads the same trained model bundle as live.py.
# By default it checks every CSV in the training folder.
buildDir = Path(__file__).resolve().parents[1]
compactModelPath = buildDir / "outputs" / "models" / "motorPumpClassifier.joblib"
rawModelPath = buildDir / "outputs" / "models" / "motorPumpRawAxisClassifier.joblib"
inputPath = buildDir / "data" / "training" / "main"
expectedLabelNames = [
    "good",
    "bad_leak",
]
modelMode = "compact"
if len(sys.argv) > 1 and sys.argv[1] in ["compact", "raw"]:
    modelMode = sys.argv[1]
if modelMode == "raw":
    modelPath = rawModelPath
else:
    modelPath = compactModelPath

################################################################################
# helpers                                                                      #
################################################################################


# Converts one CSV file into feature rows for offline inference.
# This repeats the training signal path for one file:
# CSV -> clean -> windows -> time signals -> FFT signals -> model input rows.
def featureMatrixFromCsv(csvPath, bundle):
    sampleRate = bundle["sampleRate"]
    df = data.readCsv(csvPath)
    df = data.cleanFrame(df)
    windows = data.windowFrames(
        df,
        sampleRate,
        bundle["winSecs"],
        bundle["stepSecs"],
    )

    featureRows = []

    # Each one-second window becomes one prediction row.
    for windowFrame in windows:
        if bundle.get("modelMode") == "raw":
            featureRows.append(signals.rawModelInput(windowFrame))
        else:
            timeData = signals.timeSignals(windowFrame, sampleRate)
            freqData = signals.fftSignals(timeData, sampleRate, signals.fftConfig)
            featureRows.append(signals.modelInput(timeData, freqData))

    featureMatrix = np.vstack(featureRows).astype(np.float32)
    return featureMatrix


# Converts one probability vector into the strongest label and confidence.
# The model returns one probability per class in the saved label order.
def stateFromProb(probVector, labelNames):
    topIndex = int(np.argmax(probVector))
    state = labelNames[topIndex]
    confidence = float(probVector[topIndex])
    return state, confidence


# Stops inference if an old model with old labels is loaded by mistake.
def checkModelLabels(bundle):
    labelNames = list(bundle["labelNames"])
    if labelNames != expectedLabelNames:
        raise SystemExit(
            f"model labels are {labelNames}, expected {expectedLabelNames}"
        )
    if bundle.get("modelMode", "compact") != modelMode:
        raise SystemExit(
            f"model mode is {bundle.get('modelMode')}, expected {modelMode}"
        )

    modelValue = bundle["model"]
    classIndexes = list(modelValue.named_steps["logisticregression"].classes_)
    expectedIndexes = list(range(len(expectedLabelNames)))
    if classIndexes != expectedIndexes:
        raise SystemExit(
            f"model class indexes are {classIndexes}, expected {expectedIndexes}"
        )

################################################################################
# main functions                                                               #
################################################################################


# Runs offline inference on one CSV file or folder.
def main():
    # A command line path can point at one CSV or a folder of CSV files.
    sourcePath = inputPath
    if len(sys.argv) > 1 and sys.argv[1] not in ["compact", "raw"]:
        sourcePath = Path(sys.argv[1])
    if len(sys.argv) > 2:
        sourcePath = Path(sys.argv[2])

    if sourcePath.is_file():
        csvList = [sourcePath]
    else:
        csvList = sorted(sourcePath.glob("*.csv"))

    # Load the model and label names saved by train.py.
    bundle = joblib.load(modelPath)
    checkModelLabels(bundle)
    modelValue = bundle["model"]
    labelNames = bundle["labelNames"]

    for csvPath in csvList:
        # Predict every window in the file, then average the probabilities to
        # get one overall result for the recording.
        featureMatrix = featureMatrixFromCsv(csvPath, bundle)
        probMatrix = modelValue.predict_proba(featureMatrix)
        meanProb = probMatrix.mean(axis=0)
        state, confidence = stateFromProb(meanProb, labelNames)

        print("csvPath:", csvPath)
        print("state:", state)
        print("confidence:", f"{confidence * 100.0:.1f}%")
        print("")


if __name__ == "__main__":
    main()
