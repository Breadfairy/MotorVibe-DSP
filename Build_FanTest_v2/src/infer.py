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
buildDir = Path(__file__).resolve().parents[1]
modelPath = buildDir / "outputs" / "models" / "fanClassifier.joblib"
inputPath = buildDir / "data" / "training" / "main"

################################################################################
# helpers                                                                      #
################################################################################


# Converts one CSV file into feature rows for offline inference.
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
    for windowFrame in windows:
        timeData = signals.timeSignals(windowFrame, sampleRate)
        freqData = signals.fftSignals(timeData, sampleRate, signals.fftConfig)
        featureRows.append(signals.modelInput(timeData, freqData))

    featureMatrix = np.vstack(featureRows).astype(np.float32)
    return featureMatrix


# Converts one probability vector into the strongest label and confidence.
def stateFromProb(probVector, labelNames):
    topIndex = int(np.argmax(probVector))
    state = labelNames[topIndex]
    confidence = float(probVector[topIndex])
    return state, confidence

################################################################################
# main functions                                                               #
################################################################################


# Runs offline inference on one CSV file or folder.
def main():
    sourcePath = inputPath
    if len(sys.argv) > 1:
        sourcePath = Path(sys.argv[1])

    if sourcePath.is_file():
        csvList = [sourcePath]
    else:
        csvList = sorted(sourcePath.glob("*.csv"))

    bundle = joblib.load(modelPath)
    modelValue = bundle["model"]
    labelNames = bundle["labelNames"]

    for csvPath in csvList:
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
