################################################################################
# Imports                                                                      #
################################################################################
from pathlib import Path

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
buildDir = Path(__file__).resolve().parents[1]
trainDir = buildDir / "data" / "training" / "main"
modelPath = buildDir / "outputs" / "models" / "fanClassifier.joblib"
classNames = ["good", "voltSag"]
sampleRate = 1000.0
winSecs = 1.0
stepSecs = 0.25
valFileCount = 1

################################################################################
# helpers                                                                      #
################################################################################


# Finds the CSV recordings for one fixed class label.
def classCsvs(className):
    classDir = trainDir / className
    if classDir.exists():
        return sorted(classDir.glob("*.csv"))
    return sorted(trainDir.glob(f"{className}*.csv"))


# Keeps the newest CSV file for validation and uses the rest for training.
def splitCsvs(csvPaths):
    if len(csvPaths) <= valFileCount:
        return csvPaths, []
    trainCsvs = csvPaths[:-valFileCount]
    valCsvs = csvPaths[-valFileCount:]
    return trainCsvs, valCsvs


# Converts CSV recordings into labelled feature rows.
def addCsvFeatures(csvPaths, classIndex, featureRows, classRows):
    for csvPath in csvPaths:
        df = data.readCsv(csvPath)
        df = data.cleanFrame(df)
        windows = data.windowFrames(df, sampleRate, winSecs, stepSecs)
        for windowFrame in windows:
            timeData = signals.timeSignals(windowFrame, sampleRate)
            freqData = signals.fftSignals(
                timeData,
                sampleRate,
                signals.fftConfig,
            )
            featureRows.append(signals.modelInput(timeData, freqData))
            classRows.append(classIndex)


# Converts Python lists into numpy arrays for sklearn.
def matrixFromRows(featureRows, classRows):
    featureMatrix = np.vstack(featureRows).astype(np.float32)
    classVector = np.array(classRows, dtype=np.int64)
    return featureMatrix, classVector

################################################################################
# main functions                                                               #
################################################################################


# Trains the sklearn model and saves the model bundle.
def main():
    featureRows = []
    classRows = []
    valFeatureRows = []
    valClassRows = []

    for classIndex, className in enumerate(classNames):
        csvPaths = classCsvs(className)
        trainCsvs, valCsvs = splitCsvs(csvPaths)
        addCsvFeatures(trainCsvs, classIndex, featureRows, classRows)
        addCsvFeatures(valCsvs, classIndex, valFeatureRows, valClassRows)

    featureMatrix, classVector = matrixFromRows(featureRows, classRows)

    modelValue = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
    )
    modelValue.fit(featureMatrix, classVector)
    trainAcc = modelValue.score(featureMatrix, classVector)
    trainPred = modelValue.predict(featureMatrix)

    print("trainShape:", featureMatrix.shape)
    print("trainAcc:", f"{trainAcc * 100.0:.2f}%")
    print("trainConfusion:")
    print(
        confusion_matrix(
            classVector,
            trainPred,
            labels=list(range(len(classNames))),
        )
    )
    print("trainReport:")
    print(
        classification_report(
            classVector,
            trainPred,
            labels=list(range(len(classNames))),
            target_names=classNames,
            zero_division=0,
        )
    )

    valAcc = None
    if len(valFeatureRows) > 0:
        valFeatureMatrix, valClassVector = matrixFromRows(
            valFeatureRows,
            valClassRows,
        )
        valAcc = modelValue.score(valFeatureMatrix, valClassVector)
        valPred = modelValue.predict(valFeatureMatrix)
        print("valShape:", valFeatureMatrix.shape)
        print("valAcc:", f"{valAcc * 100.0:.2f}%")
        print("valConfusion:")
        print(
            confusion_matrix(
                valClassVector,
                valPred,
                labels=list(range(len(classNames))),
            )
        )
        print("valReport:")
        print(
            classification_report(
                valClassVector,
                valPred,
                labels=list(range(len(classNames))),
                target_names=classNames,
                zero_division=0,
            )
        )
    else:
        print("valShape: none")
        print("valAcc: no held-out files")

    bundle = {
        "model": modelValue,
        "labelNames": classNames,
        "sampleRate": sampleRate,
        "winSecs": winSecs,
        "stepSecs": stepSecs,
        "featureNames": signals.featureNames(),
        "featureMode": "motor_compact_low_order_bearing_orders",
        "modelMode": "compact",
        "trainAcc": trainAcc,
        "valAcc": valAcc,
    }

    modelPath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, modelPath)
    print("modelPath:", modelPath)


if __name__ == "__main__":
    main()
