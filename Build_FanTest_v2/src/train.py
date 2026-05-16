################################################################################
# Imports                                                                      #
################################################################################
from pathlib import Path
import math
import os
import random
import re

buildDir = Path(__file__).resolve().parents[1]
mplConfigDir = buildDir / "outputs" / ".matplotlib"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(mplConfigDir))

import joblib
import matplotlib.pyplot as plt
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
trainDir = buildDir / "data" / "training" / "main"
modelPath = buildDir / "outputs" / "models" / "fanClassifier.joblib"
chartDir = buildDir / "outputs" / "ML-charts"
labelGroups = [
    ("off", "off", None),
    ("good", "good", "withVoltage"),
    ("voltSag8p5", "voltSag", 8.5),
    ("voltSag8p0", "voltSag", 8.0),
    ("voltSag7p5", "voltSag", 7.5),
    ("obstruction", "obstruction", "withoutLight"),
    ("imbalance", "imbalance", None),
]
holdoutGroups = [
    ("good", "good"),
    ("off", "off"),
    ("voltSag", "voltSag"),
]
sampleRate = 1000.0
winSecs = 1.0
stepSecs = 0.25
trainFraction = 0.25
splitSeed = 42
voltageTolerance = 0.15
baseVoltages = {
    "voltSag8p5": 8.5,
    "voltSag8p0": 8.0,
    "voltSag7p5": 7.5,
}

################################################################################
# helpers                                                                      #
################################################################################


# Finds the CSV recordings for one fixed source folder.
def sourceCsvs(sourceLabel):
    classDir = trainDir / sourceLabel
    if classDir.exists():
        csvPaths = sorted(classDir.glob("*.csv"))
    else:
        csvPaths = sorted(trainDir.glob(f"{sourceLabel}*.csv"))

    goodPaths = []
    for csvPath in csvPaths:
        if hasEnoughRows(csvPath):
            goodPaths.append(csvPath)
        else:
            print("skipping short CSV:", csvPath)
    return goodPaths


# Finds the CSV recordings for one model class.
def groupCsvs(sourceLabel, targetVoltage):
    csvPaths = sourceCsvs(sourceLabel)
    if targetVoltage == "withoutLight":
        return [
            csvPath
            for csvPath in csvPaths
            if not isLightObstruction(csvPath)
        ]
    if targetVoltage == "off":
        return [csvPath for csvPath in csvPaths if "off" in csvPath.stem.lower()]
    if targetVoltage == "withVoltage":
        return [
            csvPath
            for csvPath in csvPaths
            if filenameVoltage(csvPath) is not None
            and "off" not in csvPath.stem.lower()
        ]
    if targetVoltage is None:
        return csvPaths

    paths = []
    for csvPath in csvPaths:
        voltage = filenameVoltage(csvPath)
        if voltage is None:
            print("skipping voltage-class CSV without voltage:", csvPath)
            continue
        if abs(voltage - targetVoltage) <= voltageTolerance:
            paths.append(csvPath)
    return paths


# Extracts voltages from names like v8p5, v8.5, v10p0, or v10.0.
def filenameVoltage(csvPath):
    match = re.search(r"(?:^|_)v(\d+(?:p\d+|\.\d+)?)", csvPath.stem)
    if match is None:
        return None
    return float(match.group(1).replace("p", "."))


# Light obstruction files identified by current-model good predictions.
def isLightObstruction(csvPath):
    return csvPath.stem in {
        "r19_obst_1",
        "r19_obst_2",
        "r22_obst_4",
        "r23_obst_5",
    }


# Returns true when the CSV can produce at least one training window.
def hasEnoughRows(csvPath):
    minLines = int(sampleRate * winSecs) + 1
    with csvPath.open("r", newline="") as csvFile:
        for lineIndex, _ in enumerate(csvFile, start=1):
            if lineIndex >= minLines:
                return True
    return False


# Splits files for one class into deterministic train and validation sets.
def splitCsvs(csvPaths):
    paths = list(csvPaths)
    random.Random(splitSeed).shuffle(paths)
    if len(paths) == 1:
        return paths, []
    trainFileCount = max(1, math.ceil(len(paths) * trainFraction))
    trainFileCount = min(trainFileCount, len(paths) - 1)
    return paths[:trainFileCount], paths[trainFileCount:]


# Returns the trailing obstruction index from names like r30_obst_13.
def obstructionIndex(csvPath):
    match = re.search(r"_obst_(\d+)$", csvPath.stem)
    if match is None:
        return -1
    return int(match.group(1))


# Applies the current source-separated training plan.
def plannedCsvSplit(label, csvPaths):
    paths = sorted(csvPaths)
    if label == "good":
        trainPaths = [path for path in paths if "_warm" in path.stem]
        valPaths = [path for path in paths if path not in trainPaths]
        return trainPaths, valPaths

    if label in baseVoltages:
        baseVoltage = baseVoltages[label]
        trainPaths = [
            path
            for path in paths
            if filenameVoltage(path) is not None
            and abs(filenameVoltage(path) - baseVoltage) < 0.01
        ]
        valPaths = [path for path in paths if path not in trainPaths]
        return trainPaths, valPaths

    if label == "obstruction":
        trainNames = {
            "r25_obst_7",
            "r26_obst_8",
            "r28_obst_11",
        }
        trainPaths = [path for path in paths if path.stem in trainNames]
        valPaths = [path for path in paths if path not in trainPaths]
        return trainPaths, valPaths

    if label == "imbalance":
        trainPaths = [path for path in paths if "_2band" in path.stem]
        valPaths = [path for path in paths if path not in trainPaths]
        return trainPaths, valPaths

    return splitCsvs(paths)


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


# Converts CSV recordings into feature rows without assigning class labels.
def csvFeatures(csvPaths):
    featureRows = []
    sourceRows = []
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
            sourceRows.append(csvPath)
    if len(featureRows) == 0:
        return None, []
    return np.vstack(featureRows).astype(np.float32), sourceRows


# Splits already-extracted feature rows into deterministic train/validation sets.
def splitFeatureMatrix(featureMatrix):
    rowIndexes = list(range(featureMatrix.shape[0]))
    random.Random(splitSeed).shuffle(rowIndexes)
    if len(rowIndexes) == 1:
        return featureMatrix, None

    trainRowCount = max(1, math.ceil(len(rowIndexes) * trainFraction))
    trainRowCount = min(trainRowCount, len(rowIndexes) - 1)
    trainIndexes = rowIndexes[:trainRowCount]
    valIndexes = rowIndexes[trainRowCount:]
    return featureMatrix[trainIndexes], featureMatrix[valIndexes]


# Adds already-extracted feature rows to one labelled output set.
def addFeatureMatrix(featureMatrix, classIndex, featureRows, classRows):
    for row in featureMatrix:
        featureRows.append(row)
        classRows.append(classIndex)


# Finds original voltage-less files reserved from train/validation.
def holdoutCsvs(sourceLabel):
    csvPaths = [
        csvPath
        for csvPath in sourceCsvs(sourceLabel)
        if filenameVoltage(csvPath) is None
    ]
    return csvPaths


# Filters holdout files by the class expected from filenames.
def expectedHoldoutCsvs(expectedLabel, sourceLabel):
    csvPaths = holdoutCsvs(sourceLabel)
    if expectedLabel == "off":
        return [csvPath for csvPath in csvPaths if "off" in csvPath.stem.lower()]
    if expectedLabel == "good":
        return [csvPath for csvPath in csvPaths if "off" not in csvPath.stem.lower()]
    return csvPaths


# Prints holdout predictions using exact labels where possible.
def printHoldoutReport(model, labels):
    print("holdoutReport:")
    for expectedLabel, sourceLabel in holdoutGroups:
        csvPaths = expectedHoldoutCsvs(expectedLabel, sourceLabel)
        if len(csvPaths) == 0:
            print(f"{expectedLabel}: no holdout files")
            continue

        featureMatrix, sourceRows = csvFeatures(csvPaths)
        if featureMatrix is None:
            print(f"{expectedLabel}: no holdout windows")
            continue

        predIndexes = model.predict(featureMatrix)
        predLabels = np.array([labels[index] for index in predIndexes])
        if expectedLabel == "voltSag":
            correct = np.char.startswith(predLabels.astype(str), "voltSag")
        else:
            correct = predLabels == expectedLabel
        print(
            f"{expectedLabel}: {correct.sum()}/{correct.size} "
            f"windows correct ({correct.mean() * 100.0:.2f}%)")

        for csvPath in csvPaths:
            fileMask = np.array([path == csvPath for path in sourceRows])
            filePreds = predLabels[fileMask]
            fileLabels, counts = np.unique(filePreds, return_counts=True)
            summary = ", ".join(
                f"{label}:{count}" for label, count in zip(fileLabels, counts)
            )
            print(f"  {csvPath.name}: {summary}")


# Saves the validation confusion matrix as a PNG chart.
def saveConfusionPlot(matrix, labels):
    rowTotals = matrix.sum(axis=1, keepdims=True)
    percent = np.divide(
        matrix,
        rowTotals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=rowTotals != 0,
    ) * 100.0

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(percent, cmap="Blues", vmin=0.0, vmax=100.0)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("predicted label")
    ax.set_ylabel("actual label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if percent[row, col] >= 50.0 else "#222222"
            ax.text(
                col,
                row,
                f"{matrix[row, col]}\n{percent[row, col]:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("row percentage")
    fig.tight_layout()

    plotPath = chartDir / "confusion_matrix.png"
    fig.savefig(plotPath, dpi=180)
    plt.close(fig)
    return plotPath

################################################################################
# main functions                                                               #
################################################################################


# Trains the sklearn model and saves the model bundle.
def main():


    featureRows = []
    classRows = []
    valFeatureRows = []
    valClassRows = []
    labelFiles = {}


    # BUILD TRAINING AND VALIDATION FEATURE SETS
    for label, sourceLabel, targetVoltage in labelGroups:
        csvPaths = groupCsvs(sourceLabel, targetVoltage)
        if len(csvPaths) >= 1:
            labelFiles[label] = csvPaths

    labels = list(labelFiles.keys())
    if len(labels) < 2:
        raise SystemExit("need at least two labels with two or more CSV files each")

    print("active labels:", labels)

    for classIndex, label in enumerate(labels):
        trainCsvs, valCsvs = plannedCsvSplit(label, labelFiles[label])
        print(
            f"{label}: {len(trainCsvs)} train files, "
            f"{len(valCsvs)} validation files")
        print("  train:", ", ".join(path.name for path in trainCsvs))
        if len(valCsvs) > 0:
            print("  val:", ", ".join(path.name for path in valCsvs))

        addCsvFeatures(trainCsvs, classIndex, featureRows, classRows)
        addCsvFeatures(valCsvs, classIndex, valFeatureRows, valClassRows)


    # CORE ML TRANINING
    featureMatrix = np.vstack(featureRows).astype(np.float32)
    classVector = np.array(classRows, dtype=np.int64)
    labelIndexes = list(range(len(labels)))

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
    )
    model.fit(featureMatrix, classVector)

    trainAcc = model.score(featureMatrix, classVector)
    trainPred = model.predict(featureMatrix)


    # PRINT TRAINING RESULTS AND SAVE MODEL BUNDLE
    print("trainShape:", featureMatrix.shape)
    print("trainAcc:", f"{trainAcc * 100.0:.2f}%")
    print("trainConfusion:")
    print(
        confusion_matrix(
            classVector,
            trainPred,
            labels=labelIndexes,))

    print("trainReport:")
    print(
        classification_report(
            classVector,
            trainPred,
            labels=labelIndexes,
            target_names=labels,
            zero_division=0,))

    valAcc = None
    if len(valFeatureRows) > 0:
        valFeatureMatrix = np.vstack(valFeatureRows).astype(np.float32)
        valClassVector = np.array(valClassRows, dtype=np.int64)
        valAcc = model.score(valFeatureMatrix, valClassVector)
        valPred = model.predict(valFeatureMatrix)
        valMatrix = confusion_matrix(
            valClassVector,
            valPred,
            labels=labelIndexes,)

        print("valShape:", valFeatureMatrix.shape)
        print("valAcc:", f"{valAcc * 100.0:.2f}%")
        print("valConfusion:")
        print(valMatrix)
        print("valReport:")
        print(
            classification_report(
                valClassVector,
                valPred,
                labels=labelIndexes,
                target_names=labels,
                zero_division=0,))


        chartDir.mkdir(parents=True, exist_ok=True)
        plotPath = saveConfusionPlot(valMatrix, labels)
        print("confusionPlot:", plotPath)
    else:
        print("valShape: none")
        print("valAcc: no held-out files")

    bundle = {
        "model": model,
        "labelNames": labels,
        "sampleRate": sampleRate,
        "winSecs": winSecs,
        "stepSecs": stepSecs,
        "featureNames": signals.featureNames(),
        "featureMode": "motor_compact_low_order_bearing_orders",
        "modelMode": "compact",
        "trainAcc": trainAcc,
        "valAcc": valAcc,}

    modelPath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, modelPath)
    print("modelPath:", modelPath)
    printHoldoutReport(model, labels)


if __name__ == "__main__":
    main()
