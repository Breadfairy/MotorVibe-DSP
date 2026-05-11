################################################################################
# Imports                                                                      #
################################################################################
import os
from pathlib import Path

buildDir = Path(__file__).resolve().parents[1]
mplConfigDir = buildDir / "outputs" / ".matplotlib"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(mplConfigDir))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data
import ml
import signals

################################################################################
# variables/constants                                                          #
################################################################################
# Training paths.
trainDir = buildDir / "data" / "training" / "main"
modelPath = buildDir / "outputs" / "models" / "motorPumpClassifier.joblib"
chartDir = buildDir / "outputs" / "ML-charts"
classNames = [
    "good",
    "bad_leak",
]

# Window settings.
sampleRate = 1000.0
winSecs = 1.0
stepSecs = 0.25

# Time-order split for each CSV.
trainFraction = 0.35
valFraction = 0.25
holdoutFraction = 1.0 - trainFraction - valFraction

################################################################################
# helpers                                                                      #
################################################################################


# Finds recordings for one class.
def classCsvs(className):
    classDir = trainDir / className
    if classDir.exists():
        return sorted(classDir.glob("*.csv"))
    return sorted(trainDir.glob(f"{className}*.csv"))


# Splits one CSV by time.
def splitFrame(df):
    rowCount = df.shape[0]
    trainEnd = int(rowCount * trainFraction)
    valEnd = trainEnd + int(rowCount * valFraction)

    trainDf = df.iloc[:trainEnd].reset_index(drop=True)
    valDf = df.iloc[trainEnd:valEnd].reset_index(drop=True)
    holdoutDf = df.iloc[valEnd:].reset_index(drop=True)
    return trainDf, valDf, holdoutDf


# Adds feature rows from one split.
def addFrameFeatures(df, classIndex, featureRows, classRows):
    windows = data.windowFrames(df, sampleRate, winSecs, stepSecs)

    for windowFrame in windows:
        featureRows.append(featureRowFromWindow(windowFrame))
        classRows.append(classIndex)


# Builds one compact feature row.
def featureRowFromWindow(windowFrame):
    return ml.featureRowFromWindow(windowFrame, sampleRate)


# Reads CSVs into the train, validation, and holdout lists.
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


# Converts rows into sklearn arrays.
def matrixFromRows(featureRows, classRows):
    featureMatrix = np.vstack(featureRows).astype(np.float32)
    classVector = np.array(classRows, dtype=np.int64)
    return featureMatrix, classVector


# Prints metrics for one split.
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


# Builds a confusion matrix for one split.
def buildConfusion(classifier, splitName):
    trueRows = []
    predRows = []

    for classIndex, className in enumerate(classNames):
        csvPaths = classCsvs(className)
        for csvPath in csvPaths:
            df = data.readCsv(csvPath)
            df = data.cleanFrame(df)
            trainDf, valDf, holdoutDf = splitFrame(df)
            if splitName == "holdout":
                splitDf = holdoutDf
            else:
                splitDf = valDf

            featureMatrix = classifier.featureMatrixFromFrame(splitDf, clean=False)
            pred = classifier.model.predict(featureMatrix)
            for predIndex in pred:
                trueRows.append(classIndex)
                predRows.append(int(predIndex))

    matrix = np.zeros((len(classNames), len(classNames)), dtype=np.int64)
    for index in range(len(trueRows)):
        matrix[trueRows[index], predRows[index]] += 1
    return matrix


# Saves one confusion matrix chart.
def plotConfusion(matrix, title, fileName):
    rowTotals = matrix.sum(axis=1, keepdims=True)
    percent = np.divide(
        matrix,
        rowTotals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=rowTotals != 0,
    ) * 100.0

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(percent, cmap="Blues", vmin=0.0, vmax=100.0)
    ax.set_title(title)
    ax.set_xlabel("predicted label")
    ax.set_ylabel("actual label")
    ax.set_xticks(np.arange(len(classNames)))
    ax.set_yticks(np.arange(len(classNames)))
    ax.set_xticklabels(classNames, rotation=35, ha="right")
    ax.set_yticklabels(classNames)

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
    fig.savefig(chartDir / fileName, dpi=180)
    plt.close(fig)


# Saves validation and holdout charts.
def saveMlCharts(bundle):
    chartDir.mkdir(parents=True, exist_ok=True)
    classifier = ml.MotorPumpClassifier(bundle)
    valMatrix = buildConfusion(classifier, "val")
    holdoutMatrix = buildConfusion(classifier, "holdout")

    valDf = pd.DataFrame(valMatrix, index=classNames, columns=classNames)
    holdoutDf = pd.DataFrame(holdoutMatrix, index=classNames, columns=classNames)
    valDf.to_csv(chartDir / "confusion_matrix_validation_counts.csv")
    holdoutDf.to_csv(chartDir / "confusion_matrix_holdout_counts.csv")
    plotConfusion(valMatrix, "Validation Confusion Matrix", "confusion_matrix.png")
    plotConfusion(
        valMatrix,
        "Validation Confusion Matrix",
        "confusion_matrix_validation.png",
    )
    plotConfusion(
        holdoutMatrix,
        "Holdout Confusion Matrix",
        "confusion_matrix_holdout.png",
    )

    print("chartDir:", chartDir)

################################################################################
# main functions                                                               #
################################################################################


# Trains the sklearn model and saves the model bundle.
def main():
    featureRows = []
    classRows = []
    valFeatureRows = []
    valClassRows = []
    holdoutFeatureRows = []
    holdoutClassRows = []

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

    modelValue = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
    )
    modelValue.fit(featureMatrix, classVector)

    trainAcc = printScores("train", modelValue, featureRows, classRows)
    valAcc = printScores("val", modelValue, valFeatureRows, valClassRows)
    holdoutAcc = printScores(
        "holdout",
        modelValue,
        holdoutFeatureRows,
        holdoutClassRows,
    )

    bundle = {
        "model": modelValue,
        "labelNames": classNames,
        "sampleRate": sampleRate,
        "winSecs": winSecs,
        "stepSecs": stepSecs,
        "featureNames": ml.featureNames(),
        "featureMode": "motor_compact_low_order_bearing_orders",
        "modelMode": "compact",
        "trainAcc": trainAcc,
        "valAcc": valAcc,
        "holdoutAcc": holdoutAcc,
        "trainFraction": trainFraction,
        "valFraction": valFraction,
        "holdoutFraction": holdoutFraction,
    }

    modelPath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, modelPath)
    saveMlCharts(bundle)
    print("modelPath:", modelPath)


if __name__ == "__main__":
    main()
