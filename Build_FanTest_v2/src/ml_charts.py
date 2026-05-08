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

import data
import signals

################################################################################
# variables/constants                                                          #
################################################################################
modelPath = buildDir / "outputs" / "models" / "fanClassifier.joblib"
inputDir = buildDir / "data" / "training" / "main"
outDir = buildDir / "outputs" / "ML-charts"
classNames = ["good", "voltSag"]
valFileCount = 1

################################################################################
# helpers                                                                      #
################################################################################


# Finds the CSV recordings for one fixed class label.
def csvPathsForClass(className):
    classDir = inputDir / className
    if classDir.exists():
        return sorted(classDir.glob("*.csv"))
    return sorted(inputDir.glob(f"{className}*.csv"))


# Selects the held-out validation CSVs for one class.
def validationCsvs(csvPaths):
    if len(csvPaths) <= valFileCount:
        return csvPaths
    return csvPaths[-valFileCount:]


# Converts one validation CSV into feature rows.
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
        freqData = signals.fftSignals(
            timeData,
            sampleRate,
            signals.fftConfig,
        )
        featureRows.append(signals.modelInput(timeData, freqData))

    return np.vstack(featureRows).astype(np.float32)


# Builds the held-out confusion matrix from validation CSV files.
def buildConfusion(bundle):
    modelValue = bundle["model"]
    trueRows = []
    predRows = []

    for classIndex, className in enumerate(classNames):
        csvPaths = validationCsvs(csvPathsForClass(className))
        for csvPath in csvPaths:
            featureMatrix = featureMatrixFromCsv(csvPath, bundle)
            pred = modelValue.predict(featureMatrix)
            for predIndex in pred:
                trueRows.append(classIndex)
                predRows.append(int(predIndex))

    matrix = np.zeros((len(classNames), len(classNames)), dtype=np.int64)
    for index in range(len(trueRows)):
        matrix[trueRows[index], predRows[index]] += 1
    return matrix

################################################################################
# plots                                                                        #
################################################################################


# Saves the confusion matrix chart as a PNG image.
def plotConfusion(matrix):
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
    fig.savefig(outDir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

################################################################################
# main functions                                                               #
################################################################################


# Builds and saves the only ML chart needed for this build.
def main():
    outDir.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(modelPath)
    matrix = buildConfusion(bundle)

    matrixDf = pd.DataFrame(matrix, index=classNames, columns=classNames)
    matrixDf.to_csv(outDir / "confusion_matrix_counts.csv")
    plotConfusion(matrix)

    print("outDir:", outDir)
    print("confusionMatrix:")
    print(matrix)


if __name__ == "__main__":
    main()
