################################################################################
# Imports                                                                      #
################################################################################
import os
from pathlib import Path

buildDir = Path(__file__).resolve().parents[1]
mplConfigDir = buildDir / "outputs" / ".matplotlib"

# Force a file-based matplotlib backend so this script can run without opening
# a GUI window.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(mplConfigDir))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import signals
import train

################################################################################
# variables/constants                                                          #
################################################################################
# The chart script evaluates the held-out validation files using the current
# saved model bundle.
modelPath = train.modelPath
outDir = buildDir / "outputs" / "ML-charts"
classNames = train.classNames

################################################################################
# helpers                                                                      #
################################################################################


# Converts one DataFrame split into feature rows.
# Uses the same feature building path as training and offline inference.
def featureMatrixFromFrame(df, bundle):
    sampleRate = bundle["sampleRate"]
    windows = data.windowFrames(
        df,
        sampleRate,
        bundle["winSecs"],
        bundle["stepSecs"],
    )

    featureRows = []

    # Predict at the same one-second window level used by training.
    for windowFrame in windows:
        if bundle.get("modelMode") == "raw":
            featureRows.append(signals.rawModelInput(windowFrame))
        else:
            timeData = signals.timeSignals(windowFrame, sampleRate)
            freqData = signals.fftSignals(
                timeData,
                sampleRate,
                signals.fftConfig,
            )
            featureRows.append(signals.modelInput(timeData, freqData))

    return np.vstack(featureRows).astype(np.float32)


# Builds a confusion matrix from either the validation or holdout split.
# Rows are true labels and columns are predicted labels.
def buildConfusion(bundle, splitName):
    modelValue = bundle["model"]
    trueRows = []
    predRows = []

    # Loop through each trusted main CSV, split it the same way as train.py,
    # then evaluate either its validation segment or its holdout segment.
    for classIndex, className in enumerate(classNames):
        csvPaths = train.classCsvs(className)
        for csvPath in csvPaths:
            df = data.readCsv(csvPath)
            df = data.cleanFrame(df)
            trainDf, valDf, holdoutDf = train.splitFrame(df)
            if splitName == "holdout":
                splitDf = holdoutDf
            else:
                splitDf = valDf

            featureMatrix = featureMatrixFromFrame(splitDf, bundle)
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
# The image is shown as row percentages, while the text inside each cell also
# includes the raw window count.
def plotConfusion(matrix, title, fileName):
    # Convert counts into percentages inside each true-label row.
    rowTotals = matrix.sum(axis=1, keepdims=True)
    percent = np.divide(
        matrix,
        rowTotals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=rowTotals != 0,
    ) * 100.0

    # Draw the matrix and label the axes with the class names.
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(percent, cmap="Blues", vmin=0.0, vmax=100.0)
    ax.set_title(title)
    ax.set_xlabel("predicted label")
    ax.set_ylabel("actual label")
    ax.set_xticks(np.arange(len(classNames)))
    ax.set_yticks(np.arange(len(classNames)))
    ax.set_xticklabels(classNames, rotation=35, ha="right")
    ax.set_yticklabels(classNames)

    # Put both the count and percentage in each square.
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
    fig.savefig(outDir / fileName, dpi=180)
    plt.close(fig)

################################################################################
# main functions                                                               #
################################################################################


# Builds and saves the only ML chart needed for this build.
def main():
    # Load the model, build validation and holdout matrices, then save charts.
    outDir.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(modelPath)
    valMatrix = buildConfusion(bundle, "val")
    holdoutMatrix = buildConfusion(bundle, "holdout")

    valDf = pd.DataFrame(valMatrix, index=classNames, columns=classNames)
    holdoutDf = pd.DataFrame(holdoutMatrix, index=classNames, columns=classNames)
    valDf.to_csv(outDir / "confusion_matrix_validation_counts.csv")
    holdoutDf.to_csv(outDir / "confusion_matrix_holdout_counts.csv")
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

    print("splitFractions:")
    print("train:", train.trainFraction)
    print("val:", train.valFraction)
    print("holdout:", train.holdoutFraction)

    print("outDir:", outDir)
    print("valConfusionMatrix:")
    print(valMatrix)
    print("holdoutConfusionMatrix:")
    print(holdoutMatrix)


if __name__ == "__main__":
    main()
