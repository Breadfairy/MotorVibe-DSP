################################################################################
# Imports                                                                      #
################################################################################
from pathlib import Path
import os

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
labels = ["good", "voltSag", "obstruction"]
sampleRate = 1000.0
winSecs = 1.0
stepSecs = 0.25
valFileCount = 1

################################################################################
# helpers                                                                      #
################################################################################


# Finds the CSV recordings for one fixed class label.
def labelCsvs(label):
    classDir = trainDir / label
    if classDir.exists():
        return sorted(classDir.glob("*.csv"))
    return sorted(trainDir.glob(f"{label}*.csv"))


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


# Saves the validation confusion matrix as a PNG chart.
def saveConfusionPlot(matrix):
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


    # BUILD TRAINING AND VALIDATION FEATURE SETS
    for label in labels:
        csvPaths = labelCsvs(label)
        if len(csvPaths) < valFileCount + 1:
            raise SystemExit(
                f"need at least {valFileCount + 1} CSV files for label: {label}")

    for classIndex, label in enumerate(labels):
        csvPaths = labelCsvs(label)
        trainCsvs = csvPaths[:-valFileCount]
        valCsvs = csvPaths[-valFileCount:]

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
        plotPath = saveConfusionPlot(valMatrix)
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


if __name__ == "__main__":
    main()
