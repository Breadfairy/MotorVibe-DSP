################################################################################
# Imports                                                                      #
################################################################################
import os
from pathlib import Path

# Matplotlib handling 
buildDir = Path(__file__).resolve().parents[1]
mplConfigDir = buildDir / "outputs" / ".matplotlib"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(mplConfigDir))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data
import ml

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

# Time-order split for:wq each CSV.
trainFraction = 0.35
valFraction = 0.25
holdoutFraction = 1.0 - trainFraction - valFraction

################################################################################
# helpers                                                                      #
################################################################################


# Finds recordings for one class label.
def findCsvs(labelName):
    labelDir = trainDir / labelName
    if labelDir.exists():
        return sorted(labelDir.glob("*.csv"))
    return sorted(trainDir.glob(f"{labelName}*.csv"))


# Splits one CSV by time.
def splitByTime(capture):
    sampleCount = capture.shape[0]
    trainEnd = int(sampleCount * trainFraction)
    valEnd = trainEnd + int(sampleCount * valFraction)

    trainCapture = capture.iloc[:trainEnd].reset_index(drop=True)
    valCapture = capture.iloc[trainEnd:valEnd].reset_index(drop=True)
    holdoutCapture = capture.iloc[valEnd:].reset_index(drop=True)
    return trainCapture, valCapture, holdoutCapture


# Adds one dataframe split to one feature/label list pair.
def addWindowFeatures(capture, label, features, labels):
    oneSecondWindows = data.windowsFromCapture(
        capture,
        sampleRate,
        winSecs,
        stepSecs,
    )
    for oneSecondWindow in oneSecondWindows:
        windowFeatures = ml.featuresFromWindow(
            oneSecondWindow,
            sampleRate,
        )
        features.append(windowFeatures)
        labels.append(label)


# Converts window features and raw labels into sklearn arrays.
def makeDataset(features, rawLabels):
    featureMatrix = np.vstack(features).astype(np.float32)
    labels = np.array(rawLabels, dtype=np.int64)
    return featureMatrix, labels


# Scores one split without printing a full report.
def scoreDataset(model, dataset):
    if len(dataset["features"]) == 0:
        return None

    featureMatrix, labels = makeDataset(
        dataset["features"],
        dataset["labels"],
    )
    return model.score(featureMatrix, labels)


# Builds the final confusion matrix from one split.
def confusionForDataset(model, dataset):
    featureMatrix, labels = makeDataset(
        dataset["features"],
        dataset["labels"],
    )
    pred = model.predict(featureMatrix)
    return confusion_matrix(
        labels,
        pred,
        labels=list(range(len(classNames))),
    )


# Converts counts into row percentages.
def rowPercentMatrix(matrix):
    rowTotals = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        rowTotals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=rowTotals != 0,
    ) * 100.0


# Saves one row-percentage confusion matrix chart.
def plotConfusion(matrix, title, fileName):
    percentMatrix = rowPercentMatrix(matrix)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(percentMatrix, cmap="Blues", vmin=0.0, vmax=100.0)
    ax.set_title(title)
    ax.set_xlabel("predicted label")
    ax.set_ylabel("actual label")
    ax.set_xticks(np.arange(len(classNames)))
    ax.set_yticks(np.arange(len(classNames)))
    ax.set_xticklabels(classNames, rotation=35, ha="right")
    ax.set_yticklabels(classNames)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if percentMatrix[row, col] >= 50.0 else "#222222"
            ax.text(
                col,
                row,
                f"{matrix[row, col]}\n{percentMatrix[row, col]:.1f}%",
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


# Saves the final confusion matrix chart, counts, and percentages.
def saveConfusionChart(matrix, splitName):
    chartDir.mkdir(parents=True, exist_ok=True)
    matrixDf = pd.DataFrame(matrix, index=classNames, columns=classNames)
    matrixDf.to_csv(chartDir / "confusion_matrix_counts.csv")

    percentDf = pd.DataFrame(
        rowPercentMatrix(matrix),
        index=classNames,
        columns=classNames,
    )
    percentDf.to_csv(chartDir / "confusion_matrix_percent.csv")

    plotConfusion(
        matrix,
        f"{splitName.title()} Confusion Matrix",
        "confusion_matrix.png",
    )

    print("chartDir:", chartDir)

################################################################################
# main functions                                                               #
################################################################################


# Trains the sklearn model and saves the model bundle.
def main():
    # Collect window features and labels separately for each time split.
    datasets = {
        "train": {"features": [], "labels": []},
        "val": {"features": [], "labels": []},
        "holdout": {"features": [], "labels": []},
    }

    # Read each labelled CSV, split it by time, then extract window features.
    for label, labelName in enumerate(classNames):
        csvFiles = findCsvs(labelName)
        if len(csvFiles) == 0:
            raise SystemExit(
                f"no training CSVs found for {labelName} in {trainDir}"
            )

        for csvPath in csvFiles:
            capture = data.cleanCapture(data.readCsv(csvPath))
            trainCapture, valCapture, holdoutCapture = splitByTime(capture)
            capturesBySplit = {
                "train": trainCapture,
                "val": valCapture,
                "holdout": holdoutCapture,
            }
            for splitName, splitCapture in capturesBySplit.items():
                dataset = datasets[splitName]
                addWindowFeatures(
                    splitCapture,
                    label,
                    dataset["features"],
                    dataset["labels"],
                )

    # Convert the training portions into sklearn arrays.
    featureMatrix, labels = makeDataset(
        datasets["train"]["features"],
        datasets["train"]["labels"],
    )

    # Build the complete sklearn model pipeline.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
    )

    # Train the scaler and classifier from the labelled window features.
    model.fit(featureMatrix, labels)

    # Score each split, but keep console output compact.
    trainAcc = scoreDataset(model, datasets["train"])
    valAcc = scoreDataset(model, datasets["val"])
    holdoutAcc = scoreDataset(model, datasets["holdout"])

    # Use the most independent available split for the final confusion matrix.
    finalSplitName = "holdout"
    if len(datasets[finalSplitName]["features"]) == 0:
        finalSplitName = "val"
    if len(datasets[finalSplitName]["features"]) == 0:
        finalSplitName = "train"
    finalMatrix = confusionForDataset(model, datasets[finalSplitName])

    # Print a short training summary for the terminal.
    print("featureShape:", featureMatrix.shape)
    print("trainAcc:", f"{trainAcc * 100.0:.2f}%")
    if valAcc is not None:
        print("valAcc:", f"{valAcc * 100.0:.2f}%")
    else:
        print("valAcc: no features")
    if holdoutAcc is not None:
        print("holdoutAcc:", f"{holdoutAcc * 100.0:.2f}%")
    else:
        print("holdoutAcc: no features")
    print(f"finalConfusionSplit: {finalSplitName}")
    print("finalConfusion:")
    print(finalMatrix)

    # Bundle the trained model with the metadata needed by live inference.
    bundle = {
        "model": model,
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

    # Save the trained model and the final confusion chart.
    modelPath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, modelPath)
    saveConfusionChart(finalMatrix, finalSplitName)
    print("modelPath:", modelPath)


if __name__ == "__main__":
    main()
