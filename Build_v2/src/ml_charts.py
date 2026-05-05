################################################################################
# Imports                                                                      #
################################################################################
import os
from pathlib import Path

buildDir = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(buildDir / "outputs" / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import data
import signals

################################################################################
# variables/constants                                                          #
################################################################################
modelPath = buildDir / "outputs" / "models" / "rawAxisClassifier.pth"
inputDir = buildDir / "data" / "training" / "main"
outDir = buildDir / "outputs" / "ML-charts"
classColors = ["#2f6fdd", "#d1495b", "#2a9d8f", "#f4a261", "#8e5ea2", "#6c757d"]

################################################################################
# helpers                                                                      #
################################################################################


def loadBundle(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def trueLabel(csvPath, labelNames):
    stem = csvPath.stem.lower()
    for labelName in sorted(labelNames, key=len, reverse=True):
        if stem.startswith(labelName.lower()):
            return labelName
    raise ValueError(f"cannot infer label from {csvPath.name}")


def windowProbabilities(csvPath, bundle, modelValue):
    df = data.cleanFrame(data.readCsv(csvPath))
    windows = data.windowFrames(
        df,
        bundle["sampleRate"],
        bundle["winSecs"],
        bundle["stepSecs"],
    )

    featureRows = []
    for windowFrame in windows:
        x = signals.timeSignals(windowFrame, bundle["sampleRate"])
        y = signals.fftSignals(x, bundle["sampleRate"], signals.fftConfig)
        featureRows.append(signals.modelInput(x, y))

    featureMatrix = np.vstack(featureRows).astype(np.float32)
    scaledMatrix = (featureMatrix - bundle["featureMean"]) / bundle["featureStd"]
    featureTensor = torch.tensor(scaledMatrix, dtype=torch.float32)

    with torch.no_grad():
        outputTensor = modelValue(featureTensor)
        probTensor = torch.softmax(outputTensor, dim=1)

    return probTensor.detach().cpu().numpy(), windows


def confusionMatrix(windowDf, labelNames):
    labelToIndex = {name: i for i, name in enumerate(labelNames)}
    matrix = np.zeros((len(labelNames), len(labelNames)), dtype=np.int64)
    for row in windowDf.itertuples():
        matrix[labelToIndex[row.true_label], labelToIndex[row.predicted_label]] += 1
    return matrix


def classAccuracy(windowDf, fileDf, labelNames):
    matrix = confusionMatrix(windowDf, labelNames)
    rows = []
    for i, labelName in enumerate(labelNames):
        fileRows = fileDf[fileDf["true_label"] == labelName]
        windowTotal = int(matrix[i].sum())
        windowCorrect = int(matrix[i, i])
        fileTotal = int(fileRows.shape[0])
        fileCorrect = int(fileRows["averaged_correct"].sum())
        rows.append(
            {
                "label": labelName,
                "window_total": windowTotal,
                "window_correct": windowCorrect,
                "window_accuracy": windowCorrect / windowTotal if windowTotal else 0.0,
                "file_total": fileTotal,
                "file_correct_after_averaging": fileCorrect,
                "file_accuracy_after_averaging": (
                    fileCorrect / fileTotal if fileTotal else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)

################################################################################
# evaluation                                                                   #
################################################################################


def evaluate():
    bundle = loadBundle(modelPath)
    labelNames = list(bundle["labelNames"])
    labelToIndex = {name: i for i, name in enumerate(labelNames)}

    modelValue = nn.Linear(bundle["inputSize"], len(labelNames))
    modelValue.load_state_dict(bundle["stateDict"])
    modelValue.eval()

    windowRows = []
    fileRows = []

    for csvPath in sorted(inputDir.glob("*.csv")):
        actualLabel = trueLabel(csvPath, labelNames)
        actualIndex = labelToIndex[actualLabel]
        probMatrix, windows = windowProbabilities(csvPath, bundle, modelValue)
        predIndexes = np.argmax(probMatrix, axis=1)
        topProbs = np.max(probMatrix, axis=1)
        meanProbs = probMatrix.mean(axis=0)
        meanPredIndex = int(np.argmax(meanProbs))

        fileRows.append(
            {
                "file": csvPath.name,
                "true_label": actualLabel,
                "window_count": int(len(windows)),
                "averaged_predicted_label": labelNames[meanPredIndex],
                "averaged_top_prob": float(meanProbs[meanPredIndex]),
                "averaged_correct": bool(meanPredIndex == actualIndex),
                "window_accuracy": float(np.mean(predIndexes == actualIndex)),
            }
        )

        for windowIndex, predIndex in enumerate(predIndexes):
            windowFrame = windows[windowIndex]
            row = {
                "file": csvPath.name,
                "true_label": actualLabel,
                "true_index": int(actualIndex),
                "window_index": int(windowIndex),
                "start_s": float(windowFrame["t_s"].iloc[0]),
                "end_s": float(windowFrame["t_s"].iloc[-1]),
                "predicted_label": labelNames[int(predIndex)],
                "predicted_index": int(predIndex),
                "top_prob": float(topProbs[windowIndex]),
                "true_class_prob": float(probMatrix[windowIndex, actualIndex]),
                "correct": bool(predIndex == actualIndex),
            }
            for labelIndex, labelName in enumerate(labelNames):
                row[f"{labelName}_prob"] = float(probMatrix[windowIndex, labelIndex])
            windowRows.append(row)

    return pd.DataFrame(windowRows), pd.DataFrame(fileRows), labelNames

################################################################################
# plots                                                                        #
################################################################################


def plotConfusion(matrix, labelNames):
    rowTotals = matrix.sum(axis=1, keepdims=True)
    percent = np.divide(
        matrix,
        rowTotals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=rowTotals != 0,
    ) * 100.0

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(percent, cmap="Blues", vmin=0.0, vmax=100.0)
    ax.set_title("Per-Window Confusion Matrix")
    ax.set_xlabel("predicted label")
    ax.set_ylabel("actual label")
    ax.set_xticks(np.arange(len(labelNames)))
    ax.set_yticks(np.arange(len(labelNames)))
    ax.set_xticklabels(labelNames, rotation=35, ha="right")
    ax.set_yticklabels(labelNames)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            colour = "white" if percent[row, col] >= 50.0 else "#222222"
            ax.text(
                col,
                row,
                f"{matrix[row, col]}\n{percent[row, col]:.1f}%",
                ha="center",
                va="center",
                color=colour,
                fontsize=8,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("row percentage")
    fig.tight_layout()
    fig.savefig(outDir / "01_confusion_matrix.png", dpi=180)
    plt.close(fig)


def plotPerFile(fileDf, labelNames):
    labelToIndex = {name: i for i, name in enumerate(labelNames)}
    x = np.arange(fileDf.shape[0])
    accuracy = fileDf["window_accuracy"].to_numpy(dtype=np.float64) * 100.0
    averagedProb = fileDf["averaged_top_prob"].to_numpy(dtype=np.float64) * 100.0
    colours = [classColors[labelToIndex[name]] for name in fileDf["true_label"]]
    edgeColours = ["#222222" if ok else "#d1495b" for ok in fileDf["averaged_correct"]]

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(x, accuracy, color=colours, edgecolor=edgeColours, linewidth=1.4)
    ax.scatter(
        x,
        averagedProb,
        color="#111111",
        marker="D",
        s=28,
        label="averaged top probability",
        zorder=3,
    )

    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(103.0, bar.get_height() + 2.0),
            f"{accuracy[i]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title("Per-File Window Accuracy and Averaged Inference Confidence")
    ax.set_xlabel("file")
    ax.set_ylabel("percentage")
    ax.set_ylim(0.0, 108.0)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [Path(name).stem for name in fileDf["file"]],
        rotation=45,
        ha="right",
    )
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(outDir / "02_per_file_accuracy.png", dpi=180)
    plt.close(fig)

################################################################################
# main functions                                                               #
################################################################################


def main():
    outDir.mkdir(parents=True, exist_ok=True)
    for chartPath in outDir.glob("[0-9][0-9]_*.png"):
        chartPath.unlink()

    windowDf, fileDf, labelNames = evaluate()
    matrix = confusionMatrix(windowDf, labelNames)
    classDf = classAccuracy(windowDf, fileDf, labelNames)
    rowTotals = matrix.sum(axis=1, keepdims=True)
    rowFractions = np.divide(
        matrix,
        rowTotals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=rowTotals != 0,
    )

    windowDf.to_csv(outDir / "window_predictions.csv", index=False)
    fileDf.to_csv(outDir / "file_summary.csv", index=False)
    classDf.to_csv(outDir / "class_accuracy.csv", index=False)
    pd.DataFrame(matrix, index=labelNames, columns=labelNames).to_csv(
        outDir / "confusion_matrix_counts.csv"
    )
    pd.DataFrame(rowFractions, index=labelNames, columns=labelNames).to_csv(
        outDir / "confusion_matrix_row_fraction.csv"
    )

    with (outDir / "metrics_summary.txt").open("w", encoding="utf-8") as outFile:
        outFile.write("Per-class window accuracy\n")
        for row in classDf.itertuples():
            outFile.write(
                f"{row.label}: {row.window_accuracy * 100.0:.2f}% "
                f"({row.window_correct}/{row.window_total})\n"
            )
        outFile.write("\nPer-class file accuracy after averaging 1-second windows\n")
        for row in classDf.itertuples():
            outFile.write(
                f"{row.label}: {row.file_accuracy_after_averaging * 100.0:.2f}% "
                f"({row.file_correct_after_averaging}/{row.file_total})\n"
            )

    plotConfusion(matrix, labelNames)
    plotPerFile(fileDf, labelNames)

    print("outDir:", outDir)
    print("")
    print("Per-class window accuracy:")
    for row in classDf.itertuples():
        print(
            f"{row.label:7s}",
            f"{row.window_accuracy * 100.0:6.2f}%",
            f"({row.window_correct}/{row.window_total})",
        )
    print("")
    print("Per-file averaged inference:")
    for row in fileDf.itertuples():
        print(
            f"{row.file:12s}",
            f"actual={row.true_label:7s}",
            f"avgPred={row.averaged_predicted_label:7s}",
            f"avgProb={row.averaged_top_prob:.3f}",
            f"windowAcc={row.window_accuracy * 100.0:6.2f}%",
        )


if __name__ == "__main__":
    main()
